import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import similarity_learning.tokenize as tokenizers
from similarity_learning import sample as samplers
from similarity_learning.config import DirConf
from similarity_learning.utils import underscore_to_camel

pd.set_option('display.expand_frame_repr', False)
tqdm.pandas()


class Dataset:

    def __init__(self,
                 name: str = 'all_countries_filtered_rows_&_columns.csv',
                 verbose: int = 0,
                 max_chars: int = 32,
                 n_rows: int = 10_000,
                 save_tokenizer: bool = True,
                 val_size: float = 0.25,
                 tokenizer_params: Optional[dict] = None,
                 train_sampler_params: Optional[dict] = None,
                 val_sampler_params: Optional[dict] = None):
        """

        Parameters
        ----------
        name
        verbose
        max_chars
        n_rows
        save_tokenizer
        val_size
        tokenizer_params
        """
        self.path = os.path.join(DirConf.DATA_DIR, name)
        assert os.path.exists(self.path)
        assert isinstance(n_rows, int)
        assert n_rows > -2

        if n_rows == -1:
            n_rows = None

        self.verbose = verbose
        self.max_chars = max_chars

        self.n_rows = n_rows
        self.save_tokenizer = save_tokenizer

        self.data: Optional[pd.DataFrame] = None
        self.cols = ['name', 'alternate_names']
        self.val_size = val_size

        self.train = None
        self.val = None
        self.test = None

        self.tokenizer_params = {'name': 'ngram_tokenizer', 'maxlen': 30,
                                 'filters': '', 'lower': True, 'split': ' ',
                                 'char_level': False, 'num_words': 20_000,
                                 'oov_token': '<OOV>'}

        if tokenizer_params:
            self.tokenizer_params.update(tokenizer_params)

        TokenizerCLass = getattr(tokenizers,
                                 underscore_to_camel(
                                     self.tokenizer_params.pop('name')))
        self.tokenizer = TokenizerCLass(**self.tokenizer_params)

        self.train_sampler_params = train_sampler_params
        self.val_sampler_params = val_sampler_params

        self.train_sampler = None
        self.val_sampler = None

        self.model = None

    def load_data(self):
        """

        Returns
        -------

        """
        if self.data is None:
            self.data = pd.read_csv(self.path, nrows=self.n_rows,
                                    usecols=self.cols)

            self.data.dropna(inplace=True)
            self.data['name'] = self.data['name'].str.lower()

            self.data['alternate_names'] = self.data[
                'alternate_names'].str.lower()

            self.data['n_alternate_names'] = self.data[
                'alternate_names'].str.split(',').apply(len)

            self.data['len_name'] = self.data['name'].str.len()

            # get rid of toponyms that have more than "max_chars" characters.
            self.data = self.data[
                (self.data['len_name'] <= self.max_chars)
                & (self.data['len_name'] > 2)].reset_index(drop=True)

        return self.data

    def split_data(self):
        """
        Splits the data into training and validation and test.
        Returns
        -------

        """
        if not self.data.empty:

            self.data['first_char'] = self.data['name'].str[0]

            test_index = len(self.data) // 2

            train_val = self.data[:test_index]
            test = self.data[test_index:]

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size,
                                         random_state=0)

            train, val, y_train, y_val = None, None, None, None

            cols_to_be_used = self.cols + ['len_name']

            X = train_val[cols_to_be_used].copy()
            y = train_val[['len_name']]  # 'first_char'

            # split the data based on the number of characters of the toponym
            # and their initial character
            for train_index, val_index in sss.split(X, y):
                train, val = X.loc[train_index], X.loc[val_index]
                # don't really need the y's
                # y_train, y_val = y.loc[train_index], y.loc[val_index]

            train.reset_index(drop=True, inplace=True)
            val.reset_index(drop=True, inplace=True)
            test = test[cols_to_be_used].reset_index(drop=True)

            self.train = train
            self.val = val
            self.test = test

    def tokenize_data(self):
        """
        1) Creates n-grams from the toponyms
        2) Creates n-grams for each of the variations
        Returns
        -------

        """
        # convert each toponym to it's ngram representation
        self.train['toponym_ngrams'] = self.train['name'].progress_apply(
            self.tokenizer.get_ngrams)

        # convert each variation of each toponym to it's n-gram representation
        self.train['alternate_names'] = self.train[
            'alternate_names'].str.split(',')

        self.train['variations_ngrams'] = self.train[
            'alternate_names'].progress_apply(self.tokenizer.texts_to_ngrams)

        # collect (flatten out) all the n-grams (toponyms and variations)
        # these are needed in order to fit it to the tokenizer.
        all_train_names = list()
        for row in self.train['variations_ngrams']:
            all_train_names.extend(row)

        all_train_names += list(self.train['toponym_ngrams'])

        # fitting all the training texts on the instantiated tokenizer
        # this will create all the necessary tools that we will need.
        self.tokenizer.fit_on_texts(texts=all_train_names)

        # using the fitted tokenizer, convert the train toponyms to sequences
        self.train['toponym_seqs'] = self.train[
            'toponym_ngrams'].progress_apply(
            lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

        # pad the sequences to the max length
        self.train['toponym_seqs'] = self.train[
            'toponym_seqs'].progress_apply(self.tokenizer.pad_single)

        # using the fitted tokenizer, convert each of the variations of all the
        # toponyms sequences
        self.train['variations_seqs'] = self.train[
            'variations_ngrams'].progress_apply(
            self.tokenizer.texts_to_sequences)

        self.train['variations_seqs'] = self.train[
            'variations_seqs'].progress_apply(self.tokenizer.pad)

        #  ========== Same Procedure for the Validation Set ===================
        self.val['toponym_ngrams'] = self.val['name'].progress_apply(
            self.tokenizer.get_ngrams)

        self.val['alternate_names'] = self.val[
            'alternate_names'].str.split(',')

        self.val['variations_ngrams'] = self.val[
            'alternate_names'].progress_apply(self.tokenizer.texts_to_ngrams)

        self.val['toponym_seqs'] = self.val['toponym_ngrams'].progress_apply(
            lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

        self.val['toponym_seqs'] = self.val['toponym_seqs'].progress_apply(
            self.tokenizer.pad_single)

        self.val['variations_seqs'] = self.val[
            'variations_ngrams'].progress_apply(
            self.tokenizer.texts_to_sequences)

        self.val['variations_seqs'] = self.val[
            'variations_seqs'].progress_apply(self.tokenizer.pad)

        if self.verbose > 0:
            print(f'N-gram index length: {len(self.tokenizer.word_index)}',
                  end='\n\n')
            print('Example Transformation')
            print(self.train.loc[0])

            print(self.train.loc[0]['variations_seqs'])

        if self.save_tokenizer:
            print('Saving Tokenizer')
            self.tokenizer.save()
            print('Tokenizer saved')

    def create_samplers(self):
        """

        Returns
        -------

        """

        train_params = {'data': self.train,
                        'name': 'sampler',
                        'n_positives': 1,
                        'n_negatives': 3,
                        'neg_samples_size': 30,
                        'batch_size': 2048,
                        'shuffle': True}

        val_params = {'data': self.val,
                      'name': 'sampler',
                      'n_positives': 1,
                      'n_negatives': 3,
                      'neg_samples_size': 30,
                      'batch_size': 2048,
                      'shuffle': True}

        if self.train_sampler_params:
            train_params.update(self.train_sampler_params)

        if self.val_sampler_params:
            val_params.update(self.val_sampler_params)

        if self.train_sampler is None and self.val_sampler is None:
            TrainSamplerClass = getattr(
                samplers, underscore_to_camel(train_params.pop('name')))

            ValSamplerClass = getattr(
                samplers, underscore_to_camel(val_params.pop('name')))

            self.train_sampler = TrainSamplerClass(**train_params)
            self.val_sampler = ValSamplerClass(**val_params)

        return self.train_sampler, self.val_sampler

    def run_data_preparation(self):
        """

        Returns
        -------

        """
        self.load_data()
        self.split_data()
        self.tokenize_data()
        self.create_samplers()


if __name__ == "__main__":
    trainer = Dataset(
        name='unbiased_preshuffled_dataset-string-similarity-global-train-original.csv',
        save_tokenizer=True,
        n_rows=10000,
        verbose=0)
    trainer.run_data_preparation()
