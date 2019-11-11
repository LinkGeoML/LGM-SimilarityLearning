import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from similarity_learning.config import DirConf
from similarity_learning.model import SimilarityV2
from similarity_learning.sample import Sampler
from similarity_learning.utils_text import NgramTokenizer

pd.set_option('display.expand_frame_repr', False)

tqdm.pandas()


class Trainer:

    def __init__(self,
                 fname='all_countries_filtered_rows_&_columns.csv',
                 verbose=1,
                 maxlen=30,
                 max_chars=32,
                 num_words=20_000,
                 n_rows=10_000,
                 save_tokenizer=False,
                 val_size=0.25,
                 tokenizer_params: dict = None
                 ):
        """

        Parameters
        ----------
        fname
        verbose
        maxlen
        max_chars : Number of characters for the first filtering of the toponyms
        num_words
        n_rows
        save_tokenizer
        """
        self.path = os.path.join(DirConf.DATA_DIR, fname)
        assert os.path.exists(self.path)

        self.verbose = verbose
        self.maxlen = maxlen
        self.max_chars = max_chars
        self.num_words = num_words
        self.n_rows = n_rows
        self.save_tokenizer = save_tokenizer

        self.data: Optional[pd.DataFrame] = None
        self.cols = ['name', 'alternate_names']
        self.val_size = val_size

        self.train = None
        self.val = None
        self.test = None

        if tokenizer_params is None:
            self.tokenizer_params = dict(
                maxlen=self.maxlen, filters='', lower=True, split=' ',
                char_level=False, num_words=self.num_words, oov_token='<OOV>')

        self.tokenizer = None

        self.train_sampler = None
        self.val_sampler = None

        self.model = None

    def load_data(self):
        """

        Returns
        -------

        """
        if self.data is None:
            self.data = pd.read_csv(
                self.path, nrows=self.n_rows, usecols=self.cols)
            self.data.dropna(inplace=True)
            self.data['name'] = self.data['name'].str.lower()
            self.data['alternate_names'] = self.data[
                'alternate_names'].str.lower()
            self.data['n_alternate_names'] = self.data[
                'alternate_names'].str.split(',').progress_apply(len)

            self.data['len_name'] = self.data['name'].str.len()
            # get rid of toponyms that have more than 32 characters.
            self.data = self.data[
                (self.data['len_name'] <= self.max_chars)
                & (self.data['len_name'] > 2)].reset_index(drop=True)

        return self.data

    def split_data(self):
        """

        Returns
        -------

        """
        if not self.data.empty:

            self.data['first_char'] = self.data['name'].str[0]

            test_index = len(self.data) // 2

            train_val = self.data[:test_index]
            test = self.data[test_index:]

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_size, random_state=0)

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

        Returns
        -------

        """
        tokenizer = NgramTokenizer(**self.tokenizer_params)

        # convert each toponym to it's ngram representation
        self.train['toponym_ngrams'] = self.train['name'].progress_apply(
            tokenizer.get_ngrams)

        # convert each variation of each toponym to it's n-gram representation
        self.train['alternate_names'] = self.train[
            'alternate_names'].str.split(',')

        self.train['variations_ngrams'] = self.train[
            'alternate_names'].progress_apply(tokenizer.texts_to_ngrams)

        # collect (flatten out) all the n-grams (toponyms and variations)
        # these are needed in order to fit it to the tokenizer.
        all_train_names = list()
        for row in tqdm(self.train['variations_ngrams']):
            all_train_names.extend(row)

        all_train_names += list(self.train['toponym_ngrams'])

        # fitting all the texts on the instantiated tokenizer
        # this will create all the necessary tools that we will need.
        tokenizer.fit_on_texts(texts=all_train_names)

        # using the fitted tokenizer, convert the train toponyms to sequences
        self.train['toponym_seqs'] = self.train[
            'toponym_ngrams'].progress_apply(
            lambda x: tokenizer.texts_to_sequences(texts=[x])[0])

        # pad the sequence to the max length
        self.train['toponym_seqs'] = self.train['toponym_seqs'].progress_apply(
            tokenizer.pad_single)

        # using the fitted tokenizer, convert each of the variations of all the
        # toponyms sequences
        self.train['variations_seqs'] = self.train[
            'variations_ngrams'].progress_apply(
            tokenizer.texts_to_sequences)

        self.train['variations_seqs'] = self.train[
            'variations_seqs'].progress_apply(
            tokenizer.pad)

        #  ========== Same Procedure for the Validation Set ===================
        self.val['toponym_ngrams'] = self.val['name'].progress_apply(
            tokenizer.get_ngrams)

        self.val['alternate_names'] = self.val['alternate_names'].str.split(
            ',')

        self.val['variations_ngrams'] = self.val[
            'alternate_names'].progress_apply(tokenizer.texts_to_ngrams)

        self.val['toponym_seqs'] = self.val['toponym_ngrams'].progress_apply(
            lambda x: tokenizer.texts_to_sequences(texts=[x])[0])

        self.val['toponym_seqs'] = self.val['toponym_seqs'].progress_apply(
            tokenizer.pad_single)

        self.val['variations_seqs'] = self.val[
            'variations_ngrams'].progress_apply(
            tokenizer.texts_to_sequences)

        self.val['variations_seqs'] = self.val[
            'variations_seqs'].progress_apply(
            tokenizer.pad)

        if self.verbose > 0:
            print(f'N-gram index length: {len(tokenizer.word_index)}',
                  end='\n\n')
            print('Example Transformation')
            print(self.train.loc[0])

            print(self.train.loc[0]['variations_seqs'])

        if self.save_tokenizer:
            print('Saving Tokenizer')
            tokenizer.save()
            print('Tokenizer saved')

    def build_samplers(self):
        """

        Returns
        -------

        """
        batch_size = 1024

        if self.train_sampler is None and self.val_sampler is None:
            train_params = {'data': self.train,
                            'n_positives': 1,
                            'n_negatives': 3,
                            'neg_samples_size': 30,
                            'batch_size': batch_size,
                            'shuffle': True}

            val_params = {'data': self.val,
                          'n_positives': 1,
                          'n_negatives': 3,
                          'neg_samples_size': 30,
                          'batch_size': batch_size,
                          'shuffle': True}

            self.train_sampler = Sampler(**train_params)
            self.val_sampler = Sampler(**val_params)
            self.batch_size = batch_size

        return self.train_sampler, self.val_sampler

    def build_model(self):
        """

        Returns
        -------

        """
        self.model = SimilarityV2()

        self.model.build(max_features=self.num_words,
                         maxlen=self.maxlen,
                         emb_dim=300,
                         n_hidden=100)

        self.model.compile()

        history = self.model.fit(train_gen=self.train_sampler,
                                 val_gen=self.val_sampler,
                                 batch_size=self.batch_size,
                                 e=3,
                                 multi_process=False)

        return history

    def run(self):
        """

        Returns
        -------

        """
        self.load_data()
        self.split_data()
        self.tokenize_data()
        self.build_samplers()
        self.build_model()


if __name__ == "__main__":
    trainer = Trainer(save_tokenizer=True,
                      n_rows=None,
                      maxlen=30,
                      verbose=0,
                      num_words=50_000)
    trainer.run()
