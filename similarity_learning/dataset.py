import os
from typing import Optional

import pandas as pd
from tqdm import tqdm

import similarity_learning.tokenize as tokenizers
from similarity_learning import sample as samplers
from similarity_learning.config import DirConf
from similarity_learning.logger import exp_logger as logger
from similarity_learning.utils import underscore_to_camel

pd.set_option('display.expand_frame_repr', False)
tqdm.pandas()


class Dataset:

    def __init__(self,
                 train_fname: str,
                 val_fname: str,
                 verbose: int = 0,
                 max_chars: int = 32,
                 save_tokenizer: bool = True,
                 tokenizer_params: Optional[dict] = None,
                 train_sampler_params: Optional[dict] = None,
                 val_sampler_params: Optional[dict] = None):
        """

        Parameters
        ----------
        train_fname
        val_fname
        verbose
        max_chars
        save_tokenizer
        tokenizer_params
        train_sampler_params
        val_sampler_params
        """
        self.train_path = os.path.join(DirConf.DATA_DIR, train_fname)
        self.val_path = os.path.join(DirConf.DATA_DIR, val_fname)

        assert os.path.exists(self.train_path)
        assert os.path.exists(self.val_path)

        self.verbose = verbose
        self.max_chars = max_chars

        self.save_tokenizer = save_tokenizer

        self.cols = ['name', 'alternate_names']

        self.train_data_: Optional[pd.DataFrame] = None
        self.val_data_: Optional[pd.DataFrame] = None

        self.tokenizer_params = {'name': 'ngram_tokenizer', 'maxlen': 30,
                                 'filters': '', 'lower': True, 'split': ' ',
                                 'char_level': False, 'num_words': 20_000,
                                 'oov_token': '<OOV>'}

        if tokenizer_params:
            self.tokenizer_params.update(tokenizer_params)

        logger.info('Loading tokenizer')
        TokenizerCLass = getattr(tokenizers,
                                 underscore_to_camel(
                                     self.tokenizer_params.pop('name')))
        self.tokenizer = TokenizerCLass(**self.tokenizer_params)

        self.train_sampler_params = train_sampler_params
        self.val_sampler_params = val_sampler_params

        self.train_sampler = None
        self.val_sampler = None

        self.model = None

    @property
    def train_data(self) -> pd.DataFrame:
        """

        Returns
        -------

        """
        if self.train_data_ is None:
            logger.info('Loading training dataset')
            self.train_data_ = pd.read_csv(self.train_path, usecols=self.cols)

            self.train_data_.dropna(inplace=True)
            self.train_data_['name'] = self.train_data_['name'].str.lower()

            self.train_data_['alternate_names'] = self.train_data_[
                'alternate_names'].str.lower()

            self.train_data_['n_alternate_names'] = self.train_data_[
                'alternate_names'].str.split(',').apply(len)

            self.train_data_['len_name'] = self.train_data_['name'].str.len()

            # get rid of toponyms that have more than "max_chars" characters.
            self.train_data_ = self.train_data_[
                (self.train_data_['len_name'] <= self.max_chars)
                & (self.train_data_['len_name'] > 2)].reset_index(drop=True)

        return self.train_data_

    @property
    def val_data(self) -> pd.DataFrame:
        """

        Returns
        -------

        """
        if self.val_data_ is None:
            logger.info('Loading validation dataset')

            self.val_data_ = pd.read_csv(self.val_path, usecols=self.cols)

            self.val_data_.dropna(inplace=True)
            self.val_data_['name'] = self.val_data_['name'].str.lower()

            self.val_data_['alternate_names'] = self.val_data_[
                'alternate_names'].str.lower()

            self.val_data_['n_alternate_names'] = self.val_data_[
                'alternate_names'].str.split(',').apply(len)

            self.val_data_['len_name'] = self.val_data_['name'].str.len()

            # get rid of toponyms that have more than "max_chars" characters.
            self.val_data_ = self.val_data_[
                (self.val_data_['len_name'] <= self.max_chars)
                & (self.val_data_['len_name'] > 2)].reset_index(drop=True)

        return self.val_data_

    def tokenize_data(self):
        """
        1) Creates n-grams from the toponyms
        2) Creates n-grams for each of the variations
        Returns
        -------

        """

        logger.info('Creating N-grams for training toponyms')
        # convert each toponym to it's ngram representation
        self.train_data['toponym_ngrams'] = self.train_data[
            'name'].progress_apply(self.tokenizer.get_ngrams)

        # convert each variation of each toponym to it's n-gram representation
        self.train_data['alternate_names'] = self.train_data[
            'alternate_names'].str.split(',')

        logger.info('Creating N-grams for training alternate-names')
        self.train_data['variations_ngrams'] = self.train_data[
            'alternate_names'].progress_apply(self.tokenizer.texts_to_ngrams)

        # collect (flatten out) all the n-grams (toponyms and variations)
        # these are needed in order to fit it to the tokenizer.
        all_train_names = list()
        for row in self.train_data['variations_ngrams']:
            all_train_names.extend(row)

        all_train_names += list(self.train_data['toponym_ngrams'])

        # fitting all the training texts on the instantiated tokenizer
        # this will create all the necessary tools that we will need.
        logger.info('Fitting tokenizer to training-data')
        self.tokenizer.fit_on_texts(texts=all_train_names)

        # using the fitted tokenizer, convert the train toponyms to sequences
        logger.info('Converting training toponyms to sequences')
        self.train_data['toponym_seqs'] = self.train_data[
            'toponym_ngrams'].progress_apply(
            lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

        logger.info('Padding training toponym sequences')
        # pad the sequences to the max length
        self.train_data['toponym_seqs'] = self.train_data[
            'toponym_seqs'].progress_apply(self.tokenizer.pad_single)

        # using the fitted tokenizer, convert each of the variations of all the
        # toponyms sequences
        logger.info('Converting training alternate-names to sequences')
        self.train_data['variations_seqs'] = self.train_data[
            'variations_ngrams'].progress_apply(
            self.tokenizer.texts_to_sequences)

        logger.info('Padding training alternate-names sequences')
        self.train_data['variations_seqs'] = self.train_data[
            'variations_seqs'].progress_apply(self.tokenizer.pad)

        #  ========== Same Procedure for the Validation Set ===================
        logger.info('Creating N-grams for validation toponyms')

        self.val_data['toponym_ngrams'] = self.val_data['name'].progress_apply(
            self.tokenizer.get_ngrams)

        self.val_data['alternate_names'] = self.val_data[
            'alternate_names'].str.split(',')

        logger.info('Creating N-grams for validation alternate-names')
        self.val_data['variations_ngrams'] = self.val_data[
            'alternate_names'].progress_apply(self.tokenizer.texts_to_ngrams)

        logger.info('Converting validation toponyms to sequences')

        self.val_data['toponym_seqs'] = self.val_data[
            'toponym_ngrams'].progress_apply(
            lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

        logger.info('Padding validation toponym sequences')
        self.val_data['toponym_seqs'] = self.val_data[
            'toponym_seqs'].progress_apply(
            self.tokenizer.pad_single)

        logger.info('Converting validation alternate-names to sequences')
        self.val_data['variations_seqs'] = self.val_data[
            'variations_ngrams'].progress_apply(
            self.tokenizer.texts_to_sequences)

        logger.info('Padding validation alternate-names sequences')
        self.val_data['variations_seqs'] = self.val_data[
            'variations_seqs'].progress_apply(self.tokenizer.pad)

        if self.verbose > 0:
            print(f'N-gram index length: {len(self.tokenizer.word_index)}')
            print('\nExample Transformation')
            print(self.val_data.loc[0])
            print(self.val_data.loc[0]['variations_seqs'])

        if self.save_tokenizer:
            print('Saving Tokenizer')
            self.tokenizer.save()
            print('Tokenizer saved')

    def create_samplers(self):
        """

        Returns
        -------

        """

        train_params = {'data': self.train_data,
                        'name': 'sampler',
                        'n_positives': 1,
                        'n_negatives': 3,
                        'neg_samples_size': 30,
                        'batch_size': 2048,
                        'shuffle': True}

        val_params = {'data': self.val_data,
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
        # the datasets are implicitly loaded
        self.tokenize_data()
        self.create_samplers()


if __name__ == "__main__":
    trainer = Dataset(
        train_fname='n_alternates_1+_latin_stratified_split_x_train.csv',
        val_fname='n_alternates_1+_latin_stratified_split_x_val.csv',
        save_tokenizer=True,
        verbose=1)
    trainer.run_data_preparation()
