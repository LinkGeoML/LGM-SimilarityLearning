import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import similarity_learning.tokenize as tokenizers
from similarity_learning import sample as samplers
from similarity_learning.config import DirConf
from similarity_learning.logger import exp_logger as logger
from similarity_learning.utils import underscore_to_camel

pd.set_option('display.expand_frame_repr', False)
tqdm.pandas()

np.random.seed(2020)

TEST_ROWS = None


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

    @property
    def train_data(self) -> pd.DataFrame:
        """

        Returns
        -------

        """
        if self.train_data_ is None:
            logger.info('Loading training dataset')
            self.train_data_ = pd.read_csv(self.train_path, usecols=self.cols,
                                           nrows=TEST_ROWS)

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

            self.val_data_ = pd.read_csv(self.val_path, usecols=self.cols,
                                         nrows=TEST_ROWS)

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
                        'n_negatives': 3,
                        'neg_samples_size': 30,
                        'batch_size': 2048,
                        'shuffle': True}

        val_params = {'data': self.val_data,
                      'name': 'sampler',
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


class TestDataset:
    """
    This class is responsible for handling the test data pipeline.
    Given the parameters it does the following
    - Given the filename it loads the dataset
    - Given the tokenizer filename if loads the already fitted tokenizer,
    - If use_external is passed it will expect two columns to be present in
      order to create the sequences. If not, it will expect a dataset that
      needs pre-processing to create a "one-to-one" dataset.
    """

    def __init__(self,
                 fname: str,
                 tokenizer_fname: str,
                 use_external: bool = False,
                 verbose: int = 0,
                 max_chars: int = 32,
                 sampler_params: Optional[dict] = None,
                 ):
        """
        Parameters
        ----------
        fname : str
            The filename of the dataset that we want to prepare for model
            evaluation.
        tokenizer_fname : str
            The filename of the already fitted tokenizer tha we want to use
            for the conversion of the toponyms and the alternate_names to
            sequences.
        use_external : bool
        verbose : int
        max_chars : int
        """
        self.path = os.path.join(DirConf.DATA_DIR, fname)

        assert os.path.exists(self.path)

        self.verbose = verbose
        self.max_chars = max_chars

        self.cols = ['name', 'alternate_names']

        self.data_: Optional[pd.DataFrame] = None

        logger.info(f'Loading fitted tokenizer: {tokenizer_fname}')
        self.tokenizer = tokenizers.load_tokenizer(name=tokenizer_fname)

        self.test_sampler = None

        self.use_external = use_external
        self.sampler_params = sampler_params

        self.external_cols = ['name', 'alternate_name', 'target']

        self.output_cols = ['name', 'alternate_name', 'name_seq',
                            'alternate_name_seq', 'target']

    @property
    def data(self) -> pd.DataFrame:
        """
        This method loads lazily the dataset that we will use for the model
        evaluation.
        Returns
        -------
        pd.DataFrame :
            A pandas dataframe that
        """
        if self.data_ is None:
            if self.use_external:
                logger.info('Loading external test dataset')
                self.data_ = pd.read_csv(self.path,
                                         usecols=self.external_cols,
                                         nrows=TEST_ROWS, sep='\t')

                self.data_['target'] = self.data_['target'].astype(int)
            else:
                logger.info('Loading test dataset')
                self.data_ = pd.read_csv(self.path, usecols=self.cols,
                                         nrows=TEST_ROWS)

            self.data_.dropna(inplace=True)

            logger.info(f'Dataset size: {len(self.data_)}')

            self.data_['name'] = self.data_['name'].str.lower()

            self.data_['len_name'] = self.data_['name'].str.len()

            # get rid of toponyms that have more than "max_chars" characters.
            self.data_ = self.data_[
                (self.data_['len_name'] <= self.max_chars)
                & (self.data_['len_name'] > 2)].reset_index(drop=True)

            if not self.use_external:
                # since we don't have a ready external dataset for testing
                #  we need to pre-process the raw dataset.
                self.data_['alternate_names'] = self.data_[
                    'alternate_names'].str.lower()

                self.data_['n_alternate_names'] = self.data_[
                    'alternate_names'].str.split(',').apply(len)

        return self.data_

    def tokenize_external_data(self):
        """

        Returns
        -------

        """
        logger.info('Creating Sequences for External Test dataset')
        #  ========== Procedure for the Test Set ===================

        for col_name in self.external_cols:
            if col_name == 'target':
                # obviously, we don't want to tokenize the target
                continue

            logger.info(f'Creating N-grams for column name: "{col_name}".')

            self.data[f'{col_name}_ngrams'] = self.data[
                col_name].progress_apply(self.tokenizer.get_ngrams)

            logger.info(f'Converting column name: "{col_name}" to sequences')

            self.data[f'{col_name}_seq'] = self.data[
                f'{col_name}_ngrams'].progress_apply(
                lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

            logger.info(f'Padding sequences for column name: "{col_name}".')
            self.data[f'{col_name}_seq'] = self.data[
                f'{col_name}_seq'].progress_apply(self.tokenizer.pad_single)

        self.data_ = self.data_[self.output_cols]

        return self.data_

    def tokenize_raw_data(self):
        """
        1) Creates n-grams from the toponyms
        2) Creates n-grams for each of the variations
        Returns
        -------

        """
        #  ========== Procedure for the Test Set ===================
        logger.info('Creating N-grams for test toponyms')

        self.data['toponym_ngrams'] = self.data['name'].progress_apply(
            self.tokenizer.get_ngrams)

        self.data['alternate_names'] = self.data['alternate_names'].str.split(
            ',')

        logger.info('Creating N-grams for test alternate-names')
        self.data['variations_ngrams'] = self.data[
            'alternate_names'].progress_apply(self.tokenizer.texts_to_ngrams)

        logger.info('Converting test toponyms to sequences')

        self.data['toponym_seqs'] = self.data['toponym_ngrams'].progress_apply(
            lambda x: self.tokenizer.texts_to_sequences(texts=[x])[0])

        logger.info('Padding test toponym sequences')
        self.data['toponym_seqs'] = self.data['toponym_seqs'].progress_apply(
            self.tokenizer.pad_single)

        logger.info('Converting test alternate-names to sequences')
        self.data['variations_seqs'] = self.data[
            'variations_ngrams'].progress_apply(
            self.tokenizer.texts_to_sequences)

        logger.info('Padding test alternate-names sequences')
        self.data['variations_seqs'] = self.data[
            'variations_seqs'].progress_apply(self.tokenizer.pad)

        if self.verbose > 0:
            logger.info(
                f'N-gram index length: {len(self.tokenizer.word_index)}')
            logger.info('\nExample Transformation')
            logger.info(self.data.loc[0])
            logger.info(self.data.loc[0]['variations_seqs'])

    @staticmethod
    def get_alternate_index(idx: int, max_idx: int) -> int:
        """
        Given the max index of a dataframe it creates an index that is
        different than the one given.
        Parameters
        ----------
        idx: int
        max_idx: int

        Returns
        -------
        int
        """

        alt_index = np.random.randint(0, max_idx, 1)[0]

        if idx != alt_index:
            return alt_index

        return alt_index + 1

    def create_toponym_2_alternate_random_dataset(self):
        """
        Creates a dataset at random.
        Returns
        -------

        """
        data = self.data.reset_index()

        indexes = data.progress_apply(
            lambda row: [row['index']] * row['n_alternate_names'], axis=1)

        names = data.progress_apply(
            lambda row: [row['name']] * row['n_alternate_names'], axis=1)

        names_seq = data.progress_apply(
            lambda row: [row['toponym_seqs']] * row['n_alternate_names'],
            axis=1)

        pos_df = pd.DataFrame()
        pos_df['index'] = pd.Series(itertools.chain(*indexes))
        pos_df['alt_index'] = pos_df['index']
        pos_df['name'] = pd.Series(itertools.chain(*names))
        pos_df['name_seq'] = pd.Series(itertools.chain(*names_seq))

        positive_alt_name = pd.Series(
            (itertools.chain(*data['alternate_names'])))

        positive_alt_name_seq = pd.Series(
            itertools.chain(*data['variations_seqs']))

        pos_df['alternate_name'] = positive_alt_name
        pos_df['alternate_name_seq'] = positive_alt_name_seq

        pos_df['target'] = 1

        neg_df = pos_df[['index', 'name', 'name_seq']].copy()

        neg_sampling_indexes = pos_df['index'].progress_apply(
            self.get_alternate_index, args=(len(self.data) - 1,))

        neg_df['alt_index'] = neg_sampling_indexes

        negatives_right = data[['name', 'toponym_seqs']].loc[
            neg_sampling_indexes].reset_index(drop=True)

        negatives_right.columns = ['alternate_name', 'alternate_name_seq']

        neg_df = pd.concat([neg_df, negatives_right], axis=1)
        neg_df['target'] = 0

        final_df = pd.concat([pos_df, neg_df], sort=True).reset_index(
            drop=True)

        cols = ['name', 'alternate_name', 'name_seq',
                'alternate_name_seq', 'target']

        self.data_ = final_df[cols]

        del final_df

        return self.data_

    def run_data_preparation(self):
        """

        Returns
        -------

        """
        if self.use_external:
            self.tokenize_external_data()
        else:
            self.tokenize_raw_data()
            self.create_toponym_2_alternate_random_dataset()

        sampler_params = {'data': self.data,
                          'batch_size': 1024,
                          'shuffle': True}

        if self.sampler_params:
            sampler_params.update(self.sampler_params)

        if self.test_sampler is None:
            self.test_sampler = samplers.EvaluationSampler(**sampler_params)


if __name__ == "__main__":
    # trainer = Dataset(
    #     train_fname='n_alternates_1+_latin_stratified_split_x_train.csv',
    #     val_fname='n_alternates_1+_latin_stratified_split_x_val.csv',
    #     save_tokenizer=True,
    #     verbose=1)
    # trainer.run_data_preparation()
    #
    # evaluator = TestDataset(
    #     fname='n_alternates_1+_latin_stratified_split_x_test.csv',
    #     tokenizer_fname='unigram_tokenizer_nw_20000_ml_32.json',
    #     use_external=False)
    #
    # evaluator.run_data_preparation()
    # print(evaluator.data)
    # evaluator.data[['name',
    #                 'alternate_name',
    #                 'target']].to_csv('testing_external_dataset.csv')

    evaluator = TestDataset(
        fname='testing_external_dataset.csv',
        tokenizer_fname='unigram_tokenizer_nw_20000_ml_32.json',
        use_external=True)

    evaluator.run_data_preparation()
    print(evaluator.data)
