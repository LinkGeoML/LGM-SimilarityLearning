import os
from itertools import product
from typing import Union, Optional, List, Dict, NoReturn

import matplotlib.pyplot as plt
import pandas as pd
from alphabet_detector import AlphabetDetector
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from similarity_learning.config import DirConf
from similarity_learning.logger import exp_logger

pd.set_option('display.expand_frame_repr', False)
tqdm.pandas()


class RawDataPreprocessor:
    """

    """
    all_cols = ['geonameid', 'name', 'asciiname', 'alternate_names',
                'latitude', 'longitude', 'feature class', 'feature code',
                'country_code', 'cc2', 'admin1 code', 'admin2 code',
                'admin3 code', 'admin4 code', 'population', 'elevation', 'dem',
                'timezone', 'modification date']
    basic_cols = ['geonameid',
                  'name',
                  'asciiname',
                  'alternate_names']
    # instantiate and check detector
    ab_detector = AlphabetDetector()
    data_dir = DirConf.DATA_DIR

    def __init__(self, fname='allCountries.txt', show_plots: bool = False,
                 n_rows: Optional[int] = None, n_alternates: int = 1,
                 max_name_chars: Optional[int] = None,
                 only_latin: bool = False, stratified_split: bool = False,
                 splits: Optional[List[float]] = None,
                 save_data: bool = True) -> NoReturn:
        """

        Parameters
        ----------
        fname : str
            The file name of the dataset
        show_plots : bool
            Whether to show plots regarding the dataset
        n_rows : Int or None
            The number of rows to load from the dataset
        n_alternates : Int
            Keep the records with the minimum number of alternate_names
        only_latin : bool
            Whether to keep only the records that are written in Latin
        stratified_split : bool
            - If True, and only Latin is also True, it uses the length of the
            name as the stratification split factor.
            - If True, and only Latin is also False, it uses the Alphabet of
            the name as the stratification split factor.
            - If False is uses random shuffling and splitting.
        splits : List[float]
            A list of three floats representing the split ratios in training,
            development and testing respectively. The sum of the three values
            must add to 1. Default is [0.8, 0.1, 0.1]
        """

        self.max_name_chars = 50 if max_name_chars is None else max_name_chars

        if splits is None:
            splits = [0.8, 0.1, 0.1]

        assert sum(splits) == 1
        assert len(splits) == 3

        self.train_size, self.val_size, self.test_size = splits

        self.fname = fname
        self.raw_data_path = os.path.join(self.data_dir, fname)
        self.n_rows = n_rows
        self.show_plots = show_plots
        self.n_alternates = n_alternates
        self.only_latin = only_latin
        self.stratified_split = stratified_split
        self.save_data = save_data
        self.data_: Optional[pd.DataFrame] = None

    def detect_alphabet(self, geoname: Union[str, None]) -> str:
        """
        Detects the alphabet of the text.
        Parameters
        ----------
        geoname: Str or None
            The text that we want to extract the alphabet

        Returns
        -------
        str
            The alphabet of the string.
        """
        if geoname:
            geoname = str(geoname)

            ab = self.ab_detector.detect_alphabet(geoname)
            if "CYRILLIC" in ab:
                return "CYRILLIC"
            return ab.pop() if len(ab) != 0 else 'UND'
        else:
            return 'UND'

    @property
    def data(self) -> pd.DataFrame:
        """
        Lazily loads the raw dataset
        Returns
        -------

        """
        if self.data_ is None:
            self.data_ = pd.read_csv(
                self.raw_data_path, header=None, sep='\t', names=self.all_cols,
                usecols=self.basic_cols, nrows=self.n_rows)

            # Convert np.nan to None
            self.data_ = self.data_.where((pd.notnull(self.data_)), None)

        return self.data_

    def create_plots(self) -> NoReturn:
        """
        This method creates plots regarding the dataset.
        Returns
        -------

        """
        self.data['n_alternate_names'].hist(figsize=(16, 8), bins=range(0, 25))
        plt.show()

        self.data['n_alternate_names'].value_counts().plot(kind='pie',
                                                           figsize=(8, 8))
        plt.show()

        counts = (100 * self.data['n_alternate_names'].value_counts() /
                  self.data['n_alternate_names'].sum()).sort_index()
        counts.head(10).plot('barh')
        plt.show()

        self.data['n_alternate_names'].value_counts(
        ).sort_index().head(10).plot(kind='barh')
        plt.show()

    def filter_latin_related_records(self) -> pd.DataFrame:
        """

        Returns
        -------

        """
        exp_logger.info('Filtering Latin Names')
        self.data_ = self.data[self.data['name_alphabet'] == "LATIN"]
        exp_logger.info(
            f'Number of Records after filtering: {len(self.data_)}')

        # filter only the alternate names that are written in LATIN
        exp_logger.info('Filtering Latin Alternate Names')
        self.data['alt_names_seq'] = self.data.apply(
            lambda row: [n for n, ab in zip(row['alt_names_seq'],
                                            row['alternate_names_alphabet'])
                         if ab == 'LATIN'], axis=1)

        # replace the alternate_names with those that are only written in LATIN
        self.data['alternate_names'] = self.data['alt_names_seq'].apply(
            lambda l: ', '.join(l) if l else None)

        return self.data

    def split_records(self) -> Dict[str, pd.DataFrame]:
        """

        Returns
        -------

        """
        data_size = len(self.data)
        test_size = int(self.test_size * data_size)
        val_size = int(self.val_size * data_size)

        if not self.stratified_split:

            exp_logger.info('Random Split into Train-Val and Test')
            X_train_val, X_test = train_test_split(
                self.data[self.basic_cols], test_size=test_size, shuffle=True,
                random_state=2020, stratify=None)

            exp_logger.info('Random Split into Train and Val')
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_size, shuffle=True,
                random_state=2020, stratify=None)

        else:
            if self.only_latin:
                exp_logger.info('Using Name Length as stratification factor')
                stratify_column = self.data['len_name']
            else:
                exp_logger.info('Using Name Alphabet as stratification factor')
                stratify_column = self.data['name_alphabet']

            exp_logger.info('Stratified Split into Train-Val and Test')

            # y_train_val will be used for the stratification in the second
            # split.
            X_train_val, X_test, y_train_val, _ = train_test_split(
                self.data[self.basic_cols], stratify_column,
                test_size=test_size, shuffle=True, random_state=2020,
                stratify=stratify_column)

            exp_logger.info('Stratified Split into Train and Val')
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_size, shuffle=True,
                random_state=2020, stratify=y_train_val)

        exp_logger.info(f'X_train-val size: {X_train_val.shape[0]}')
        exp_logger.info(f'X_train size: {X_train.shape[0]}')
        exp_logger.info(f'X_val size: {X_val.shape[0]}')
        exp_logger.info(f'X_test size: {X_test.shape[0]}')

        return dict(X_train_val=X_train_val, X_train=X_train, X_val=X_val,
                    X_test=X_test)

    def run(self):
        """

        Returns
        -------

        """
        # get the alternate names as a list for each record
        self.data['alt_names_seq'] = self.data['alternate_names'].apply(
            lambda x: x.split(',') if x else [])

        self.data['len_name'] = self.data['name'].apply(len)

        exp_logger.info(f'Keeping records with Name '
                        f'Length smaller than {self.max_name_chars}')

        self.data_ = self.data[
            self.data['len_name'] <= self.max_name_chars].reset_index(
            drop=True)

        exp_logger.info('Detecting Alphabet for all Names')
        # detect the alphabet for the name
        self.data['name_alphabet'] = self.data['name'].progress_apply(
            self.detect_alphabet)

        exp_logger.info('Detecting Alphabet for all Alternate Names')
        # get the alphabet for each alternate name
        self.data['alternate_names_alphabet'] = self.data[
            'alt_names_seq'].progress_apply(
            lambda l: [self.detect_alphabet(n) for n in l])

        if self.only_latin:
            self.filter_latin_related_records()  # filters self.data

        self.data['n_alt_names'] = self.data['alt_names_seq'].progress_apply(
            len)

        self.data['n_alt_gte'] = self.data['n_alt_names'] >= self.n_alternates

        if self.show_plots:
            exp_logger.info('Creating Plots')
            self.create_plots()

        datasets = self.split_records()

        if self.save_data:

            exp_logger.info('Saving Datasets')
            for data_type, data in datasets.items():
                exp_logger.info(f'Saving {data_type}')

                ab = 'latin' if self.only_latin else 'global'
                shuffle = 'stratified' if self.stratified_split else 'random'

                outfile = f'n_alternates_{self.n_alternates}+_{ab}_' \
                          f'{shuffle}_split_{data_type}.csv'.strip().lower()

                outfile = os.path.join(self.data_dir, outfile)

                data[self.basic_cols].to_csv(outfile, encoding='utf8',
                                             index=False)

        return datasets


if __name__ == "__main__":
    options = {
        'n_alternates': [1, 3],
        'only_latin': [True, False],
        'stratified_split': [True, False]
    }

    all_options = sorted(options)
    combinations = product(*(options[Name] for Name in all_options))

    for comb in combinations:
        params = dict(zip(all_options, comb))
        params['show_plots'] = False
        params['save_data'] = True
        params['n_rows'] = None

        print(params)
        processor = RawDataPreprocessor(**params)

        data_sets = processor.run()
