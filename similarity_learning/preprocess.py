import os

import pandas as pd
from tqdm import tqdm

from similarity_learning.config import DirConf

tqdm.pandas()


def clean_up_dataset(sample: bool = True, save: bool = False):
    """

    :param sample:
    :param save:
    :return:
    """
    infile = os.path.join(DirConf.DATA_DIR, 'allCountries.txt')

    if sample:

        df = pd.read_csv(infile, nrows=10000, sep='\t', header=None,
                         usecols=[1, 2, 3])

    else:
        df = pd.read_csv(infile, sep='\t', header=None, usecols=[1, 2, 3])

    df = df.where((pd.notnull(df)), None)
    df[2] = df[2].progress_apply(lambda x: {x} if x else set())
    df[3] = df[3].progress_apply(lambda x: set(x.split(',')) if x else set())
    df['variations'] = df.progress_apply(
        lambda row: (row[2] | row[3]) - {row[1]}, axis=1)
    df = df[[1, 'variations']]
    df.columns = ['toponym', 'variations']
    df['variations'] = df['variations'].progress_apply(
        lambda x: ' || '.join(sorted(x)))

    if save:
        outfile = os.path.join(DirConf.DATA_DIR, 'all_countries_cleaned.csv')
        df.to_csv(outfile, index=False, encoding='utf-8')

    return df


if __name__ == "__main__":
    data = clean_up_dataset(sample=True, save=False)
    print(data)
