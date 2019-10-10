import os
import random
from typing import List

import editdistance
import numpy as np
import pandas as pd

from similarity_learning.config import DirConf

pd.set_option('display.expand_frame_repr', False)

random.seed(5)


class SimpleSampler:

    def __init__(self,
                 toponyms,
                 variations,
                 n_positives: int = 1,
                 n_negatives: int = 3,
                 neg_samples_size: int = 30,
                 unique_samples: bool = True):
        """

        :param toponyms:
        :param variations:
        :param n_positives:
        :param n_negatives:
        :param neg_samples_size:
        """
        assert neg_samples_size % 2 == 0

        self.toponyms = toponyms
        self.variations = variations
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.neg_samples_size = neg_samples_size

        self.cache = set()
        self.unique_samples = unique_samples

    def get_closest_sample(self, anchor: str, variations: List[str]):
        """

        :param anchor:
        :param variations:
        :return:
        """
        if variations:

            furthest_positives = sorted(
                variations,
                key=lambda x: editdistance.eval(anchor, x),
                reverse=True)

            return furthest_positives[:self.n_positives]

        else:
            return [anchor]

    def get_furthest_sample(self, toponyms, index: int):
        """

        :param toponyms:
        :param index:
        :return:
        """
        n = self.neg_samples_size // 2

        bottom = max(0, index - n)
        top = min(index + n + 1, len(toponyms))

        index_range = list(range(bottom, top))
        index_range.remove(index)

        anchor = toponyms[index]
        negatives = toponyms[index_range]
        negatives = sorted(negatives,
                           key=lambda x: editdistance.eval(anchor, x))

        return negatives[:self.n_negatives]

    def generate_samples(self):
        """

        :return:
        """
        index = random.randint(0, len(self.toponyms) - 1)
        if self.unique_samples:

            while index in self.cache:
                index = random.randint(0, len(self.toponyms))

            self.cache.add(index)

        left = list()
        right = list()
        targets = list()

        anchor = self.toponyms[index]
        toponym_variations = self.variations[index]

        positives = self.get_closest_sample(anchor, toponym_variations)
        negatives = self.get_furthest_sample(self.toponyms, index)

        left.extend((1 + self.n_negatives) * [anchor])

        right.extend(positives)
        targets.extend(len(positives) * [1])

        right.extend(negatives)
        targets.extend(len(negatives) * [0])

        return np.array(left), np.array(right), np.array(targets)

    def generate_batch(self, batch_size: int = 128):
        """

        :param batch_size:
        :return:
        """

        left, right, targets = [], [], []

        n_samples = batch_size // (self.n_negatives + self.n_positives)

        for i in range(n_samples):
            x_left, x_right, y_targets = self.generate_samples()
            left.append(x_left)
            right.append(x_right)
            targets.append(y_targets)

        return (np.concatenate(left),
                np.concatenate(right),
                np.concatenate(targets))


if __name__ == "__main__":
    path = os.path.join(DirConf.DATA_DIR, 'all_countries_cleaned.csv')
    df = pd.read_csv(path, nrows=100_000)
    df = df.where((pd.notnull(df)), None)
    df.sort_values(['toponym', 'variations'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['variations'] = df['variations'].apply(
        lambda x: x.split(' || ') if x else [])

    sampler = SimpleSampler(toponyms=df['toponym'],
                            variations=df['variations'],
                            n_positives=1,
                            n_negatives=3,
                            neg_samples_size=30)

    # for i in range(12):
    #     adf = sampler.generate_samples()
    #     print(adf)

    batch_sample = sampler.generate_batch(batch_size=4 * 1024)

    print(pd.DataFrame(batch_sample).T)
