import os
import random
from typing import List, Tuple, NoReturn, Optional

import editdistance
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from similarity_learning.config import DirConf
from similarity_learning.utils_text import NgramTokenizer

pd.set_option('display.expand_frame_repr', False)

random.seed(5)


class SimpleSampler(Sequence):
    """

    """

    def __init__(
            self, toponyms: pd.Series, variations: pd.Series, batch_size=128,
            tokenizer: Optional[NgramTokenizer] = None, n_positives: int = 1,
            n_negatives: int = 3, neg_samples_size: int = 30,
            shuffle: bool = True) -> NoReturn:
        """

        :param toponyms:
        :param variations:
        :param batch_size:
        :param tokenizer:
        :param n_positives:
        :param n_negatives:
        :param neg_samples_size:
        :param shuffle:
        """
        assert neg_samples_size % 2 == 0
        assert batch_size // (n_negatives + n_positives) % 2 == 0

        self.batch_size = batch_size

        # every index produces n_negatives + n_positives.
        self.n_samples = batch_size // (n_negatives + n_positives)

        self.toponyms = toponyms
        self.variations = variations
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.neg_samples_size = neg_samples_size

        self.indexes: np.ndarray = None

        self.shuffle = shuffle
        self.on_epoch_end()

        self.tokenizer = tokenizer

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        # we are diving by the number of samples instead of the batch size
        # this is because for each row in the dataset, we create
        # n_positives + n_negatives samples.
        return int(np.floor(len(self.indexes) / self.n_samples))

    def get_positive_samples(self, anchor: str, variations: List[str]):
        """
        Given an anchor and it's variations, it calculates the Edit Distance
        between the two, and selects those that have the maximum distance

        In case the variations list is empty, then the anchor itself is
        returned.

        :param anchor: str
        :param variations: List[str]
        :return: List[str]
        """
        if variations:

            furthest_positives = sorted(
                variations,
                key=lambda x: editdistance.eval(anchor, x),
                reverse=True)

            return furthest_positives[:self.n_positives]

        else:
            return [anchor]

    def get_negative_samples(self, index: int) -> List[str]:
        """
        :param index:
        :return:
        """

        neg_samples_indexes = list(np.random.choice(self.toponyms.index,
                                                    size=self.neg_samples_size,
                                                    replace=True))

        if index in neg_samples_indexes:
            neg_samples_indexes.remove(index)

        anchor = self.toponyms[index]
        negatives = self.toponyms[neg_samples_indexes]

        # sort the samples, by calculating the edit distance between the
        # anchor and the negative samples. Minimum distance first.
        negatives = sorted(negatives,
                           key=lambda x: editdistance.eval(anchor, x))

        return negatives[:self.n_negatives]

    def generate_samples(self, index):
        """
        Generates n_positives + n_negatives samples for a given index.
        :return:
        """
        left = list()
        right = list()
        targets = list()

        anchor = self.toponyms[index]
        toponym_variations = self.variations[index]

        positives = self.get_positive_samples(anchor, toponym_variations)
        negatives = self.get_negative_samples(index)

        left.extend((1 + self.n_negatives) * [anchor])

        right.extend(positives)
        targets.extend(len(positives) * [1])

        right.extend(negatives)
        targets.extend(len(negatives) * [0])

        return np.array(left), np.array(right), np.array(targets)

    def __getitem__(self, index) -> Tuple[Tuple[np.ndarray,
                                                np.ndarray],
                                          np.ndarray]:
        """
        Generate one batch of data
        :return:
        """

        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.n_samples: (index + 1) * self.n_samples]

        left, right, targets = [], [], []

        for index in indexes:
            x_left, x_right, y_targets = self.generate_samples(index)
            left.append(x_left)
            right.append(x_right)
            targets.append(y_targets)

        if self.tokenizer:
            left = self.tokenizer.texts_to_ngrams(texts=left)
            left = self.tokenizer.pad(left)

            right = self.tokenizer.texts_to_ngrams(texts=right)
            right = self.tokenizer.pad(right)

        return ((np.concatenate(left),
                 np.concatenate(right)),
                np.concatenate(targets))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.toponyms))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class SimpleSamplerV2(SimpleSampler):
    """
    The difference with the SimpleSampler is on the negative sampling.
    Instead of taking a random sample from all the indexes, we filter the
    indexes to those that have the same length with the toponym.
    """

    def __init__(
            self, toponyms: pd.Series, variations: pd.Series, batch_size=128,
            tokenizer: Optional[NgramTokenizer] = None, n_positives: int = 1,
            n_negatives: int = 3, neg_samples_size: int = 30,
            shuffle: bool = True) -> NoReturn:
        """

        :param toponyms:
        :param variations:
        :param batch_size:
        :param tokenizer:
        :param n_positives:
        :param n_negatives:
        :param neg_samples_size:
        :param shuffle:
        """
        super().__init__(toponyms, variations, batch_size, tokenizer,
                         n_positives, n_negatives, neg_samples_size, shuffle)

        self.length_indexes = self.toponyms.str.len().to_frame().groupby(
            'name').groups

        # print(self.length_indexes.get(3))

    def get_negative_samples(self, index: int) -> List[str]:
        """
        :param index:
        :return:
        """
        anchor = self.toponyms[index]

        # calculate the anchor length
        anchor_length = len(anchor)
        # get the indexes of the toponyms that have the same index
        same_length_indexes = self.length_indexes.get(anchor_length)

        # get k samples from the toponyms that share the same length
        neg_samples_indexes = list(np.random.choice(same_length_indexes,
                                                    size=self.neg_samples_size,
                                                    replace=True))

        if index in neg_samples_indexes:
            neg_samples_indexes.remove(index)

        negatives = self.toponyms[neg_samples_indexes]

        # sort the samples, by calculating the edit distance between the
        # anchor and the negative samples. Minimum distance first.
        negatives = sorted(negatives,
                           key=lambda x: editdistance.eval(anchor, x))

        return negatives[:self.n_negatives]


# # Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# Datasets
# partition = # IDs
# labels = # Labels

# Generators
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)

#
# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)
if __name__ == "__main__":
    path = os.path.join(DirConf.DATA_DIR, 'all_countries_filtered_columns.csv')
    df = pd.read_csv(path, nrows=1000)
    df = df.where((pd.notnull(df)), None)
    df.sort_values(['name', 'alternate_names'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    tokeniZer = NgramTokenizer(num_words=10,
                               oov_token='<OOV>',
                               lower=True,
                               char_level=False,
                               split=' ',
                               filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', )

    df['alternate_names'] = df['alternate_names'].apply(
        lambda x: x.split(' || ') if x else [])

    params = {'toponyms': df['name'],
              'variations': df['alternate_names'],
              'n_positives': 1,
              'n_negatives': 3,
              'neg_samples_size': 30,
              'batch_size': 16,
              'shuffle': True}

    sampler = SimpleSamplerV2(**params)

    for i in range(len(sampler)):
        t = sampler.__getitem__(i)
        print(t)
        break
