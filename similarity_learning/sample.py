import random
from typing import List, Tuple, NoReturn, Optional, Union

import editdistance
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from similarity_learning.tokenize import TrigramTokenizer, UnigramTokenizer

pd.set_option('display.expand_frame_repr', False)

random.seed(5)

tokenizers = Optional[Union[UnigramTokenizer, TrigramTokenizer]]


class SimpleSampler(Sequence):
    """

    """

    def __init__(
            self, toponyms: pd.Series, variations: pd.Series, batch_size=128,
            tokenizer: tokenizers = None, n_positives: int = 1,
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
            tokenizer: tokenizers = None, n_positives: int = 1,
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


class Sampler(Sequence):
    def __init__(
            self, data, batch_size=128, n_negatives: int = 3,
            neg_samples_size: int = 30, n_positives: int = 1,
            tokenizer: tokenizers = None,
            shuffle: bool = True) -> NoReturn:
        """

        Parameters
        ----------
        data
        batch_size
        n_negatives
        neg_samples_size
        n_positives
        tokenizer
        shuffle
        """

        assert neg_samples_size % 2 == 0
        assert batch_size // (n_negatives + n_positives) % 2 == 0

        self.data = data
        self.batch_size = batch_size

        # every index produces n_negatives + n_positives.
        self.n_samples = batch_size // (n_negatives + n_positives)

        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.neg_samples_size = neg_samples_size

        self.indexes: Optional[np.ndarray] = None

        self.shuffle = shuffle

        self.tokenizer = tokenizer

        self.length_indexes = self.data.groupby('len_name').groups

        self.names = self.data['name']  # str
        self.alternate_names = self.data['alternate_names']  # list of str
        self.name_seqs = self.data['toponym_seqs']
        self.alt_name_seqs = self.data['variations_seqs']

        del self.data
        self.on_epoch_end()

        self.steps_per_epoch = len(self.names) // self.n_samples

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        # we are diving by the number of samples instead of the batch size
        # this is because for each row in the dataset, we create
        # n_positives + n_negatives samples.
        return int(np.floor(len(self.indexes) / self.n_samples))

    def get_positive_samples(self, index):
        """
        Given an anchor and it's variations, it calculates the Edit Distance
        between the two, and selects those that have the maximum distance

        In case the variations list is empty, then the anchor itself is
        returned.

        :param index: int
        :return: List[List[int]
        """
        anchor = self.names[index]
        variations = self.alternate_names[index]
        anchor_seq = self.name_seqs[index]
        variations_seqs = self.alt_name_seqs[index]

        if variations:

            distances = [editdistance.eval(anchor, n) for n in variations]

            # indexes of the furthest positives for furthest to closest
            indexes = np.argsort(distances)[::-1][:self.n_positives]

            var_seqs = [variations_seqs[idx] for idx in indexes]
            return var_seqs

        else:
            return [anchor_seq]

    def get_negative_samples(self, index: int) -> List[str]:
        """
        :param index:
        :return:
        """
        anchor = self.names[index]

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

        # get the negatives strings
        negatives = list(self.names[neg_samples_indexes])
        # get the negatives sequences for the same indexes
        negatives_seqs = list(self.name_seqs[neg_samples_indexes])

        # sort the samples, by calculating the edit distance between the
        # anchor and the negative samples. Minimum distance first.
        distances = [editdistance.eval(anchor, n) for n in negatives]

        # indexes of the negatives for closest to furthest
        indexes = np.argsort(distances)[:self.n_negatives]

        # for each index in the closest index get the sequence and not the
        # name
        negatives_seqs = [negatives_seqs[idx] for idx in indexes]

        return negatives_seqs

    def generate_samples(self, index):
        """
        Generates n_positives + n_negatives samples for a given index.
        :return:
        """
        left = list()
        right = list()
        targets = list()

        anchor_seq = self.name_seqs[index]

        positives = self.get_positive_samples(index)
        negatives = self.get_negative_samples(index)

        left.extend((1 + self.n_negatives) * [anchor_seq])

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
        self.indexes = np.arange(len(self.names))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class SamplerV2(Sequence):
    def __init__(self, data, batch_size=32, n_negatives: int = 3,
                 neg_samples_size: int = 30, shuffle: bool = True,
                 tokenizer: tokenizers = None) -> NoReturn:
        """

        Parameters
        ----------
        data
        batch_size :
            This is not the actual batch size. It's the number of indexes taken
            for each step for each epoch

        n_negatives
        neg_samples_size
        shuffle
        tokenizer
        """

        assert neg_samples_size % 2 == 0

        self.data = data

        self.names = self.data['name']  # str
        self.alternate_names = self.data['alternate_names']  # list of str
        self.name_seqs = self.data['toponym_seqs']
        self.alt_name_seqs = self.data['variations_seqs']

        self.n_positives = self.alternate_names.apply(len).sum()
        self.n_negatives = n_negatives * self.n_positives

        self.total_samples = self.n_positives + n_negatives

        self.neg_samples_size = neg_samples_size

        self.indexes: Optional[np.ndarray] = None

        self.shuffle = shuffle

        self.tokenizer = tokenizer

        self.length_indexes = self.data.groupby('len_name').groups

        del self.data
        self.on_epoch_end()

        self.batch_size = batch_size

        self.steps_per_epoch = len(self.names) // self.batch_size

    def __len__(self):
        """
        Denotes the number of batches per epoch

        :return:
        """
        # we are diving by the number of samples instead of the batch size
        # this is because for each row in the dataset, we create
        # n_positives + n_negatives samples.
        return self.steps_per_epoch

    def get_positive_samples(self, index):
        """
        Given an anchor and it's variations, it calculates the Edit Distance
        between the two, and selects those that have the maximum distance

        In case the variations list is empty, then the anchor itself is
        returned.

        :param index: int
        :return: List[List[int]
        """

        anchor_seq = self.name_seqs[index]
        variations_seqs = self.alt_name_seqs[index]

        if len(variations_seqs) > 0:
            return variations_seqs

        else:
            return [anchor_seq]

    def get_negative_samples(self, index: int, n_samples: int) -> List[str]:
        """

        Parameters
        ----------
        index
        n_samples

        Returns
        -------

        """
        anchor = self.names[index]

        # calculate the anchor length
        anchor_length = len(anchor)

        # get the indexes of the toponyms that have the same length -1
        minus_1_length_indexes = self.length_indexes.get(anchor_length - 1, [])

        # get the indexes of the toponyms that have the same length
        same_length_indexes = self.length_indexes.get(anchor_length)

        # get the indexes of the toponyms that have the same length + 1
        plus_1_length_indexes = self.length_indexes.get(anchor_length + 1, [])

        neg_samples_indexes = list()

        denominator = 3

        if len(minus_1_length_indexes) == 0:
            denominator -= 1
        else:
            # Sample from the three pools
            # get k/3 samples from the toponyms that share the same (length -1)
            minus_one = list(np.random.choice(
                minus_1_length_indexes,
                size=self.neg_samples_size // denominator,
                replace=True))
            neg_samples_indexes.extend(minus_one)

        if len(plus_1_length_indexes) == 0:
            denominator -= 1
        else:
            # get k/3 samples from the toponyms that share the same length
            plus_one = list(np.random.choice(
                plus_1_length_indexes,
                size=self.neg_samples_size // denominator,
                replace=True))

            neg_samples_indexes.extend(plus_one)

        # get samples from the toponyms that share the same (length + 1)
        same = list(np.random.choice(same_length_indexes,
                                     size=self.neg_samples_size // denominator,
                                     replace=True))

        neg_samples_indexes.extend(same)

        if index in neg_samples_indexes:
            neg_samples_indexes.remove(index)

        # get the negatives strings
        negatives = list(self.names[neg_samples_indexes])
        # get the negatives sequences for the same indexes
        negatives_seqs = list(self.name_seqs[neg_samples_indexes])

        # sort the samples, by calculating the edit distance between the
        # anchor and the negative samples. Minimum distance first.
        distances = [editdistance.eval(anchor, n) for n in negatives]

        # indexes of the negatives for closest to furthest
        indexes = np.argsort(distances)[:n_samples]

        # for each index in the closest index get the sequence and not the
        # name
        negatives_seqs = [negatives_seqs[idx] for idx in indexes]

        return negatives_seqs

    def generate_samples(self, index):
        """
        Generates n_positives + n_negatives samples for a given index.
        :return:
        """

        left = list()
        right = list()

        targets = list()

        # the anchor sequence
        anchor_seq = self.name_seqs[index]

        # at least one sample
        positives = self.get_positive_samples(index)

        n_pos_samples = len(positives)
        n_neg_samples = n_pos_samples * self.n_negatives
        # at least one sample
        negatives = self.get_negative_samples(index=index,
                                              n_samples=n_neg_samples)

        n_neg_samples = len(negatives)  # may end up being fewer than actual

        total_index_samples = n_pos_samples + n_neg_samples

        # fill up the anchor sequences `total_index_samples` for the left part
        left.extend(total_index_samples * [anchor_seq])

        # add the positives to the right bucket
        right.extend(positives)
        # add the 1's for the positive samples
        targets.extend(n_pos_samples * [1])

        # add the negatives to the right bucket
        right.extend(negatives)
        # add the 0's for the negative samples

        targets.extend(n_neg_samples * [0])

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
                  index * self.batch_size: (index + 1) * self.batch_size]

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
        self.indexes = np.arange(len(self.names))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class EvaluationSampler(Sequence):
    """
    This is a simple Sampler that works only for the evaluation process.
    This sampler yields all the positives and chooses the same amount of
    negatives totally in random.
    """

    def __init__(self,
                 data,
                 batch_size: int = 1024,
                 shuffle: bool = True) -> NoReturn:
        """

        Parameters
        ----------
        data
        batch_size
        shuffle
        """
        self.data = data
        self.batch_size = batch_size
        self.indexes: Optional[np.ndarray] = None
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        Returns
        -------
        int
        """

        n_batches = int(np.floor(len(self.data) / self.batch_size))

        return n_batches

    def __getitem__(self, index) -> Tuple[Tuple[np.ndarray,
                                                np.ndarray],
                                          np.ndarray]:
        """
        Generates one batch of data
        :return:
        """
        # np.array(left), np.array(right), np.array(targets)
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]

        names = np.vstack(self.data['name_seq'].loc[indexes])

        alt_names = np.vstack(self.data['alternate_name_seq'].loc[indexes])

        targets = self.data['target'].loc[indexes].values

        return (names, alt_names), targets

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            # shuffles without assignment
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    from similarity_learning.dataset import Dataset

    # TODO: clean this up
    dataset = Dataset(train_fname='', val_fname='')
    dataset.tokenize_data()

    params = {'data': dataset.train_data_,
              'n_negatives': 3,
              'neg_samples_size': 30,
              'batch_size': 16,
              'shuffle': False}

    sampler = SamplerV2(**params)

    print('Steps: ', sampler.steps_per_epoch)
    print('Names len: ', len(sampler.names))
    # x = sampler.__getitem__(0)[1]
    # print(len(x))

    for i in range(len(sampler)):
        t = sampler.__getitem__(i)
        # print(t[0][0])
        # print()
        # print(t[0][1])
        # print()
        # print(t[1])
        # print(t[0][0].shape)
        # print(t[0][0].shape)
        print(t[1].shape)
