from typing import List

import numpy as np
import pandas as pd
from more_itertools import windowed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class NgramTokenizer(Tokenizer):
    def __init__(self, maxlen, **kwargs):
        """

        :param maxlen:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.maxlen = maxlen

    @staticmethod
    def get_ngrams(text: str, n: int = 3, step=1) -> str:
        """

        :param text:
        :param n:
        :param step:
        :return:
        """

        output = []

        # split the sentence in tokens.
        tokens = text.split()

        # if only one token, then we only have BOS and EOS tags
        if len(tokens) == 1:

            chars = ['<BOS>'] + list(text) + ['<EOS>']
            text = ' '.join(
                [''.join(t) for t in windowed(seq=chars, n=3, step=1)])
            output.append(text)

        # We have more than 1 tokens. So we need 3 kind of tags:
        # BOS: beginning of sentence
        # IOS: inside of sentence
        # EOS: end of sentence
        else:
            # extracting the first token, a list of the inside tokens, and the
            # last token. We handle each one differently
            first, *inside, last = tokens

            # in the first token we put BOS tag in the beginning of the token
            # and IOS at the end, since the sentence is not over.
            # We also split to first token to it's characters, so we can get
            # the n-grams.
            first_chars = ['<BOS>'] + list(first) + ['<IOS>']

            # create the n-gram texts and join them back together with a ' '
            first_chars = ' '.join(
                [''.join(t)
                 for t in windowed(seq=first_chars, n=n, step=step)])

            # append the "n-gramed" token to the output list
            output.append(first_chars)

            for ins_token in inside:
                # for each of the inside tokens use only the IOS tags
                # we do the same procedure as in the first token.
                inside_chars = ['<IOS>'] + list(ins_token) + ['<IOS>']

                inside_chars = ' '.join(
                    [''.join(t) for t in
                     windowed(seq=inside_chars, n=n, step=step)])

                output.append(inside_chars)

            # for the last token we use IOS and EOS tags.
            # Same procedure as before.
            last_chars = ['<IOS>'] + list(last) + ['<EOS>']

            last_chars = ' '.join(
                [''.join(t) for t in windowed(seq=last_chars, n=3, step=1)])

            output.append(last_chars)

        return ' '.join(output)

    def texts_to_ngrams(self, texts, n: int = 3, step=1) -> List[str]:
        output = list()

        for text in texts:
            output.append(self.get_ngrams(text, n, step))

        return output

    def pad(self, seqs):
        return pad_sequences(seqs,
                             maxlen=self.maxlen,
                             padding='post',
                             truncating='post')

    def pad_single(self, sequence):
        return pad_sequences(sequences=[sequence],
                             maxlen=self.maxlen,
                             padding='post',
                             truncating='post')[0]


def raw_code():
    max_features = 20_000
    X_train = pd.DataFrame({'toponymX1': [],
                            'toponymX2': []})

    X_val = pd.DataFrame({'toponymX1': [],
                          'toponymX2': []})

    tokenizer = Tokenizer(num_words=max_features,
                          oov_token='<OOV>',
                          lower=True,
                          char_level=False)

    # fitting on the train dataset only
    tokenizer.fit_on_texts(
        list(X_train['toponymX1']) + list(X_train['toponymX2']))

    X_train['toponymX1_seqs'] = tokenizer.texts_to_sequences(
        X_train['toponymX1'])
    X_train['toponymX2_seqs'] = tokenizer.texts_to_sequences(
        X_train['toponymX2'])

    X_val['toponymX1_seqs'] = tokenizer.texts_to_sequences(X_val['toponymX1'])
    X_val['toponymX2_seqs'] = tokenizer.texts_to_sequences(X_val['toponymX2'])

    all_train_lengths = list(X_train.toponymX1_seqs.apply(len)) + list(
        X_train.toponymX2_seqs.apply(len))

    max_len = int(np.percentile(all_train_lengths, q=90))
    print('Max Length: {}'.format(max_len))

    X_train_t1 = pad_sequences(X_train['toponymX1_seqs'],
                               maxlen=max_len,
                               padding='post',
                               truncating='post')

    X_train_t2 = pad_sequences(X_train['toponymX2_seqs'],
                               maxlen=max_len,
                               padding='post',
                               truncating='post')

    X_val_t1 = pad_sequences(X_val['toponymX1_seqs'],
                             maxlen=max_len,
                             padding='post',
                             truncating='post')

    X_val_t2 = pad_sequences(X_val['toponymX2_seqs'],
                             maxlen=max_len,
                             padding='post',
                             truncating='post')

    # Make sure everything is ok
    assert X_train_t1.shape == X_train_t2.shape
    assert X_val_t1.shape == X_val_t2.shape

    # assert len(X_train_t1) == len(y_train)
    # assert len(X_train_q2) == len(y_train)


if __name__ == "__main__":
    toponyms = ['athens',
                'athens greece',
                'athens gr greece',
                'athina',
                'athina gr']

    tokeniZer = NgramTokenizer(num_words=None,
                               oov_token='<OOV>',
                               lower=True,
                               char_level=False,
                               split=' ',
                               maxlen=30,
                               filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', )

    toponyms = tokeniZer.texts_to_ngrams(texts=toponyms)
    print(toponyms)
    tokeniZer.fit_on_texts(toponyms)
    sequences = tokeniZer.texts_to_sequences(toponyms)
    print(toponyms)
    print(sequences)

    from pprint import pprint

    pprint(tokeniZer.word_index)
    # x = NgramTokenizer(maxlen=None).texts_to_ngrams(texts=toponyms)
    # for i in zip(toponyms, x):
    #     print(i)
