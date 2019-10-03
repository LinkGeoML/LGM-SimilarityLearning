from typing import List

import numpy as np
import pandas as pd
from more_itertools import windowed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class NgramTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_ngrams(self, texts, n: int = 3, step=1) -> List[List[str]]:
        output = list()

        for text in texts:
            chars = ['<BOS>'] + list(text) + ['<EOS>']
            text = [''.join(t) for t in windowed(seq=chars, n=n, step=step)]
            output.append(text)

        return output


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
    toponyms = ['athens', 'athens greece', 'athina', 'athina gr']

    tokeniZer = NgramTokenizer(num_words=10,
                               oov_token='<OOV>',
                               lower=True,
                               char_level=False,
                               split=' ',
                               filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', )

    toponyms = tokeniZer.create_ngrams(texts=toponyms)
    tokeniZer.fit_on_texts(toponyms)
    sequences = tokeniZer.texts_to_sequences(toponyms)
    print(toponyms)
    print(sequences)

    from pprint import pprint

    pprint(tokeniZer.word_index)
