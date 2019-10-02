import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_features = 20_000
X_train = pd.DataFrame({'toponymX1': [],
                        'toponymX2': []})

X_val = pd.DataFrame({'toponymX1': [],
                      'toponymX2': []})

tokenizer = Tokenizer(num_words=max_features,
                      oov_token='<OOV>')

# fitting on the train dataset only
tokenizer.fit_on_texts(list(X_train['toponymX1']) + list(X_train['toponymX2']))

X_train['toponymX1_seqs'] = tokenizer.texts_to_sequences(X_train['toponymX1'])
X_train['toponymX2_seqs'] = tokenizer.texts_to_sequences(X_train['toponymX2'])

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
