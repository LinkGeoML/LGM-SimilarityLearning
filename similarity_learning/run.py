import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from similarity_learning.config import DirConf
from similarity_learning.utils_text import NgramTokenizer

pd.set_option('display.expand_frame_repr', False)

tqdm.pandas()


def run_preprocessing(fname='all_countries_filtered_rows_&_columns.csv',
                      verbose=1,
                      maxlen=30,
                      max_chars=32,
                      num_words=20_000,
                      n_rows=10_000,
                      save_tokenizer=False):
    """

    :param fname:
    :param verbose:
    :param maxlen:
    :param max_chars:
    :param num_words:
    :param n_rows:
    :param save_tokenizer:
    :return:
    """

    path = os.path.join(DirConf.DATA_DIR, fname)

    cols = ['name', 'alternate_names']

    data = pd.read_csv(path, nrows=n_rows, usecols=cols)
    data.dropna(inplace=True)
    data['len_name'] = data['name'].str.len()
    data['name'] = data['name'].str.lower()
    data['alternate_names'] = data['alternate_names'].str.lower()
    data['first_char'] = data['name'].str[0]

    # get rid of toponyms that have more than 32 characters.
    data = data[(data['len_name'] <= max_chars)
                & (data['len_name'] > 2)].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train, val, y_train, y_val = None, None, None, None

    X = data[cols].copy()
    y = data[['len_name']]

    # split the data based on the number of characters of the toponym and their
    # initial character
    for train_index, val_index in sss.split(X, y):
        train, val = X.loc[train_index], X.loc[val_index]
        # y_train, y_val = y.loc[train_index], y.loc[val_index]

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    token_params = dict(maxlen=maxlen, filters='', lower=True, split=' ',
                        char_level=False, num_words=num_words,
                        oov_token='<OOV>')

    tokenizer = NgramTokenizer(**token_params)

    # convert each toponym to it's ngram representation
    train['toponym_ngrams'] = train['name'].progress_apply(
        tokenizer.get_ngrams)

    # convert each variation of each toponym to it's n-gram representation
    train['variations_ngrams'] = train[
        'alternate_names'].str.split(',').progress_apply(
        tokenizer.texts_to_ngrams)

    # collect (flatten out) all the n-grams (toponyms and variations)
    # these are needed in order to fit it to the tokenizer.
    all_train_names = list()
    for row in tqdm(train['variations_ngrams']):
        all_train_names.extend(row)

    all_train_names += list(train['toponym_ngrams'])

    # fitting all the texts on the instantiated tokenizer
    # this will create all the necessary tools that we will need.
    tokenizer.fit_on_texts(texts=all_train_names)

    # using the fitted tokenizer, convert the train toponyms to sequences
    train['toponym_seqs'] = train['toponym_ngrams'].progress_apply(
        lambda x: tokenizer.texts_to_sequences(texts=[x])[0])

    # pad the sequence to the max length
    train['toponym_seqs'] = train['toponym_seqs'].progress_apply(
        tokenizer.pad_single)

    # using the fitted tokenizer, convert each of the variations of all the
    # toponyms sequences
    train['variations_seqs'] = train['variations_ngrams'].progress_apply(
        tokenizer.texts_to_sequences)

    train['variations_seqs'] = train['variations_seqs'].progress_apply(
        tokenizer.pad)

    #  ========== Same Procedure for the Validation Set ====================
    val['toponym_ngrams'] = val['name'].progress_apply(
        tokenizer.get_ngrams)

    val['variations_ngrams'] = val['alternate_names'].str.split(
        ',').progress_apply(tokenizer.texts_to_ngrams)

    val['toponym_seqs'] = val['toponym_ngrams'].progress_apply(
        lambda x: tokenizer.texts_to_sequences(texts=[x])[0])

    val['toponym_seqs'] = val['toponym_seqs'].progress_apply(
        tokenizer.pad_single)

    val['variations_seqs'] = val['variations_ngrams'].progress_apply(
        tokenizer.texts_to_sequences)

    val['variations_seqs'] = val['variations_seqs'].progress_apply(
        tokenizer.pad)

    if verbose > 0:
        print(f'N-gram index length: {len(tokenizer.word_index)}', end='\n\n')
        print('Example Transformation')
        print(train.loc[0])

    if save_tokenizer:
        print('Saving Tokenizer')
        tokenizer.save()
        print('Tokenizer saved')

    return {'tokenizer': tokenizer,
            'tokenizer_params': token_params,
            'x_train': train,
            'x_val': val}


if __name__ == "__main__":
    meta = run_preprocessing(save_tokenizer=True,
                             n_rows=100_000,
                             maxlen=30,
                             num_words=30_000)

    print(meta['x_val'])
