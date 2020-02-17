import io
import json
import os
import sys
from typing import List, Optional, NoReturn, Union

from more_itertools import windowed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from similarity_learning.config import DirConf
from similarity_learning.logger import exp_logger


class CustomTokenizer(Tokenizer):
    def __init__(self, maxlen=None, exp_name: Optional[str] = None, **kwargs):
        """

        :param maxlen:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.exp_name = exp_name
        self.name = f'tokenizer_nw_{self.num_words}_ml_{self.maxlen}.json'

    def build_outfile(self) -> str:
        """

        Returns
        -------

        """
        outfile = []

        if self.exp_name:
            outfile.append(self.exp_name)

        outfile.append('tokenizer')

        if self.num_words is None:
            self.num_words = len(self.index_word)

        outfile.extend(['nw', str(self.num_words)])

        outfile.extend(['ml', str(self.maxlen)])

        self.name = '_'.join(outfile) + '.json'

        return self.name

    def pad(self, seqs):
        """

        Parameters
        ----------
        seqs

        Returns
        -------

        """
        return pad_sequences(seqs,
                             maxlen=self.maxlen,
                             padding='post',
                             truncating='post')

    def pad_single(self, sequence):
        """

        Parameters
        ----------
        sequence

        Returns
        -------

        """
        return pad_sequences(sequences=[sequence],
                             maxlen=self.maxlen,
                             padding='post',
                             truncating='post')[0]

    def to_json(self, **kwargs):
        """Returns a JSON string containing the tokenizer configuration.
        To load a tokenizer from a JSON string, use
        `keras.preprocessing.text.tokenizer_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        config['maxlen'] = self.maxlen

        tokenizer_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(tokenizer_config, **kwargs)

    def save(self) -> NoReturn:
        """

        :return:
        """

        self.name = self.build_outfile()

        tokenizer_json = self.to_json()

        path = os.path.join(DirConf.MODELS_DIR, self.name)

        exp_logger.info(f'Storing Tokenizer at: {path}')

        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=True))


class TrigramTokenizer(CustomTokenizer):
    def __init__(self, maxlen=None, exp_name: Optional[str] = None, **kwargs):
        """

        :param maxlen:
        :param kwargs:
        """
        super().__init__(maxlen, exp_name, **kwargs)
        self.maxlen = maxlen
        self.name = f'trigram_tokenizer_nw_{self.num_words}_ml_{self.maxlen}.json'

    def build_outfile(self) -> str:
        """

        Returns
        -------

        """
        outfile = []

        if self.exp_name:
            outfile.append(self.exp_name)

        outfile.append('trigram')
        outfile.append('tokenizer')

        if self.num_words is None:
            self.num_words = len(self.index_word)

        outfile.extend(['nw', str(self.num_words)])

        outfile.extend(['ml', str(self.maxlen)])

        self.name = '_'.join(outfile) + '.json'

        return self.name

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
                [''.join(t) for t in windowed(seq=chars, n=n, step=1)])
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

    def save(self):
        """

        :return:
        """
        self.name = self.build_outfile()

        tokenizer_json = self.to_json()

        path = os.path.join(DirConf.MODELS_DIR, self.name)

        exp_logger.info(f'Storing Trigram Tokenizer at: {path}')

        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=True))


class UnigramTokenizer(CustomTokenizer):
    def __init__(self, maxlen=None, exp_name: Optional[str] = None, **kwargs):
        """

        :param maxlen:
        :param kwargs:
        """
        super().__init__(maxlen, exp_name, **kwargs)

        self.maxlen = maxlen
        self.name = f'unigram_tokenizer_nw_{self.num_words}_ml_{self.maxlen}.json'

    def build_outfile(self) -> str:
        """

        Returns
        -------

        """
        outfile = []

        if self.exp_name:
            outfile.append(self.exp_name)

        outfile.append('unigram')
        outfile.append('tokenizer')

        if self.num_words is None:
            self.num_words = len(self.index_word)

        outfile.extend(['nw', str(self.num_words)])

        outfile.extend(['ml', str(self.maxlen)])

        self.name = '_'.join(outfile) + '.json'

        return self.name

    @staticmethod
    def get_ngrams(text: str) -> str:
        """

        :param text:
        :param n:
        :param step:
        :return:
        """
        output = []

        # split the sentence in tokens.
        tokens = text.strip().split()

        # if only one token, then we only have BOS and EOS tags
        if len(tokens) == 1:

            chars = ['<BOS>'] + list(text) + ['<EOS>']
            output.append(' '.join(chars))

        elif len(tokens) == 2:
            chars = ['<BOS>'] + list(tokens[0]) + ['<IOS>'] + list(
                tokens[1]) + ['<EOS>']

            output.append(' '.join(chars))

        # We have more than 1 tokens. So we need 3 kind of tags:
        # BOS: beginning of sentence
        # IOS: inside of sentence
        # EOS: end of sentence
        else:
            # extracting the first token, a list of the inside tokens, and the
            # last token. We handle each one differently
            first, *inside, last = tokens

            chars = ['<BOS>'] + list(first) + ['<IOS>']
            for token in inside:
                chars += list(token)
                chars += ['<IOS>']
            chars += list(last)
            chars += ['<EOS>']

            output.append(' '.join(chars))

        return ' '.join(output)

    def texts_to_ngrams(self, texts, n: int = 1, step=1) -> List[str]:
        output = list()

        for text in texts:
            output.append(self.get_ngrams(text))

        return output

    def save(self):
        """

        :return:
        """
        self.name = self.build_outfile()

        tokenizer_json = self.to_json()

        path = os.path.join(DirConf.MODELS_DIR, self.name)

        exp_logger.info(f'Storing Unigram Tokenizer at: {path}')

        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=True))


def tokenizer_from_json(json_string):
    """
    Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.

    :param json_string: JSON string encoding a tokenizer configuration.
    :return: A Keras Tokenizer instance
    """

    tokenizer_config = json.loads(json_string)

    class_name = tokenizer_config['class_name']

    class_config = tokenizer_config.get('config')

    word_counts = json.loads(class_config.pop('word_counts'))
    word_docs = json.loads(class_config.pop('word_docs'))
    index_docs = json.loads(class_config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(class_config.pop('index_word'))
    # Integer indexing gets converted to strings with json.dumps()
    index_word = {int(k): v for k, v in index_word.items()}

    word_index = json.loads(class_config.pop('word_index'))

    TokenizerCLass = getattr(sys.modules[__name__], class_name)

    tokenizer = TokenizerCLass(**class_config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    tokenizer.num_words = class_config['num_words']
    tokenizer.filters = class_config['filters']
    tokenizer.lower = class_config['lower']
    tokenizer.split = class_config['split']
    tokenizer.char_level = class_config['char_level']
    tokenizer.oov_token = class_config['oov_token']
    tokenizer.document_count = class_config['document_count']

    return tokenizer


def load_tokenizer(name: str) -> Union[UnigramTokenizer, TrigramTokenizer]:
    """

    :return:
    """
    path = os.path.join(DirConf.MODELS_DIR, name)

    with open(path) as f:
        json_string = json.load(f)
        tokenizer = tokenizer_from_json(json_string)

    return tokenizer


if __name__ == "__main__":
    from pprint import pprint

    toponyms = ['athens',
                'athens greece',
                'athens gr greece',
                'athina',
                'athina gr']

    unigram_tokenizer = UnigramTokenizer(
        num_words=None, oov_token='<OOV>', lower=True, char_level=False,
        split=' ', maxlen=30, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')

    unigrams = unigram_tokenizer.texts_to_ngrams(texts=toponyms)

    pprint(unigrams)

    unigram_tokenizer.fit_on_texts(unigrams)
    unigram_sequences = unigram_tokenizer.texts_to_sequences(unigrams)

    print(unigram_sequences)
    print(unigram_tokenizer.index_word)

    trigram_params = dict(num_words=None, oov_token='<OOV>', lower=True,
                          char_level=False, split=' ', maxlen=30,
                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    trigram_tokenizer = TrigramTokenizer(**trigram_params)

    toponyms_3grams = trigram_tokenizer.texts_to_ngrams(texts=toponyms)

    pprint(toponyms_3grams)
    trigram_tokenizer.fit_on_texts(toponyms_3grams)
    trigram_sequences = trigram_tokenizer.texts_to_sequences(toponyms_3grams)
    print(trigram_tokenizer.index_word)
    print(trigram_sequences)

    trigram_tokenizer.save()

    trigram_tokenizer = TrigramTokenizer(**trigram_params).load()
    print(trigram_tokenizer)
    print(trigram_tokenizer.maxlen)
    print(trigram_tokenizer.num_words)
    print(trigram_tokenizer.index_word)
