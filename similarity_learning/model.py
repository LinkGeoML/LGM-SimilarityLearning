import os
import time
from typing import Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, \
    Dense, Input, Lambda, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adadelta

from similarity_learning.config import DirConf
from similarity_learning.distance import exponent_neg_manhattan_distance
from similarity_learning.utils import camel_to_underscore


class BaseNetMeta(type):
    """Meta class for injecting class method properties to BaseNet class"""

    @property
    def name(cls) -> str:
        """Returns the class name in underscored format."""
        return camel_to_underscore(cls.__name__)

    @property
    def filename(self) -> str:
        """Returns the filename corresponding to the specific model class"""

        return '{}.pth'.format(self.name)

    @property
    def default_path(self) -> str:
        """Returns the default path to model weights"""
        return os.path.join(DirConf.MODELS_DIR, self.filename)


class BaseNet(metaclass=BaseNetMeta):
    """Abstract class for each Neural Network model definition.

    Contains some utility class and instance methods helpful for
    basic model operations.
    """

    def __init__(self):
        self.__model: Optional[Model] = None

    @property
    def name(self) -> str:
        """Returns the class name in underscored format."""

        return self.__class__.name

    @property
    def filename(self) -> str:
        """Returns the filename corresponding to the specific model class"""

        return self.__class__.filename

    @property
    def default_path(self) -> str:
        """Returns the default path to model weights"""

        return self.__class__.default_path

    def load_weights(self, path: str = None):
        weights_path = path if path else self.default_path

        self.__model.load_weights(filepath=weights_path, by_name=True)

    def save_weights(self, path: str = None):
        """

        Parameters
        ----------
        path

        Returns
        -------

        """
        weights_path = path if path else self.default_path
        # torch.save(self.state_dict(), weights_path)
        self.__model.save_weights(filepath=weights_path,
                                  overwrite=True)

    def build(self):
        pass

    def plot_summary(self):
        print(self.__model.summary())


class SimilarityV1:

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.__model: Optional[Model] = None

    def build(self, dr: 0.3, learning_rate=0.0001):
        """

        :param dr:
        :param learning_rate:
        :return:
        """

        input_shape = ((self.__IMG_DIMENSIONS ** 2) * 3,)
        convolution_shape = (self.__IMG_DIMENSIONS, self.__IMG_DIMENSIONS, 3)

        seq_conv_model = Sequential()

        seq_conv_model.add(
            Reshape(input_shape=input_shape, target_shape=convolution_shape))
        seq_conv_model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        seq_conv_model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        seq_conv_model.add(MaxPooling2D(pool_size=(2, 2)))
        seq_conv_model.add(Dropout(dr))
        seq_conv_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        seq_conv_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        seq_conv_model.add(MaxPooling2D(pool_size=(2, 2)))
        seq_conv_model.add(Dropout(dr))
        seq_conv_model.add(Flatten())
        seq_conv_model.add(Dense(128, activation='relu'))
        seq_conv_model.add(Dropout(0.5))
        seq_conv_model.add(Dense(64, activation='sigmoid'))

        print(seq_conv_model.summary())

        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)

        output_x1 = seq_conv_model(input_x1)
        output_x2 = seq_conv_model(input_x2)

        distance_l1 = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))(
            [output_x1, output_x2])

        outputs = Dense(1, activation='sigmoid')(distance_l1)

        self.__model = Model([input_x1, input_x2], outputs)

        self.__model.compile(loss='binary_crossentropy',
                             optimizer=optimizers.Adam(lr=learning_rate),
                             metrics=['binary_accuracy'])

        self.__model.summary()

        return self.__model

    def fit(self, X, Y, hyper_parameters):
        initial_time = time.time()
        self.__model.fit(X, Y,
                         batch_size=hyper_parameters['batch_size'],
                         epochs=hyper_parameters['epochs'],
                         callbacks=hyper_parameters['callbacks'],
                         validation_split=0.2,
                         verbose=1
                         )
        final_time = time.time()
        eta = (final_time - initial_time)

        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'

        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(
            hyper_parameters['epochs'], eta, time_unit))

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def summary(self):
        self.__model.summary()

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model = models.load_model(file_path)


class SimilarityV2(BaseNet):

    def __init__(self):
        super().__init__()

    def build(self,
              max_seq_len,
              emb_matrix,
              emb_dim,
              grad_clip_norm,
              n_hidden: int = 50) -> Model:
        """

        :param max_seq_len:
        :param emb_matrix:
        :param emb_dim:
        :param grad_clip_norm:
        :param n_hidden:
        :return:
        """
        # The visible layer
        left_input = Input(shape=(max_seq_len,),
                           dtype='int32',
                           name='left_input')

        right_input = Input(shape=(max_seq_len,),
                            dtype='int32',
                            name='right_input')

        embedding_layer = Embedding(len(emb_matrix),
                                    emb_dim,
                                    weights=[emb_matrix],
                                    input_length=max_seq_len,
                                    trainable=False,
                                    name='emb_layer')

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(
            function=lambda tensors: exponent_neg_manhattan_distance(tensors),
            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        malstm = Model(inputs=[left_input, right_input],
                       outputs=[malstm_distance])

        # Ada-delta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=grad_clip_norm)

        malstm.compile(loss='mean_squared_error',
                       optimizer=optimizer,
                       metrics=['accuracy'])

        self.__model = malstm
        return malstm
