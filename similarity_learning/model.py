import os
from typing import Optional, NoReturn

import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Lambda, Embedding, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.callbacks import (EarlyStopping,
                                               ReduceLROnPlateau,
                                               ModelCheckpoint, TensorBoard)

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

        return f'{self.name}.h5'

    @property
    def default_path(self) -> str:
        """Returns the default path to model weights"""
        return os.path.join(DirConf.MODELS_DIR, self.filename)


class BaseNet(metaclass=BaseNetMeta):
    """Abstract class for each Neural Network model definition.

    Contains some utility class and instance methods helpful for
    basic model operations.
    """

    def __init__(self, encoder: Sequential):
        self._model: Optional[Model] = None
        self.encoder = encoder

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

    def load_weights(self, path: str = None) -> NoReturn:
        """

        Parameters
        ----------
        path : str
            Use as specific path for loading pre-trained weights

        Returns
        -------
        NoReturn
        """
        weights_path = path if path else self.default_path
        weights_path = str(weights_path)

        self._model.load_weights(filepath=weights_path, by_name=True)

    def save_weights(self, path: str = None) -> NoReturn:
        """

        Parameters
        ----------
        path

        Returns
        -------

        """
        weights_path = path if path else self.default_path
        self._model.save_weights(filepath=weights_path, overwrite=True)

    def save_model(self, file_path):
        """
        Save model architecture and weights
        Parameters
        ----------
        file_path

        Returns
        -------

        """
        self._model.save(file_path)

    def load_model(self, file_path):
        """
        Load model architecture and weights
        Parameters
        ----------
        file_path

        Returns
        -------

        """
        self._model = models.load_model(file_path)

    def build(self, max_features: int, maxlen: int, emb_dim: int,
              n_hidden: int = 50):
        """

        Parameters
        ----------
        max_features
        maxlen
        emb_dim
        n_hidden

        Returns
        -------

        """
        raise NotImplementedError

    def set_callbacks(self) -> list:
        """

        Returns
        -------

        """
        monitor = 'val_loss'

        es = EarlyStopping(monitor=monitor,
                           patience=3,
                           verbose=1,
                           restore_best_weights=True)

        rop = ReduceLROnPlateau(monitor=monitor,
                                patience=5,
                                verbose=1)

        checkpoint = ModelCheckpoint(filepath=self.default_path,
                                     monitor=monitor,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     load_weights_on_restart=False)

        logs_dir = os.path.join(DirConf.LOG_DIR, 'experiment')
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)

        tb = TensorBoard(log_dir=logs_dir)
        return [es, rop, checkpoint, tb]

    def plot_summary(self) -> NoReturn:
        """

        Returns
        -------

        """
        print(self._model.summary())

    def plot_model_architecture(self, fname: str) -> NoReturn:
        """

        Parameters
        ----------
        fname

        Returns
        -------

        """
        plot_model(self._model,
                   to_file=fname,
                   show_shapes=True,
                   show_layer_names=True)

    def fit(self, train_gen, val_gen, e: int, multi_process: bool = False):
        """

        Parameters
        ----------
        train_gen
        val_gen
        e
        multi_process

        Returns
        -------

        """

        history = self._model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.steps_per_epoch,
            verbose=1,
            validation_data=val_gen,
            validation_steps=val_gen.steps_per_epoch,
            use_multiprocessing=multi_process,
            workers=1,
            callbacks=self.set_callbacks(),
            epochs=e)

        return history

    def evaluate(self, test_gen, y_true: np.ndarray, show_info: bool = True,
                 multi_process: bool = False) -> np.ndarray:
        """

        Parameters
        ----------
        test_gen
        y_true
        show_info
        multi_process

        Returns
        -------

        """
        # make predictions on the testing toponym pairs, finding the index
        # of the label with the corresponding largest predicted probability
        predictions = self._model.evaluate(
            test_gen,
            verbose=1,
            use_multiprocessing=multi_process,
            workers=1)

        return predictions


class SiameseNet(BaseNet):

    def __init__(self, encoder: Sequential):
        """

        Parameters
        ----------
        encoder
        """
        super().__init__(encoder)

    def build(self,
              max_features,
              maxlen,
              emb_dim,
              n_hidden: int = 50) -> Model:
        """

        Parameters
        ----------
        max_features
        maxlen
        emb_dim
        n_hidden

        Returns
        -------

        """
        # The visible layer
        left_input = Input(shape=(maxlen,), dtype='int32', name='left_input')

        right_input = Input(shape=(maxlen,), dtype='int32', name='right_input')

        # input_dim: int > 0. Size of the vocabulary,
        #  i.e. maximum integer index + 1.
        #  output_dim: int >= 0. Dimension of the dense embedding.
        embedding_layer = Embedding(
            input_dim=max_features + 1, output_dim=emb_dim, trainable=True,
            mask_zero=True, name='emb_layer')

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same encoder
        shared_encoder = self.encoder(n_hidden)

        # # Since this is a siamese network, both sides share the same encoder
        left_output = shared_encoder(encoded_left)
        right_output = shared_encoder(encoded_right)

        # Calculates the distance between the vectors
        distance = Lambda(
            function=lambda tensors: exponent_neg_manhattan_distance(tensors),
            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        model = Model(inputs=[left_input, right_input], outputs=[distance])

        self._model = model

        print(model.summary())
        return model


class SiameseNetV2(BaseNet):

    def __init__(self, encoder: Sequential):
        """

        Parameters
        ----------
        encoder
        """
        super().__init__(encoder)

    def build(self,
              max_features,
              maxlen,
              emb_dim,
              n_hidden: int = 50) -> Model:
        """

        Parameters
        ----------
        max_features
        maxlen
        emb_dim
        n_hidden

        Returns
        -------

        """
        # The visible layer
        left_input = Input(shape=(maxlen,), dtype='int32', name='left_input')

        right_input = Input(shape=(maxlen,), dtype='int32', name='right_input')

        # input_dim: int > 0. Size of the vocabulary,
        #  i.e. maximum integer index + 1.
        #  output_dim: int >= 0. Dimension of the dense embedding.
        embedding_layer = Embedding(
            input_dim=max_features + 1, output_dim=emb_dim, trainable=True,
            mask_zero=True, name='emb_layer')

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same encoder
        shared_encoder = self.encoder(n_hidden)

        # # Since this is a siamese network, both sides share the same encoder
        left_output = shared_encoder(encoded_left)
        right_output = shared_encoder(encoded_right)

        # Calculates the distance between the vectors
        distance = Lambda(
            function=lambda tensors: exponent_neg_manhattan_distance(tensors),
            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Add a dense layer with a sigmoid unit to generate the similarity
        # score
        prediction = Dense(1, activation='sigmoid')(distance)

        # Pack it all up into a model
        model = Model(inputs=[left_input, right_input], outputs=[prediction])

        self._model = model

        print(model.summary())
        return model
