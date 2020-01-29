from datetime import datetime
from pathlib import Path
from typing import Dict

from tensorflow.keras import losses as tf_losses
from tensorflow.keras import optimizers as optim
from tensorflow.keras.optimizers import Optimizer

import similarity_learning.encode as encoders
import similarity_learning.loss as losses
import similarity_learning.metric as custom_metrics
import similarity_learning.model as models
from similarity_learning.config import DirConf
from similarity_learning.dataset import Dataset
from similarity_learning.logger import exp_logger
from similarity_learning.utils import underscore_to_camel


class Components:
    """Class that handles all the preparation steps and loads all the necessary
     objects for the training procedure."""

    def __init__(self, logger=None):
        """

        Parameters
        ----------
        logger
        """
        self._logger = logger

    @property
    def logger(self):
        """

        Returns
        -------

        """
        if self._logger is None:
            self._logger = exp_logger

        return self._logger

    def get_model(self, **params):
        """
        This function obtains 2 models.
        The encoder models and the Similarity Model
        It then injects the encoder model within the Similarity Model
        Parameters
        ----------
        params : dict
            Optional parameters that we use for the definition of the model

        Returns
        -------
            An instantiated custom model object. All models inherit from the
            Base class
        """

        # Prepare models working directory
        # NOTE: Creates model directory if not already exists!
        models_dir = Path(DirConf.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        encoder_name = params.pop('encoder')
        self.logger.info('Encoder name: {}'.format(encoder_name))
        encoder = getattr(encoders, encoder_name)

        model_name = underscore_to_camel(params.pop('name'))
        self.logger.info('Model name: {}'.format(model_name))

        ModelClass = getattr(models, model_name)

        # Instantiate chosen model
        model = ModelClass(encoder=encoder)

        return model

    def get_criterion(self, **params):
        """
        This method get the loss criterion. At first it searches in the custom
        losses. If not found it searches in the usual tensorflow losses.
        Parameters
        ----------
        params: dict
            Parameters that we need in order to search for the criterion.
        Returns
        -------

        """
        criterion_name = params.pop('name')
        criterion_name_camel = underscore_to_camel(criterion_name)

        self.logger.info('Selected criterion name: {}'.format(criterion_name))

        try:
            criterion = getattr(losses, criterion_name)
        except AttributeError:
            self.logger.warning('Criterion not found in custom losses')

            CriterionClass = getattr(tf_losses, criterion_name_camel)
            criterion = CriterionClass(**params)
            self.logger.info('Criterion found in tf losses')

        return criterion

    def get_optimizer(self, **params) -> Optimizer:
        """
        This function instantiates the custom model's optimizer.

        Parameters
        ----------
        params : dict
            Optional parameters that we use for the definition of
            the optimizer name

        Returns
        -------
            An instantiated optimizer
        """
        optimizer_name = underscore_to_camel(params.pop('name'))

        self.logger.info('Optimizer name: {}'.format(optimizer_name))

        OptimizerClass = getattr(optim, optimizer_name)

        # Collect trainable parameters and pass them through the optimizer
        # instance
        optimizer = OptimizerClass(**params)

        return optimizer

    def get_metrics(self, metric_names: Dict[str, str]) -> dict:
        """
        For each key (metric name) in the metric_names dictionary try and get
        the metric class.

        metric_names example:
        {'accuracy': 'primary',
         'AUC': 'primary',
         'precision': 'secondary'}

        Parameters
        ----------
        metric_names : dict
        Returns
        -------
        dict
        """
        metrics = {}
        for metric_name in metric_names:
            try:
                # at first check the custom metrics
                metrics[metric_name] = getattr(custom_metrics, metric_name)
                self.logger.info(f'Using custom metric: {metric_name}')

            except AttributeError:
                self.logger.warning(
                    f'Metric not found in custom metrics: {metric_name}')

                self.logger.info(f'Checking Keras Metrics for: {metric_name}')
                metrics[metric_name] = metric_name

        return metrics


class Experiment:

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        """
        self.dataset_params = kwargs['dataset']
        self.tokenizer_params = kwargs['tokenizer']
        self.train_sampler_params = kwargs['train_sampler']
        self.val_sampler_params = kwargs['val_sampler']
        self.model_params = kwargs['model']
        self.criterion_params = kwargs['criterion']
        self.optimizer_params = kwargs.get('optimizer')
        self.metrics_params = kwargs.get('metrics', {})
        self.training_params = kwargs['training']

        self._logger = None
        # Obtain a name for the current experiment
        self.name = kwargs.get('name') or self.build_name()

        # Lazily loads the logger
        self.logger.info('Model name: {}'.format(self.name))

        self.components = Components(logger=self.logger)

        # Prepare Trainer to be ready for running!
        self.trainer = self.prepare()

    @property
    def logger(self):
        """

        Returns
        -------

        """
        if self._logger is None:
            # =========== Setting the Application logger ==============
            # put the  version and the model_name to the env variables
            # in order to be able to use it throughout all the modules
            self._logger = exp_logger
            self._logger.info('Experiment logger initialized')
            # =========================================================

        return self._logger

    @property
    def hyperparams(self) -> Dict[str, dict]:
        """
        Property that provides all the hyper parameters of the experiment.
        Returns
        -------
        Dict[str, dict]
            The components names and the corresponding params
        """

        return dict(
            dataset=dict(**self.dataset_params),
            model=dict(**self.model_params),
            criterion=dict(**self.criterion_params),
            optimizer=dict(**self.optimizer_params),
            training=dict(**self.training_params),
            metrics=self.metrics_params)

    def build_name(self) -> str:
        """
        This method creates the models name based on the date of the experiment
        Returns
        -------
        str
        """
        name = '{}-{}'.format(
            datetime.now().strftime('%Y-%m-%d'), self.model_params['name'])

        return name

    def prepare(self) -> dict:
        """

        Returns
        -------

        """
        self.logger.info('Preparing the datasets')

        dataset_params = self.dataset_params
        dataset_params['tokenizer_params'] = self.tokenizer_params
        dataset_params['train_sampler_params'] = self.train_sampler_params
        dataset_params['val_sampler_params'] = self.val_sampler_params

        dataset = Dataset(**dataset_params)

        self.logger.info('Creating model')

        model = self.components.get_model(**self.model_params)

        # loss
        criterion = self.components.get_criterion(**self.criterion_params)

        # usually adam
        optimizer = self.components.get_optimizer(**self.optimizer_params)

        # usually accuracy and distance_accuracy
        metrics = self.components.get_metrics(self.metrics_params)

        primary_metrics = [name
                           for name, pri in self.metrics_params.items()
                           if pri == 'primary']

        self.logger.info(
            'Selected primary metrics: {}'.format(' | '.join(primary_metrics)))

        output = dict(dataset=dataset,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      metrics=metrics,
                      primary_metrics=primary_metrics)

        return output

    def run(self):
        self.logger.info('Hyper-params: {}'.format(self.hyperparams))

        # load the data
        # split in train val test
        # create tokenizer and fit on data
        # create the two samplers
        self.trainer['dataset'].run_data_preparation()

        # the model is already instantiated.
        # we need to build the actual model
        num_words = self.trainer['dataset'].tokenizer_params['num_words']
        maxlen = self.trainer['dataset'].tokenizer_params['maxlen']

        model = self.trainer['model']

        model.build(max_features=num_words,
                    maxlen=maxlen,
                    emb_dim=100,
                    n_hidden=50)

        metrics = [self.trainer['metrics'][metric_name] for metric_name in
                   self.trainer['primary_metrics']]

        model._model.compile(loss=self.trainer['criterion'],
                             optimizer=self.trainer['optimizer'],
                             metrics=metrics)

        history = model.fit(
            train_gen=self.trainer['dataset'].train_sampler,
            val_gen=self.trainer['dataset'].val_sampler,
            e=self.training_params['num_epochs'],
            multi_process=self.training_params['multi_process'])

        return history


if __name__ == "__main__":
    parameters = dict(
        dataset=dict(
            train_fname="n_alternates_1+_latin_stratified_split_x_train.csv",
            val_fname="n_alternates_1+_latin_stratified_split_x_val.csv",
            max_chars=32),
        tokenizer=dict(
            name="ngram_tokenizer",
            maxlen=30,
            num_words=50000),
        train_sampler=dict(
            name="sampler",
            batch_size=2048,
            n_positives=1,
            n_negatives=3,
            neg_samples_size=30,
            shuffle=True),
        val_sampler=dict(
            name="sampler",
            batch_size=2048,
            n_positives=1,
            n_negatives=3,
            neg_samples_size=30,
            shuffle=True),
        model=dict(
            name="siamese_net",
            encoder="lstm2"),
        criterion=dict(
            name="binary_crossentropy"),
        optimizer=dict(
            name="adam",
            lr=0.001),
        metrics=dict(
            accuracy="primary",
            AUC='primary'),
        training=dict(
            num_epochs=50,
            num_workers=1,
            multi_process=False
        )
    )

    experiment = Experiment(**parameters)

    experiment.run()
