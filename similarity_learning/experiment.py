from pathlib import Path
from typing import Dict

from tensorflow.keras import losses as tf_losses
from tensorflow.keras import optimizers as optim
from tensorflow.keras.optimizers import Optimizer

import similarity_learning.dataset as datasets
import similarity_learning.loss as losses
import similarity_learning.metric as metrics_module
import similarity_learning.model as models
from similarity_learning.config import DirConf
from similarity_learning.logger import DummyLogger
from similarity_learning.utils import timer, underscore_to_camel


class ExperimentSetup:
    """Class that handles all the preparation steps and loads all the necessary
     objects for the training procedure
     """

    def __init__(self, logger=None):

        self._logger = logger

    @property
    def logger(self):
        """

        Returns
        -------

        """
        if self._logger is None:
            self._logger = DummyLogger()

        return self._logger

    def get_datasets(self, **params) -> dict:
        """
        This method creates the train and validation datasets.

        Parameters
        ----------
        params : dict

        Returns
        -------
        Dict[str, Dataset]

        """

        dataset_name = underscore_to_camel(params.pop('name'))
        DatasetClass = getattr(datasets, dataset_name)

        train_path = params.pop('train_path', None)
        val_path = params.pop('val_path', None)

        with timer('loading train split'):
            self.logger.info('Loading train split')

            train_ds = DatasetClass(mode='train', path=train_path, **params)

        with timer('loading validation split'):
            self.logger.info('Loading validation split')

            val_ds = DatasetClass(mode='val', path=val_path, **params)

        return {'train': train_ds, 'val': val_ds}

    def get_model(self, **params):
        """
        This function obtains the Custom model

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

        model_name = underscore_to_camel(params.pop('name'))

        self.logger.debug('Model name: {}'.format(model_name))

        ModelClass = getattr(models, model_name)

        # Freeze layers based on the provided layers list.
        # NOTE: "params.pop" is used on purpose in oder to remove
        # keys which should not pass through to class instantiation.
        layers_whitelist = params.pop('layers_whitelist', None)

        # Instantiate chosen model
        model = ModelClass(**params)

        # TODO: Implement freezing
        # if layers_whitelist:
        #     self.logger.debug(
        #         'Freezing the following '
        #         'layers: {}'.format(' | '.join(layers_whitelist)))
        #
        #     model.freeze_params(layers_whitelist)

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
        criterion_name = underscore_to_camel(params.pop('name'))

        self.logger.debug('Selected criterion name: {}'.format(criterion_name))

        try:
            CriterionClass = getattr(losses, criterion_name)
        except AttributeError:

            self.logger.warning(
                'Criterion not found in custom losses. Fallback')

            CriterionClass = getattr(tf_losses, criterion_name)

        criterion = CriterionClass(**params)

        return criterion

    def get_optimizer(self, model, **params) -> Optimizer:
        """
        This function instantiates the custom model's optimizer.

        Parameters
        ----------
        model : obj
            An instantiated tensorflow model
        params : dict
            Optional parameters that we use for the definition of
            the optimizer name

        Returns
        -------
            An instantiated optimizer

        """
        optimizer_name = underscore_to_camel(params.pop('name'))

        self.logger.debug('Optimizer name: {}'.format(optimizer_name))

        OptimizerClass = getattr(optim, optimizer_name)

        # Collect trainable parameters and pass them through
        # the optimizer instance
        optimizer = OptimizerClass(model.trainable_params, **params)

        return optimizer

    def get_metrics(self, metric_names: Dict[str, str]) -> dict:
        """
        For each key (metric name) in the metric_names dictionary try and get
        the metric class.

        metric_names example:
        {'accuracy': 'primary',
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
                metrics[metric_name] = getattr(metrics_module, metric_name)

                self.logger.debug('Using metric: {}'.format(metric_name))

            except AttributeError:
                self.logger.warning('Metric not found: {}'.format(metric_name))
                metrics[metric_name] = None
        return metrics


class Experiment:

    def __init__(self, **kwargs):
        pass

    def run(self):
        pass

    def evaluate(self):
        pass
