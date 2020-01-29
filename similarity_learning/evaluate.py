from datetime import datetime
from typing import Dict

from similarity_learning.dataset import TestDataset
from similarity_learning.experiment import Components
from similarity_learning.logger import exp_logger


class ExperimentEvaluator:
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        """
        self.dataset_params = kwargs['dataset']
        self.sampler_params = kwargs['test_sampler']
        self.model_params = kwargs['model']
        self.criterion_params = kwargs['criterion']
        self.optimizer_params = kwargs.get('optimizer')
        self.metrics_params = kwargs.get('metrics', {})

        self._logger = None
        # Obtain a name for the current experiment
        self.exp_name = kwargs.get('exp_name') or self.build_name()

        # Lazily loads the logger
        self.logger.info('Experiment name: {}'.format(self.exp_name))

        self.components = Components(logger=self.logger)

        # Prepare Evaluator to be ready for running!
        self.evaluator = self.prepare()

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

    @property
    def logger(self):
        """

        Returns
        -------

        """
        if self._logger is None:
            # =========== Setting the Experiment logger ==============
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
            metrics=self.metrics_params)

    def prepare(self) -> dict:
        """

        Returns
        -------

        """
        self.logger.info('Preparing the Test Dataset')

        dataset_params = self.dataset_params
        dataset_params['sampler_params'] = self.sampler_params

        dataset = TestDataset(**dataset_params)

        self.logger.info('Creating model')

        model = self.components.get_model(**self.model_params)

        # # loss
        # criterion = self.components.get_criterion(**self.criterion_params)

        # # usually adam
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
                      # criterion=criterion,
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
        self.evaluator['dataset'].run_data_preparation()

        print(self.evaluator['dataset'].data_)
