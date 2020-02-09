"""Command line interface for operation management"""
import argparse
import os
from typing import NoReturn

import yaml

from similarity_learning.evaluate import ExperimentEvaluator
from similarity_learning.experiment import Experiment
from similarity_learning.scripts.handle_raw_dataset import RawDataPreprocessor


class LGMInterface:
    """Command Line Interface for the LGM-Interlinking"""

    def __init__(self) -> NoReturn:
        """
        The ComLInt is the main CLI of the Similarity Approach.
        It is responsible for handling different groups of actions with diff
        arguments per action.
        """
        self.parser = argparse.ArgumentParser(
            description='Run various training and evaluation ops.')

        self.subparsers = self.parser.add_subparsers(
            help='sub-command help', required=True, dest='action')

        # create parser for "train" command
        self.train_parser = self.subparsers.add_parser(
            'train',
            help='triggers the training pipeline of the selected model')

        self.dataset_parser = self.subparsers.add_parser(
            'dataset',
            help='triggers the raw data pre-processing')

        self.evaluation_parser = self.subparsers.add_parser(
            'evaluate',
            help='triggers the evaluation pipeline of the selected model')

    def run(self) -> NoReturn:
        """
        This method instantiates all previous methods in order to parse all
        the arguments for all the actions in the CLI.

        Then returns a tuple containing the actual action and the parameters
        needed for that action to run.
        Returns
        -------
        None
        """

        # Experiment related arguments
        self.train_parser.add_argument(
            '--exp_name', type=str, help='experiment name. Eg. LGM_SiameseNet')

        self.train_parser.add_argument(
            '--settings', type=str, default='similarity2.yml',
            help='path for YAML configuration file containing default params')

        # Raw Dataset Split Related Arguments
        self.dataset_parser.add_argument(
            '--dataset_name', type=str, default='allCountries.txt',
            help='dataset name')

        self.dataset_parser.add_argument(
            '--n_alternates', type=int, default=1,
            help='Minimum number of alternate names')

        self.dataset_parser.add_argument(
            '--only_latin', type=bool, default=False,
            help='Whether to use only Latin Alphabet records')

        self.dataset_parser.add_argument(
            '--stratified_split', type=bool, default=False,
            help='Whether to use stratified shuffle splint when breaking in'
                 ' train-val-test sets')

        self.evaluation_parser.add_argument(
            '--settings', type=str, default='test_similarity.yml',
            help='path for YAML configuration file containing default params')

        cmd_args = self.parser.parse_args()

        if cmd_args.action == 'train':

            if cmd_args.settings:
                path = os.path.join('config', 'train', cmd_args.settings)
                # Load default configurations from YAML file
                with open(path, 'r') as f:
                    experiment_params = yaml.safe_load(f)

            # experiment name
            experiment_params['exp_name'] = cmd_args.exp_name

            if cmd_args.action == 'train':
                exp = Experiment(**experiment_params)
                # exp.run()

        elif cmd_args.action == 'dataset':

            options = {'fname': cmd_args.dataset_name,
                       'n_alternates': cmd_args.n_alternates,
                       'only_latin': cmd_args.only_latin,
                       'stratified_split': cmd_args.stratified_split}

            preprocessor = RawDataPreprocessor(**options)
            preprocessor.run()

        elif cmd_args.action == 'evaluate':

            if cmd_args.settings:
                path = os.path.join('config', 'test', cmd_args.settings)
                # Load default configurations from YAML file
                with open(path, 'r') as f:
                    test_params = yaml.safe_load(f)

                evaluator = ExperimentEvaluator(**test_params)
                evaluator.run()


if __name__ == "__main__":
    interface = LGMInterface()
    interface.run()
