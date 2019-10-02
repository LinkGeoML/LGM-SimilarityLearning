"""Command line interface for operation management"""
import argparse
from typing import Tuple

import yaml

from similarity_learning.experiment import Experiment


class ComLInt:
    """Command Line Interface for the LGM-Interlinking"""

    TRAINING_GROUP_NAMES = [
        'dataset',
        'model',
        'criterion',
        'optimizer',
        'training',
        'metrics']

    EVALUATION_GROUP_NAMES = [
        'dataset',
        'model',
        'criterion',
        'training',
        'metrics']

    def __init__(self) -> None:
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
            help='triggers the training operation of the selected model')

        # create parser for "eval" command
        self.eval_parser = self.subparsers.add_parser(
            'evaluate',
            help='evaluate help')

    def experiment_arguments(self) -> None:
        """
        This method create arguments needed only for the experimentation
        procedures/actions. These actions are the training and the evaluation
        of a particular nn model.
        Returns
        -------
        None
        """

        self.train_parser.add_argument(
            '--exp_name', type=str,
            help='experiment name  E.g. similarity_classifier_v1')

        self.eval_parser.add_argument(
            '--exp_name', type=str,
            help='experiment name  E.g. similarity_classifier_v1')

        self.train_parser.add_argument(
            '--settings', type=str, default='config/triplet.yml',
            help='path for YAML configuration file containing default params')

        self.eval_parser.add_argument(
            '--settings', type=str,
            help='path for YAML configuration file containing default params')

    def parse_dataset_options(self) -> None:
        """
        This method creates arguments/flags both for training and evaluation
        phase about any dataset options.
        Returns
        -------
        None
        """
        # --- Dataset options ---

        for parser in ['train_parser', 'eval_parser']:
            dataset_group = vars(self)[parser].add_argument_group('dataset')

            dataset_group.add_argument(
                '--dataset-name', type=str,
                help='the name of dataset class to be used')

            dataset_group.add_argument(
                '--dataset-train_path', type=str,
                help='the path towards training split file')

            dataset_group.add_argument(
                '--dataset-val_path', type=str,
                help='the path towards validation split file')

            dataset_group.add_argument(
                '--dataset-eval_path', type=str,
                help='the path towards evaluation set file')

    def parse_model_options(self) -> None:
        """
        This method creates arguments/flags both for training and evaluation
        phase about any model options.
        Returns
        -------
        None
        """
        # --- Model options ---
        for parser in ['train_parser', 'eval_parser']:
            model_group = vars(self)[parser].add_argument_group('model')

            model_group.add_argument(
                '--model-name', type=str,
                help='the name of the training model to be used')

            model_group.add_argument(
                '--model-embeddings_dim', type=int,
                help='the dimensions of toponym n-gram embeddings')

            model_group.add_argument(
                '--model-num_classes', type=int,
                help='the number of target classes in a classification model')

    def parse_criterion_options(self) -> None:
        """
        This method creates arguments/flags both for training and evaluation
        phase about any criterion options.
        Returns
        -------
        None
        """
        # --- Criterion options ---
        for parser in ['train_parser', 'eval_parser']:
            criter_group = vars(self)[parser].add_argument_group('criterion')

            criter_group.add_argument(
                '--criterion-name', type=str,
                help='the name of the loss function to be used')

    def parse_optimizer_options(self) -> None:
        """
        This method creates arguments/flags for the training phase about any
        optimizer options.
        Returns
        -------
        None
        """
        # --- Optimizer options ---
        # only for the trainer
        optimizer_group = self.train_parser.add_argument_group('optimizer')

        optimizer_group.add_argument(
            '--optimizer-name', type=str,
            help='the name of the optimizer algorithm to be used')

        optimizer_group.add_argument(
            *['--optimizer-lr', '--lr'],
            type=float, dest='optimizer_lr',
            help='the initial learning rate of optimizer')

        optimizer_group.add_argument(
            '--optimizer-momentum', type=float,
            help='the initial momentum value of optimizer (if supported)')

    def parse_general_training_options(self) -> None:
        """
        This method creates arguments/flags both for training and evaluation
        phase about any general training options.
        Returns
        -------
        None
        """
        # --- General Training options ---
        training_group = self.train_parser.add_argument_group('training')
        eval_training_group = self.eval_parser.add_argument_group('training')

        training_group.add_argument(
            *['--training-num_epochs', '--num_epochs'],
            type=int, dest='training_num_epochs',
            help='the number of epochs training should last')

        for group in [training_group, eval_training_group]:
            group.add_argument(
                *['--training-batch_size', '--batch_size'],
                type=int, dest='training_batch_size',
                help='number of samples in each mini-batch (train/eval)')

            group.add_argument(
                *['--training-emb_path', '--emb_path'],
                type=str, dest='training_emb_path',
                help='path for storing the toponym n-grams embeddings')

            group.add_argument(
                *['--training-num_categories', '--num_categories'],
                type=int, dest='training_num_categories',
                help="number of categories for training")

            group.add_argument(
                *['--training-num_workers', '--num_workers'],
                type=int, dest='training_num_workers',
                help='the number of workers to use when loading data')

            group.add_argument(
                *['--training-resume', '--resume'],
                dest='training_resume', type=str,
                help='training should resume from this checkpoint path')

    def parse_metrics_options(self) -> None:
        """
        This method creates arguments/flags both for training and evaluation
        phase about any metrics options.
        Returns
        -------

        """
        # --- Metrics options ---
        for parser in ['train_parser', 'eval_parser']:
            metrics_group = vars(self)[parser].add_argument_group('metrics')

            metrics_group.add_argument(
                *['--metrics-accuracy', '--accuracy'],
                type=str, choices=['primary', 'secondary'],
                help='accuracy metric')

            metrics_group.add_argument(
                *['--metrics-precision', '--precision'],
                type=str, choices=['primary', 'secondary'],
                help='metric: precision')

            metrics_group.add_argument(
                *['--metrics-recall', '--recall'],
                type=str, choices=['primary', 'secondary'],
                help='metric: recall')

            metrics_group.add_argument(
                *['--metrics-f1', '--f1'],
                type=str, choices=['primary', 'secondary'],
                help='metric: f1')

    def setup_cli(self) -> Tuple[str, dict]:
        """
        This method instantiates all previous methods in order to parse all
        the arguments for all the actions in the CLI.

        Then returns a tuple containing the actual action and the parameters
        needed for that action to run.
        Returns
        -------
        Tuple[str, dict]
            A string containing the action. A dict containing parameters about
            that specific action.
        """
        self.experiment_arguments()

        # === Parse Optional Arguments ===
        self.parse_dataset_options()
        self.parse_model_options()
        self.parse_criterion_options()
        self.parse_optimizer_options()
        self.parse_general_training_options()
        self.parse_metrics_options()

        cmd_args = self.parser.parse_args()
        # returns the cmd_args arguments in a dict format
        cmd_params = vars(cmd_args)

        if cmd_args.action in ['train', 'evaluate']:

            group_names = self.TRAINING_GROUP_NAMES
            if cmd_args.action == 'evaluate':
                # no optimizer or scheduler
                group_names = self.EVALUATION_GROUP_NAMES

            # Assign group settings to nested param dictionaries
            exp_params = {group_name: {} for group_name in group_names}

            for key, val in cmd_params.items():
                if val is None:
                    continue

                tokens = key.split('_')
                prefix = tokens[0]
                suffix = "_".join(tokens[1:])

                if prefix in group_names:
                    exp_params[prefix][suffix] = val
                else:
                    exp_params[key] = val

            default_params = {}
            if cmd_args.settings:
                # Load default configurations from YAML file
                with open(cmd_args.settings, 'r') as f:
                    default_params = yaml.safe_load(f)

            # for each group of parameters if there exists a cmd parameter,
            # then overwrite the YAML file parameter.
            # Iff a cmd parameter is not passed and there exists a YAML
            # parameter, then use the YAML parameter

            experiment_params = {
                group: {**default_params.get(group, {}), **exp_params[group]}
                for group in group_names}

            # experiment name
            experiment_params['name'] = cmd_args.exp_name

            if cmd_args.action == 'train':
                return 'train', experiment_params

            elif cmd_args.action == 'evaluate':
                return 'evaluate', experiment_params

    def run(self):
        """
        This method performs the following:
        - Runs the preparation step in order to create all flags/args for all
        actions
        - Create all needed parameters needed for the particular actions
        - Run the actual action
        Returns
        -------
        None
        """
        action, params = self.setup_cli()

        if action == 'train':
            exp = Experiment(**params)
            exp.run()

        elif action == 'evaluate':
            exp = Experiment(**params)
            exp.evaluate()


if __name__ == "__main__":
    interface = ComLInt()
    interface.run()
