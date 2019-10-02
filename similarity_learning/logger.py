"""
Basic logging configuration for application and for experiments logging
"""
from logging import config as logger_config
from logging import getLogger

from similarity_learning.config import LoggerConf


def setup_app_logger(name: str = None):
    """
    Set appropriate name for module/class logger
    Configures the logger
    Parameters
    ----------
    name

    Returns
    -------
    A configured logger

    """

    if name is not None:
        full_name = '{}.{}'.format(LoggerConf.LOGGER_NAME, name)
    else:
        full_name = LoggerConf.LOGGER_NAME

    logger = getLogger(full_name)

    # Set the logging configuration
    logger_config.dictConfig(LoggerConf.LOGGING_CONF)

    return logger


class DummyLogger:

    def info(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass


app_logger = setup_app_logger(name='app_logger')
