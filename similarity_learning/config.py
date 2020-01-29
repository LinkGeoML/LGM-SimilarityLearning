"""
Configuration Module
"""
import os
from typing import Any

ENV_PREFIX = "LGM_"


def _env_get(name: str, default=None) -> Any:
    """

    Parameters
    ----------
    name : str
        The name of the environment variable

    default : any
        The default value of the environment variable

    Returns
    -------
    any
    The value of the requested env variable

    """

    return os.environ.get("{}{}".format(ENV_PREFIX, name), default)


class DirConf:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    REPORTS_DIR = 'reports'
    LOG_DIR = 'log'


# === LOGGING ===


class LoggerConf:
    """Logger Configuration class"""

    LOG_DIR = DirConf.LOG_DIR

    EXPERIMENT_LOG_DIR = os.path.join(LOG_DIR, 'experiment')

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.exists(EXPERIMENT_LOG_DIR):
        os.mkdir(EXPERIMENT_LOG_DIR)

    # This will be necessary for syslog (if used)
    LOGGER_NAME = 'LGM_Similarity_Learning'

    # Logger configuration
    LOGGING_CONF = {
        'version': 1,  # required
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {'format': '%(asctime)-15s: %(message)s',
                       'datefmt': '%Y-%m-%d %H:%M:%S'},
            'devlog': {
                'format': '%(asctime)s - PID:%(process)d - '
                          '%(name)s.py:%(funcName)s:%(lineno)d - '
                          '%(levelname)s - %(message)s'}
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'formatter': 'simple',
                'class': 'logging.StreamHandler',
            },
            'devlog': {
                'level': 'DEBUG',
                'formatter': 'devlog',
                'filename': os.path.join(
                    EXPERIMENT_LOG_DIR, 'development.log'),
                'class': 'logging.handlers.RotatingFileHandler'
            }
        },
        'loggers': {
            LOGGER_NAME: {
                'level': 'INFO',
                'handlers': ['devlog', 'console']
            },
        },
    }
