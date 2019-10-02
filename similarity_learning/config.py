"""
Configuration Module
"""
import os
from distutils.util import strtobool
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


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return bool(strtobool(value))


# This should be one of 'development', 'test'
RUN_ENV = _env_get('RUN_ENV', default='development')
assert RUN_ENV in ['development', 'test']


class DirConf:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    REPORTS_DIR = 'reports'


class ModelConf:
    """Model Training Configuration class"""

    SEED = 42
    EPOCHS = 10

    # Default dataset names for each task
    DS_TRAIN = 'train.csv'
    DS_VAL = 'val.csv'
    DS_TEST = 'test.csv'

    # Default embedding vector dimensions
    EMBEDDINGS_DIM = 128

    # Default name for whole dataset embeddings
    EMBEDDINGS_FNAME = 'embeddings.npy'

    # Default Batch Size
    DEFAULT_BS = 32

    # # NOTE: Using the env var CUDA_VISIBLE_DEVICES one can filter the list
    # # of available CUDA devices!
    # if torch.cuda.is_available():
    #     TORCH_DEVICE = torch.device("cuda")
    # else:
    #     TORCH_DEVICE = torch.device("cpu")
    #
    # TORCH_CUDA_DEVICES_CNT = torch.cuda.device_count()

    # Trained models
    EMBEDDINGS_MODEL_NAME = _env_get(
        'EMBEDDINGS_MODEL', default='test_model')

    MODEL_NUM_CLASSES = 2


# === LOGGING ===
if RUN_ENV == 'development':
    root_handlers = ['devlog']
    root_level = 'INFO'

elif RUN_ENV == 'test':
    root_handlers = ['devlog']
    root_level = 'INFO'

else:
    root_handlers = ['console', 'file']
    # This can be set in a more fine grained way per handler.
    root_level = 'INFO'


class LoggerConf:
    """Logger Configuration class"""

    LOG_DIR = 'log'

    APP_LOG_DIR = os.path.join(LOG_DIR, 'app')

    EXPERIMENT_LOG_DIR = os.path.join(LOG_DIR, 'experiment')

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.exists(APP_LOG_DIR):
        os.mkdir(APP_LOG_DIR)

    if not os.path.exists(EXPERIMENT_LOG_DIR):
        os.mkdir(EXPERIMENT_LOG_DIR)

    # Training defaults
    LOG_INTERVAL = 20

    # This will be necessary for syslog (if used)
    LOGGER_NAME = 'beholder'

    LOG_NAMES = dict(development='development.log',
                     test='test.log',
                     production='production.log')

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
                'filename': os.path.join(APP_LOG_DIR, LOG_NAMES[RUN_ENV]),
                'class': 'logging.handlers.RotatingFileHandler'
            }
        },
        'loggers': {
            LOGGER_NAME: {
                'level': root_level,
                'handlers': root_handlers
            },
        },
    }

    EXP_LOG_CONF = {
        'disable_existing_loggers': False,
        'level': root_level,

        'formatter': '%(asctime)s - PID:%(process)d - %(name)s.py:'
                     '%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
    }
