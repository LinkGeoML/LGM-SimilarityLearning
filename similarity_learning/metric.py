import numpy as np
from tensorflow.keras import backend as K


def distance_accuracy(y_true, y_pred):
    """
    Compute classification accuracy with a fixed threshold on distances.

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy with a fixed threshold on distances.
    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
