import tensorflow.keras.backend as K


def exponent_neg_manhattan_distance(tensors):
    """

    :param tensors:
    :return:
    """
    return K.exp(-K.sum(K.abs(tensors[0] - tensors[1]), axis=1, keepdims=True))


def l1_distance(tensors):
    """

    :param tensors:
    :return:
    """
    return K.abs(tensors[0] - tensors[1])
