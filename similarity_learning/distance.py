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


def euclidean_distance(tensors):
    x, y = tensors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
