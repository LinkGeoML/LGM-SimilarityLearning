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


def l2_distance(tensors):
    x, y = tensors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

#
# # Add a customized layer to compute the absolute difference between the encodings
# distance_layer = Lambda(lambda tensors: l1_distance(tensors))
# distance = distance_layer([encoded_l, encoded_r])
#
# # Calculates the distance between the vectors
# distance_layer = Lambda(
#     lambda tensors: exponent_neg_manhattan_distance(tensors))
# distance = distance_layer([encoded_l, encoded_r])
