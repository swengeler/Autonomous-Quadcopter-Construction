import numpy as np


def simple_distance(pos_1, pos_2, compared_positions=None):
    """
    Return the distance between the two positions (possibly in fewer dimensions).

    :param pos_1: the first position vector
    :param pos_2: the second position vector
    :param compared_positions: specifies the number of dimensions to compare
    :return: the distance between the two positions
    """

    min_length = min(len(pos_1), len(pos_2))
    if compared_positions is not None and compared_positions < min_length:
        min_length = compared_positions
    pos_1 = pos_1[:min_length]
    pos_2 = pos_2[:min_length]
    return np.sqrt(sum([(e[0] - e[1]) ** 2 for e in zip(pos_1, pos_2)]))


def interval_overlaps(min_1, max_1, min_2, max_2):
    """
    Check whether the given intervals overlap.

    :param min_1: start of the first interval
    :param max_1: end of the first interval
    :param min_2: start of the second interval
    :param max_2: end of the second interval
    :return: True if the intervals overlap, False otherwise
    """

    return min_1 <= min_2 <= max_1 or min_2 <= min_1 <= max_2
