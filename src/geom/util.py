import numpy as np


def simple_distance(pos_1, pos_2):
    assert len(pos_1) == len(pos_2)
    return np.sqrt(sum([(e[0] - e[1]) ** 2 for e in zip(pos_1, pos_2)]))


def interval_overlaps(min_1, max_1, min_2, max_2):
    return min_1 <= min_2 <= max_1 or min_2 <= min_1 <= max_2
