import numpy as np


def rotation_2d(vector, angle):
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([vector[0] * cs - vector[1] * sn, vector[0] * sn + vector[1] * cs, vector[2]])


def rotation_2d_experimental(vector, angle):
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([vector[0] * cs - vector[2] * sn, vector[1], vector[0] * sn + vector[2] * cs])


def simple_distance(pos_1, pos_2, compared_positions=None):
    min_length = min(len(pos_1), len(pos_2))
    if compared_positions is not None and compared_positions < min_length:
        min_length = compared_positions
    pos_1 = pos_1[:min_length]
    pos_2 = pos_2[:min_length]
    return np.sqrt(sum([(e[0] - e[1]) ** 2 for e in zip(pos_1, pos_2)]))


def interval_overlaps(min_1, max_1, min_2, max_2):
    return min_1 <= min_2 <= max_1 or min_2 <= min_1 <= max_2
