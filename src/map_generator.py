import numpy as np
from experiments import scale_map

SAVE_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/experiment_maps/"


def create_hole_scale_maps():
    hole_scale_1 = np.array([[[1, 2, 1, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 1, 1, 1]]])
    np.save(SAVE_DIRECTORY + "hole_scale_1.npy", hole_scale_1, allow_pickle=False)

    hole_scale_2 = scale_map(hole_scale_1, 2, (1, 2))
    np.save(SAVE_DIRECTORY + "hole_scale_2.npy", hole_scale_2, allow_pickle=False)

    hole_scale_3 = scale_map(hole_scale_1, 3, (1, 2))
    np.save(SAVE_DIRECTORY + "hole_scale_3.npy", hole_scale_3, allow_pickle=False)

    hole_scale_4 = scale_map(hole_scale_1, 4, (1, 2))
    np.save(SAVE_DIRECTORY + "hole_scale_4.npy", hole_scale_4, allow_pickle=False)


def create_combined_maps():
    base_side_length = 6
    max_side_length = 16
    for i in range(base_side_length, max_side_length + 1, 2):
        center = int(i / 2) - 1
        combined_map = np.zeros((i, i, i))
        np.place(combined_map[0:2], combined_map[0:2] == 0, 1)
        combined_map[0, center, center] = 2
        for j in range(2, i - 2):
            for x, y in ((0, 0), (0, i - 2), (i - 2, 0), (i - 2, i - 2)):
                combined_map[j, y, x] = 1
                combined_map[j, y + 1, x] = 1
                combined_map[j, y, x + 1] = 1
                combined_map[j, y + 1, x + 1] = 1
        np.place(combined_map[(i - 2):i], combined_map[(i - 2):i] == 0, 1)
        np.place(combined_map[(i - 2):i, 2:(i - 2), 2:(i - 2)],
                 combined_map[(i - 2):i, 2:(i - 2), 2:(i - 2)] == 1, 0)
        np.save(SAVE_DIRECTORY + "combined_map_{}.npy".format(i), combined_map, allow_pickle=False)


def create_pyramid_maps():
    base_side_length = 4
    max_side_length = 16
    for i in range(base_side_length, max_side_length + 1, 2):
        center = int(i / 2) - 1
        height = int((i - 2) / 2) + 1
        pyramid_map = np.zeros((height, i, i))
        for j in range(height):
            side_length = 2 ** (height - j)
            offset = int((i - side_length) / 2)
            pyramid_map[height, offset:(offset + side_length), offset:(offset + side_length)] = 1
        pyramid_map[0, center, center] = 2


def main():
    # create_hole_scale_maps()
    create_combined_maps()


if __name__ == "__main__":
    main()
