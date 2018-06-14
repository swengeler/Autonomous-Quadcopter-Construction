import numpy as np

from deprecated_experiments import scale_map
from geom.util import simple_distance

SAVE_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_experiment_maps/"

PYRAMID_SINGLE = 0
PYRAMID_DIAMOND = 1
PYRAMID_HOURGLASS = 2


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


def create_pyramid_maps(mode=PYRAMID_SINGLE):
    base_side_length = 4
    max_side_length = 16
    for i in range(base_side_length, max_side_length + 1, 4):
        center = int(i / 2) - 1
        height = int(i / 2)

        # constructing the map with the necessary height
        if mode == PYRAMID_SINGLE:
            pyramid_map = np.zeros((height, i, i))
        else:
            pyramid_map = np.zeros((height * 2, i, i))

        # filling in the "normal" pyramid (with a z-offset for the diamond shape)
        for j in range(height):
            side_length = i - j * 2
            offset = int((i - side_length) / 2)
            if mode == PYRAMID_DIAMOND:
                pyramid_map[j + height, offset:(offset + side_length), offset:(offset + side_length)] = 1
            else:
                pyramid_map[j, offset:(offset + side_length), offset:(offset + side_length)] = 1

        # filling in the inverted pyramid (with a z-offset for the hourglass shape)
        if mode in [PYRAMID_DIAMOND, PYRAMID_HOURGLASS]:
            for j in range(height):
                side_length = i - (height - j - 1) * 2
                offset = int((i - side_length) / 2)
                if mode == PYRAMID_DIAMOND:
                    pyramid_map[j, offset:(offset + side_length), offset:(offset + side_length)] = 1
                else:
                    pyramid_map[j + height, offset:(offset + side_length), offset:(offset + side_length)] = 1

        pyramid_map[0, center, center] = 2
        save_string = "NONE"
        if mode == PYRAMID_SINGLE:
            save_string = "pyramid_map_{}.npy".format(i)
        elif mode == PYRAMID_DIAMOND:
            save_string = "diamond_map_{}.npy".format(i)
        elif mode == PYRAMID_HOURGLASS:
            save_string = "hourglass_map_{}.npy".format(i)
        np.save(SAVE_DIRECTORY + save_string, pyramid_map, allow_pickle=False)


def create_hole_maps():
    base_side_length = 8
    max_side_length = 32
    for i in range(base_side_length, max_side_length + 1, 4):
        center = int(i / 2) - 1
        quarter = int(i / 4)
        hole_map = np.ones((1, i, i))
        hole_map[0, 2:(i - 2), 2:(i - 2)] = 0
        coords = np.where(hole_map == 1)
        coords = list(zip(coords[2], coords[1]))
        coords = sorted(coords, key=lambda e: simple_distance(e, (center, center)))
        hole_map[0, coords[0][1], coords[0][0]] = 2
        np.save(SAVE_DIRECTORY + "hole_same_{}".format(i), hole_map, allow_pickle=False)
        if quarter != 2:
            other_hole_map = np.ones((1, i, i))
            other_hole_map[0, quarter:(i - quarter), quarter:(i - quarter)] = 0
            other_coords = np.where(other_hole_map == 1)
            other_coords = list(zip(other_coords[2], other_coords[1]))
            other_coords = sorted(other_coords, key=lambda e: simple_distance(e, (center, center)))
            other_hole_map[0, other_coords[0][1], other_coords[0][0]] = 2
            np.save(SAVE_DIRECTORY + "hole_diff_{}".format(i), other_hole_map, allow_pickle=False)


def create_hole_size_maps():
    base_side_length = 16
    center = int(base_side_length / 2) - 1
    hole_map = np.ones((1, base_side_length, base_side_length))
    for i in range(base_side_length - 2, 1, -2):
        offset = int((base_side_length - i) / 2)
        copy = np.copy(hole_map)
        copy[0, offset:(offset + i), offset:(offset + i)] = 0
        coords = np.where(copy == 1)
        coords = list(zip(coords[2], coords[1]))
        coords = sorted(coords, key=lambda e: simple_distance(e, (center, center)))
        copy[0, coords[0][1], coords[0][0]] = 2
        np.save(SAVE_DIRECTORY + "hole_size_{}".format(i), copy, allow_pickle=False)


def create_block_size_maps():
    sizes = [(4, 4, 4), (6, 5, 5), (7, 6, 6), (8, 7, 7), (9, 8, 8), (10, 9, 9), (10, 10, 10)]
    pad_to_size = [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (28, 28), (32, 32)]
    padding = False
    for s_idx, s in enumerate(sizes):
        if padding:
            width = pad_to_size[s_idx][1]
            height = pad_to_size[s_idx][0]
        else:
            width = s[2]
            height = s[1]
        center_x = int(width / 2) - 1
        center_y = int(height / 2) - 1

        if padding:
            block_map = np.zeros((s[0], pad_to_size[s_idx][0], pad_to_size[s_idx][1]))
            offset_x = int((pad_to_size[s_idx][1] - s[2]) / 2)
            offset_y = int((pad_to_size[s_idx][0] - s[1]) / 2)
            block_map[:, offset_y:(offset_y + s[1]), offset_x:(offset_x + s[2])] = 1
        else:
            block_map = np.ones(s)

        block_map[0, center_y, center_x] = 2
        np.save(SAVE_DIRECTORY + "block_{}x{}x{}{}".format(s[2], s[1], s[0], "_padded.npy" if padding else ".npy"),
                block_map, allow_pickle=False)


def create_blocks_8x8x():
    heights = [2, 3, 4, 6, 8]
    for h in heights:
        block_map = np.ones((h, 8, 8))
        block_map[0, 3, 3] = 2
        np.save(SAVE_DIRECTORY + "block_{}x{}x{}.npy".format(8, 8, h), block_map, allow_pickle=False)


def create_component_map():
    component_size = 3
    component_number = (2, 2)
    component_spacing = 1
    height = 4
    component_map = np.zeros(
        (height, component_size * component_number[1] + component_spacing * (component_number[1] - 1),
         component_size * component_number[0] + component_spacing * (component_number[0] - 1)))
    for x in range(component_number[0]):
        for y in range(component_number[1]):
            x_start = x * (component_spacing + component_size)
            x_end = x_start + component_size
            y_start = y * (component_spacing + component_size)
            y_end = y_start + component_size
            component_map[:, y_start:y_end, x_start:x_end] = 1
    center_x = int(component_map.shape[2] / 2)
    center_y = int(component_map.shape[1] / 2)
    coords = np.where(component_map[0] == 1)
    coords = list(zip(coords[1], coords[0]))
    coords = sorted(coords, key=lambda c: simple_distance(c, (center_x, center_y)))
    component_map[0, coords[0][1], coords[0][0]] = 2
    np.save(SAVE_DIRECTORY + "seed_comp_{}x{}_{}_{}.npy"
            .format(component_number[0], component_number[1], component_size, component_spacing),
            component_map, allow_pickle=False)


def main():
    # create_hole_scale_maps()
    # create_combined_maps()
    # create_pyramid_maps(PYRAMID_HOURGLASS)
    # create_hole_maps()
    # create_hole_size_maps()
    # create_block_size_maps()
    # create_blocks_8x8x()
    create_component_map()


if __name__ == "__main__":
    main()
