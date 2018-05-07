import numpy as np
import collections
from enum import Enum
from typing import List, Tuple
from env.block import *


class Occupancy(Enum):
    UNOCCUPIED = 0
    OCCUPIED = 1
    SEED = 2


class BlockType(Enum):
    INERT = 0


class BlockGeneratorInfo:
    def __init__(self, block_type: BlockType, size: float, color: str, count: int):
        self.type = block_type
        self.size = size
        self.color = color
        self.count = count


class WrongInputException(Exception):
    pass


def create_block(block_type, *args, **kwargs):
    logger = logging.getLogger(__name__)
    if block_type is BlockType.INERT:
        logger.info("Creating Block.")
        return Block(*args, **kwargs)
    else:
        logger.warning("No block type specified.")
        return None


def create_block_list(block_generator_info):
    blocks = []
    for bgi in block_generator_info:
        for _ in range(0, bgi.count):
            blocks.append(create_block(bgi.type, size=bgi.size, color=bgi.color))
    return blocks


def shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0] and grid[y2, x2] != 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return path


def shortest_path_3d_in_2d(lattice: np.ndarray, start: Tuple, goal: Tuple):
    if len(start) == 3:
        start = start[0:2]
    if len(goal) == 3:
        goal = goal[0:2]

    # determining a grid of locations occupied below that can be used for orientation
    grid = np.zeros_like(lattice[0])
    for z in range(lattice.shape[0]):
        grid[lattice[z] != 0] = 1
    grid[start[1], start[0]] = 1
    grid[goal[1], goal[0]] = 1

    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            # determine the highest layer at which there is a block,
            # i.e. how high the agent has to fly to avoid all collisions
            highest_layer = 0
            for x, y in path:
                for z in range(lattice.shape[0] - 1, -1, -1):
                    if lattice[z, y, x] >= 1:
                        highest_layer = max(z, highest_layer)
                        break
            return path, highest_layer
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0] and grid[y2, x2] != 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    highest_layer = 0
    for x, y in path:
        for z in range(lattice.shape[0] - 1, -1, -1):
            if lattice[z, y, x] >= 1:
                highest_layer = max(z, highest_layer)
                break
    return path, highest_layer


def shortest_path_3d(lattice: np.ndarray, start, goal):
    # Note that the way I'm doing this right now basically means the resulting
    # shortest path tries to stick as close to the structure as possible
    processed_lattice = np.zeros_like(lattice)
    processed_lattice = np.concatenate((processed_lattice, np.array([processed_lattice[0]])))
    for z in range(processed_lattice.shape[0]):
        for y in range(processed_lattice.shape[1]):
            for x in range(processed_lattice.shape[2]):
                if z == 0 and lattice[z, y, x] == 0 and any([0 <= x2 < lattice.shape[2] and 0 <= y2 < lattice.shape[1]
                                                             and lattice[z, y2, x2] != 0 for x2, y2
                                                             in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1))]):
                    processed_lattice[z, y, x] = 1
                elif z != 0 and lattice[z - 1, y, x] != 0 and (z == lattice.shape[0] or lattice[z, y, x] == 0):
                    processed_lattice[z, y, x] = 1
    print("\nORIGINAL MAP:")
    print_map(lattice)
    # print("\nINTERMEDIATE LEGAL PATHWAY MAP:")
    # print_map(processed_lattice)
    for z in range(processed_lattice.shape[0]):
        for y in range(processed_lattice.shape[1]):
            for x in range(processed_lattice.shape[2]):
                if z != 0 and processed_lattice[z - 1, y, x] != 0 \
                        and any([0 <= x2 < processed_lattice.shape[2]
                                 and 0 <= y2 < processed_lattice.shape[1]
                                 and processed_lattice[z, y2, x2] == 1 for x2, y2
                                 in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1))]):
                    processed_lattice[z, y, x] = 2
    # print("\nFINAL LEGAL PATHWAY MAP:")
    # print_map(processed_lattice, custom_symbols={2: "."})
    # TODO: how to solve problem of accounting for multiple layer "climbs"?

    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y, z = path[-1]
        if x == goal[0] and y == goal[1] and z == goal[2]:
            return path
        for x2, y2, z2 in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
            if 0 <= x2 < processed_lattice.shape[2] and 0 <= y2 < processed_lattice.shape[1] \
                    and 0 <= z2 < processed_lattice.shape[0] and processed_lattice[z2, y2, x2] != 0 \
                    and (x2, y2, z2) not in seen:
                queue.append(path + [(x2, y2, z2)])
                seen.add((x2, y2, z2))
    return path


def print_map_layer(environment: np.ndarray, path=None, custom_symbols=None):
    symbols = {
        0: " ",
        1: ".",
        2: "X",
        "corner": "C"
    }

    for x in range(0, (environment.shape[1] + 2) * 2 - 1):
        print("―", end="")
    print("")
    for y in range(environment.shape[0] - 1, -1, -1):
        print("| ", end="")
        for x in range(environment.shape[1]):
            if path is not None and (x, y) in path and environment[y, x] != 0:
                if path.index((x, y)) == 0:
                    print("S", end=" ")
                elif path.index((x, y)) == len(path) - 1:
                    print("E", end=" ")
                else:
                    print("#", end=" ")
            elif custom_symbols is not None and environment[y, x] in custom_symbols.keys():
                print(custom_symbols[environment[y, x]], end=" ")
            elif environment[y, x] in symbols.keys():
                print(symbols[environment[y, x]], end=" ")
            elif environment[y, x] < 0:
                print(symbols["corner"], end=" ")
            else:
                print("?", end=" ")
        print("|")
    for x in range(0, (environment.shape[1] + 2) * 2 - 1):
        print("―", end="")
    print("")


def print_map(environment: np.ndarray, path=None, layer=None, custom_symbols=None):
    if path is not None and layer is None and len(path[0]) != 3:
        raise WrongInputException("Variable 'layer' has to be provided if 2D path is given.")
    elif path is None and layer is not None:
        print("Warning: variable 'path' is None even though 'layer' was provided.")

    if environment.ndim == 2:
        environment = np.array([environment])
    for z in range(environment.shape[0]):
        for x in range(0, (environment.shape[2] + 2) * 2 - 1):
            print("―", end="")
        print("\n| Layer {} |".format(str(z).ljust(environment.shape[2] * 2 - 7)))
        if path is not None and ((len(path[0]) == 2 and z == layer) or len(path[0]) == 3):
            print_map_layer(environment[z], path=path, custom_symbols=custom_symbols)
        else:
            print_map_layer(environment[z], custom_symbols=custom_symbols)


def legal_attachment_sites(target_map: np.ndarray, occupancy_map: np.ndarray, component_marker=None):
    # input is supposed to be a 2-dimensional layer
    # setting all values larger than 1 (seeds) to 1
    if component_marker is not None:
        occupancy_map = np.copy(occupancy_map)
        np.place(occupancy_map, target_map != component_marker, 0)
        target_map = np.copy(target_map)
        np.place(target_map, target_map != component_marker, 0)

    # identifying all sites that are adjacent to the structure built so far
    legal_sites = np.zeros_like(target_map)
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # can only be legal if a block is actually supposed to be placed there and has not been placed yet
            if target_map[y, x] >= 1 and occupancy_map[y, x] == 0:
                # needs to be adjacent to existing part of structure, but only to 2 or fewer blocks
                counter = 0
                for y2 in (y - 1, y + 1):
                    if 0 <= y2 < legal_sites.shape[0] and occupancy_map[y2, x] >= 1:
                        counter += 1
                for x2 in (x - 1, x + 1):
                    if 0 <= x2 < legal_sites.shape[1] and occupancy_map[y, x2] >= 1:
                        counter += 1
                if 1 <= counter < 3:
                    legal_sites[y, x] = 1

    # identifying those sites where the row rule would be violated
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # only matters for those sites which have already been identified as potential attachment sites
            if legal_sites[y, x] == 1:
                # check whether the following pattern exists around that block (or in rotated form):
                # [B] [E] ... [A]
                # where (B = block already placed), (E = other potential attachment site), (A = current site)
                # and where at "..." there are only E's
                for diff in (-1, 1):
                    # making it through this loop without a break means that in the x-row, y-column where the block
                    # could be placed, there is either only a block immediately adjacent or any blocks already placed
                    # are separated from the current site by a gap

                    counter = 1
                    while 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] == 0 \
                            and target_map[y + counter * diff, x] > 0:
                        counter += 1
                    if counter > 1 and 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] > 0 and target_map[y + counter * diff, x] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        # print("Info for x = {}, y = {} (y):\ndiff = {}\ncounter = {}\noccupancy: {}\ntarget: {}"
                        #       .format(x, y, diff, counter, occupancy_map[y + counter * diff, x],
                        #               target_map[y + counter * diff, x]))
                        break

                    counter = 1
                    while 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] == 0 \
                            and target_map[y, x + counter * diff] > 0:
                        counter += 1
                    if counter > 1 and 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] > 0 and target_map[y, x + counter * diff] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        # print("Info for x = {}, y = {} (x):\ndiff = {}\ncounter = {}\noccupancy: {}\ntarget: {}"
                        #       .format(x, y, diff, counter, occupancy_map[y + counter * diff, x],
                        #               target_map[y + counter * diff, x]))
                        break

                    # if 0 <= y + 2 * diff < legal_sites.shape[0] and occupancy_map[y + 2 * diff, x] >= 1 \
                    #         and legal_sites[y + diff, x] == 1:
                    #     legal_sites[y, x] = 0
                    #     break
                    # if 0 <= x + 2 * diff < legal_sites.shape[1] and occupancy_map[y, x + 2 * diff] >= 1 \
                    #         and legal_sites[y, x + diff] == 1:
                    #     legal_sites[y, x] = 0
                    #     break

    return legal_sites


def legal_attachment_sites_3d(target_map: np.ndarray, occupancy_map: np.ndarray, safety_radius=2):
    # input is supposed to be a 3-dimensional layer

    # for each layer, identify the potential attachment sites in 2D, the 3D version can only get more restrictive
    legal_sites = np.empty_like(target_map, dtype="int64")
    for z in range(target_map.shape[0]):
        legal_sites[z] = legal_attachment_sites(target_map[z], occupancy_map[z])

    # in addition to attaching to adjacent blocks on the next level, attachment where supported from below is also
    # possible; however, it still has to be possible to attach other blocks (row/plane rule?)
    # -> allowed if supported and does not violate row rule on its layer?

    for z in range(1, target_map.shape[0]):
        for y in range(target_map.shape[1]):
            for x in range(target_map.shape[2]):
                # not yet "registered" as legal attachment site and supported from below
                if legal_sites[z, y, x] == 0 and occupancy_map[z, y, x] == 0 \
                        and target_map[z, y, x] >= 1 and occupancy_map[z - 1, y, x] >= 1:
                    legal_sites[z, y, x] = 1
                    # check whether row rule is violated
                    for diff in (-1, 1):
                        counter = 1
                        while 0 <= y + counter * diff < legal_sites.shape[1] \
                                and occupancy_map[z, y + counter * diff, x] == 0 \
                                and target_map[z, y + counter * diff, x] >= 0:
                            counter += 1
                        if counter > 1 and 0 <= y + counter * diff < legal_sites.shape[1] \
                                and occupancy_map[z, y + counter * diff, x] >= 0 \
                                and target_map[z, y + counter * diff, x] >= 0:
                            # have encountered a block already in this row
                            legal_sites[z, y, x] = 0
                            break

                        counter = 1
                        while 0 <= x + counter * diff < legal_sites.shape[2] \
                                and occupancy_map[z, y, x + counter * diff] == 0 \
                                and target_map[z, y, x + counter * diff] >= 0:
                            counter += 1
                        if counter > 1 and 0 <= x + counter * diff < legal_sites.shape[2] \
                                and occupancy_map[z, y, x + counter * diff] >= 0 \
                                and target_map[z, y, x + counter * diff] >= 0:
                            # have encountered a block already in this row
                            legal_sites[z, y, x] = 0
                            break
                if legal_sites[z, y, x] > 0:
                    # check whether quadcopter can still get to other sites, i.e. whether there is any layer below
                    # the current one where a block has to be placed within a 2 block radius of the (x, y) position
                    done = False
                    for z2 in range(z):
                        if not done:
                            for y2 in range(y - safety_radius, y + safety_radius + 1):
                                if not done:
                                    for x2 in range(x - safety_radius, x + safety_radius + 1):
                                        # check if: not the same (x, y) position, within range,
                                        # block supposed to go there and block not yet there
                                        if not (y2 == y and x2 == x) \
                                                and 0 <= y2 < legal_sites.shape[1] \
                                                and 0 <= x2 < legal_sites.shape[2] \
                                                and target_map[z2, y2, x2] >= 1 \
                                                and occupancy_map[z2, y2, x2] == 0:
                                            legal_sites[z, y, x] = 0
                                            done = True
                                            break

    return legal_sites


def neighbourhood(arr: np.ndarray, position, flatten=False):
    # position given as [x, y, z] or [x, y]
    x = position[0]
    y = position[1]
    x_lower = max(x - 1, 0)
    x_upper = min(x + 2, arr.shape[1 if len(position) == 2 else 2])
    y_lower = max(y - 1, 0)
    y_upper = min(y + 2, arr.shape[0 if len(position) == 2 else 1])
    if len(position) == 3:
        z = position[2]
        z_lower = max(z - 1, 0)
        z_upper = min(z + 2, arr.shape[0])
        result = arr[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        return result.flatten() if flatten else result
    else:
        result = arr[y_lower:y_upper, x_lower:x_upper]
        return result.flatten() if flatten else result
