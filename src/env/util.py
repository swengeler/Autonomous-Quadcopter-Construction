import numpy as np
import collections
from enum import Enum
from typing import List
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


def shortest_path(grid: np.ndarray, start, goal):
    queue = collections.deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0] and grid[y2, x2] != 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def print_map(environment: np.ndarray, path=None, layer=None):
    if path is not None and layer is None:
        raise WrongInputException("Variable 'layer' has to be provided if 'path' is None.")
    elif path is None and layer is not None:
        print("Warning: variable 'path' is None even though 'layer' was provided.")

    symbols = {
        0: " ",
        1: ".",
        2: "X",
        "corner": "C"
    }

    for z in range(environment.shape[0]):
        for x in range(0, (environment.shape[2] + 2) * 2 - 1):
            print("―", end="")
        print("\n| Layer {} |".format(str(z).ljust(environment.shape[2] * 2 - 7)))
        for x in range(0, (environment.shape[2] + 2) * 2 - 1):
            print("―", end="")
        print("")
        for y in range(environment.shape[1] - 1, -1, -1):
            print("| ", end="")
            for x in range(environment.shape[2]):
                if path is not None and (x, y) in path and environment[z, y, x] != 0:
                    if path.index((x, y)) == 0:
                        print("S", end=" ")
                    elif path.index((x, y)) == len(path) - 1:
                        print("E", end=" ")
                    else:
                        print("#", end=" ")
                elif environment[z, y, x] in symbols.keys():
                    print(symbols[environment[z, y, x]], end=" ")
                elif environment[z, y, x] < 0:
                    print(symbols["corner"], end=" ")
                else:
                    print("?", end=" ")
            print("|")
        for x in range(0, (environment.shape[2] + 2) * 2 - 1):
            print("―", end="")
        print("")


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
