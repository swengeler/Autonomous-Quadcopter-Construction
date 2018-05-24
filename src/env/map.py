import numpy as np
import seaborn as sns
import logging
import env.block
from typing import List, Tuple
from env.util import cw_angle_and_distance


class Map:

    def __init__(self,
                 target_map: np.ndarray,
                 offset_origin: Tuple[float, float] = (0.0, 0.0),
                 environment_extent: List[float] = None):
        # logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # occupancy matrices
        self.target_map = target_map
        self.occupancy_map = np.zeros_like(target_map)
        self.occupancy_map[np.where(self.target_map == 2)] = 1

        # entities in the space
        self.agents = []
        self.blocks = []
        self.placed_blocks = []

        # information about block and seed positions (only x, y coordinates)
        self.block_stashes = {}
        self.seed_stashes = {}

        # other information
        self.offset_origin = offset_origin  # might instead want to just pad the maps
        self.environment_extent = environment_extent
        if environment_extent is not None:
            min_x_extent = self.offset_origin[0] + self.target_map.shape[2] * env.block.Block.SIZE
            min_y_extent = self.offset_origin[1] + self.target_map.shape[1] * env.block.Block.SIZE
            min_z_extent = self.target_map.shape[0] * env.block.Block.SIZE
            if environment_extent[0] < min_x_extent:
                environment_extent[0] = min_x_extent
            if environment_extent[1] < min_y_extent:
                environment_extent[1] = min_y_extent
            if environment_extent[2] < min_z_extent:
                environment_extent[2] = min_z_extent

        # global information available?
        self.global_information = False

    def add_agents(self, agents):
        self.agents.extend(agents)

        # place agents according to some scheme, for now just specified positions
        # I guess it makes sense that agents, by definition, actually encapsulate their own positions, right?

    def add_blocks(self, blocks: List[env.block.Block]):
        self.blocks.extend(blocks)
        if self.environment_extent is None:
            x_extent = self.offset_origin[0] + self.target_map.shape[2] * (env.block.Block.SIZE - 1)
            y_extent = self.offset_origin[1] + self.target_map.shape[1] * (env.block.Block.SIZE - 1)
            self.environment_extent = (x_extent, y_extent)

        original_seed_position = self.original_seed_position()
        for b in blocks:
            if b.is_seed and (b.geometry.position[0], b.geometry.position[1]) not in list(self.seed_stashes.keys()) \
                    and (b.geometry.position[0], b.geometry.position[1]) != tuple(original_seed_position[:2]):
                self.seed_stashes[(b.geometry.position[0], b.geometry.position[1])] = [b]
            elif b.is_seed and (b.geometry.position[0], b.geometry.position[1]) != tuple(original_seed_position[:2]):
                self.seed_stashes[(b.geometry.position[0], b.geometry.position[1])].append(b)
            elif not b.is_seed and (b.geometry.position[0], b.geometry.position[1]) not in list(self.block_stashes.keys()):
                self.block_stashes[(b.geometry.position[0], b.geometry.position[1])] = [b]
            elif not b.is_seed:
                self.block_stashes[(b.geometry.position[0], b.geometry.position[1])].append(b)

        # place blocks according to some scheme, for now just specified positions

        # might have to select one block, designate and place it as seed
        # (seed is already in correct position for this first test)

    def required_blocks(self):
        # return either number of blocks (for starters) or some other specification of blocks,
        # e.g. this many of a certain type
        return np.count_nonzero(self.target_map)

    def original_seed_position(self):
        y, x = np.where(self.target_map[0] == 2)
        return x[0] * env.block.Block.SIZE + self.offset_origin[0], \
               y[0] * env.block.Block.SIZE + self.offset_origin[1], \
               env.block.Block.SIZE / 2  # + env.block.Block.SIZE

    def original_seed_grid_position(self):
        z, y, x = np.where(self.target_map == 2)
        return np.array([x[0], y[0], z[0]], dtype="int32")

    def place_block(self, grid_position, block):
        self.occupancy_map[tuple(reversed(grid_position))] = 2 if block.is_seed else 1
        self.placed_blocks.append(block)
        if self.global_information:
            for a in self.agents:
                a.local_occupancy_map[tuple(reversed(grid_position))] = 1

    def check_occupancy_map(self, position, comparator=lambda x: x != 0):
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if any(position < 0):
            return comparator(0)
        try:
            temp = self.occupancy_map[tuple(np.flip(position, 0))]
        except IndexError:
            return comparator(0)
        else:
            val = comparator(temp)
            return val

    def block_below(self, position, structure_level=None, radius=0.0):
        closest_x = int((position[0] - self.offset_origin[0]) / env.block.Block.SIZE)
        closest_y = int((position[1] - self.offset_origin[1]) / env.block.Block.SIZE)
        if structure_level is None:
            for height_level in range(self.occupancy_map.shape[0]):
                if 0 <= closest_x < self.occupancy_map.shape[2] and 0 <= closest_y < self.occupancy_map.shape[1] \
                        and self.check_occupancy_map(np.array([closest_x, closest_y, height_level])):
                    temp = [closest_x, closest_y, height_level]
                    for b in self.placed_blocks:
                        if all(b.grid_position[i] == temp[i] for i in range(3)):
                            return b
        elif 0 <= closest_x < self.occupancy_map.shape[2] and 0 <= closest_y < self.occupancy_map.shape[1] \
                and self.check_occupancy_map(np.array([closest_x, closest_y, structure_level])):
            temp = [closest_x, closest_y, structure_level]
            for b in self.placed_blocks:
                if all(b.grid_position[i] == temp[i] for i in range(3)):
                    return b
        return None

    def block_at_position(self, position, grid=True):
        for b in self.placed_blocks:
            if grid and all(b.grid_position[i] == position[i] for i in range(3)):
                return b
        return None

    def ccw_block_stash_locations(self):
        stash_positions = list(self.block_stashes.keys())
        ordered_stash_positions = sorted(stash_positions, key=lambda x: cw_angle_and_distance(
            x, (self.offset_origin[0] + env.block.Block.SIZE * int(self.target_map.shape[2] / 2),
                self.offset_origin[1] + env.block.Block.SIZE * int(self.target_map.shape[1] / 2)),
            stash_positions[0]))
        return ordered_stash_positions[::-1]

    def ccw_seed_stash_locations(self):
        stash_positions = list(self.seed_stashes.keys())
        ordered_stash_positions = sorted(stash_positions, key=lambda x: cw_angle_and_distance(
            x, (self.offset_origin[0] + env.block.Block.SIZE * int(self.target_map.shape[2] / 2),
                self.offset_origin[1] + env.block.Block.SIZE * int(self.target_map.shape[1] / 2)),
            stash_positions[0]))
        return ordered_stash_positions[::-1]

    def check_over_construction_area(self, position):
        return self.offset_origin[0] <= position[0] < self.offset_origin[0] \
               + env.block.Block.SIZE * self.target_map.shape[2] \
               and self.offset_origin[0] <= position[0] < self.offset_origin[0] \
               + env.block.Block.SIZE * self.target_map.shape[2]

    def distance_to_construction_area(self, position):
        if self.check_over_construction_area(position):
            return 0
        # otherwise, find the distance to the closest side
        width = env.block.Block.SIZE * self.target_map.shape[2]
        height = env.block.Block.SIZE * self.target_map.shape[1]
        dx = max(abs(position[0] - self.offset_origin[0] + width / 2) - width / 2, 0)
        dy = max(abs(position[0] - self.offset_origin[1] + height / 2) - height / 2, 0)
        return np.sqrt(dx ** 2 + dy ** 2)

