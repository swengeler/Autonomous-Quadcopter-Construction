import numpy as np
import logging
import env.block
from typing import List, Tuple
from tkinter import *


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

    def add_agents(self, agents):
        self.agents = agents

        # place agents according to some scheme, for now just specified positions
        # I guess it makes sense that agents, by definition, actually encapsulate their own positions, right?

    def add_blocks(self, blocks: List[env.block.Block]):
        self.blocks = blocks
        if self.environment_extent is None:
            x_extent = self.offset_origin[0] + self.target_map.shape[2] * (env.block.Block.SIZE - 1)
            y_extent = self.offset_origin[1] + self.target_map.shape[1] * (env.block.Block.SIZE - 1)
            self.environment_extent = (x_extent, y_extent)

        # place blocks according to some scheme, for now just specified positions

        # might have to select one block, designate and place it as seed
        # (seed is already in correct position for this first test)

    def update(self):
        pass

    def required_blocks(self):
        # return either number of blocks (for starters) or some other specification of blocks,
        # e.g. this many of a certain type
        counts = np.bincount(self.target_map.flatten())
        indices = np.nonzero(counts)[0]
        counts = list(zip(counts, counts[indices]))
        # return [env.BlockGeneratorInfo(env.BlockType.INERT, 15, "white", np.count_nonzero(self.target_map))]
        return np.count_nonzero(self.target_map)

    def seed_position(self):
        z, y, x = np.where(self.target_map == 2)
        return x[0] * env.block.Block.SIZE + self.offset_origin[0], \
               y[0] * env.block.Block.SIZE + self.offset_origin[1], \
               env.block.Block.SIZE / 2  # + env.block.Block.SIZE

    def seed_grid_position(self):
        z, y, x = np.where(self.target_map == 2)
        return np.array([x[0], y[0], z[0]], dtype="int32")

    def place_block(self, grid_position, block):
        print("Placing block at: {}".format(grid_position))
        self.occupancy_map[tuple(reversed(grid_position))] = 2 if block.is_seed else 1
        self.placed_blocks.append(block)

    def check_occupancy_map(self, position, comparator=lambda x: x != 0):
        if any(position < 0):
            return comparator(0)
        try:
            temp = self.occupancy_map[tuple(np.flip(position, 0))]
        except IndexError:
            return comparator(0)
        else:
            val = comparator(temp)
            return val

    def check_over_structure(self, position):
        closest_x = int((position[0] - self.offset_origin[0]) / env.block.Block.SIZE)
        closest_y = int((position[1] - self.offset_origin[1]) / env.block.Block.SIZE)
        print(closest_x, closest_y)
        for height_level in range(self.occupancy_map.shape[0]):
            if 0 <= closest_x < self.occupancy_map.shape[2] and 0 <= closest_y < self.occupancy_map.shape[1] \
                    and self.check_occupancy_map(np.array([closest_x, closest_y, height_level])):
                return True
        return False

    '''
    def draw_grid(self, canvas: Canvas):
        size = env.block.Block.SIZE
        for i in range(self.target_map.shape[2] + 1):
            x_const = self.offset_origin[0] + (i - 0.5) * size
            y_start = self.environment_extent[1] - (self.offset_origin[1] - size / 2)
            y_end = y_start - size * (self.target_map.shape[1])
            canvas.create_line(x_const, y_start, x_const, y_end)

        for i in range(self.target_map.shape[1] + 1):
            x_start = self.offset_origin[0] - size / 2
            x_end = x_start + size * (self.target_map.shape[2])
            y_const = self.environment_extent[1] - (self.offset_origin[1] + (i - 0.5) * size)
            canvas.create_line(x_start, y_const, x_end, y_const)

    def draw_agents(self, canvas: Canvas):
        for a in self.agents:
            # x_base = a.geometry.position[0] - a.geometry.size[0] / 2
            # y_base = self.environment_extent[1] - (a.geometry.position[1] - a.geometry.size[1] / 2)
            # canvas.create_rectangle(x_base, y_base, x_base + a.geometry.size[0], y_base - a.geometry.size[1], fill="blue")
            points = np.concatenate(a.geometry.corner_points_2d()).tolist()
            for p_idx, p in enumerate(points):
                if p_idx % 2 != 0:
                    points[p_idx] = self.environment_extent[1] - p
            canvas.create_polygon(points, fill="blue", outline="black")

    def draw_blocks(self, canvas: Canvas):
        size = env.block.Block.SIZE
        for b in self.blocks:
            # x_base = b.geometry.position[0] - size / 2
            # y_base = self.environment_extent[1] - (b.geometry.position[1] - size / 2)
            # canvas.create_rectangle(x_base, y_base, x_base + size, y_base - size, fill=b.color)
            points = np.concatenate(b.geometry.corner_points_2d()).tolist()
            for p_idx, p in enumerate(points):
                if p_idx % 2 != 0:
                    points[p_idx] = self.environment_extent[1] - p
            canvas.create_polygon(points, fill=b.color, outline="black")
    '''
