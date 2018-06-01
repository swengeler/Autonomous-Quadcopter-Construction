import numpy as np
import seaborn as sns
import logging
import env.block
from typing import List, Tuple
from geom.util import simple_distance
from env.util import cw_angle_and_distance, ccw_angle_and_distance


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
        self.center = (self.offset_origin[0] + env.block.Block.SIZE * int(self.target_map.shape[2] / 2),
                       self.offset_origin[1] + env.block.Block.SIZE * int(self.target_map.shape[1] / 2))

        # global information available?
        self.global_information = False

        # some things to keep track of
        self.highest_block_z = 0
        self.component_target_map = None
        self.component_info = {}

    def add_agents(self, agents):
        self.agents.extend(agents)
        self.store_component_coordinates(self.agents[0].split_into_components())

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
        # if self.occupancy_map[tuple(reversed(grid_position))] != 0:
        #     # self.logger.error("Block at {} (is_seed = {}) placed in already occupied location: {}"
        #     #                   .format(block.grid_position, block.is_seed, grid_position))
        #     print("Block at {} (is_seed = {}) placed in already occupied location: {}"
        #           .format(block.grid_position, block.is_seed, grid_position))
        self.occupancy_map[tuple(reversed(grid_position))] = 2 if block.is_seed else 1
        self.placed_blocks.append(block)
        if grid_position[2] > self.highest_block_z:
            self.highest_block_z = grid_position[2]
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
            for height_level in range(self.occupancy_map.shape[0], -1, -1):
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
        ordered_stash_positions = sorted(stash_positions,
                                         key=lambda x: cw_angle_and_distance(x, self.center, stash_positions[0]))
        return ordered_stash_positions[::-1]

    def ccw_seed_stash_locations(self):
        stash_positions = list(self.seed_stashes.keys())
        ordered_stash_positions = sorted(stash_positions,
                                         key=lambda x: cw_angle_and_distance(x, self.center, stash_positions[0]))
        return ordered_stash_positions[::-1]

    def ccw_corner_locations(self, position, offset=0.0):
        corner_locations = [(self.offset_origin[0] - offset, self.offset_origin[1] - offset),
                            (self.offset_origin[0] + env.block.Block.SIZE * self.target_map.shape[2] + offset,
                             self.offset_origin[1] - offset),
                            (self.offset_origin[0] + env.block.Block.SIZE * self.target_map.shape[2] + offset,
                             self.offset_origin[1] + env.block.Block.SIZE * self.target_map.shape[1] + offset),
                            (self.offset_origin[0] - offset,
                             self.offset_origin[1] + env.block.Block.SIZE * self.target_map.shape[1] + offset)]
        ordered_corner_locations = sorted(corner_locations,
                                          key=lambda x: ccw_angle_and_distance(x, self.center, position))
        if all(ordered_corner_locations[0][i] == position[i] for i in range(2)):
            temp = ordered_corner_locations[0]
            ordered_corner_locations.remove(temp)
            ordered_corner_locations.append(temp)
        return ordered_corner_locations

    def check_over_construction_area(self, position):
        return self.offset_origin[0] <= position[0] < self.offset_origin[0] \
               + env.block.Block.SIZE * self.target_map.shape[2] \
               and self.offset_origin[1] <= position[1] < self.offset_origin[1] \
               + env.block.Block.SIZE * self.target_map.shape[1]

    def count_over_construction_area(self):
        agent_count = 0
        for a in self.agents:
            if self.check_over_construction_area(a.geometry.position):
                agent_count += 1
        return agent_count

    def count_over_component(self, component_marker):
        if self.highest_block_z > self.component_info[component_marker]["layer"] \
                or self.highest_block_z < self.component_info[component_marker]["layer"]:
            return 0

        agent_count = 0
        for a in self.agents:
            closest_x = int((a.geometry.position[0] - self.offset_origin[0]) / env.block.Block.SIZE)
            closest_y = int((a.geometry.position[1] - self.offset_origin[1]) / env.block.Block.SIZE)
            if closest_x in self.component_info[component_marker]["coordinates_np"][2] \
                    and closest_y in self.component_info[component_marker]["coordinates_np"][1]:
                return agent_count
        return 0

    def component_started(self, component_marker):
        return np.count_nonzero(self.occupancy_map[self.component_info[component_marker]["coordinates_np"]]) > 0

    def component_finished(self, component_marker):
        return np.count_nonzero(self.occupancy_map[self.component_info[component_marker]["coordinates_np"]]) == \
               len(self.component_info[component_marker]["coordinates_np"][0])

    def layer_started(self, layer):
        return np.count_nonzero(self.occupancy_map[layer]) > 0

    def layer_finished(self, layer):
        return np.count_nonzero(self.occupancy_map[layer]) == np.count_nonzero(self.target_map[layer])

    def density_over_construction_area(self):
        construction_area = self.target_map.shape[2] * self.target_map.shape[1]
        # assuming that 1 agent over an area of 16 blocks is roughly balanced (because it takes up that space)
        # -> therefore normalise using that number
        required_area = (np.round(self.agents[0].required_distance / env.block.Block.SIZE) + 1) ** 2
        return (self.count_over_construction_area() / construction_area) / (1 / required_area)

    def count_at_stash(self, stash_position, min_distance=50.0):
        agent_count = 0
        for a in self.agents:
            pass
        pass

    def distance_to_construction_area(self, position):
        if self.check_over_construction_area(position):
            return 0
        # otherwise, find the distance to the closest side
        width = env.block.Block.SIZE * self.target_map.shape[2]
        height = env.block.Block.SIZE * self.target_map.shape[1]
        dx = max(abs(position[0] - (self.offset_origin[0] + width / 2)) - width / 2, 0)
        dy = max(abs(position[1] - (self.offset_origin[1] + height / 2)) - height / 2, 0)
        return np.sqrt(dx ** 2 + dy ** 2)

    def collision_potential_with_structure(self, agent, check_z=False):
        max_index = 3 if check_z else 2
        # should check only in x, y direction by default
        collision_potential_blocks = []
        for b in self.placed_blocks:
            # if there is collision potential, return true, use required distance to check
            if simple_distance(b.geometry.position, agent.geometry.position) \
                    < b.geometry.size[0] + agent.required_distance / 2:
                collision_potential_blocks.append(b)
        # the block information can then also be used to identify the highest block
        # then something has to be done to rise to that level and possibly go closer to said block
        # -> if there is still a block in the way then we should rise higher
        # perhaps could just do a while block in the way, rise sorta thing
        return collision_potential_blocks

    def store_component_coordinates(self, component_target_map):
        self.component_target_map = component_target_map
        for m in [cm for cm in np.unique(self.component_target_map) if cm != 0]:
            coords = np.where(self.component_target_map == m)
            coord_tuples = tuple(zip(coords[2], coords[1], coords[0]))
            layer = coords[2][0]
            self.component_info[m] = {
                "coordinates_np": coords,
                "coordinate_tuples": coord_tuples,
                "layer": layer
            }

