from typing import List, Tuple

import numpy as np

import env.block
from env.util import ccw_angle_and_distance
from geom.util import simple_distance


class Map:
    """
    The container for all relevant elements for the construction task. Most importantly this includes a
    target occupancy matrix for the structure to be built, information about the environment extent, all
    of the building blocks and the agents.
    """

    def __init__(self,
                 target_map: np.ndarray,
                 offset_origin: Tuple[float, float] = (0.0, 0.0),
                 environment_extent: List[float] = None):
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
        """
        Add a number of agents to keep track of.

        This method also uses one of the agents' methods to get access to a component map of the structure
        for later use.

        :param agents: a list of agents
        :return:
        """

        self.agents.extend(agents)
        self.store_component_coordinates(self.agents[0].split_into_components())

    def add_blocks(self, blocks: List[env.block.Block]):
        """
        Add a number of blocks to keep track of.

        This method also sets the extent of the environment to match the most extreme coordinates of any
        of the blocks if necessary. In addition, a number of block stashes is created from the list of
        blocks, which are assumed to have been placed before. If block stashes were used in which the blocks
        do not share the same (x, y) coordinates, some other method should be used for this purpose.

        :param blocks: a list of building blocks
        """

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
        """
        Return the number of blocks required to build the structure defined by the map.

        :return: the number of blocks required for the structure
        """

        return np.count_nonzero(self.target_map)

    def original_seed_position(self):
        y, x = np.where(self.target_map[0] == 2)
        return x[0] * env.block.Block.SIZE + self.offset_origin[0], \
               y[0] * env.block.Block.SIZE + self.offset_origin[1], \
               env.block.Block.SIZE / 2  # + env.block.Block.SIZE

    def original_seed_grid_position(self):
        """
        Return the grid position of the original seed on the lowest layer of the structure,
        defined by the numpy array used to construct the map object.

        :return: grid position of the initial seed as a tuple
        """

        z, y, x = np.where(self.target_map == 2)
        return np.array([x[0], y[0], z[0]], dtype="int32")

    def place_block(self, grid_position, block):
        """
        Place the given block at the specified grid position.

        The occupancy matrix is changed to reflect this changed and the block is added to the
        list of placed blocks.

        :param grid_position: position in the grid to place the block at
        :param block: block to place
        """

        self.occupancy_map[tuple(reversed(grid_position))] = 2 if block.is_seed else 1
        self.placed_blocks.append(block)
        if grid_position[2] > self.highest_block_z:
            self.highest_block_z = grid_position[2]
        if self.global_information:
            for a in self.agents:
                a.local_occupancy_map[tuple(reversed(grid_position))] = 1

    def check_occupancy_map(self, position, comparator=lambda x: x != 0):
        """
        Check whether the specified condition holds at the given position.

        Using the occupancy matrix, it is determined whether the expression specified by the
        comparator function is true for the entry at the specified position.

        :param position: grid position to check
        :param comparator: expression evaluating to True or False which is applied to the entry at the position
        :return: True if the condition holds, False otherwise
        """

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

    def block_below(self, position, structure_level=None):
        """
        Return an already placed block below the specified position, if there is one.

        :param position: coordinates in 3D space
        :param structure_level: the level of the structure to check for a block below the position
        (if omitted all levels are checked)
        :return: the block below the specified position or None if there is none
        """

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

    def block_at_position(self, position):
        """
        Return the block located at a specified grid position, if there is one.

        :param position: the grid position for which to return the block
        :return: the block at the specified grid position or None if there is none
        """

        for b in self.placed_blocks:
            if all(b.grid_position[i] == position[i] for i in range(3)):
                return b
        return None

    def ccw_block_stash_locations(self):
        """
        Return the (x, y) coordinates of all block stashes on the map in counter-clockwise order.

        :return: locations of all block stashes
        """

        stash_positions = list(self.block_stashes.keys())
        ordered_stash_positions = sorted(stash_positions,
                                         key=lambda x: ccw_angle_and_distance(x, self.center, stash_positions[0]))
        return ordered_stash_positions

    def ccw_seed_stash_locations(self):
        """
        Return the (x, y) coordinates of all seed stashes on the map in counter-clockwise order.

        :return: locations of all seed stashes
        """

        stash_positions = list(self.seed_stashes.keys())
        ordered_stash_positions = sorted(stash_positions,
                                         key=lambda x: ccw_angle_and_distance(x, self.center, stash_positions[0]))
        return ordered_stash_positions

    def ccw_corner_locations(self, position, offset=0.0):
        """
        Return the (x, y) coordinates of the corners of the construction area in counter-clockwise order.

        :param position: the position to take as reference for ordering
        :param offset: can be used to add a diagonal offset to the corner coordinates
        :return: locations of the construction area corners
        """

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
        """
        Check whether the specified position is within the construction area.

        :param position: the position to check
        :return: True if the position is in the construction area, False otherwise
        """

        return self.offset_origin[0] <= position[0] < self.offset_origin[0] \
               + env.block.Block.SIZE * self.target_map.shape[2] \
               and self.offset_origin[1] <= position[1] < self.offset_origin[1] \
               + env.block.Block.SIZE * self.target_map.shape[1]

    def count_over_construction_area(self):
        """
        Count the number of agents currently over the construction area.

        :return: number of agents over the construction area
        """

        agent_count = 0
        for a in self.agents:
            if self.check_over_construction_area(a.geometry.position):
                agent_count += 1
        return agent_count

    def count_over_component(self, component_marker):
        """
        Count the number of agents currently over the component corresponding to the specified marker.

        :param component_marker: the marker of the component
        :return:
        """

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
        """
        Check whether at least one position of the component corresponding to the given marker is occupied.

        :param component_marker: the marker of the component to check
        :return: True if at least one position is occupied, False otherwise
        """

        return np.count_nonzero(self.occupancy_map[self.component_info[component_marker]["coordinates_np"]]) > 0

    def component_finished(self, component_marker):
        """
        Check whether all positions of the component corresponding to the given marker are occupied.

        :param component_marker: the marker of the component to check
        :return: True if all positions are occupied, False otherwise
        """

        return np.count_nonzero(self.occupancy_map[self.component_info[component_marker]["coordinates_np"]]) == \
               len(self.component_info[component_marker]["coordinates_np"][0])

    def layer_started(self, layer):
        """
        Check whether at least one position of the specified layer is occupied.

        :param layer: the layer to check
        :return: True if at least one position is occupied, False otherwise
        """

        return np.count_nonzero(self.occupancy_map[layer]) > 0

    def layer_finished(self, layer):
        """
        Check whether all positions of the specified layer are occupied.

        :param layer: the layer to check
        :return: True if all positions are occupied, False otherwise
        """

        return np.count_nonzero(self.occupancy_map[layer]) == np.count_nonzero(self.target_map[layer])

    def density_over_construction_area(self, required_area_side_length=4):
        """
        Return the density of agents over the construction area.

        The density is defined as the number of agents over the construction area divided by the area
        of that region in terms of building blocks. Instead of returning just the density, it is
        divided by a reference density which is deemed not to be too crowded, so that values lower than
        1 returned by this method represent an "acceptable" density.

        :return: normalised density of agents over the construction area
        """

        # originally, the density was normalised using the collision cloud of the agent (when used for waiting on the
        # perimeter), but performance actually suffered from that; using the new default resulted in some improvements
        # on structures with small horizontal size
        construction_area = self.target_map.shape[2] * self.target_map.shape[1]
        required_area = required_area_side_length ** 2
        return (self.count_over_construction_area() / construction_area) / (1 / required_area)

    def distance_to_construction_area(self, position):
        """
        Return the distance of the specified position to the construction area (or 0 if inside the area).

        :param position: the position for which to return the distance
        :return: the distance of the position to the construction area
        """

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
        """
        Extracts and saves information about components of the structure for later use.

        :param component_target_map: the map of components in the structure
        """

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

