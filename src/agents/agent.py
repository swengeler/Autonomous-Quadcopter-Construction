import numpy as np
import logging
import env.map
from enum import Enum
from abc import ABCMeta, abstractmethod
from env.block import Block
from geom.shape import *
from geom.util import simple_distance

np.seterr(divide='ignore', invalid='ignore')


class Task(Enum):
    FETCH_BLOCK = 0
    PICK_UP_BLOCK = 2
    TRANSPORT_BLOCK = 3
    MOVE_TO_PERIMETER = 4
    FIND_ATTACHMENT_SITE = 5
    PLACE_BLOCK = 6
    FIND_NEXT_COMPONENT = 7
    SURVEY_COMPONENT = 8
    RETURN_BLOCK = 9
    AVOID_COLLISION = 10
    LAND = 11
    FINISHED = 12


class AgentStatistics:

    def __init__(self, agent):
        self.agent = agent
        self.task_counter = {
            Task.FETCH_BLOCK: 0,
            Task.PICK_UP_BLOCK: 0,
            Task.TRANSPORT_BLOCK: 0,
            Task.MOVE_TO_PERIMETER: 0,
            Task.FIND_ATTACHMENT_SITE: 0,
            Task.PLACE_BLOCK: 0,
            Task.FIND_NEXT_COMPONENT: 0,
            Task.SURVEY_COMPONENT: 0,
            Task.RETURN_BLOCK: 0,
            Task.AVOID_COLLISION: 0,
            Task.LAND: 0,
            Task.FINISHED: 0
        }
        self.previous_task = None

    def step(self, environment: env.map.Map):
        if self.previous_task != self.agent.current_task:
            aprint(self.agent.id, "Changed task to {}".format(self.agent.current_task))
            self.previous_task = self.agent.current_task
        self.task_counter[self.agent.current_task] = self.task_counter[self.agent.current_task] + 1


def check_map(map_to_check, position, comparator=lambda x: x != 0):
    if any(position < 0):
        return comparator(0)
    try:
        temp = map_to_check[tuple(np.flip(position, 0))]
    except IndexError:
        return comparator(0)
    else:
        val = comparator(temp)
        return val


def aprint(identifier, *args, **kwargs):
    if isinstance(identifier, str):
        identifier = "?"
    print("[Agent {}]: ".format(identifier), end="")
    print(*args, **kwargs)


class Agent:
    __metaclass__ = ABCMeta

    MOVEMENT_PER_STEP = 5

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.geometry = GeomBox(position, size, 0.0)
        self.collision_avoidance_geometry = GeomBox(position,
                                                    [size[0] + required_spacing * 2,
                                                     size[1] + required_spacing * 2,
                                                     size[2] + required_spacing * 2], 0.0)
        self.collision_avoidance_geometry_with_block = GeomBox([position[0], position[1],
                                                                position[2] - (Block.SIZE - size[2]) / 2 - size[2] / 2],
                                                               [size[0] + required_spacing * 2,
                                                                size[1] + required_spacing * 2,
                                                                size[2] + Block.SIZE + required_spacing * 2], 0.0)
        self.geometry.following_geometries.append(self.collision_avoidance_geometry)
        self.geometry.following_geometries.append(self.collision_avoidance_geometry_with_block)
        self.target_map = target_map
        self.component_target_map = None
        self.required_spacing = required_spacing
        self.required_distance = 100
        self.required_vertical_distance = -50
        self.known_empty_stashes = []
        self.component_target_map = self.split_into_components()
        self.closing_corners, self.hole_map, self.hole_boundaries, self.closing_corner_boundaries = \
            self.find_closing_corners()

        self.local_occupancy_map = np.zeros_like(target_map)
        self.next_seed = None
        self.next_seed_position = None
        self.current_seed = None
        self.current_block = None
        self.current_trajectory = None
        self.current_task = Task.FETCH_BLOCK
        self.current_path = None
        self.current_structure_level = 0
        self.current_grid_position = None  # closest grid position if at structure
        self.current_grid_direction = None
        self.current_row_started = False
        self.current_component_marker = 2
        self.current_visited_sites = None
        self.current_seed_grid_positions = None
        self.current_seed_grid_position_index = 0
        self.current_grid_positions_to_be_seeded = None
        self.current_grid_position_to_be_seeded = None
        self.current_seeded_positions = None
        self.current_block_type_seed = False
        self.current_collision_avoidance_counter = 0
        self.backup_grid_position = None
        self.previous_task = Task.FETCH_BLOCK
        self.transporting_to_seed_site = False
        self.path_before_collision_avoidance_none = False

        self.collision_using_geometries = False
        self.task_history = []
        self.task_history.append(self.current_task)
        self.LAND_CALLED_FIRST_TIME = False
        self.id = -1
        self.collision_possible = True
        self.repeated = 0

        self.agent_statistics = AgentStatistics(self)

    @abstractmethod
    def advance(self, environment: env.map.Map):
        pass

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

    def move(self, environment: env.map.Map):
        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a) \
                        and a.geometry.position[2] <= self.geometry.position[2] - self.required_vertical_distance:
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if any(np.isnan(current_direction)):
            current_direction = np.array([0, 0, 0])

        return next_position, current_direction

    def collision_potential(self, other):
        # check whether self and other have other geometries attached
        # find midpoint of collective geometries
        # compute minimum required distance to not physically collide
        # check whether distance is large enough

        # OR: use collision box and check overlap
        if self.collision_using_geometries:
            if self.current_block is not None and self.current_block.geometry in self.geometry.following_geometries:
                # block is attached, use collision_avoidance_geometry_with_block geometry
                if other.current_block is not None and other.current_block.geometry in other.geometry.following_geometries:
                    return self.collision_avoidance_geometry_with_block.overlaps(
                        other.collision_avoidance_geometry_with_block)
                else:
                    return self.collision_avoidance_geometry_with_block.overlaps(other.collision_avoidance_geometry)
            else:
                # no block attached, check only quadcopter
                if other.current_block is not None and other.current_block.geometry in other.geometry.following_geometries:
                    return self.collision_avoidance_geometry.overlaps(other.collision_avoidance_geometry_with_block)
                else:
                    return self.collision_avoidance_geometry.overlaps(other.collision_avoidance_geometry)
        else:
            if simple_distance(self.geometry.position + np.array([0, 0, -self.geometry.size[2]]),
                               other.geometry.position) \
                    < self.required_distance:
                return True
        return False

    def collision_potential_visible(self, other):
        if not self.collision_potential(other):
            return False
        # check whether other agent is within view, i.e. below this agent or in view of one of the cameras

    def update_local_occupancy_map(self, environment: env.map.Map):
        # update knowledge of the map
        for y_diff in (-1, 0, 1):
            for x_diff in (-1, 0, 1):
                if environment.check_occupancy_map(np.array([self.current_grid_position[0] + x_diff,
                                                             self.current_grid_position[1] + y_diff,
                                                             self.current_grid_position[2]])):
                    self.local_occupancy_map[self.current_grid_position[2],
                                             self.current_grid_position[1] + y_diff,
                                             self.current_grid_position[0] + x_diff] = 1

        # if there are any sites to fill in the target map which are surrounded by blocks on 3 sides
        # in the local occupancy map, then the site is also assumed to be filled because the rules
        # of construction would not allow anything else, e.g. with this as the local map:
        # [B] [O] [B] [A] ...
        # [B] [O] [B] [B] ...
        # [B] [B] [B] [B] ...
        # ... ... ... ... ...
        # where B = known block positions, A = attachment site, O = empty sites (to be occupied)
        # in this case, the agent may have attached the upper-left most block when the adjacent two
        # empty (but to-be-occupied) blocks were still empty, and then came back later when all the
        # other blocks (marked here as B) had been filled out already -> since anything else would
        # not be permitted, the agent knows that the previously empty sites have to be occupied at
        # this point
        # this would actually also be the case if there is a gap of more than 1:
        # TODO: make this work for bigger gaps than 1
        # any continuous row/column in the target map between two occupied locations in the local occupancy
        # map should be assumed to be filled out already
        current_occupancy_map = self.local_occupancy_map[self.current_grid_position[2]]
        current_target_map = self.target_map[self.current_grid_position[2]]
        for y in range(current_target_map.shape[0]):
            for x in range(current_target_map.shape[1]):
                if current_occupancy_map[y, x] != 0:
                    counter = 0
                    for diff in (-1, 1):
                        y2 = y + diff
                        if 0 <= y2 < self.target_map.shape[1]:
                            if self.local_occupancy_map[self.current_grid_position[2], y2, x] != 0:
                                counter += 1
                        x2 = x + diff
                        if 0 <= x2 < self.target_map.shape[2]:
                            if self.local_occupancy_map[self.current_grid_position[2], y, x2] != 0:
                                counter += 1
                    if counter >= 3:
                        self.local_occupancy_map[self.current_grid_position[2], y, x] = 1

                    # for diff in (-1, 1):
                    #     # making it through this loop without a break means that in the x-row, y-column where the block
                    #     # could be placed, there is either only a block immediately adjacent or any blocks already
                    #     # placed are separated from the current site by an intended gap
                    #
                    #     counter = 1
                    #     while 0 <= y + counter * diff < current_occupancy_map.shape[0] \
                    #             and current_occupancy_map[y + counter * diff, x] == 0 \
                    #             and current_target_map[y + counter * diff, x] > 0:
                    #         counter += 1
                    #     if counter > 1 and 0 <= y + counter * diff < current_occupancy_map.shape[0] \
                    #             and current_occupancy_map[y + counter * diff, x] > 0 and \
                    #             current_target_map[y + counter * diff, x] > 0:
                    #         # have encountered a block in this column, mark all in between
                    #         # this position and the end of that column as occupied
                    #         other_y = y + counter * diff
                    #         if other_y > y:
                    #             self.local_occupancy_map[self.current_grid_position[2], y:other_y, x] = 1
                    #         else:
                    #             self.local_occupancy_map[self.current_grid_position[2], other_y:y, x] = 1
                    #
                    #     counter = 1
                    #     while 0 <= x + counter * diff < current_occupancy_map.shape[1] \
                    #             and current_occupancy_map[y, x + counter * diff] == 0 \
                    #             and current_target_map[y, x + counter * diff] > 0:
                    #         counter += 1
                    #     if counter > 1 and 0 <= x + counter * diff < current_occupancy_map.shape[1] \
                    #             and current_occupancy_map[y, x + counter * diff] > 0 and \
                    #             current_target_map[y, x + counter * diff] > 0:
                    #         # have encountered a block in this row, mark all in between
                    #         # this position and the end of that row as occupied
                    #         other_x = x + counter * diff
                    #         if other_x > x:
                    #             self.local_occupancy_map[self.current_grid_position[2], y, x:other_x] = 1
                    #         else:
                    #             self.local_occupancy_map[self.current_grid_position[2], y, other_x:x] = 1

    def check_component_finished(self, compared_map: np.ndarray, component_marker=None):
        if component_marker is None:
            component_marker = self.current_component_marker
        tm = np.zeros_like(self.target_map[self.current_structure_level])
        np.place(tm, self.component_target_map[self.current_structure_level] == component_marker, 1)
        om = np.copy(compared_map[self.current_structure_level])
        np.place(om, om > 0, 1)
        np.place(om, self.component_target_map[self.current_structure_level] != component_marker, 0)
        return np.array_equal(om, tm)

    def check_layer_finished(self, compared_map: np.ndarray):
        tm = np.copy(self.target_map[self.current_structure_level])
        np.place(tm, tm > 0, 1)
        om = np.copy(compared_map[self.current_structure_level])
        np.place(om, om > 0, 1)
        return np.array_equal(om, tm)

    def check_structure_finished(self, compared_map: np.ndarray):
        tm = np.copy(self.target_map)
        np.place(tm, tm > 0, 1)
        om = np.copy(compared_map)
        np.place(om, om > 0, 1)
        return np.array_equal(om, tm)

    def unfinished_component_markers(self, compared_map: np.ndarray, level=None):
        if level is None:
            level = self.current_structure_level
        candidate_components = []
        for marker in np.unique(self.component_target_map[level]):
            if marker != 0:  # != self.current_component_marker:
                subset_indices = np.where(
                    self.component_target_map[level] == marker)
                candidate_values = compared_map[level][subset_indices]
                # the following check means that on the occupancy map, this component still has all
                # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                # for placing the currently transported seed there
                if np.count_nonzero(candidate_values == 0) > 0:
                    candidate_components.append(marker)
        return candidate_components

    def unseeded_component_markers(self, compared_map: np.ndarray, level=None):
        if level is None:
            level = self.current_structure_level
        candidate_components = []
        for marker in np.unique(self.component_target_map[level]):
            if marker != 0:  # != self.current_component_marker:
                subset_indices = np.where(
                    self.component_target_map[level] == marker)
                candidate_values = compared_map[level][subset_indices]
                if np.count_nonzero(candidate_values) == 0:
                    candidate_components.append(marker)
        return candidate_components

    def component_seed_location(self, component_marker, level=None):
        if level is None:
            level = self.current_structure_level

        # according to whichever strategy is currently being employed for placing the seed, return that location
        # this location is the grid location, not the absolute spatial position
        occupied_locations = np.where(self.component_target_map[level] == component_marker)
        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
        supported_locations = np.nonzero(self.target_map[level - 1])
        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
        occupied_locations = [x for x in occupied_locations if x in supported_locations]
        occupied_locations = [x for x in occupied_locations if (x[1], x[0], level)
                              not in self.closing_corners[level][self.current_component_marker]]

        # TODO: while this now uses the most SOUTH-WESTERN position, something else might be even better
        sorted_by_y = sorted(occupied_locations, key=lambda e: e[0])
        sorted_by_x = sorted(sorted_by_y, key=lambda e: e[1])

        seed_location = [sorted_by_x[0][1], sorted_by_x[0][0], level]
        return seed_location

    def check_loop_corner(self, environment: env.map.Map, position=None):
        if position is None:
            position = self.current_grid_position

        loop_corner_attachable = False
        at_loop_corner = False
        if tuple(position) \
                in self.closing_corners[self.current_structure_level][self.current_component_marker]:
            at_loop_corner = True
            # need to check whether the adjacent blocks have been placed already
            counter = 0
            surrounded_in_y = False
            surrounded_in_x = False
            index = self.closing_corners[self.current_structure_level][self.current_component_marker].index(
                tuple(position))
            possible_boundaries = self.closing_corner_boundaries[self.current_structure_level][
                self.current_component_marker][index]
            if environment.check_occupancy_map(position + np.array([0, -1, 0])) and \
                    tuple(position + np.array([0, -1, 0])) in possible_boundaries:
                counter += 1
                surrounded_in_y = True
            if not surrounded_in_y and environment.check_occupancy_map(
                    position + np.array([0, 1, 0])) and tuple(position + np.array([0, 1, 0])) in possible_boundaries:
                counter += 1
            if environment.check_occupancy_map(position + np.array([-1, 0, 0])) and \
                    tuple(position + np.array([-1, 0, 0])) in possible_boundaries:
                counter += 1
                surrounded_in_x = True
            if not surrounded_in_x and environment.check_occupancy_map(
                    position + np.array([1, 0, 0])) and tuple(position + np.array([1, 0, 0])) in possible_boundaries:
                counter += 1
            if counter >= 2:
                # aprint(self.id, "CORNER ALREADY SURROUNDED BY ADJACENT BLOCKS")
                loop_corner_attachable = True
            else:
                # aprint(self.id, "CORNER NOT SURROUNDED BY ADJACENT BLOCKS YET")
                pass
        else:
            loop_corner_attachable = True

        return at_loop_corner, loop_corner_attachable

    def split_into_components(self):
        def flood_fill(layer, i, j, marker):
            if layer[i, j] == 1:
                layer[i, j] = marker

                if i > 0:
                    flood_fill(layer, i - 1, j, marker)
                if i < layer.shape[0] - 1:
                    flood_fill(layer, i + 1, j, marker)
                if j > 0:
                    flood_fill(layer, i, j - 1, marker)
                if j < layer.shape[1] - 1:
                    flood_fill(layer, i, j + 1, marker)

        # go through the target map layer by layer and split each one into disconnected components
        # how to store these components? could be separate target map, using numbers to denote each component
        # for
        self.component_target_map = np.copy(self.target_map)
        np.place(self.component_target_map, self.component_target_map > 1, 1)
        component_marker = 2
        for z in range(self.component_target_map.shape[0]):
            # use flood fill to identify components of layer
            for y in range(self.component_target_map.shape[1]):
                for x in range(self.component_target_map.shape[2]):
                    if self.component_target_map[z, y, x] == 1:
                        flood_fill(self.component_target_map[z], y, x, component_marker)
                        component_marker += 1
        return self.component_target_map

    def find_closing_corners(self):
        # for each layer, check whether there are any holes, i.e. 0's that - when flood-filled - only connect to 1's
        def flood_fill(layer, i, j, marker):
            if layer[i, j] == 0:
                layer[i, j] = marker
                if i > 0:
                    flood_fill(layer, i - 1, j, marker)
                if i < layer.shape[0] - 1:
                    flood_fill(layer, i + 1, j, marker)
                if j > 0:
                    flood_fill(layer, i, j - 1, marker)
                if j < layer.shape[1] - 1:
                    flood_fill(layer, i, j + 1, marker)

        hole_map = np.copy(self.target_map)
        np.place(hole_map, hole_map > 1, 1)
        hole_marker = 2
        valid_markers = []
        for z in range(hole_map.shape[0]):
            valid_markers.append([])
            # use flood fill to identify hole(s)
            for y in range(hole_map.shape[1]):
                for x in range(hole_map.shape[2]):
                    if hole_map[z, y, x] == 0:
                        flood_fill(hole_map[z], y, x, hole_marker)
                        valid_markers[z].append(hole_marker)
                        hole_marker += 1

        # TODO: need to check whether hole is between two different components, because then it's not a hole

        for z in range(len(valid_markers)):
            temp_copy = valid_markers[z].copy()
            for m in valid_markers[z]:
                locations = np.where(hole_map == m)
                if 0 in locations[1] or 0 in locations[2] or hole_map.shape[1] - 1 in locations[1] \
                        or hole_map.shape[2] - 1 in locations[2]:
                    temp_copy.remove(m)
                    hole_map[locations] = 0
            valid_markers[z][:] = temp_copy

        # now need to find the enclosing cycles/loops for each hole
        def boundary_search(layer, z, i, j, marker, visited, c_list):
            if visited is None:
                visited = []
            if not (i, j) in visited and layer[i, j] == marker:
                visited.append((i, j))
                if i > 0:
                    boundary_search(layer, z, i - 1, j, marker, visited, c_list)
                if i < layer.shape[0] - 1:
                    boundary_search(layer, z, i + 1, j, marker, visited, c_list)
                if j > 0:
                    boundary_search(layer, z, i, j - 1, marker, visited, c_list)
                if j < layer.shape[1] - 1:
                    boundary_search(layer, z, i, j + 1, marker, visited, c_list)
            elif layer[i, j] != marker:
                c_list.append((z, i, j))

        hole_boundaries = []
        removable_holes = []
        for z in range(hole_map.shape[0]):
            hole_boundaries.append([])
            for m in valid_markers[z]:
                coord_list = []
                locations = np.where(hole_map == m)
                boundary_search(hole_map[z], z, locations[1][0], locations[2][0], m, None, coord_list)
                first_component_marker = self.component_target_map[z, coord_list[0][1], coord_list[0][0]]
                for _, y, x in coord_list:
                    if self.component_target_map[z, y, x] != first_component_marker:
                        removable_holes.append((z, m))
                        break
                coord_list = tuple(np.moveaxis(np.array(coord_list), -1, 0))
                hole_boundaries[z].append(coord_list)

        # remove those holes between components
        for z, m in removable_holes:
            index = valid_markers[z].index(m)
            np.place(hole_map[z], hole_map[z] == m, 0)
            del valid_markers[z][index]
            del hole_boundaries[z][index]

        a_dummy_copy = np.copy(hole_map)
        hole_corners = []
        closing_corners = []
        closing_corner_boundaries = []
        hole_boundary_coords = dict()
        for z in range(hole_map.shape[0]):
            hole_corners.append([])
            closing_corners.append({})
            closing_corner_boundaries.append({})
            for cm in np.unique(self.component_target_map):
                # for each component marker, make new entry in closing_corners dictionary
                closing_corners[z][cm] = []
                closing_corner_boundaries[z][cm] = []
            # this list is used to keep track of which (closing) corners (and boundaries) belong to which components
            for m_idx, m in enumerate(valid_markers[z]):
                # find corners as coordinates that are not equal to the marker and adjacent to
                # two of the boundary coordinates (might only want to look for outside corners though)
                boundary_coord_tuple_list = list(zip(
                    hole_boundaries[z][m_idx][2], hole_boundaries[z][m_idx][1], hole_boundaries[z][m_idx][0]))
                hole_boundary_coords[m] = hole_boundaries[z][m_idx]
                outer_corner_coord_list = []
                outer_open_corner_coord_list = []
                inner_corner_coord_list = []
                corner_boundary_list = []

                component_marker_list_outer = []
                component_marker_list_outer_open = []
                component_marker_list_inner = []
                for y in range(hole_map.shape[1]):
                    for x in range(hole_map.shape[2]):
                        # check for each possible orientation of an outer corner whether the current block
                        # matches that pattern, which would be of the following form:
                        # [C] [B]
                        # [B] [H]
                        # or rotated by 90 degrees (where C = corner, B = boundary, H = hole)
                        for y2 in (y - 1, y + 1):
                            for x2 in (x - 1, x + 1):
                                if 0 <= y2 < hole_map.shape[1] and 0 <= x2 < hole_map.shape[2] \
                                        and hole_map[z, y, x] == 1 and hole_map[z, y2, x2] == m \
                                        and (x, y2, z) in boundary_coord_tuple_list \
                                        and (x2, y, z) in boundary_coord_tuple_list:
                                    a_dummy_copy[z, y, x] = -m
                                    outer_corner_coord_list.append((x, y, z))
                                    corner_boundary_list.append([(x, y2, z), (x2, y, z)])
                                    component_marker_list_outer.append(self.component_target_map[z, y, x])
                        # do the same for inner corners, which have the following pattern:
                        # [C] [H]
                        # [H] [H]
                        for y2 in (y - 1, y + 1):
                            for x2 in (x - 1, x + 1):
                                if 0 <= y2 < hole_map.shape[1] and 0 <= x2 < hole_map.shape[2] \
                                        and hole_map[z, y, x] == 1 and hole_map[z, y2, x2] == m \
                                        and hole_map[z, y, x2] == m and hole_map[z, y2, x] == m:
                                    a_dummy_copy[z, y, x] = -m
                                    inner_corner_coord_list.append((x, y, z))
                                    component_marker_list_inner.append(self.component_target_map[z, y, x])

                        # [H]

                # split up the corners and boundaries into the different components
                for cm in np.unique(component_marker_list_outer):
                    current_outer = [outer_corner_coord_list[i] for i in range(len(outer_corner_coord_list))
                                     if component_marker_list_outer[i] == cm]
                    current_inner = [inner_corner_coord_list[i] for i in range(len(inner_corner_coord_list))
                                     if component_marker_list_inner[i] == cm]
                    current_boundary = [corner_boundary_list[i] for i in range(len(corner_boundary_list))
                                        if component_marker_list_outer[i] == cm]
                    # if (len(current_outer) + len(current_inner)) % 2 == 1:
                    #     # in this case, need to check whether the hole's "open" corner is where the
                    #     # closing corner would otherwise be; in that case it should work out fine
                    #     sorted_by_y = sorted(range(len(current_outer)),
                    #                          key=lambda e: current_outer[e][1], reverse=True)
                    #     sorted_by_x = sorted(sorted_by_y, key=lambda e: current_outer[e][0], reverse=True)
                    #     current_outer = [current_outer[i] for i in sorted_by_x]
                    #     current_boundary = [current_boundary[i] for i in sorted_by_x]
                    #     closing_corners[z][cm].append(current_outer[0])
                    #     closing_corner_boundaries[z][cm].append(current_boundary[0])
                    if (len(current_outer) + len(current_inner)) % 2 == 0:
                        sorted_by_y = sorted(range(len(current_outer)), key=lambda e: current_outer[e][1], reverse=True)
                        sorted_by_x = sorted(sorted_by_y, key=lambda e: current_outer[e][0], reverse=True)
                        current_outer = [current_outer[i] for i in sorted_by_x]
                        current_boundary = [current_boundary[i] for i in sorted_by_x]
                        closing_corners[z][cm].append(current_outer[0])
                        closing_corner_boundaries[z][cm].append(current_boundary[0])
                    else:
                        self.logger.warning("The structure contains open corners, "
                                            "which cannot be built using the PerimeterFollowingAgent.")
                    # if (len(current_outer) + len(current_inner)) % 2 == 0:
                    #     sorted_by_x = sorted(range(len(current_outer)), key=lambda e: current_outer[e][0], reverse=True)
                    #     sorted_by_y = sorted(sorted_by_x, key=lambda e: current_outer[e][1], reverse=True)
                    #     current_outer = [current_outer[i] for i in sorted_by_y]
                    #     current_boundary = [current_boundary[i] for i in sorted_by_y]
                    #     closing_corners[z][cm].append(current_outer[0])
                    #     closing_corner_boundaries[z][cm].append(current_boundary[0])
                    # else:
                    #     self.logger.warning("The structure contains open corners, "
                    #                         "which cannot be built using the PerimeterFollowingAgent.")
                    # if the NORTH-WESTERN corner is an open one and it does not lie next to another hole,
                    # then we should be able to just ignore it
                    # TODO (possibly): not sure if this will always work, might be better to still have a closing
                    # corner; if the open corner is adjacent to another hole, I am not sure it always works
                hole_corners[z].append(outer_corner_coord_list)

        all_hole_boundaries = []
        for z in range(hole_map.shape[0]):
            boundary_locations = []
            for m_idx, m in enumerate(valid_markers[z]):
                # pass 1: get all immediately adjacent boundaries
                # pass 2: get corners
                # empty map with all boundaries of that hole being 1:
                boundary_map = np.zeros_like(hole_map)
                boundary_map[hole_boundaries[z][m_idx]] = 1
                for y in range(boundary_map.shape[1]):
                    for x in range(boundary_map.shape[2]):
                        if 0 < hole_map[z, y, x] < 2:
                            counter = 0
                            for x_diff, y_diff in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                                if 0 <= x_diff < boundary_map.shape[2] and 0 <= y_diff < boundary_map.shape[1] and \
                                        boundary_map[z, y_diff, x_diff] == 1:
                                    counter += 1
                            if counter >= 2:
                                boundary_map[z, y, x] = 1
                boundary_coords = np.where(boundary_map == 1)
                boundary_coord_tuple_list = list(zip(boundary_coords[2], boundary_coords[1], boundary_coords[0]))
                boundary_locations.extend(boundary_coord_tuple_list)
            all_hole_boundaries.append(boundary_locations)

        boundary_map = np.zeros_like(hole_map)
        for z in range(hole_map.shape[0]):
            for x, y, z in all_hole_boundaries[z]:
                boundary_map[z, y, x] = 1

        return closing_corners, hole_map, hole_boundary_coords, closing_corner_boundaries
