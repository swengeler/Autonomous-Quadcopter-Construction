import numpy as np
import logging
import env.map
import random
from collections import deque as dq
from enum import Enum
from abc import ABCMeta, abstractmethod
from env.block import Block
from env.util import print_map, shortest_path
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
    CHECK_STASHES = 11
    LAND = 12
    FINISHED = 13
    WAIT_ON_PERIMETER = 14
    REJOIN_SWARM = 15
    HOVER_OVER_COMPONENT = 16


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
            Task.CHECK_STASHES: 0,
            Task.LAND: 0,
            Task.FINISHED: 0,
            Task.WAIT_ON_PERIMETER: 0,
            Task.REJOIN_SWARM: 0,
            Task.HOVER_OVER_COMPONENT: 0
        }
        self.previous_task = None
        self.collision_danger = []
        self.collision_avoidance_contribution = []
        self.attachment_interval = []

    def step(self, environment: env.map.Map):
        if self.previous_task != self.agent.current_task:
            # print("[Agent {}]: Changed task to {}".format(self.agent.id, self.agent.current_task))
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


class Agent:
    __metaclass__ = ABCMeta

    MOVEMENT_PER_STEP = 5

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float,
                 printing_enabled=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.printing_enabled = printing_enabled

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
        self.initial_position = None
        self.target_map = target_map
        self.component_target_map = None
        self.required_spacing = required_spacing
        self.required_distance = 90
        self.required_vertical_distance = -50
        self.known_empty_stashes = []

        self.local_occupancy_map = np.zeros_like(target_map)
        self.next_seed = None
        self.next_seed_position = None
        self.current_seed = None
        self.current_block = None
        self.current_trajectory = None
        self.current_task = Task.FETCH_BLOCK
        self.current_path = None
        self.previous_path = None
        self.current_static_location = self.geometry.position
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
        self.current_stash_path = None
        self.current_stash_path_index = 0
        self.current_stash_position = None
        self.current_waiting_height = 0

        # parameters that can change for all agent types
        self.waiting_on_perimeter_enabled = False
        self.avoiding_crowded_stashes_enabled = False
        self.transport_avoid_others_enabled = False

        # performance metrics
        self.step_count = 0  # happens in advance method
        self.stuck_count = 0
        self.returned_blocks = 0
        self.per_task_step_count = dict([(task, 0) for task in Task])  # happens in move method
        self.per_task_collision_avoidance_count = dict([(task, 0) for task in Task])  # happens in move method
        self.per_task_distance_travelled = dict([(task, 0) for task in Task]) # has to happen in every method
        self.attachment_frequency_count = []
        self.components_seeded = []  # this and the following one to see how balanced the work load is
        self.components_attached = []
        self.per_search_attachment_site_count = {
            "possible": [],
            "total": []
        }
        self.drop_out_statistics = {
            "drop_out_of_swarm": [],
            "wait_for_rejoining": [],
            "rejoining_swarm": []
        }

        self.backup_grid_position = None
        self.previous_task = Task.FETCH_BLOCK
        self.transporting_to_seed_site = False
        self.path_before_collision_avoidance_none = False
        self.wait_for_rejoining = False
        self.rejoining_swarm = False
        self.position_queue = dq(maxlen=20)
        self.collision_queue = dq(maxlen=100)
        self.collision_average_queue = dq(maxlen=100)
        self.path_finding_contribution_queue = dq(maxlen=100)
        self.collision_avoidance_contribution_queue = dq(maxlen=100)
        self.collision_count = 0
        self.collision_average = 0
        self.non_static_count = 0
        self.count_since_last_attachment = 0
        self.reference_position = np.copy(self.geometry.position)
        self.drop_out_of_swarm = False
        self.close_to_seed_count = 0
        self.seed_arrival_delay_queue = dq(maxlen=10)

        self.max_agent_count = 20
        self.area_density_restricted = True
        self.stash_min_distance = 100

        self.collision_using_geometries = False
        self.task_history = []
        self.task_history.append(self.current_task)
        self.LAND_CALLED_FIRST_TIME = False
        self.id = -1
        self.collision_possible = True
        self.repeated = 0

        self.agent_statistics = AgentStatistics(self)

        self.component_target_map = self.split_into_components()
        # self.multi_layer_component_target_map = self.merge_multi_layer_components()
        self.closing_corners, self.hole_map, self.hole_boundaries, self.closing_corner_boundaries, \
            self.closing_corner_orientations = self.find_closing_corners()

    @abstractmethod
    def advance(self, environment: env.map.Map):
        pass

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

    def move(self, environment: env.map.Map, react_only=False):
        self.per_task_step_count[self.current_task] += 1

        if not react_only:
            next_position = self.current_path.next()
            current_direction = self.current_path.direction_to_next(self.geometry.position)
            original_direction = self.current_path.direction_to_next(self.geometry.position)
        else:
            next_position = self.current_static_location
            current_direction = next_position - self.geometry.position
            original_direction = next_position - self.geometry.position
            current_direction = np.array([0.0, 0.0, 0.0])
            original_direction = np.array([0.0, 0.0, 0.0])

        if sum(np.sqrt(current_direction ** 2)) > 0:
            current_direction /= sum(np.sqrt(current_direction ** 2))

        # scale force vector by angle compared to direction of movement?
        # if angle is small, then the influence of the force vector should also be smaller because
        # the movement itself will already contribute to the collision avoidance
        # otherwise, the force vector has to do most of the work (e.g. also stop the agent movement forwards
        # and should therefore be scaled up)
        # collision_count_updated = False or not self.current_task in [Task.FIND_NEXT_COMPONENT,
        #                                                              Task.FIND_ATTACHMENT_SITE,
        #                                                              Task.MOVE_TO_PERIMETER,
        #                                                              Task.SURVEY_COMPONENT,
        #                                                              Task.PLACE_BLOCK]
        collision_count_updated = False
        total_ff_vector = np.array([0.0, 0.0, 0.0])
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a) and self.collision_potential_visible(a, react_only):
                    # position_difference = a.geometry.position - self.geometry.position
                    # position_signed_angle = np.arctan2(position_difference[1], position_difference[0]) - \
                    #                         np.arctan2(original_direction[1], original_direction[0])

                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, random.random() * np.pi / 8)
                    if not react_only:
                        force_field_vector *= 100 / simple_distance(self.geometry.position, a.geometry.position)
                    else:
                        force_field_vector *= 200 / simple_distance(self.geometry.position, a.geometry.position)
                        force_field_vector[2] /= 2

                    current_direction += force_field_vector
                    total_ff_vector += force_field_vector

                    if not collision_count_updated:
                        self.collision_queue.append(1)
                        self.agent_statistics.collision_danger.append(1)
                        self.collision_count += 1
                        collision_count_updated = True
                        self.per_task_collision_avoidance_count[self.current_task] += 1

        if not collision_count_updated:
            self.agent_statistics.collision_danger.append(0)
            self.collision_queue.append(0)

        ca_contribution = sum(np.sqrt(total_ff_vector ** 2))
        pf_contribution = 1.0
        self.collision_avoidance_contribution_queue.append(ca_contribution)
        self.path_finding_contribution_queue.append(pf_contribution)

        self.agent_statistics.collision_avoidance_contribution.append(ca_contribution)

        if sum(np.sqrt(current_direction ** 2)) > 0:
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

    def collision_potential_visible(self, other, view_above=False):
        # check whether other agent is within view, i.e. below this agent or in view of one of the cameras
        # get list of agents for which there is collision potential/danger
        self_corner_points = self.geometry.corner_points_2d()
        self_x = [p[0] for p in self_corner_points]
        self_y = [p[1] for p in self_corner_points]
        self_min_x = min(self_x)
        self_max_x = min(self_x)
        self_min_y = min(self_y)
        self_max_y = min(self_y)
        if self is not other:
            # the following checks whether the other quadcopter is below ourselves (in which case the assumption
            # is that, due to downward-facing cameras, it is visible) or if it is above use but visible, which
            # is fairly unrealistic but hopefully good enough
            # this check assumes that the quadcopters don't rotate (which may not be desirable)
            other_corner_points = other.geometry.corner_points_2d()
            other_x = [p[0] for p in other_corner_points]
            other_y = [p[1] for p in other_corner_points]
            other_min_x = min(other_x)
            other_max_x = min(other_x)
            other_min_y = min(other_y)
            other_max_y = min(other_y)
            if view_above or other.geometry.position[2] <= self.geometry.position[2] \
                    or (other_min_x >= self_max_x and other_min_y >= self_max_y) \
                    or (other_min_x >= self_max_x and other_max_y <= self_min_y) \
                    or (other_max_x <= self_min_x and other_min_y >= self_max_y) \
                    or (other_max_x <= self_min_x and other_max_y <= self_min_y):
                return True
        return False

    def count_in_direction(self,
                           environment: env.map.Map,
                           directions=None,
                           angle=np.pi / 4,
                           max_distance=500,
                           max_vertical_distance=200):
        # should this method only return the positions of "reasonably" visible agents or should it do more
        # and e.g. just give directions (angles) and distances?

        # maybe a good place to start would be to check for each direction whether agents are within x distance
        if directions is None:
            directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        agents_counts = [0] * len(directions)
        for a in environment.agents:
            if a is not self and self.collision_potential_visible(a) \
                    and a.geometry.position[2] <= self.geometry.position[2] + max_vertical_distance \
                    and simple_distance(self.geometry.position[:2], a.geometry.position[:2]) < max_distance \
                    and environment.offset_origin[0] - a.geometry.size[0] <= a.geometry.position[0] <= \
                    Block.SIZE * self.target_map.shape[2] + environment.offset_origin[0] + a.geometry.size[0] \
                    and environment.offset_origin[1] - a.geometry.size[1] <= a.geometry.position[1] <= \
                    Block.SIZE * self.target_map.shape[1] + environment.offset_origin[1] + a.geometry.size[1]:
                # should probably also check whether agent is (likely to be) over structure/grid at all
                for d_idx, d in enumerate(directions):
                    difference = a.geometry.position[:2] - self.geometry.position[:2]
                    dot_product = d[0] * difference[0] + d[1] * difference[1]
                    point_on_line = self.geometry.position[:2] + dot_product * d[:2]
                    # if simple_distance(point_on_line, a.geometry.position[:2]) < 40:
                    # in that case the agent is close enough to be counted as possibly interfering
                    # agents_counts[d_idx] = agents_counts[d_idx] + 1
                    # probably better to do this with angles instead
                    position_difference = a.geometry.position - self.geometry.position
                    position_signed_angle = np.arctan2(position_difference[1], position_difference[0]) - \
                                            np.arctan2(d[1], d[0])
                    # self.aprint("Direction {} signed angle to agent {}: {}".format(d[:2], a.id, np.rad2deg(position_signed_angle)))
                    if abs(position_signed_angle) < angle:
                        agents_counts[d_idx] = agents_counts[d_idx] + 1

        return tuple(agents_counts)

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
        # any continuous row/column in the target map between two occupied locations in the local occupancy
        # map should be assumed to be filled out already
        current_occupancy_map = self.local_occupancy_map[self.current_grid_position[2]]
        current_target_map = self.target_map[self.current_grid_position[2]]
        for y in range(current_target_map.shape[0]):
            for x in range(current_target_map.shape[1]):
                if current_occupancy_map[y, x] != 0:
                    # counter = 0
                    # for diff in (-1, 1):
                    #     y2 = y + diff
                    #     if 0 <= y2 < self.target_map.shape[1]:
                    #         if self.local_occupancy_map[self.current_grid_position[2], y2, x] != 0:
                    #             counter += 1
                    #     x2 = x + diff
                    #     if 0 <= x2 < self.target_map.shape[2]:
                    #         if self.local_occupancy_map[self.current_grid_position[2], y, x2] != 0:
                    #             counter += 1
                    # if counter >= 3:
                    #     self.local_occupancy_map[self.current_grid_position[2], y, x] = 1

                    for diff in (-1, 1):
                        # making it through this loop without a break means that in the x-row, y-column where the block
                        # could be placed, there is either only a block immediately adjacent or any blocks already
                        # placed are separated from the current site by an intended gap

                        counter = 1
                        while 0 <= y + counter * diff < current_occupancy_map.shape[0] \
                                and current_occupancy_map[y + counter * diff, x] == 0 \
                                and current_target_map[y + counter * diff, x] > 0:
                            counter += 1
                        if counter > 1 and 0 <= y + counter * diff < current_occupancy_map.shape[0] \
                                and current_occupancy_map[y + counter * diff, x] > 0 and \
                                current_target_map[y + counter * diff, x] > 0:
                            # have encountered a block in this column, mark all in between
                            # this position and the end of that column as occupied
                            other_y = y + counter * diff
                            if other_y > y:
                                self.local_occupancy_map[self.current_grid_position[2], y:other_y, x] = 1
                            else:
                                self.local_occupancy_map[self.current_grid_position[2], other_y:y, x] = 1

                        counter = 1
                        while 0 <= x + counter * diff < current_occupancy_map.shape[1] \
                                and current_occupancy_map[y, x + counter * diff] == 0 \
                                and current_target_map[y, x + counter * diff] > 0:
                            counter += 1
                        if counter > 1 and 0 <= x + counter * diff < current_occupancy_map.shape[1] \
                                and current_occupancy_map[y, x + counter * diff] > 0 and \
                                current_target_map[y, x + counter * diff] > 0:
                            # have encountered a block in this row, mark all in between
                            # this position and the end of that row as occupied
                            other_x = x + counter * diff
                            if other_x > x:
                                self.local_occupancy_map[self.current_grid_position[2], y, x:other_x] = 1
                            else:
                                self.local_occupancy_map[self.current_grid_position[2], y, other_x:x] = 1

        # for y in range(3, 7):
        #     if self.local_occupancy_map[0, y, 0] != 0 and environment.occupancy_map[0, y, 0] == 0:
        #         self.aprint("Local occupancy map occupied at {} where environment not occupied."
        #                .format((0, y, 0)))
        #         self.aprint("Current position: {}".format(self.current_grid_position))
        #         self.aprint("Local occupancy map at this level:\n{}".format(self.local_occupancy_map[0]))
        #         self.aprint("Global occupancy map at this level:\n{}".format(environment.occupancy_map[0]))
        #         print()
        #         break

        for z in range(self.local_occupancy_map.shape[0]):
            for y in range(self.local_occupancy_map.shape[1]):
                for x in range(self.local_occupancy_map.shape[2]):
                    if self.local_occupancy_map[z, y, x] != 0 and environment.occupancy_map[z, y, x] == 0:
                        self.aprint("Local occupancy map occupied at {} where environment not occupied."
                               .format((x, y, z)))
                        self.aprint("Current position: {}".format(self.current_grid_position))
                        self.aprint("Local occupancy map at this level:\n{}".format(self.local_occupancy_map[z]))
                        self.aprint("Global occupancy map at this level:\n{}".format(environment.occupancy_map[z]))
                        self.aprint("")
                        break

    def check_component_finished(self, compared_map: np.ndarray, component_marker=None):
        if component_marker is None:
            component_marker = self.current_component_marker
        tm = np.zeros_like(self.target_map[self.current_structure_level])
        np.place(tm, self.component_target_map[self.current_structure_level] == component_marker, 1)
        om = np.copy(compared_map[self.current_structure_level])
        np.place(om, om > 0, 1)
        np.place(om, self.component_target_map[self.current_structure_level] != component_marker, 0)
        # print("CHECKING COMPONENT {} FINISHED".format(component_marker))
        # print_map(tm)
        # print_map(om)
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

    def component_seed_location(self, component_marker, level=None, include_closing_corners=False):
        # if component_marker == 2:
        #     location = np.where(self.target_map == 2)
        #     return location[2][0], location[1][0], location[0][0]

        if level is None:
            level = self.current_structure_level
            locations = np.where(self.component_target_map == component_marker)
            level = locations[0][0]

        # according to whichever strategy is currently being employed for placing the seed, return that location
        # this location is the grid location, not the absolute spatial position
        occupied_locations = np.where(self.component_target_map[level] == component_marker)
        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
        supported_locations = np.nonzero(self.target_map[level - 1])
        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
        occupied_locations = [x for x in occupied_locations if x in supported_locations]
        if not include_closing_corners:
            occupied_locations = [x for x in occupied_locations if (x[1], x[0], level)
                                  not in self.closing_corners[level][self.current_component_marker]]

        min_y = min([l[0] for l in occupied_locations])
        max_y = max([l[0] for l in occupied_locations])
        min_x = min([l[1] for l in occupied_locations])
        max_x = max([l[1] for l in occupied_locations])
        center_point = np.array([max_y + min_y, max_x + min_x]) / 2
        min_distance = float("inf")
        min_location = None
        for l in occupied_locations:
            temp = simple_distance(center_point, l)
            if temp < min_distance:
                min_distance = temp
                min_location = l
        seed_location = [min_location[1], min_location[0], level]

        return seed_location

    def shortest_direction_to_perimeter(self,
                                        compared_map:np.ndarray,
                                        start_position,
                                        level=None,
                                        component_marker=None):
        if level is None:
            level = self.current_structure_level

        if component_marker is None:
            component_marker = self.current_component_marker

        directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        start_position = np.array(start_position)
        min_distance = 0
        min_direction = None
        for d in directions:
            current_position = start_position.copy()
            current_distance = 0
            done = False
            while not done:
                current_position += d
                if 0 <= current_position[0] < compared_map.shape[1] \
                        and 0 <= current_position[1] < compared_map.shape[0]:
                    # check whether on outside of current component's perimeter, i.e. not in a hole
                    current_distance += 1
                else:
                    done = True
            if current_distance > min_distance:
                min_distance = current_distance
                min_direction = d
            elif current_distance == min_distance:
                min_direction = random.sample([min_direction, d], 1)[0]
        return np.array([min_direction[0], min_direction[1], 0])

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
                # self.aprint("CORNER ALREADY SURROUNDED BY ADJACENT BLOCKS")
                loop_corner_attachable = True
            else:
                # self.aprint("CORNER NOT SURROUNDED BY ADJACENT BLOCKS YET")
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

    def merge_multi_layer_components(self):
        # merge components if they are only connected to each other and one other component on a different layer
        # (i.e. the top and bottom most layers get special treatment for not being connected on both sides)

        # that means any component which is connected to at least two other components
        # on one of the adjacent layers can automatically be counted out

        # go layer-by-layer: start with all components on layer 0 and then go up and check whether any of the
        # components on layer 1 should be added to those identified from layer 0 or if they should become the new
        # components to compare to (?)
        multi_layer_component_markers = list(np.unique(self.component_target_map[0]))
        if 0 in multi_layer_component_markers:
            multi_layer_component_markers.remove(0)
        for layer in range(1, self.target_map.shape[0]):
            current_components = list(np.unique(self.component_target_map[layer]))
            if 0 in current_components:
                current_components.remove(0)
            # check for the existing components whether it works?

        # repeat the stuff below until identifying that no more components can be merged
        # for all components:
        #     go through all other components and check whether they are adjacent (vertically)
        #         if they are, check whether components can be merged or not
        #         this means that they are both only connected to one other component (?)
        # TODO: maybe ignore components that are "alone" on a layer because it wouldn't change anything?
        # TODO: make this a hierarchical (think initial example) -> at least good suggestion for future work
        # TODO: need to check the distance to adjacent components
        # if it is too small, then the multi-layer component thing shouldn't be attempted because it will hinder
        # construction otherwise; might be better to do this check "online" though instead of putting it into
        # the ml_component_map (?)
        ml_component_markers = [x for x in list(np.unique(self.component_target_map)) if x != 0]
        mergeable_pairs = []
        for m1 in ml_component_markers:
            # go through all other components and check whether they are adjacent
            m1_locations = np.where(self.component_target_map == m1)
            m1_height = m1_locations[0]
            m1_min_height = np.min(m1_height)
            m1_max_height = np.max(m1_height)
            m1_adjacent_components = []
            for z, y, x in tuple(zip(m1_locations[0], m1_locations[1], m1_locations[2])):
                for z2 in (z - 1, z + 1):
                    if 0 <= z2 < self.component_target_map.shape[0] \
                            and self.component_target_map[z2, y, x] not in (0, m1) \
                            and self.component_target_map[z2, y, x] not in m1_adjacent_components:
                        m1_adjacent_components.append(self.component_target_map[z2, y, x])
            for m2 in ml_component_markers:
                if m2 != m1 and (m1, m2) not in mergeable_pairs and (m2, m1) not in mergeable_pairs:
                    m2_locations = np.where(self.component_target_map == m2)
                    m2_height = m2_locations[0]
                    m2_min_height = np.min(m2_height)
                    m2_max_height = np.max(m2_height)
                    m2_adjacent_components = []
                    for z, y, x in tuple(zip(m2_locations[0], m2_locations[1], m2_locations[2])):
                        for z2 in (z - 1, z + 1):
                            if 0 <= z2 < self.component_target_map.shape[0] \
                                    and self.component_target_map[z2, y, x] not in (0, m2) \
                                    and self.component_target_map[z2, y, x] not in m2_adjacent_components:
                                m2_adjacent_components.append(self.component_target_map[z2, y, x])

                    # check if the components are adjacent
                    if abs(m2_min_height - m1_max_height) == 1 or abs(m1_min_height - m2_max_height) == 1:
                        # check if the components are each only connected to one other component (or fewer)
                        # and if these components which they are connected to are on the "other" side of the
                        # respective component (not on the same level as m1 or m2)
                        m1_other = [x for x in m1_adjacent_components if x != m2]
                        m2_other = [x for x in m2_adjacent_components if x != m1]
                        if len(m1_other) > 1 or len(m2_other) > 1:
                            continue
                        m1_clear = True
                        if len(m1_other) == 1:
                            # check whether that other component is on the same side as m2
                            other_height = np.where(self.component_target_map == m1_other[0])[0]
                            if any([oh in m2_height for oh in other_height]):
                                m1_clear = False

                        m2_clear = True
                        if len(m2_other) == 1:
                            # check whether that other component is on the same side as m1
                            other_height = np.where(self.component_target_map == m2_other[0])[0]
                            if any([oh in m1_height for oh in other_height]):
                                m2_clear = False

                        if m1_clear and m2_clear:
                            # print("Component {} and {} can be merged.".format(m1, m2))
                            mergeable_pairs.append((m1, m2))
        self.aprint("Mergeable pairs:\n{}".format(mergeable_pairs))

        # merge all components which share one component in the mergeable pairs list
        # first group all those pairs which have share components
        mergeable_groups = []
        for pair in mergeable_pairs:
            # try to find an entry in mergeable_groups that pair would belong to
            found = False
            for group in mergeable_groups:
                if found:
                    break
                for other_pair in group:
                    if pair[0] in other_pair or pair[1] in other_pair:
                        found = True
                        group.append(pair)
                        break
            if not found:
                mergeable_groups.append([pair])

        backup = []
        all_ml_components = []
        for group in mergeable_groups:
            # for all of the components in the group, check whether they have a dangerous distance
            # to any other part of the structure (doesn't matter which), so that this multi-layered component
            # might have to be discarded
            # actually, since one might still be able to build a subpart of the multi-layered component in this way
            # it would probably be best to check whether that's possible too (i.e. identify the components which
            # make things problematic and then remove them if they are on the "top" or "bottom" of the ml component)
            unique_values = []
            for m1, m2 in group:
                if m1 not in unique_values:
                    unique_values.append(m1)
                if m2 not in unique_values:
                    unique_values.append(m2)
            backup.append(unique_values)
            all_ml_components.extend(unique_values)
        mergeable_groups = backup

        self.aprint("Mergeable groups:\n{}".format(mergeable_groups))

        ml_index = 2
        ml_component_map = np.zeros_like(self.component_target_map)
        non_ml_components = [x for x in ml_component_markers if x not in all_ml_components]
        for group in mergeable_groups:
            while ml_index in non_ml_components:
                ml_index += 1
            for m in group:
                np.place(ml_component_map, self.component_target_map == m, ml_index)
            ml_index += 1
        for m in non_ml_components:
            np.place(ml_component_map, self.component_target_map == m, m)

        return ml_component_map

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
        removable_holes = {}
        for z in range(hole_map.shape[0]):
            hole_boundaries.append([])
            for m in valid_markers[z]:
                coord_list = []
                locations = np.where(hole_map == m)
                boundary_search(hole_map[z], z, locations[1][0], locations[2][0], m, None, coord_list)
                first_component_marker = self.component_target_map[z, coord_list[0][1], coord_list[0][2]]
                for _, y, x in coord_list:
                    current_marker = self.component_target_map[z, y, x]
                    if current_marker != first_component_marker:
                        if (z, m) not in removable_holes.keys():
                            removable_holes[(z, m)] = [first_component_marker]
                        if current_marker not in removable_holes[(z, m)]:
                            removable_holes[(z, m)].append(current_marker)
                coord_list = tuple(np.moveaxis(np.array(coord_list), -1, 0))
                hole_boundaries[z].append(coord_list)
                # should already only include components in hole boundaries which are actually AROUND the hole
                # i.e. the hole does not enclose that component

        # a) the hole having to be removed because it is between components
        # b) the hole encircling a component, but still being a valid hole
        #    -> in this case it's important not to select a closing corner from any enclosed component

        # remove those holes between components
        # for each hole, if any adjacent component(s) only have sites adjacent that are "themselves" or that hole,
        # then they are enclosed by the hole and the hole itself should still count as such?

        # actually, if the component is in a hole between components, this does not hold, therefore the above has
        # to be true for all components except for one (which is the enclosing component)

        enclosed_components = {}
        # for every hole that is adjacent to two (or more) components:
        for z, m in removable_holes:
            # decide whether the hole itself should be removed because it is only between components or
            # whether the enclosed adjacent component(s) should be recorded to be avoided for closing corners
            local_enclosed_components = []
            adjacent_components = removable_holes[(z, m)]
            hole_positions = np.where(hole_map == m)
            hole_min_x = np.min(hole_positions[2])
            hole_max_x = np.max(hole_positions[2])
            hole_min_y = np.min(hole_positions[1])
            hole_max_y = np.max(hole_positions[1])
            # print("HOLE {} BOUNDARIES\nmin_x = {}, max_x = {}, min_y = {}, max_y = {}"
            #       .format(m, hole_min_x, hole_max_x, hole_min_y, hole_max_y))
            for ac in adjacent_components:
                component_positions = np.where(self.component_target_map == ac)
                component_min_x = np.min(component_positions[2])
                component_max_x = np.max(component_positions[2])
                component_min_y = np.min(component_positions[1])
                component_max_y = np.max(component_positions[1])
                # print("COMPONENT {} BOUNDARIES\nmin_x = {}, max_x = {}, min_y = {}, max_y = {}"
                #       .format(ac, component_min_x, component_max_x, component_min_y, component_max_y))
                # print(component_positions)
                if hole_min_x <= component_min_x <= hole_max_x \
                        and hole_min_x <= component_max_x <= hole_max_x \
                        and hole_min_y <= component_min_y <= hole_max_y \
                        and hole_min_y <= component_max_y <= hole_max_y:
                    # the component is (hopefully) enclosed by the hole
                    local_enclosed_components.append(ac)
            if len(local_enclosed_components) <= len(adjacent_components) - 2:
                self.aprint("HOLE {} IS BETWEEN COMPONENTS AND THEREFORE REMOVABLE.".format(m))
                index = valid_markers[z].index(m)
                np.place(hole_map[z], hole_map[z] == m, 0)
                del valid_markers[z][index]
                del hole_boundaries[z][index]
            else:
                self.aprint("ADJACENT COMPONENTS ENCLOSED BY HOLE {}:\n{}".format(m, local_enclosed_components))
                enclosed_components[(z, m)] = local_enclosed_components

        # self.aprint("COMPONENT MAP:\n{}".format(self.component_target_map))
        # self.aprint("ENCLOSED COMPONENTS: {}".format(enclosed_components))

        for z, m in enclosed_components:
            index = valid_markers[z].index(m)
            temp = np.zeros_like(self.target_map)
            temp[hole_boundaries[z][index]] = 1
            for c in enclosed_components[(z, m)]:
                temp[self.component_target_map == c] = 0
            hole_boundaries[z][index] = np.where(temp == 1)

        a_dummy_copy = np.copy(hole_map)
        hole_corners = []
        closing_corners = []
        closing_corner_boundaries = []
        closing_corner_orientations = []
        hole_boundary_coords = dict()
        for z in range(hole_map.shape[0]):
            hole_corners.append([])
            closing_corners.append({})
            closing_corner_boundaries.append({})
            closing_corner_orientations.append({})
            for cm in np.unique(self.component_target_map):
                # for each component marker, make new entry in closing_corners dictionary
                closing_corners[z][cm] = []
                closing_corner_boundaries[z][cm] = []
                closing_corner_orientations[z][cm] = []
            # this list is used to keep track of which (closing) corners (and boundaries) belong to which components
            for m_idx, m in enumerate(valid_markers[z]):
                # find corners as coordinates that are not equal to the marker and adjacent to
                # two of the boundary coordinates (might only want to look for outside corners though)
                boundary_coord_tuple_list = list(zip(
                    hole_boundaries[z][m_idx][2], hole_boundaries[z][m_idx][1], hole_boundaries[z][m_idx][0]))

                # there should only be one adjacent component, so this SHOULD work
                adjacent_components = np.unique(self.component_target_map[hole_boundaries[z][m_idx]])
                if len(adjacent_components) > 1:
                    self.logger.error("More than 1 adjacent component to hole {}: {}".format(m, adjacent_components))

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

                # split up the corners and boundaries into the different components
                for cm in np.unique(component_marker_list_outer):
                    current_outer = [outer_corner_coord_list[i] for i in range(len(outer_corner_coord_list))
                                     if component_marker_list_outer[i] == cm]
                    current_inner = [inner_corner_coord_list[i] for i in range(len(inner_corner_coord_list))
                                     if component_marker_list_inner[i] == cm]
                    current_boundary = [corner_boundary_list[i] for i in range(len(corner_boundary_list))
                                        if component_marker_list_outer[i] == cm]

                    # now the seed location for that component can be determined, meaning that
                    # it should also be possible to determine the correct closing corner

                    # need to check whether the component contains the original seed
                    if cm in self.component_target_map[self.target_map == 2]:
                        # in this case, the original seed is in this component and needs to be chosen as the position
                        position = np.where(self.target_map == 2)
                        position = tuple(zip(position[2], position[1], position[0]))
                        if len(position) > 1:
                            self.logger.warning("Too many seeds when creating hole map.")
                        adjacent_seed_location = position[0]
                    else:
                        adjacent_seed_location = self.component_seed_location(cm, include_closing_corners=True)

                    # self.aprint("\nSEED LOCATION FOR HOLE {} IN COMPONENT {}: {}".format(m, cm, adjacent_seed_location))

                    # determine the location of the hole with respect to the seed (NW, NE, SW, SE)
                    # does this depend on the min/max? I think it can, but doesnt have to
                    # or rather: it depends on the corners?
                    # -> probably best to take whichever is not in the same x or y position and is the furthest away

                    # maybe do this:
                    # check whether hole is completely on some side of the seed (both in horizontal and vertical
                    # direction) and pick at least the one where it's clear
                    # for the other direction(s) (I think) any (extreme) closing corner should be fine, but picking
                    # the one furthest away would probably not be a bad idea because more building can get done
                    # between the seed and that corner?
                    # -> could actually use furthest closing corner in any case -> might be a good idea

                    # actually, there might be situations where the choice matters a lot, e.g. if a closing corner
                    # (or the closing corner region) overlaps with a different hole

                    current_sp_lengths = []
                    for corner in current_outer:
                        temp = shortest_path(
                            self.target_map[corner[2]], tuple(adjacent_seed_location[:2]), tuple(corner[:2]))
                        current_sp_lengths.append(len(temp))
                    # self.aprint("SHORTEST PATH LENGTHS FOR HOLE {} IN COMPONENT {}:\n{}"
                    #             .format(m, cm, current_sp_lengths))
                    ordered_idx = sorted(range(len(current_sp_lengths)), key=lambda i: current_sp_lengths[i])
                    ordered_outer = [current_outer[i] for i in ordered_idx]
                    ordered_boundary = [current_boundary[i] for i in ordered_idx]
                    # self.aprint("ORDERED POSSIBLE CORNERS:\n{}\n{}"
                    #             .format([current_sp_lengths[i] for i in ordered_idx], ordered_outer))

                    # other possibility: choose the corner that excludes the smallest area?

                    # regardless of what method is used to determine the corners, set the closing corner here
                    closing_corner = ordered_outer[-1]
                    closing_corner_boundary = ordered_boundary[-1]

                    hole_positions = np.where(hole_map == m)
                    hole_min_x = np.min(hole_positions[2])
                    hole_max_x = np.max(hole_positions[2])
                    hole_min_y = np.min(hole_positions[1])
                    hole_max_y = np.max(hole_positions[1])
                    orientation = []
                    if hole_max_x < adjacent_seed_location[0]:
                        orientation.append("W")
                    if hole_min_x > adjacent_seed_location[0]:
                        orientation.append("E")
                    if hole_min_y < adjacent_seed_location[1]:
                        orientation.append("S")
                    if hole_max_y > adjacent_seed_location[1]:
                        orientation.append("N")

                    indices = range(len(current_outer))
                    if len(orientation) > 0:
                        # sorted by y
                        if "S" in orientation:
                            indices = sorted(indices, key=lambda e: current_outer[e][1])
                        elif "N" in orientation:
                            indices = sorted(indices, key=lambda e: current_outer[e][1], reverse=True)
                        # sort by x
                        if "W" in orientation:
                            indices = sorted(indices, key=lambda e: current_outer[e][0])
                        elif "E" in orientation:
                            indices = sorted(indices, key=lambda e: current_outer[e][0], reverse=True)
                        closing_corner = current_outer[indices[0]]
                        closing_corner_boundary = current_boundary[indices[0]]

                    # determine the orientation of the corner with respect to the hole
                    x = closing_corner[0]
                    y = closing_corner[1]
                    corner_direction = None
                    for x2, y2 in ((x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)):
                        if 0 <= x2 < hole_map.shape[1] and 0 <= y2 < hole_map.shape[2] and hole_map[z, y2, x2] == m:
                            if x2 < x:
                                if y2 < y:
                                    corner_direction = "NE"
                                else:
                                    corner_direction = "SE"
                            else:
                                if y2 < y:
                                    corner_direction = "NW"
                                else:
                                    corner_direction = "SW"

                    # TODO: important note (see image hole_problem in Images folder as well)
                    # => there might actually be some cases in which it is not possible to do use the corner/region
                    #    technique to guarantee correct construction -> should include that in the thesis (at least
                    #    if I find cases where this is indeed the case)

                    if (len(current_outer) + len(current_inner)) % 2 == 0:
                        # THE BLOCK BELOW IS THE OLD METHOD OF DOING THINGS AND IT WORKED ON RESTRICTED STRUCTURES
                        # sorted_by_y = sorted(range(len(current_outer)),
                        #                      key=lambda e: current_outer[e][1], reverse=True)
                        # sorted_by_x = sorted(sorted_by_y, key=lambda e: current_outer[e][0], reverse=True)
                        # current_outer = [current_outer[i] for i in sorted_by_x]
                        # current_boundary = [current_boundary[i] for i in sorted_by_x]
                        # closing_corners[z][cm].append(current_outer[0])
                        # closing_corner_boundaries[z][cm].append(current_boundary[0])
                        # closing_corner_orientations[z][cm].append("NE")

                        # THIS BLOCK IS THE NEW METHOD
                        closing_corners[z][cm].append(closing_corner)
                        closing_corner_boundaries[z][cm].append(closing_corner_boundary)
                        closing_corner_orientations[z][cm].append(corner_direction)
                    else:
                        self.logger.warning("The structure contains open corners, "
                                            "which cannot be built using the PerimeterFollowingAgent.")

                    # need to actually register open corners as such, I think, because otherwise a corner might e.g.
                    # be chosen as a closing corner even though it is in the SOUTH-EAST of the hole, which given the
                    # attachment and hole rules that we have could result in trouble

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

        # self.aprint("\nCLOSING CORNERS:\n{}".format(closing_corners))
        # self.aprint("CLOSING CORNER ORIENTATIONS:\n{}".format(closing_corner_orientations))

        return closing_corners, hole_map, hole_boundary_coords, closing_corner_boundaries, closing_corner_orientations

    def aprint(self, *args, print_as_map=False, override_global_printing_enabled=False, **kwargs):
        if self.printing_enabled or override_global_printing_enabled:
            if print_as_map:
                print("[Agent {}]: ".format(self.id))
                print_map(*args, **kwargs)
            else:
                print("[Agent {}]: ".format(self.id), end="")
                print(*args, **kwargs)

