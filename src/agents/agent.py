import numpy as np
import random
from typing import List
from abc import ABCMeta, abstractmethod
from collections import deque as dq
from enum import Enum

import env.map
from env import Block, print_map, shortest_path
from geom import *

np.seterr(divide='ignore', invalid='ignore')


class Task(Enum):
    """
    A enumeration specifying different tasks which the agents perform during construction.
    This could be extended if other tasks are required (e.g. for planned collision avoidance).
    """

    FETCH_BLOCK = 0
    PICK_UP_BLOCK = 2
    TRANSPORT_BLOCK = 3
    MOVE_TO_PERIMETER = 4
    FIND_ATTACHMENT_SITE = 5
    PLACE_BLOCK = 6
    FIND_NEXT_COMPONENT = 7
    SURVEY_COMPONENT = 8
    RETURN_BLOCK = 9
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
            self.previous_task = self.agent.current_task
        self.task_counter[self.agent.current_task] = self.task_counter[self.agent.current_task] + 1


def check_map(map_to_check, position, comparator=lambda x: x != 0):
    """
    Check whether the specified condition holds at the given position on the given map.

    :param map_to_check: the occupancy matrix to check for the condition
    :param position: the grid position to check the condition at
    :param comparator: an expression evaluating to True or False which is applied to the entry at the position
    :return: True if the condition holds, False otherwise
    """

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
    """
    A super class representing an agent encapsulating all the important information and functionality that is
    expected to be used by all or most agent types that can be implemented based on it.
    """

    __metaclass__ = ABCMeta

    MOVEMENT_PER_STEP = 5
    AGENT_ID = 0

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float,
                 printing_enabled=True):
        self.printing_enabled = printing_enabled

        self.geometry = Geometry(position, size, 0.0)
        self.collision_avoidance_geometry = Geometry(position,
                                                     [size[0] + required_spacing * 2,
                                                     size[1] + required_spacing * 2,
                                                     size[2] + required_spacing * 2], 0.0)
        self.collision_avoidance_geometry_with_block = Geometry([position[0], position[1],
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
        self.avoiding_crowded_stashes_enabled = True
        self.transport_avoid_others_enabled = True

        self.order_only_one_metric = False

        # performance metrics to keep track of
        self.step_count = 0
        self.stuck_count = 0
        self.seeded_blocks = 0
        self.attached_blocks = 0
        self.returned_blocks = 0
        self.per_task_step_count = dict([(task, 0) for task in Task])
        self.per_task_collision_avoidance_count = dict([(task, 0) for task in Task])
        self.per_task_distance_travelled = dict([(task, 0) for task in Task])
        self.attachment_frequency_count = []
        self.components_seeded = []
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
        # store tuples declaring whether it ended in finding an attachment site,
        # moving to a different component and finding one there or whether it was returned
        # start counting (i.e. resetting to 0) when attaching a block
        self.sp_search_count = []
        self.current_sp_search_count = 0

        # components visited/actually considered as targets until some block attached at the final choice,
        # can again be reset when block is placed
        self.next_component_count = []
        self.current_next_component_count = 0

        # time to go from all agents trying to seed to all trying to attach (global vs local)
        # -> should be at level of main loop though: count from end of old layer to all agents having attached once?
        #    probably good if they attached anything, the point would be that they actually figured out their component

        # steps per layer/component (with normal attachment and seeding)
        # what if trying to seed/attach to component and then switching to other?
        # -> count for first until "realisation", then switch
        self.steps_per_layer = {}
        self.steps_per_component = {}

        # per attachment (with both algorithms), count total blocks travelled per attachment/decision to return
        # separated in components: start counter when attachment site stuff is called and keep counting
        self.blocks_per_attachment = []
        self.current_blocks_per_attachment = 0
        self.steps_per_attachment = []
        self.current_steps_per_attachment = 0

        # time difference between component being completed (can be checked globally) and them actually switching to
        # different component -> yay, this one should be easy-ish
        self.complete_to_switch_delay = {}
        self.current_component_switch_marker = -1

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

        self.task_history = []
        self.task_history.append(self.current_task)
        self.id = Agent.AGENT_ID
        Agent.AGENT_ID += 1
        self.collision_possible = True

        self.agent_statistics = AgentStatistics(self)

        self.component_target_map = self.split_into_components()
        # self.multi_layer_component_target_map = self.merge_multi_layer_components()
        self.closing_corners, self.hole_map, self.hole_boundaries, self.closing_corner_boundaries, \
            self.closing_corner_orientations = self.find_closing_corners()

    @abstractmethod
    def advance(self, environment: env.map.Map):
        """
        Abstract method to be overridden by subclasses.

        :param environment: the environment the agent operates in
        """

        pass

    def overlaps(self, other):
        """
        Check whether this agent's geometry and the other agent's geometry overlap.

        :param other: the other agent to check against
        :return: True if the geometries overlap, False otherwise
        """

        return self.geometry.overlaps(other.geometry)

    def move(self, environment: env.map.Map, react_only=False):
        """
        Calculate the direction for the next step to take on the current path, possibly avoiding collisions.

        This method uses the current path to determine the next way point to move to and calculate a vector
        of movement in that direction. If there are other agents which are in risk of colliding with this agent
        (and which are visible to it), a force vector pointing in the opposite direction of the vector between
        the two agents is added to the vector of movement (scaled by their distance). The summed vector is then
        normalised and its length extended to match the maximum range of movement per step for an agent.

        :param environment: the environment in which the agent is operating
        :param react_only: determines whether the agent will follow the current path or just try to stay
        at the current position
        :return: the next position on the current path and a the determined vector of movement
        """

        self.per_task_step_count[self.current_task] += 1

        if not react_only:
            next_position = self.current_path.next()
            current_direction = self.current_path.direction_to_next(self.geometry.position)
        else:
            next_position = self.current_static_location
            current_direction = np.array([0.0, 0.0, 0.0])

        # normalising the direction vector first, to be able to reason about the size of the force vector more easily
        if sum(np.sqrt(current_direction ** 2)) > 0:
            current_direction /= sum(np.sqrt(current_direction ** 2))

        collision_count_updated = False
        total_ff_vector = np.array([0.0, 0.0, 0.0])
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a) and self.collision_potential_visible(a, react_only):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))

                    # the coefficients were determined experimentally, others might be better
                    if not react_only:
                        force_field_vector *= 100 / simple_distance(self.geometry.position, a.geometry.position)
                    else:
                        force_field_vector *= 200 / simple_distance(self.geometry.position, a.geometry.position)
                        force_field_vector[2] = 0

                    current_direction += force_field_vector
                    total_ff_vector += force_field_vector

                    if not collision_count_updated:
                        self.collision_queue.append(1)
                        self.agent_statistics.collision_danger.append(1)
                        self.collision_count += 1
                        self.per_task_collision_avoidance_count[self.current_task] += 1
                        collision_count_updated = True

        if not collision_count_updated:
            self.agent_statistics.collision_danger.append(0)
            self.collision_queue.append(0)

        ca_contribution = sum(np.sqrt(total_ff_vector ** 2))
        pf_contribution = 1.0
        self.collision_avoidance_contribution_queue.append(ca_contribution)
        self.path_finding_contribution_queue.append(pf_contribution)

        self.agent_statistics.collision_avoidance_contribution.append(ca_contribution)

        # normalising and scaling the vector
        if sum(np.sqrt(current_direction ** 2)) > 0:
            current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if any(np.isnan(current_direction)):
            current_direction = np.array([0, 0, 0])

        return next_position, current_direction

    def land(self, environment: env.map.Map):
        """
        Move with the goal of landing.

        This method is called if the current task is LAND. If the agent has not planned a path yet, it constructs a
        path to its original position in the environment and then proceeds to land there. If the agent does not land
        because it is leaving the swarm and planning to rejoin it later, the task changes to FINISHED and the agent
        does not take part in construction any more. It may be desirable to choose the place to land differently
        (e.g. outside of the area of movement of the other agents), but this has not been implemented.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            land_x = self.initial_position[0]
            land_y = self.initial_position[1]
            land_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + self.required_spacing

            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], land_z])
            self.current_path.add_position([land_x, land_y, land_z])
            self.current_path.add_position([land_x, land_y, self.geometry.size[2] / 2])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                if abs(self.geometry.position[2] - Block.SIZE / 2) > Block.SIZE / 2:
                    self.aprint("Error: finished without landing.")
                if self.current_block is not None:
                    self.aprint("Error: finished with block still attached.")
                if self.drop_out_of_swarm:
                    self.wait_for_rejoining = True
                    self.current_task = Task.REJOIN_SWARM
                else:
                    self.current_task = Task.FINISHED
                self.task_history.append(self.current_task)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.LAND] += simple_distance(position_before, self.geometry.position)

    def collision_potential(self, other):
        """
        Check whether the other agent is at risk of colliding with this agent.

        This method could be adapted to initiate collision avoidance differently than it is currently done.

        :param other: the other agent to check for a risk of collision with
        :return: True if there is a risk of colliding soon, False if there is not
        """

        # agents which have already landed (and are considered to be "taken out of" the simulation) are ignored
        if self.current_task == Task.FINISHED or other.current_task == Task.FINISHED:
            return False

        # make a simple distance-based decision (a vector is added because the agent might have a block attached)
        if simple_distance(self.geometry.position + np.array([0, 0, -self.geometry.size[2]]), other.geometry.position) \
                < self.required_distance:
            return True

        return False

    def collision_potential_visible(self, other, view_above=False):
        """
        Check whether the other agent/quadcopter is visible to potentially initiate collision avoidance.

        This method could be overridden to allow for different sensing capabilities of the quadcopters.
        Currently it is assumed that agents all around can be recognised, except if they are immediately
        above the current agent (unless specified by the view_above parameter).

        :param other: the other agent
        :param view_above:
        :return: what
        """

        # check whether other agent is within view, i.e. below this agent or in view of one of the cameras
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
        """
        Count the number of agents visible in the specified directions (or North, South, East, West if none are given)

        :param environment: the environment the agent operates in
        :param directions: the directions in which to count agents
        :param angle: the angle around the directions within which agents are counted for a direction
        :param max_distance: the maximum viewing distance (agents further away are not counted)
        :param max_vertical_distance: the maximum vertical distance of other agents to this agent
        :return: a list of agent counts for each specified direction
        """

        # if no directions are given, check in the four directions of movement along the grid
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
                # it may in addition be desirable to check whether the agent is even above the structure/grid
                for d_idx, d in enumerate(directions):
                    position_difference = a.geometry.position - self.geometry.position
                    position_signed_angle = np.arctan2(position_difference[1], position_difference[0]) - \
                                            np.arctan2(d[1], d[0])
                    if abs(position_signed_angle) < angle:
                        agents_counts[d_idx] = agents_counts[d_idx] + 1

        return tuple(agents_counts)

    def check_component_finished(self, compared_map: np.ndarray, component_marker=None):
        """
        Check whether the specified component is finished according to the provided occupancy matrix.

        :param compared_map: the occupancy matrix to check
        :param component_marker: the marker of the component to check
        :return: True if the component is finished, False otherwise
        """

        if component_marker is None:
            component_marker = self.current_component_marker
        tm = np.zeros_like(self.target_map[self.current_structure_level])
        np.place(tm, self.component_target_map[self.current_structure_level] == component_marker, 1)
        om = np.copy(compared_map[self.current_structure_level])
        np.place(om, om > 0, 1)
        np.place(om, self.component_target_map[self.current_structure_level] != component_marker, 0)
        return np.array_equal(om, tm)

    def check_layer_finished(self, compared_map: np.ndarray):
        """
        Check whether the current layer is finished according to the provided occupancy matrix.

        :param compared_map: the occupancy matrix to check
        :return: True if the current layer is finished, False otherwise
        """

        tm = np.copy(self.target_map[self.current_structure_level])
        np.place(tm, tm > 0, 1)
        om = np.copy(compared_map[self.current_structure_level])
        np.place(om, om > 0, 1)
        return np.array_equal(om, tm)

    def check_structure_finished(self, compared_map: np.ndarray):
        """
        Check whether the structure is finished according to the provided occupancy matrix.

        :param compared_map: the occupancy matrix to check
        :return: True if the structure is finished, False otherwise
        """

        tm = np.copy(self.target_map)
        np.place(tm, tm > 0, 1)
        om = np.copy(compared_map)
        np.place(om, om > 0, 1)
        return np.array_equal(om, tm)

    def unfinished_component_markers(self, compared_map: np.ndarray, level=None):
        """
        Return the markers of all components which are not finished yet according to the provided occupancy matrix.

        :param compared_map: the occupancy matrix to check
        :param level: the structure level at which to check for unfinished components
        :return: a list of component markers of unfinished components
        """

        if level is None:
            level = self.current_structure_level
        candidate_components = []
        for marker in np.unique(self.component_target_map[level]):
            if marker != 0:
                subset_indices = np.where(
                    self.component_target_map[level] == marker)
                candidate_values = compared_map[level][subset_indices]
                if np.count_nonzero(candidate_values == 0) > 0:
                    candidate_components.append(marker)
        return candidate_components

    def unseeded_component_markers(self, compared_map: np.ndarray, level=None):
        """
        Return the markers of all components which are not seeded yet according to the provided occupancy matrix.

        :param compared_map: the occupancy matrix to check
        :param level: the structure level at which to check for unseeded components
        :return: a list of component markers of unseeded components
        """

        if level is None:
            level = self.current_structure_level
        candidate_components = []
        for marker in np.unique(self.component_target_map[level]):
            if marker != 0:
                subset_indices = np.where(
                    self.component_target_map[level] == marker)
                candidate_values = compared_map[level][subset_indices]
                if np.count_nonzero(candidate_values) == 0:
                    candidate_components.append(marker)
        return candidate_components

    def component_seed_location(self, component_marker, level=None, include_closing_corners=False):
        """
        Return the seed location for the specified component according to some common (non-random) strategy.

        :param component_marker: the component marker for which to return the seed location
        :param level: the level of the structure at which to check (might be needed for multi-layered components)
        :param include_closing_corners: include the closing corners of holes as possible seed locations
        :return: the location of the specified component's seed in the grid
        """

        if level is None:
            locations = np.where(self.component_target_map == component_marker)
            level = locations[0][0]

        # this location is the grid location, not the position in 3D space
        occupied_locations = np.where(self.component_target_map[level] == component_marker)
        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
        supported_locations = np.nonzero(self.target_map[level - 1])
        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
        occupied_locations = [x for x in occupied_locations if x in supported_locations]
        if not include_closing_corners:
            occupied_locations = [x for x in occupied_locations if (x[1], x[0], level)
                                  not in self.closing_corners[level][self.current_component_marker]]

        # the center-most location is used here, other positions may be more desirable (interesting to investigate)
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
        """
        Return the direction (out of the four grid directions) in which the perimeter of the structure is reached first.

        Note that this method was intended to be used for moving to the perimeter of the structure as quickly as
        possible and thereby hopefully speeding up the search for an attachment site. This did not improve performance
        in practice, and was not used anymore. Thus this does not actually account for the component marker (and
        whether the hole is enclosed by that component, e.g. the component itself might be enclosed by a hole instead).

        :param compared_map: the occupancy matrix to check
        :param start_position: the grid position from which to start counting in each direction
        :param level: the level of the structure on which to check
        :param component_marker: the component marker for which to consider
        :return: a vector of the direction with the smallest distance to the structure perimeter
        """

        if level is None:
            level = self.current_structure_level

        if component_marker is None:
            component_marker = self.current_component_marker

        # TODO: actually check component stuff
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
        """
        Check whether the specified position is at at the closing corner of a hole and if so, whether the
        adjacent positions on the boundary of the hole have already been filled and a block could be attached there.

        :param environment: the environment the agent operates in
        :param position: the position to check
        :return: a boolean tuple, indicating whether the position is at a closing corner and whether
        attachment is allowed there
        """

        if position is None:
            position = self.current_grid_position

        loop_corner_attachable = False
        at_loop_corner = False
        if tuple(position) in self.closing_corners[self.current_structure_level][self.current_component_marker]:
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
                loop_corner_attachable = True
        else:
            loop_corner_attachable = True

        return at_loop_corner, loop_corner_attachable

    def split_into_components(self):
        """
        Determine the disconnected components for each layer, assign markers to them and return a component map.

        :return: an occupancy matrix with component markers
        """

        def component_fill_iterative(layer, i, j, marker):
            if layer[i, j] == 1:
                layer[i, j] = marker

                wave_front = []
                if i > 0:
                    wave_front.append((i - 1, j))
                if i < layer.shape[0] - 1:
                    wave_front.append((i + 1, j))
                if j > 0:
                    wave_front.append((i, j - 1))
                if j < layer.shape[1] - 1:
                    wave_front.append((i, j + 1))
                while len(wave_front) > 0:
                    new_wave_front = []
                    for i2, j2 in wave_front:
                        if layer[i2, j2] == 1:
                            layer[i2, j2] = marker

                            if i2 > 0:
                                new_wave_front.append((i2 - 1, j2))
                            if i2 < layer.shape[0] - 1:
                                new_wave_front.append((i2 + 1, j2))
                            if j2 > 0:
                                new_wave_front.append((i2, j2 - 1))
                            if j2 < layer.shape[1] - 1:
                                new_wave_front.append((i2, j2 + 1))
                    wave_front = new_wave_front

        # go through the target map layer by layer and split each one into disconnected components
        # that information is then stored in an occupancy matrix using different integers for different components
        self.component_target_map = np.copy(self.target_map)
        np.place(self.component_target_map, self.component_target_map > 1, 1)
        component_marker = 2
        for z in range(self.component_target_map.shape[0]):
            # use flood fill to identify components of layer
            for y in range(self.component_target_map.shape[1]):
                for x in range(self.component_target_map.shape[2]):
                    if self.component_target_map[z, y, x] == 1:
                        component_fill_iterative(self.component_target_map[z], y, x, component_marker)
                        component_marker += 1
        return self.component_target_map

    def merge_multi_layer_components(self):
        """
        Merge existing disconnected components into multi-layered components and return a component map of that.

        This method was never actually used in the implementation but may still be useful for future work on this.
        Its purpose is essentially to identify parts of the structure (stretching over multiple layers) that could
        be worked on in isolation from the rest of the structure, because they don't have to be connected to it
        up until some point. One could also make this hierarchical and identify progressively larger subsets of the
        structure which are "disconnected" in this way from the rest, eventually ending up with the entire structure.
        Once assigned to such a multi-layered component, agents would be able to focus on that component until it is
        finished without having to worry about the rest of the structure. After that they would have to check whether
        other (lower-level) components would have to be completed before connecting these "disconnected" components.

        :return: an occupancy map of component markers for multi-layered components
        """

        # merge components if they are only connected to each other and one other component on a different layer
        # (the top and bottom most layers get special treatment for not being connected on both sides)
        # that means any component which is connected to at least two other components on one of the adjacent layers
        # can automatically be counted out

        # NOTES FOR THE FUTURE:
        # - make this a hierarchical (would have to find different representation than single occupancy matrix)
        # - should also check the distance between multi-layered components to ensure that building one component
        #   does not make it physically impossible to reach another (because QCs need some space to operate)
        #   -> could e.g. require 3 blocks of space between components (which would still be useful for structures
        #      with pillars with large distances between them)

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
                            mergeable_pairs.append((m1, m2))

        # merge all components which share one component in the mergeable pairs list
        # first group all those pairs which share components
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
            unique_values = []
            for m1, m2 in group:
                if m1 not in unique_values:
                    unique_values.append(m1)
                if m2 not in unique_values:
                    unique_values.append(m2)
            backup.append(unique_values)
            all_ml_components.extend(unique_values)
        mergeable_groups = backup

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
        """
        Determine the closing corners of holes in the structure and return information about them.

        :return: the coordinates of the closing corners, an occupancy matrix of all holes, the boundary coordinates
        of each hole, the coordinates adjacent to closing corners and the orientations of the closing corners
        """

        # for each layer, check whether there are any holes, i.e. 0's that - when flood-filled - only connect to 1's
        def hole_fill_iterative(layer, i, j, marker):
            if layer[i, j] == 0:
                layer[i, j] = marker

                wave_front = []
                if i > 0:
                    wave_front.append((i - 1, j))
                if i < layer.shape[0] - 1:
                    wave_front.append((i + 1, j))
                if j > 0:
                    wave_front.append((i, j - 1))
                if j < layer.shape[1] - 1:
                    wave_front.append((i, j + 1))
                while len(wave_front) > 0:
                    new_wave_front = []
                    for i2, j2 in wave_front:
                        if layer[i2, j2] == 0:
                            layer[i2, j2] = marker

                            if i2 > 0:
                                new_wave_front.append((i2 - 1, j2))
                            if i2 < layer.shape[0] - 1:
                                new_wave_front.append((i2 + 1, j2))
                            if j2 > 0:
                                new_wave_front.append((i2, j2 - 1))
                            if j2 < layer.shape[1] - 1:
                                new_wave_front.append((i2, j2 + 1))
                    wave_front = new_wave_front

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
                        hole_fill_iterative(hole_map[z], y, x, hole_marker)
                        valid_markers[z].append(hole_marker)
                        hole_marker += 1

        # remove all "holes" connected to the perimeter of the structure, since they are not really holes
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
        def boundary_search_iterative(layer, z, i, j, marker):
            visited = []
            c_list = []
            if layer[i, j] == marker:
                visited.append((i, j))
                wave_front = []
                if i > 0:
                    wave_front.append((i - 1, j))
                if i < layer.shape[0] - 1:
                    wave_front.append((i + 1, j))
                if j > 0:
                    wave_front.append((i, j - 1))
                if j < layer.shape[1] - 1:
                    wave_front.append((i, j + 1))
                while len(wave_front) > 0:
                    new_wave_front = []
                    for i2, j2 in wave_front:
                        if (i2, j2) not in visited and layer[i2, j2] == marker:
                            visited.append((i2, j2))
                            if i2 > 0:
                                new_wave_front.append((i2 - 1, j2))
                            if i2 < layer.shape[0] - 1:
                                new_wave_front.append((i2 + 1, j2))
                            if j2 > 0:
                                new_wave_front.append((i2, j2 - 1))
                            if j2 < layer.shape[1] - 1:
                                new_wave_front.append((i2, j2 + 1))
                        elif layer[i2, j2] != marker:
                            c_list.append((z, i2, j2))
                    wave_front = new_wave_front
            elif layer[i, j] != marker:
                c_list.append((z, i, j))
            return c_list

        hole_boundaries = []
        removable_holes = {}
        for z in range(hole_map.shape[0]):
            hole_boundaries.append([])
            for m in valid_markers[z]:
                locations = np.where(hole_map == m)
                coord_list = boundary_search_iterative(hole_map[z], z, locations[1][0], locations[2][0], m)
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

        # there are two options why a hole might be added to the removable_holes list
        # (adjacent to two different components):
        # a) the hole has to be removed because it is between components (not really a hole)
        # b) the hole encircles a component, but is still a valid hole
        #    -> in this case it's important not to select a closing corner from any enclosed component,
        #       even though that component is adjacent to the hole

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
            for ac in adjacent_components:
                component_positions = np.where(self.component_target_map == ac)
                component_min_x = np.min(component_positions[2])
                component_max_x = np.max(component_positions[2])
                component_min_y = np.min(component_positions[1])
                component_max_y = np.max(component_positions[1])
                if hole_min_x <= component_min_x <= hole_max_x \
                        and hole_min_x <= component_max_x <= hole_max_x \
                        and hole_min_y <= component_min_y <= hole_max_y \
                        and hole_min_y <= component_max_y <= hole_max_y:
                    # any hole enclosing a component has to be larger than the component itself
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
                    self.aprint("More than 1 adjacent component to hole {}: {}".format(m, adjacent_components))

                hole_boundary_coords[m] = hole_boundaries[z][m_idx]
                outer_corner_coord_list = []
                inner_corner_coord_list = []
                corner_boundary_list = []

                component_marker_list_outer = []
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
                        adjacent_seed_location = position[0]
                    else:
                        # include_closing_corners is True here, because the closing corners have not been determined
                        # yet (since they are based on the seed position)
                        adjacent_seed_location = self.component_seed_location(cm, include_closing_corners=True)

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

                    # the closing corner is determined by the length of the shortest path to each of them
                    current_sp_lengths = []
                    for corner in current_outer:
                        temp = shortest_path(
                            self.target_map[corner[2]], tuple(adjacent_seed_location[:2]), tuple(corner[:2]))
                        current_sp_lengths.append(len(temp))
                    ordered_idx = sorted(range(len(current_sp_lengths)), key=lambda i: current_sp_lengths[i])
                    ordered_outer = [current_outer[i] for i in ordered_idx]
                    ordered_boundary = [current_boundary[i] for i in ordered_idx]

                    # OTHER IDEA FOR CHOOSING CORNER: choose the corner that excludes the smallest area

                    # regardless of what method is used to determine the corners, set the closing corner here
                    closing_corner = ordered_outer[-1]
                    closing_corner_boundary = ordered_boundary[-1]

                    # determine on which side of the seed the closing corner lies
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

                    if (len(current_outer) + len(current_inner)) % 2 == 0:
                        closing_corners[z][cm].append(closing_corner)
                        closing_corner_boundaries[z][cm].append(closing_corner_boundary)
                        closing_corner_orientations[z][cm].append(corner_direction)
                    else:
                        self.aprint("The structure contains open corners, which cannot be built.")

                hole_corners[z].append(outer_corner_coord_list)

        return closing_corners, hole_map, hole_boundary_coords, closing_corner_boundaries, closing_corner_orientations

    def aprint(self, *args, print_as_map=False, override_global_printing_enabled=False, **kwargs):
        """
        Print the input to this method, formatted so that the agent's ID is shown, if printing is enabled.

        :param args: normal arguments to the print function
        :param print_as_map: format the first element of args as a map using print_map
        :param override_global_printing_enabled: regardless of the global setting, print or do not print
        depending on this argument
        :param kwargs: keyword arguments to the print function
        """

        if self.printing_enabled or override_global_printing_enabled:
            if print_as_map:
                print("[Agent {}]: ".format(self.id))
                print_map(*args, **kwargs)
            else:
                print("[Agent {}]: ".format(self.id), end="")
                print(*args, **kwargs)

