import numpy as np
import logging
import random
import env.map
from enum import Enum
from typing import List
from abc import ABCMeta, abstractmethod
from env.block import Block
from env.util import print_map, shortest_path, neighbourhood, legal_attachment_sites
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance, rotation_2d, rotation_2d_experimental

np.seterr(divide='ignore', invalid='ignore')


class Task(Enum):
    FETCH_BLOCK = 0
    TRANSPORT_BLOCK = 1
    MOVE_TO_PERIMETER = 2
    FIND_ATTACHMENT_SITE = 3
    PLACE_BLOCK = 4
    MOVE_UP_LAYER = 5
    AVOID_COLLISION = 6
    LAND = 7
    FINISHED = 8


class AgentStatistics:

    def __init__(self, agent):
        self.agent = agent
        self.task_counter = {
            Task.FETCH_BLOCK: 0,
            Task.TRANSPORT_BLOCK: 0,
            Task.MOVE_TO_PERIMETER: 0,
            Task.FIND_ATTACHMENT_SITE: 0,
            Task.PLACE_BLOCK: 0,
            Task.MOVE_UP_LAYER: 0,
            Task.AVOID_COLLISION: 0,
            Task.LAND: 0,
            Task.FINISHED: 0
        }

    def step(self, environment):
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
        self.required_distance = 80

        self.local_occupancy_map = None
        self.next_seed = None
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
        self.backup_grid_position = None
        self.previous_task = Task.FETCH_BLOCK

        self.seed_on_perimeter = False
        self.collision_using_geometries = False
        self.task_history = []
        self.task_history.append(self.current_task)
        self.LAND_CALLED_FIRST_TIME = False
        self.id = -1

        self.agent_statistics = AgentStatistics(self)

    @abstractmethod
    def advance(self, environment: env.map.Map):
        pass

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

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


class CollisionAvoidanceAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(CollisionAvoidanceAgent, self).__init__(position, size, target_map, required_spacing)
        self.block_locations_known = True
        self.structure_location_known = True
        self.logger.setLevel(logging.DEBUG)

    def advance(self, environment: env.map.Map):
        # decide on paths randomly and follow them
        # then dodge any traffic using the experimental collision avoidance scheme
        # TODO: implement collision avoidance scheme
        # (only taking position and geometry into account first, not the current task/path to be followed)
        pass


class PerimeterFollowingAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(PerimeterFollowingAgent, self).__init__(position, size, target_map, required_spacing)
        self.block_locations_known = True
        self.structure_location_known = True
        self.collision_possible = True
        self.component_target_map = self.split_into_components()
        print("COMPONENT TARGET MAP: {}".format(self.component_target_map))
        self.closing_corners, self.hole_map, self.hole_boundaries, self.closing_corner_boundaries = \
            self.find_closing_corners()
        self.logger.setLevel(logging.DEBUG)

    def fetch_block(self, environment: env.map.Map):
        # locate block, locations may be:
        # 1. known from the start (including block type)
        # 2. known roughly (in the case of block deposits/clusters)
        # 3. completely unknown; then the search is an important part
        # whether the own location relative to all this is known is also a question

        if self.block_locations_known:
            if self.current_path is None:
                # find the closest block that is not the seed and has not been placed yet
                # alternatively (or maybe better yet) check if there are any in the construction zone
                min_block = None
                min_distance = float("inf")
                for b in environment.blocks:
                    temp = self.geometry.distance_2d(b.geometry)
                    if not (b.is_seed or b.placed or any(b is a.current_block for a in environment.agents)) \
                            and temp < min_distance:
                        min_block = b
                        min_distance = temp
                if min_block is None:
                    self.logger.info("Construction finished (4).")
                    self.current_task = Task.LAND
                    self.task_history.append(self.current_task)
                    self.current_path = None
                    return
                min_block.color = "green"

                # first add a point to get up to the level of movement for fetching blocks
                # which is one above the current construction level
                self.current_path = Path()
                fetch_level_z = Block.SIZE * (self.current_structure_level + 2) + \
                                self.geometry.size[2] / 2 + self.required_spacing
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
                self.current_path.add_position(
                    [min_block.geometry.position[0], min_block.geometry.position[1], fetch_level_z])
                self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1],
                                                min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[
                                                    2] / 2])
                self.current_block = min_block
        else:
            # TODO: instead of having scattered blocks, use block stashes
            pass

        # assuming that the if-statement above takes care of setting the path:
        # collision detection should intervene here if necessary
        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reach (i.e. the block can be picked up)
            if not ret:
                self.geometry.attached_geometries.append(self.current_block.geometry)
                self.current_task = Task.TRANSPORT_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        # fly down and pick up block, then switch to TRANSPORT_BLOCK
        pass

    def transport_block(self, environment: env.map.Map):
        # gain height to the "transport-to-structure" level

        # locate structure (and seed in particular), different ways of doing this:
        # 1. seed location is known (beacon)
        # 2. location has to be searched for

        if self.structure_location_known:
            # in this case the seed location is taken as the structure location,
            # since that is where the search for attachment sites would start anyway
            if self.current_path is None:
                # gain height, fly to seed location and then start search for attachment site
                self.current_path = Path()
                transport_level_z = Block.SIZE * (self.current_structure_level + 3) + \
                                    (self.geometry.size[2] / 2 + self.required_spacing) * 2
                seed_location = self.current_seed.geometry.position
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                transport_level_z])
                self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z])

        # assuming that the if-statement above takes care of setting the path:
        # collision detection should intervene here if necessary
        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, search for attachment site should start
            if not ret:
                self.current_grid_position = np.copy(self.current_seed.grid_position)
                self.current_path = None
                if not self.seed_on_perimeter:
                    self.current_task = Task.MOVE_TO_PERIMETER
                    self.task_history.append(self.current_task)

                    # since MOVE_TO_PERIMETER is used, the direction to go into is initialised randomly
                    # a better way of doing it would probably be to take the shortest path to the perimeter
                    # using the available knowledge about the current state of the structure (this could even
                    # include leading the agent to an area where it is likely to find an attachment site soon)
                    # self.current_grid_direction = np.array(
                    #     random.sample([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], 1)[0])
                    self.current_grid_direction = [0, -1, 0]
                else:
                    self.current_task = Task.FIND_ATTACHMENT_SITE
                    self.task_history.append(self.current_task)
                # TODO: check whether current component is finished, if yes, check whether others are not yet
        else:
            self.geometry.position = self.geometry.position + current_direction

        # determine path to structure location

        # avoid collisions until structure/seed comes in sight

    def move_to_perimeter(self, environment: env.map.Map):
        if self.current_path is None:
            # move to next block position in designated direction (which could be the shortest path or
            # just some direction chosen e.g. at the start, which is assumed here)
            self.current_path = Path()
            destination_x = (self.current_grid_position + self.current_grid_direction)[0] * Block.SIZE + \
                            environment.offset_origin[0]
            destination_y = (self.current_grid_position + self.current_grid_direction)[1] * Block.SIZE + \
                            environment.offset_origin[1]
            self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.current_grid_position += self.current_grid_direction
                # TODO: don't cheat when checking whether the hole is closed
                # could e.g. try to go to perimeter and when revisiting the same site on attachment site finding,
                # you know that you must be either in a hole or might have to move up a layer
                try:
                    # aprint(self.id, "CHECK AT {}".format(self.current_grid_position))
                    # aprint(self.id, "RESULTS IN {}".format(self.hole_map[
                    #     self.current_grid_position[2], self.current_grid_position[1], self.current_grid_position[0]]))
                    # aprint(self.id, "AND FURTHER IN {}".format(self.hole_boundaries[self.hole_map[
                    #     self.current_grid_position[2], self.current_grid_position[1], self.current_grid_position[0]]]))
                    # aprint(self.id, "GIVING: {}".format(environment.occupancy_map[self.hole_boundaries[self.hole_map[
                    #     self.current_grid_position[2], self.current_grid_position[1], self.current_grid_position[
                    #         0]]]]))
                    # aprint(self.id, "CHECK: \n{}".format(environment.occupancy_map[self.hole_boundaries[self.hole_map[
                    #     self.current_grid_position[2], self.current_grid_position[1], self.current_grid_position[
                    #         0]]]]))
                    # this is basically getting all values of the occupancy map at the locations where the hole map
                    # has the value of the hole which we are currently over
                    result = all(environment.occupancy_map[self.hole_boundaries[self.hole_map[
                        self.current_grid_position[2], self.current_grid_position[1], self.current_grid_position[
                            0]]]] != 0)
                    # aprint(self.id, "CHECKING OVER HOLE, BUT HOLE IS NOT CLOSED YET")
                except (IndexError, KeyError):
                    result = False

                if not environment.check_over_structure(self.geometry.position, self.current_structure_level) and \
                        (check_map(self.hole_map, self.current_grid_position, lambda x: x < 2) or not result):
                    # have reached perimeter
                    self.current_task = Task.FIND_ATTACHMENT_SITE
                    self.task_history.append(self.current_task)
                    self.current_grid_direction = np.array(
                        [-self.current_grid_direction[1], self.current_grid_direction[0], 0], dtype="int32")
                else:
                    destination_x = (self.current_grid_position + self.current_grid_direction)[0] * Block.SIZE + \
                                    environment.offset_origin[0]
                    destination_y = (self.current_grid_position + self.current_grid_direction)[1] * Block.SIZE + \
                                    environment.offset_origin[1]
                    self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])
        else:
            self.geometry.position = self.geometry.position + current_direction

    def find_attachment_site(self, environment: env.map.Map):
        # structure should be in view at this point

        # if allowed attachment site can be determined from visible structure:
        #   determine allowed attachment site given knowledge of target structure
        # else:
        #   use current searching scheme to find legal attachment site

        seed_block = self.current_seed

        if self.current_component_marker != -1:
            # aprint(self.id, "WORKING ON COMPONENT {}".format(self.current_component_marker))
            # checking below whether the current component (as designated by self.current_component_marker) is finished
            tm = np.zeros_like(self.target_map[self.current_structure_level])
            np.place(tm, self.component_target_map[self.current_structure_level] ==
                     self.current_component_marker, 1)
            om = np.copy(environment.occupancy_map[self.current_structure_level])
            np.place(om, om > 0, 1)
            np.place(om, self.component_target_map[self.current_structure_level] !=
                     self.current_component_marker, 0)
            if np.array_equal(om, tm):
                # aprint(self.id, "COMPONENT {} IS COMPLETED".format(self.current_component_marker))
                # current component completed, see whether there is a different one that should be constructed
                # NOTE: for now it is assumed that these components are also seeded already, therefore we
                # can immediately move on to construction

                # checking if unfinished components left
                candidate_components = []
                for marker in np.unique(self.component_target_map[self.current_structure_level]):
                    if marker != 0 and marker != self.current_component_marker:
                        subset_indices = np.where(
                            self.component_target_map[self.current_structure_level] == marker)
                        candidate_values = environment.occupancy_map[self.current_structure_level][
                            subset_indices]
                        # the following check means that on the occupancy map, this component still has all
                        # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                        # for placing the currently transported seed there
                        if np.count_nonzero(candidate_values == 0) > 0:
                            candidate_components.append(marker)

                aprint(self.id,
                       "THIS IS ONLY A TEST:\n{}".format(self.component_target_map[self.current_structure_level]))

                if len(candidate_components) > 0:
                    # choosing one of the candidate components to continue constructing
                    self.current_component_marker = random.sample(candidate_components, 1)[0]
                    aprint(self.id, "(1) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                    aprint(self.id,
                           "(FIND_ATTACHMENT_SITE) After placing block: unfinished components left, choosing {}".format(
                               self.current_component_marker))
                    # getting the coordinates of those positions where the other component already has blocks
                    correct_locations = np.where(
                        self.component_target_map[self.current_structure_level] == self.current_component_marker)
                    correct_locations = list(zip(correct_locations[0], correct_locations[1]))
                    occupied_locations = np.where(environment.occupancy_map[self.current_structure_level] != 0)
                    occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                    occupied_locations = list(set(occupied_locations).intersection(correct_locations))
                    for b in environment.placed_blocks:
                        if b.is_seed and b.grid_position[2] == self.current_structure_level \
                                and (b.grid_position[1], b.grid_position[0]) in occupied_locations:
                            self.current_seed = b
                            aprint(self.id, "New seed location: {}".format(self.current_seed))
                            self.current_path = None
                            self.current_task = Task.TRANSPORT_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_visited_sites = None
                            break
                else:
                    if self.current_structure_level >= self.target_map.shape[0] - 1:
                        self.current_task = Task.LAND
                        aprint(self.id, "LANDING INITIALISED IN FIND_ATTACHMENT_SITE 1")
                        aprint(self.id, "current_component_marker:", self.current_component_marker)
                        aprint(self.id, "TARGET MAP:\n{}\nOCCUPANCY MAP:\n{}".format(tm, om))
                        self.task_history.append(self.current_task)
                        self.current_visited_sites = None
                    else:
                        self.current_task = Task.MOVE_UP_LAYER
                        self.task_history.append(self.current_task)
                        self.current_structure_level += 1
                        self.current_visited_sites = None
                    self.current_path = None
                    self.current_component_marker = -1
                    aprint(self.id, "(2) CURRENT COMPONENT MARKER SET TO {} DUE TO {} (SL: {})".format(
                        self.current_component_marker, candidate_components,
                        self.current_structure_level if self.current_task == Task.LAND else self.current_structure_level - 1))
                return

        # orientation happens counter-clockwise -> follow seed edge in that direction once its reached
        # can either follow the perimeter itself or just fly over blocks (do the latter for now)
        if self.current_path is None:
            # path only leads to next possible site (assumption for now is that only block below is known)
            # first go to actual perimeter of structure (correct side of seed block)
            # THE FOLLOWING IS USED ONLY WHEN THE SEED IS ON THE PERIMETER AND USED AS THE ONLY REFERENCE POINT:
            seed_perimeter = np.copy(seed_block.geometry.position)
            if seed_block.seed_marked_edge == "down":
                seed_perimeter += np.array([0, -Block.SIZE, 0])
                self.current_grid_position += np.array([0, -1, 0])
                self.current_grid_direction = np.array([1, 0, 0], dtype="int32")
            elif seed_block.seed_marked_edge == "up":
                seed_perimeter += np.array([0, Block.SIZE, 0])
                self.current_grid_position += np.array([0, 1, 0])
                self.current_grid_direction = np.array([-1, 0, 0], dtype="int32")
            elif seed_block.seed_marked_edge == "right":
                seed_perimeter += np.array([Block.SIZE, 0, 0])
                self.current_grid_position += np.array([1, 0, 0])
                self.current_grid_direction = np.array([0, 1, 0], dtype="int32")
            elif seed_block.seed_marked_edge == "left":
                seed_perimeter += np.array([-Block.SIZE, 0, 0])
                self.current_grid_position += np.array([-1, 0, 0])
                self.current_grid_direction = np.array([0, -1, 0], dtype="int32")

            seed_perimeter[2] = self.geometry.position[2]
            self.current_path = Path()
            self.current_path.add_position(seed_perimeter)

            # THE FOLLOWING IS USED WHEN THE PERIMETER HAS BEEN FOUND USING MOVE_TO_PERIMETER:
            # the grid direction is assumed to be correct, so it just has to be turned counter-clockwise

        # use the following list to keep track of the combination of visited attachment sites and the direction of
        # movement at that point in time; if, during the same "attachment search" one such "site" is revisited, then
        # the agent is stuck in a loop, most likely caused by being trapped in a hole
        if self.current_visited_sites is None:
            self.current_visited_sites = []

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                # aprint(self.id, "TRYING TO ATTACH AT COMPONENT {}".format(self.current_component_marker))
                # corner of the current block reached, assess next action
                # TODO: check whether this is the corner of a loop and if so whether it can be closed
                # TODO: need to prioritise closing holes, i.e. get hole closed first, then build outside of said hole
                # otherwise it is possible that the "outer layers" can block finishing the hole
                loop_corner_attachable = False
                at_loop_corner = False
                if tuple(self.current_grid_position) \
                        in self.closing_corners[self.current_structure_level][self.current_component_marker]:
                    # aprint(self.id,
                    #        "TRYING TO ATTACH AT CORNER OF LOOP, COORDINATES {}".format(self.current_grid_position))
                    at_loop_corner = True
                    # need to check whether the adjacent blocks have been placed already
                    counter = 0
                    surrounded_in_y = False
                    surrounded_in_x = False
                    index = self.closing_corners[self.current_structure_level][self.current_component_marker].index(
                        tuple(self.current_grid_position))
                    possible_boundaries = self.closing_corner_boundaries[self.current_structure_level][
                        self.current_component_marker][index]
                    # TODO: change closing_corners and closing_corner_boundaries to be component-wise
                    if environment.check_occupancy_map(self.current_grid_position + np.array([0, -1, 0])) and \
                            tuple(self.current_grid_position + np.array([0, -1, 0])) in possible_boundaries:
                        counter += 1
                        surrounded_in_y = True
                    if not surrounded_in_y and environment.check_occupancy_map(
                            self.current_grid_position + np.array([0, 1, 0])) and \
                            tuple(self.current_grid_position + np.array([0, 1, 0])) in possible_boundaries:
                        counter += 1
                    if environment.check_occupancy_map(self.current_grid_position + np.array([-1, 0, 0])) and \
                            tuple(self.current_grid_position + np.array([-1, 0, 0])) in possible_boundaries:
                        counter += 1
                        surrounded_in_x = True
                    if not surrounded_in_x and environment.check_occupancy_map(
                            self.current_grid_position + np.array([1, 0, 0])) and \
                            tuple(self.current_grid_position + np.array([1, 0, 0])) in possible_boundaries:
                        counter += 1
                    if counter >= 2:
                        aprint(self.id, "CORNER ALREADY SURROUNDED BY ADJACENT BLOCKS")
                        loop_corner_attachable = True
                    else:
                        aprint(self.id, "CORNER NOT SURROUNDED BY ADJACENT BLOCKS YET")
                else:
                    loop_corner_attachable = True

                # check whether location is somewhere NORTH-EAST of any closing corner, i.e. the block should not be
                # placed there before closing that loop (NE because currently all closing corners are located there)
                allowable_region_attachable = True
                if not at_loop_corner:
                    for x, y, z in self.closing_corners[self.current_structure_level][self.current_component_marker]:
                        if not environment.check_occupancy_map(np.array([x, y, z])) and \
                                x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                            allowable_region_attachable = False
                            break

                current_site_tuple = (tuple(self.current_grid_position), tuple(self.current_grid_direction))
                if current_site_tuple in self.current_visited_sites:
                    aprint(self.id, "REVISITED SITE")
                    self.current_path = None
                    self.current_task = Task.MOVE_TO_PERIMETER
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    # self.current_grid_direction = np.array(
                    #     random.sample([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], 1)[0])
                    self.current_grid_direction = [1, 0, 0]
                    return

                # adding location and direction here to check for revisiting
                self.current_visited_sites.append(current_site_tuple)

                # TODO: might want to check whether behaviour should actually be different if a hole was encountered
                if loop_corner_attachable and allowable_region_attachable and \
                        check_map(self.target_map, self.current_grid_position) and \
                        (environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction) or
                         (self.current_row_started and (check_map(self.target_map, self.current_grid_position +
                                                                                   self.current_grid_direction,
                                                                  lambda x: x == 0) or
                                                        environment.check_occupancy_map(
                                                            self.current_grid_position + self.current_grid_direction +
                                                            np.array([-self.current_grid_direction[1],
                                                                      self.current_grid_direction[0], 0],
                                                                     dtype="int32"), lambda x: x == 0)))):

                    if ((environment.check_occupancy_map(self.current_grid_position + np.array([1, 0, 0])) and
                         environment.check_occupancy_map(self.current_grid_position + np.array([-1, 0, 0]))) or \
                        (environment.check_occupancy_map(self.current_grid_position + np.array([0, 1, 0])) and
                         environment.check_occupancy_map(self.current_grid_position + np.array([0, -1, 0])))) and \
                            not environment.check_occupancy_map(self.current_grid_position, lambda x: x > 0):
                        self.current_task = Task.LAND
                        aprint(self.id, "LANDING INITIALISED IN FIND_ATTACHMENT_SITE 2 ({} is occupied: {})"
                               .format(self.current_grid_position,
                                       environment.check_occupancy_map(self.current_grid_position, lambda x: x > 0)))
                        self.task_history.append(self.current_task)
                        self.current_visited_sites = None
                        self.current_path = None
                        self.logger.debug("CASE 1-3: Attachment site found, but block cannot be placed at {}."
                                          .format(self.current_grid_position))
                    else:
                        # site should be occupied AND
                        # 1. site ahead has a block (inner corner) OR
                        # 2. the current "identified" row ends (i.e. no chance of obstructing oneself)
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_visited_sites = None
                        self.current_row_started = False
                        self.current_path = None
                        log_string = "CASE 1-{}: Attachment site found, block can be placed at {}."
                        if environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction):
                            log_string = log_string.format(1, self.current_grid_position)
                        else:
                            log_string = log_string.format(2, self.current_grid_position)
                        self.logger.debug(log_string)
                else:
                    # site should not be occupied -> determine whether to turn a corner or continue, options:
                    # 1. turn right (site ahead occupied)
                    # 2. turn left
                    # 3. continue straight ahead along perimeter
                    if environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction):
                        # turn right
                        self.current_grid_direction = np.array([self.current_grid_direction[1],
                                                                -self.current_grid_direction[0], 0],
                                                               dtype="int32")
                        # self.current_grid_position += self.current_grid_direction
                        self.logger.debug("CASE 2: Position straight ahead occupied, turning clockwise.")
                    elif environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction +
                                                         np.array([-self.current_grid_direction[1],
                                                                   self.current_grid_direction[0], 0],
                                                                  dtype="int32"),
                                                         lambda x: x == 0):
                        reference_position = self.geometry.position
                        # first move forward (to the corner)
                        self.current_path.add_position(
                            self.geometry.position + Block.SIZE * self.current_grid_direction)
                        reference_position = self.current_path.positions[-1]
                        # then turn left
                        self.current_grid_position += self.current_grid_direction
                        self.current_grid_direction = np.array([-self.current_grid_direction[1],
                                                                self.current_grid_direction[0], 0],
                                                               dtype="int32")
                        self.current_grid_position += self.current_grid_direction
                        # might want to build the above ugliness into the Path class somehow
                        self.current_row_started = True
                        self.logger.debug(
                            "CASE 3: Reached corner of structure, turning counter-clockwise. {} {}".format(
                                self.current_grid_position, self.current_grid_direction))
                        self.current_path.add_position(reference_position + Block.SIZE * self.current_grid_direction)
                    else:
                        # otherwise site "around the corner" occupied -> continue straight ahead
                        self.current_grid_position += self.current_grid_direction
                        self.current_row_started = True
                        self.logger.debug("CASE 4: Adjacent positions ahead occupied, continuing to follow perimeter.")
                        self.current_path.add_position(
                            self.geometry.position + Block.SIZE * self.current_grid_direction)

                # need a way to check whether the current level has been completed already
                tm = np.copy(self.target_map[self.current_structure_level])
                np.place(tm, tm == 2, 1)
                om = np.copy(environment.occupancy_map[self.current_structure_level])
                np.place(om, om == 2, 1)
                if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                    self.current_task = Task.MOVE_UP_LAYER
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_structure_level += 1
                    self.current_path = None
                    self.next_seed = self.current_block
        else:
            self.geometry.position = self.geometry.position + current_direction

        # if interrupted by other quadcopter when moving to attachment site
        # (which should be avoidable if they all e.g. move counterclockwise),
        # avoid collision and determine new attachment site
        pass

    def place_block(self, environment: env.map.Map):
        # fly to determined attachment site, lower quadcopter and place block,
        # then switch task back to fetching blocks

        if self.current_path is None:
            init_z = Block.SIZE * (self.current_structure_level + 2) + self.required_spacing + self.geometry.size[2] / 2
            placement_x = Block.SIZE * self.current_grid_position[0] + environment.offset_origin[0]
            placement_y = Block.SIZE * self.current_grid_position[1] + environment.offset_origin[1]
            placement_z = Block.SIZE * (self.current_grid_position[2] + 1) + self.geometry.size[2] / 2
            self.current_path = Path()
            self.current_path.add_position([placement_x, placement_y, init_z])
            self.current_path.add_position([placement_x, placement_y, placement_z])

        if environment.check_occupancy_map(self.current_grid_position):
            # a different agent has already placed the block in the meantime
            self.current_path = None
            self.current_task = Task.TRANSPORT_BLOCK
            self.task_history.append(self.current_task)
            return

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # block should now be placed in the environment's occupancy matrix
            if not ret:
                if abs(self.current_block.geometry.position[2] -
                       (Block.SIZE / 2 + self.current_structure_level * Block.SIZE)) > Block.SIZE / 2:
                    placement_x = Block.SIZE * self.current_grid_position[0] + environment.offset_origin[0]
                    placement_y = Block.SIZE * self.current_grid_position[1] + environment.offset_origin[1]
                    placement_z = Block.SIZE * (self.current_grid_position[2] + 1) + self.geometry.size[2] / 2
                    aprint(self.id, "BLOCK PLACED IN WRONG Z-LOCATION")
                    aprint(self.id, "THINGS: {} - {}".format(self.current_block.geometry.position[2], placement_z))
                    self.current_path.add_position([placement_x, placement_y, placement_z])
                    # TODO: (probably) need to make sure that collision avoidance does not fuck up the pathing
                    return

                environment.place_block(self.current_grid_position, self.current_block)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                self.current_block.placed = True
                self.current_block.grid_position = self.current_grid_position
                self.current_block = None
                self.current_path = None

                if self.current_component_marker != -1:
                    # aprint(self.id, "CASE 1 AFTER PLACING")
                    tm = np.zeros_like(self.target_map[self.current_structure_level])
                    np.place(tm, self.component_target_map[self.current_structure_level] ==
                             self.current_component_marker, 1)
                    om = np.copy(environment.occupancy_map[self.current_structure_level])
                    np.place(om, om > 0, 1)
                    np.place(om, self.component_target_map[self.current_structure_level] !=
                             self.current_component_marker, 0)
                    if np.array_equal(om, tm):
                        # current component completed, see whether there is a different one that should be constructed
                        # NOTE: for now it is assumed that these components are also seeded already, therefore we
                        # can immediately move on to construction

                        # checking if unfinished components left
                        candidate_components = []
                        for marker in np.unique(self.component_target_map[self.current_structure_level]):
                            if marker != 0 and marker != self.current_component_marker:
                                subset_indices = np.where(
                                    self.component_target_map[self.current_structure_level] == marker)
                                candidate_values = environment.occupancy_map[self.current_structure_level][
                                    subset_indices]
                                # the following check means that on the occupancy map, this component still has all
                                # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                                # for placing the currently transported seed there
                                if np.count_nonzero(candidate_values == 0) > 0:
                                    candidate_components.append(marker)

                        if len(candidate_components) > 0:
                            # choosing one of the candidate components to continue constructing
                            self.current_component_marker = random.sample(candidate_components, 1)[0]
                            aprint(self.id,
                                   "(3) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                            aprint(self.id, "After placing block: unfinished components left, choosing {}".format(
                                self.current_component_marker))
                            # getting the coordinates of those positions where the other component already has blocks
                            correct_locations = np.where(self.component_target_map[
                                                             self.current_structure_level] == self.current_component_marker)
                            correct_locations = list(zip(correct_locations[0], correct_locations[1]))
                            occupied_locations = np.where(environment.occupancy_map[self.current_structure_level] != 0)
                            occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                            occupied_locations = list(set(occupied_locations).intersection(correct_locations))
                            for b in environment.placed_blocks:
                                if b.is_seed and b.grid_position[2] == self.current_structure_level \
                                        and (b.grid_position[1], b.grid_position[0]) in occupied_locations:
                                    self.current_seed = b
                                    aprint(self.id, "New seed location: {}".format(self.current_seed))
                                    self.current_path = None
                                    self.current_task = Task.FETCH_BLOCK
                                    self.task_history.append(self.current_task)
                                    break
                        else:
                            if self.current_structure_level >= self.target_map.shape[0] - 1:
                                self.current_task = Task.LAND
                                self.task_history.append(self.current_task)
                            else:
                                self.current_task = Task.MOVE_UP_LAYER
                                self.task_history.append(self.current_task)
                                self.current_structure_level += 1
                            self.current_path = None
                            self.current_component_marker = -1
                            aprint(self.id,
                                   "(4) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                    else:
                        self.current_task = Task.FETCH_BLOCK
                        self.task_history.append(self.current_task)
                else:
                    aprint(self.id, "CASE 2 AFTER PLACING")
                    tm = np.copy(self.target_map[self.current_structure_level])
                    np.place(tm, tm == 2, 1)
                    om = np.copy(environment.occupancy_map[self.current_structure_level])
                    np.place(om, om == 2, 1)
                    if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                        self.current_task = Task.MOVE_UP_LAYER
                        self.task_history.append(self.current_task)
                        self.current_structure_level += 1
                        self.current_path = None
                    elif np.array_equal(environment.occupancy_map[self.current_structure_level], tm):
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.logger.info("Construction finished (3).")
                    else:
                        self.current_task = Task.FETCH_BLOCK
                        self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

        # if interrupted, search for new attachment site again
        pass

    def move_up_layer(self, environment: env.map.Map):
        if self.current_path is None:
            if self.current_component_marker == -1:
                min_free_edges = None
                if self.seed_on_perimeter:
                    def allowed_position(y, x):
                        return 0 <= y < self.target_map[self.current_structure_level].shape[0] and \
                               0 <= x < self.target_map[self.current_structure_level].shape[1]

                    # determine one block to serve as seed (a position in the target map)
                    min_adjacent = 4
                    min_coordinates = [0, 0, self.current_structure_level]  # [x, y]
                    min_free_edges = ["up", "down", "left", "right"]
                    for i in range(0, self.target_map[self.current_structure_level].shape[0]):
                        for j in range(0, self.target_map[self.current_structure_level].shape[1]):
                            if self.target_map[self.current_structure_level, i, j] == 0:
                                continue
                            current_adjacent = 0
                            current_free_edges = ["up", "down", "left", "right"]
                            if allowed_position(i, j + 1) and self.target_map[
                                self.current_structure_level, i, j + 1] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("up")
                            if allowed_position(i, j - 1) and self.target_map[
                                self.current_structure_level, i, j - 1] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("down")
                            if allowed_position(i - 1, j) and self.target_map[
                                self.current_structure_level, i - 1, j] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("left")
                            if allowed_position(i + 1, j) and self.target_map[
                                self.current_structure_level, i + 1, j] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("right")
                            if current_adjacent < min_adjacent:
                                min_adjacent = current_adjacent
                                min_coordinates = [j, i, self.current_structure_level]
                                min_free_edges = current_free_edges
                else:
                    # pick random location on the next layer
                    # might actually want the MAXIMUM number of adjacent blocks
                    # in the case of "dangling" (what) parts of the structures (i.e. overhanging ledges etc.),
                    # should also choose one that is actually supported by the underlying structure
                    # TODO: implement better selection of location for next seed
                    # occupied_locations = np.nonzero(self.target_map[self.current_structure_level])
                    # occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                    # supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                    # supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                    # occupied_locations = [x for x in occupied_locations if x in supported_locations]
                    # coordinates = random.sample(occupied_locations, 1)
                    # min_coordinates = [coordinates[0][0], coordinates[0][1], self.current_structure_level]

                    # pick some connected component on this layer (randomly for now) and get a seed for it
                    unique_values = np.unique(self.component_target_map[self.current_structure_level]).tolist()
                    unique_values = [x for x in unique_values if x != 0]
                    self.current_component_marker = random.sample(unique_values, 1)[0]
                    aprint(self.id, "(5) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                    occupied_locations = np.where(
                        self.component_target_map[self.current_structure_level] == self.current_component_marker)
                    aprint(self.id, np.unique(self.component_target_map[self.current_structure_level]))
                    aprint(self.id, self.current_component_marker, occupied_locations)
                    occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                    # still using np.nonzero here because it does not matter what the supporting block belongs to
                    supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                    supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                    occupied_locations = [x for x in occupied_locations if x in supported_locations]
                    occupied_locations = [x for x in occupied_locations if (x[1], x[0], self.current_structure_level)
                                          not in self.closing_corners[self.current_structure_level][
                                              self.current_component_marker]]
                    coordinates = random.sample(occupied_locations, 1)
                    min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]

                    # TODO: while this now uses the most SOUTH-WESTERN position, something else might be even better
                    sorted_by_y = sorted(occupied_locations, key=lambda e: e[0])
                    sorted_by_x = sorted(sorted_by_y, key=lambda e: e[1])
                    min_coordinates = [sorted_by_x[0][1], sorted_by_x[0][0], self.current_structure_level]

                    aprint(self.id, "MIN COORDINATES: {}".format(min_coordinates))

                min_block = None
                if self.current_block is None:
                    aprint(self.id, "DECIDING ON NEW SEED BLOCK FOR NEXT LAYER")
                    min_distance = float("inf")
                    for b in environment.blocks:
                        temp = self.geometry.distance_2d(b.geometry)
                        if not (b.is_seed or b.placed or any(b is a.current_block for a in environment.agents)) \
                                and temp < min_distance:
                            min_block = b
                            min_distance = temp
                    aprint(self.id, min_block)
                    if min_block is None:
                        self.logger.info("Construction finished (2).")
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        return
                else:
                    min_block = self.current_block

                min_block.is_seed = True
                # min_block.color = Block.COLORS["seed"]
                if self.seed_on_perimeter:
                    min_block.seed_marked_edge = random.choice(min_free_edges)

                self.next_seed = min_block
                self.next_seed.grid_position = np.array(min_coordinates)
                self.backup_grid_position = self.next_seed.grid_position
                self.current_grid_position = self.next_seed.grid_position

            # first add a point to get up to the level of movement for fetching blocks
            # which is one above the current construction level
            self.current_path = Path()
            fetch_level_z = Block.SIZE * (self.current_structure_level + 1) + \
                            self.geometry.size[2] / 2 + self.required_spacing
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            if self.current_block is None:
                self.current_path.add_position(
                    [self.next_seed.geometry.position[0], self.next_seed.geometry.position[1], fetch_level_z])
                self.current_path.add_position(
                    [self.next_seed.geometry.position[0], self.next_seed.geometry.position[1],
                     self.next_seed.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[
                         2] / 2])
            else:
                transport_level_z = Block.SIZE * (self.current_structure_level + 3) + \
                                    self.required_spacing + self.geometry.size[2] / 2
                destination_x = self.current_seed.geometry.position[0]
                destination_y = self.current_seed.geometry.position[1]
                self.current_path.add_position(
                    [self.geometry.position[0], self.geometry.position[1], transport_level_z])
                self.current_path.add_position([destination_x, destination_y, transport_level_z])
        else:
            pass

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # aprint(self.id, "self.next_seed:", self.next_seed)
            # aprint(self.id, "self.next_seed.grid_position:", self.next_seed.grid_position)
            if self.next_seed.grid_position is None:
                self.next_seed.grid_position = self.backup_grid_position
            destination = [Block.SIZE * self.next_seed.grid_position[0] + environment.offset_origin[0],
                           Block.SIZE * self.next_seed.grid_position[1] + environment.offset_origin[1]]
            if self.seed_on_perimeter:
                if self.current_block is not None and all(
                        [self.geometry.position[i] == destination[i] for i in range(2)]):
                    # check whether seed block has already been placed (this assumes that all agents will choose the
                    # same location for the next seed block, which works for now but should probably be changed later)
                    if environment.check_occupancy_map(self.next_seed.grid_position):
                        self.current_block.color = "green"
                        self.current_block.is_seed = False
                        self.current_block.seed_marked_edge = "down"
                        self.current_path = None
                        self.current_seed = None
                        for b in environment.placed_blocks:
                            if b.is_seed and all(
                                    [b.grid_position[i] == self.next_seed.grid_position[i] for i in range(3)]):
                                self.current_seed = b
                                break
                        aprint(self.id, "next_seed = None bc self.seed_on_perimeter")
                        self.next_seed = None
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        return
            else:
                # the following could also be used for the other case (where the seed has to be on the perimeter)
                # however, this also assumes that knowledge of the entire structure is available immediately upon
                # arriving somewhere at the structure, which is unrealistic -> this would have to be changed into
                # surveying the structure beforehand to be sure that it is the case
                # NOTE: the above may not be true anymore, after changing to the multi-seed algorithm

                # the if-statement below checks whether the current component has already been seeded
                # if it has the count is larger than zero and the carried seed should be reused
                # otherwise it should be seeded
                current_component_subset = environment.occupancy_map[self.current_structure_level][
                    self.component_target_map[self.current_structure_level] == self.current_component_marker]
                if self.current_block is not None and environment.check_over_structure(self.geometry.position) \
                        and np.count_nonzero(current_component_subset) != 0:
                    # two possibilities for multi-seeded algorithm:
                    # 1. there are still other unseeded components left, then go seed those
                    # 2. no unseeded components are left, the block should simply be attached somewhere

                    # checking if unseeded components are left
                    candidate_components = []
                    for marker in np.unique(self.component_target_map[self.current_structure_level]):
                        if marker != 0 and marker != self.current_component_marker:
                            subset_indices = np.where(self.component_target_map[self.current_structure_level] == marker)
                            candidate_values = environment.occupancy_map[self.current_structure_level][subset_indices]
                            # the following check means that on the occupancy map, this component still has all
                            # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                            # for placing the currently transported seed there
                            if np.count_nonzero(candidate_values) == 0:
                                candidate_components.append(marker)

                    if len(candidate_components) > 0:
                        # choosing one of the candidate components and determine the desired coordinates
                        self.current_component_marker = random.sample(candidate_components, 1)[0]
                        aprint(self.id, "(6) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                        occupied_locations = np.where(
                            self.component_target_map[self.current_structure_level] == self.current_component_marker)
                        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                        supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                        occupied_locations = [x for x in occupied_locations if x in supported_locations]
                        coordinates = random.sample(occupied_locations, 1)
                        min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]

                        sorted_by_y = sorted(occupied_locations, key=lambda e: e[0])
                        sorted_by_x = sorted(sorted_by_y, key=lambda e: e[1])
                        min_coordinates = [sorted_by_x[0][1], sorted_by_x[0][0], self.current_structure_level]

                        # determining the path there
                        self.current_path = Path()
                        transport_level_z = Block.SIZE * (self.current_structure_level + 3) + \
                                            self.required_spacing + self.geometry.size[2] / 2
                        destination_x = self.current_seed.geometry.position[0]
                        destination_y = self.current_seed.geometry.position[1]
                        self.current_path.add_position(
                            [self.geometry.position[0], self.geometry.position[1], transport_level_z])
                        self.current_path.add_position([destination_x, destination_y, transport_level_z])

                        # setting the new coordinates for seeding that component
                        self.next_seed.grid_position = np.array(min_coordinates)
                        self.backup_grid_position = self.next_seed.grid_position
                        self.current_grid_position = self.next_seed.grid_position
                        return
                    else:
                        # instead place the block, preferably at the current component
                        # TODO (CONTINUE WORK HERE): finding correct component and changing task to going there
                        if self.current_block is None:
                            aprint(self.id, "SOMETHING'S FUCKY")
                        # it can happen that the agent is still hovering over the structure and "observes" that
                        # a block is being placed in the intended position -> therefore this would trigger without
                        # the agent actually carrying a block -> an element has to be added to the if-statement
                        self.current_block.color = "green"
                        self.current_block.is_seed = False
                        self.current_path = None
                        self.current_seed = None
                        # this needs a-changing:
                        y, x = np.where(self.component_target_map[self.current_structure_level] ==
                                        self.current_component_marker)
                        for b in environment.placed_blocks:
                            for c in zip(x, y):
                                if b.is_seed and b.grid_position[2] == self.current_structure_level \
                                        and all([b.grid_position[i] == c[i] for i in range(2)]):
                                    self.current_seed = b
                                    break
                        aprint(self.id, "next_seed = None bc block should be placed rather than used as seed")
                        # aprint(self.id, "current level component map:\n{}\nand coordinates: {}".format(environment))
                        aprint(self.id, "current_seed = {}".format(self.current_seed))
                        self.next_seed = None
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        return

            # TODO
            # Currently, after placing one seed block for one component, the agent will go back to the if statement
            # below immediately after. This results in it just fetching the block if it has already been placed.
            # Instead, the agent should check whether it should get another seed and fetch that.

            # if the final point on the path has been reach, search for attachment site should start
            if not ret:
                # also need to check whether there is already a seed on that layer or nah
                # -> to simplify things, can assume that agents would all choose the same seed location
                if self.current_block is None:
                    aprint(self.id, "CASE 1")
                    self.current_block = self.next_seed
                    self.geometry.attached_geometries.append(self.current_block.geometry)
                    transport_level_z = Block.SIZE * (self.current_structure_level + 3) + \
                                        self.required_spacing + self.geometry.size[2] / 2

                    # computing the seed position for the previous layer
                    destination_x = self.current_seed.geometry.position[0]
                    destination_y = self.current_seed.geometry.position[1]

                    self.current_path.add_position(
                        [self.geometry.position[0], self.geometry.position[1], transport_level_z])
                    self.current_path.add_position([destination_x, destination_y, transport_level_z])
                elif all([self.geometry.position[i] == self.current_seed.geometry.position[i] for i in range(2)]) \
                        and not all([self.current_seed.geometry.position[i] == destination[i] for i in range(2)]):
                    aprint(self.id, "CASE 2")
                    # old seed position has been reached, now the path to the next seed's location should commence
                    # what should really happen is to use the shortest path over the existing blocks to the next
                    # seed location to reach it and at the same time ensure that the location is correct

                    # to make things less tedious for now, a simple path is used
                    destination_x = Block.SIZE * self.next_seed.grid_position[0] + environment.offset_origin[0]
                    destination_y = Block.SIZE * self.next_seed.grid_position[1] + environment.offset_origin[1]
                    destination_z = Block.SIZE * (self.current_structure_level + 1) + self.geometry.size[2] / 2

                    self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])
                    self.current_path.add_position([destination_x, destination_y, destination_z])

                    # need to check at this point (or actually in between here and the next else block)
                    # whether the determined seed location is already occupied
                elif all([self.geometry.position[i] == self.current_seed.geometry.position[i] for i in range(2)]) \
                        and not abs(self.geometry.position[2] - self.geometry.size[2] / 2 - Block.SIZE -
                                    self.current_structure_level * Block.SIZE) < 0.0001:
                    aprint(self.id, "CASE 3")
                    # TODO: clean up this mess with the previous seed position
                    self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                    Block.SIZE * (self.current_structure_level + 1) +
                                                    self.geometry.size[2] / 2])
                else:
                    aprint(self.id, "CASE 4")
                    self.geometry.attached_geometries.remove(self.current_block.geometry)
                    self.current_seed = self.next_seed
                    self.current_block.placed = True
                    self.current_block.grid_position = self.current_grid_position
                    environment.place_block(self.current_grid_position, self.current_block)
                    self.current_block = None
                    self.current_path = None
                    self.next_seed = None

                    # TODO: need to check here whether next seed has to be fetched
                    # checking if unseeded components are left
                    candidate_components = []
                    for marker in np.unique(self.component_target_map[self.current_structure_level]):
                        if marker != 0 and marker != self.current_component_marker:
                            subset_indices = np.where(self.component_target_map[self.current_structure_level] == marker)
                            candidate_values = environment.occupancy_map[self.current_structure_level][subset_indices]
                            # the following check means that on the occupancy map, this component still has all
                            # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                            # for placing the currently transported seed there
                            if np.count_nonzero(candidate_values) == 0:
                                candidate_components.append(marker)

                    if len(candidate_components) > 0:
                        # choosing one of the candidate components and determine the desired coordinates
                        self.current_component_marker = random.sample(candidate_components, 1)[0]
                        aprint(self.id, "(7) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                        occupied_locations = np.where(
                            self.component_target_map[self.current_structure_level] == self.current_component_marker)
                        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                        supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                        occupied_locations = [x for x in occupied_locations if x in supported_locations]
                        coordinates = random.sample(occupied_locations, 1)
                        min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]

                        sorted_by_y = sorted(occupied_locations, key=lambda e: e[0])
                        sorted_by_x = sorted(sorted_by_y, key=lambda e: e[1])
                        min_coordinates = [sorted_by_x[0][1], sorted_by_x[0][0], self.current_structure_level]

                        min_block = None
                        min_distance = float("inf")
                        for b in environment.blocks:
                            temp = self.geometry.distance_2d(b.geometry)
                            if not (b.is_seed or b.placed or any(b is a.current_block for a in environment.agents)) \
                                    and temp < min_distance:
                                min_block = b
                                min_distance = temp
                        if min_block is None:
                            self.logger.info("Construction finished (2).")
                            self.current_task = Task.LAND
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            return
                        min_block.is_seed = True
                        # min_block.color = Block.COLORS["seed"]

                        self.next_seed = min_block
                        self.next_seed.grid_position = np.array(min_coordinates)
                        self.backup_grid_position = self.next_seed.grid_position
                        self.current_grid_position = self.next_seed.grid_position
                    else:
                        # no more components have to be seeded, can move on to construction
                        tm = np.copy(self.target_map[self.current_structure_level])
                        np.place(tm, tm == 2, 1)
                        om = np.copy(environment.occupancy_map[self.current_structure_level])
                        np.place(om, om == 2, 1)
                        if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                            self.current_task = Task.MOVE_UP_LAYER
                            self.task_history.append(self.current_task)
                            self.current_structure_level += 1
                            self.current_component_marker = -1
                            aprint(self.id,
                                   "(8) CURRENT COMPONENT MARKER SET TO {}".format(self.current_component_marker))
                            self.current_path = None
                        elif np.array_equal(om, tm):
                            self.current_task = Task.LAND
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            self.logger.info("Construction finished (1).")
                        else:
                            self.current_task = Task.FETCH_BLOCK
                            self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

        # move up one level in the structure and place the seed block

        # continue normal operation

    def avoid_collision(self, environment: env.map.Map):
        # get list of agents for which there is collision potential/danger
        collision_danger_agents = []
        for a in environment.agents:
            if self is not a and self.collision_potential(a):
                collision_danger_agents.append(a)

        not_dodging = True
        # TODO: don't do this based on distance to seed but based on some other measure
        for a in collision_danger_agents:
            not_dodging = not_dodging and \
                          simple_distance(a.geometry.position, self.current_seed.geometry.position) > \
                          simple_distance(self.geometry.position, self.current_seed.geometry.position)
            # aprint(self.id, "Distance to current goal: {:3f} (other agent) and {:3f} (self)".format(
            #     simple_distance(a.geometry.position, self.current_seed.geometry.position),
            #     simple_distance(self.geometry.position, self.current_seed.geometry.position)))

        # for a in collision_danger_agents:
        #     if a.current_path is not None and self.current_path is not None:
        #         not_dodging = not_dodging and \
        #                                     simple_distance(a.geometry.position, a.current_path.next()) > \
        #                                     simple_distance(self.geometry.position, self.current_path.next())

        # self does not even move towards the structure, evade
        if self.current_task in [Task.LAND, Task.FETCH_BLOCK, Task.FINISHED, Task.MOVE_UP_LAYER]:
            not_dodging = True

        # if self carries no block and some of the others carry one, then simply evade
        if (self.current_block is None or self.current_block.geometry not in self.geometry.following_geometries) and \
                any([a.current_block is not None and a.current_block.geometry in a.geometry.following_geometries
                     for a in collision_danger_agents]):
            not_dodging = False

        if not_dodging:
            self.current_task = self.previous_task
            self.task_history.append(self.current_task)
            # if self.current_task == Task.LAND:
            #     self.current_path = None
            return
        else:
            # decide on direction to go into
            if self.current_path is None:
                self.current_path = Path()
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                                               self.current_path.current_index)
            else:
                next_position = self.current_path.next()
                self.current_path.add_position([next_position[0], next_position[1],
                                                self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                                               self.current_path.current_index)
            # TODO: instead of simply going up, also move in direction of next goal

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                pass
        else:
            self.geometry.position = self.geometry.position + current_direction

    def land(self, environment: env.map.Map):
        if not self.LAND_CALLED_FIRST_TIME:
            self.LAND_CALLED_FIRST_TIME = True
            if self.current_path is not None:
                aprint(self.id, "WHAT THE FUCK IS GOING ON HERE")
                aprint(self.id, "breakpoint")

        if self.current_path is None:
            # find some unoccupied location on the outside of the construction zone and land there
            start_x = environment.offset_origin[0] - self.geometry.size[0]
            end_x = start_x + self.target_map.shape[2] * Block.SIZE + self.geometry.size[0] * 1.5
            start_y = environment.offset_origin[1] - self.geometry.size[1]
            end_y = start_y + self.target_map.shape[1] * Block.SIZE + self.geometry.size[1] * 1.5
            candidate_x = candidate_y = -1
            while candidate_x < 0 or (start_x <= candidate_x <= end_x and start_y <= candidate_y <= end_y):
                candidate_x = random.uniform(0.0, environment.environment_extent[0])
                candidate_y = random.uniform(0.0, environment.environment_extent[1])

            candidate_x = environment.environment_extent[0]
            candidate_y = environment.environment_extent[1]

            land_level_z = Block.SIZE * (self.current_structure_level + 2) + \
                           self.geometry.size[2] / 2 + self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, self.geometry.size[2] / 2])
            aprint(self.id, "SETTING PATH TO LAND: {}".format(self.current_path.positions))

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        # if self.collision_possible:
        #     for a in environment.agents:
        #         if self is not a and self.collision_potential(a):
        #             force_field_vector = np.array([0.0, 0.0, 0.0])
        #             force_field_vector += (self.geometry.position - a.geometry.position)
        #             force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
        #             # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
        #             # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
        #             force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
        #             current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                if abs(self.geometry.position[2] - Block.SIZE / 2) > Block.SIZE / 2:
                    aprint(self.id, "FINISHED WITHOUT LANDING")
                    aprint(self.id, "PATH POSITIONS: {}\nPATH INDEX: {}".format(self.current_path.positions,
                                                                                self.current_path.current_index))
                    aprint(self.id, "POSITION IN QUESTION: {}".format(
                        self.current_path.positions[self.current_path.current_index]))
                    aprint(self.id, "LAST 10 TASKS: {}".format(self.task_history[-10:]))
                    aprint(self.id, "HAPPENING IN AGENT: {}".format(self))
                    aprint(self.id, "placeholder")
                if self.current_block is not None:
                    aprint(self.id, "LANDING WITH BLOCK STILL ATTACHED")
                    aprint(self.id, "LAST 20 TASKS: {}".format(self.task_history[-10:]))
                    aprint(self.id, "what")
                self.current_task = Task.FINISHED
                self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

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
        for z in range(hole_map.shape[0]):
            hole_boundaries.append([])
            for m in valid_markers[z]:
                coord_list = []
                locations = np.where(hole_map == m)
                boundary_search(hole_map[z], z, locations[1][0], locations[2][0], m, None, coord_list)
                coord_list = tuple(np.moveaxis(np.array(coord_list), -1, 0))
                hole_boundaries[z].append(coord_list)

        # TODO: ACCOUNT FOR MULTIPLE COMPONENTS

        a_dummy_copy = np.copy(hole_map)
        hole_corners = []
        closing_corners = []
        closing_corner_boundaries = []
        hole_boundary_coords = dict()
        for z in range(hole_map.shape[0]):
            hole_corners.append([])
            # TODO: implement this with more fitting data structures
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
                    if (len(current_outer) + len(current_inner)) % 2 == 0:
                        sorted_by_y = sorted(range(len(current_outer)), key=lambda e: current_outer[e][1], reverse=True)
                        sorted_by_x = sorted(sorted_by_y, key=lambda e: current_outer[e][0], reverse=True)
                        current_outer = [current_outer[i] for i in sorted_by_x]
                        current_boundary = [current_boundary[i] for i in sorted_by_x]
                        closing_corners[z][cm].append(current_outer[0])
                        closing_corner_boundaries[z][cm].append(current_boundary[0])
                        # TODO (possibly): not sure if this will always work might be better to still have a closing
                        # corner; if the open corner is adjacent to another hole, I am not sure it always works

                # corner is missing -> inside corners will also have to be counted
                # if (len(outer_corner_coord_list) + len(inner_corner_coord_list)) % 2 == 0:
                #     # else there must be some "open" corner and it should not be necessary to explicitly
                #     # leave a corner open -> now choose the "right and upper"-most corner
                #     sorted_by_y = sorted(range(len(outer_corner_coord_list)),
                #                          key=lambda e: outer_corner_coord_list[e][1],
                #                          reverse=True)
                #     sorted_by_x = sorted(sorted_by_y, key=lambda e: outer_corner_coord_list[e][0], reverse=True)
                #     outer_corner_coord_list = [outer_corner_coord_list[i] for i in sorted_by_x]
                #     corner_boundary_list = [corner_boundary_list[i] for i in sorted_by_x]
                #     closing_corners[z].append(outer_corner_coord_list[0])
                #     closing_corner_boundaries[z].append(corner_boundary_list[0])

                hole_corners[z].append(outer_corner_coord_list)

        aprint(self.id, "HERE, LOOK HERE")
        print_map(a_dummy_copy)

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

        print("CLOSING CORNERS: {}".format(closing_corners))
        print("CLOSING CORNER BOUNDARIES: {}".format(closing_corner_boundaries))

        boundary_map = np.zeros_like(hole_map)
        for z in range(hole_map.shape[0]):
            for x, y, z in all_hole_boundaries[z]:
                boundary_map[z, y, x] = 1

        return closing_corners, hole_map, hole_boundary_coords, closing_corner_boundaries

    def advance(self, environment: env.map.Map):
        # determine current task:
        # fetch block
        # bring block to structure (i.e. until it comes into sight)
        # find attachment site
        # place block
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]

        self.agent_statistics.step(environment)

        if self.current_task == Task.AVOID_COLLISION:
            self.avoid_collision(environment)
        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.MOVE_TO_PERIMETER:
            self.move_to_perimeter(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.MOVE_UP_LAYER:
            self.move_up_layer(environment)
        elif self.current_task == Task.LAND:
            # aprint(self.id, "LAND ROUTINE IS ACTUALLY CALLED IN AGENT {}".format(self))
            self.land(environment)

        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    # aprint(self.id, "INITIATING HIGH-LEVEL COLLISION AVOIDANCE")
                    self.previous_task = self.current_task
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    break

        # else:
        #     for a in environment.agents:
        #         if self is not a and self.collision_potential(a):
        #             break
        #     else:
        #         self.current_task = self.previous_task
        #         self.task_history.append(self.current_task)


class ShortestPathAgent(PerimeterFollowingAgent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(ShortestPathAgent, self).__init__(position, size, target_map, required_spacing)

    # need to change the following:
    # - find_attachment_site
    # - might be something else that might not work then?

    def find_attachment_site(self, environment: env.map.Map):
        seed_block = self.current_seed

        if self.current_component_marker != -1:
            # checking below whether the current component (as designated by self.current_component_marker) is finished
            tm = np.zeros_like(self.target_map[self.current_structure_level])
            np.place(tm, self.component_target_map[self.current_structure_level] ==
                     self.current_component_marker, 1)
            om = np.copy(environment.occupancy_map[self.current_structure_level])
            np.place(om, om > 0, 1)
            np.place(om, self.component_target_map[self.current_structure_level] !=
                     self.current_component_marker, 0)
            if np.array_equal(om, tm):
                aprint("CURRENT COMPONENT FINISHED")
                # current component completed, see whether there is a different one that should be constructed
                candidate_components = []
                for marker in np.unique(self.component_target_map[self.current_structure_level]):
                    if marker != 0 and marker != self.current_component_marker:
                        subset_indices = np.where(
                            self.component_target_map[self.current_structure_level] == marker)
                        candidate_values = environment.occupancy_map[self.current_structure_level][
                            subset_indices]
                        # the following check means that on the occupancy map, this component still has all
                        # positions unoccupied, i.e. no seed has been placed -> this makes it a candidate
                        # for placing the currently transported seed there
                        if np.count_nonzero(candidate_values == 0) > 0:
                            candidate_components.append(marker)

                if len(candidate_components) > 0:
                    # choosing one of the candidate components to continue constructing
                    self.current_component_marker = random.sample(candidate_components, 1)[0]
                    # getting the coordinates of those positions where the other component already has blocks
                    correct_locations = np.where(
                        self.component_target_map[self.current_structure_level] == self.current_component_marker)
                    correct_locations = list(zip(correct_locations[0], correct_locations[1]))
                    occupied_locations = np.where(environment.occupancy_map[self.current_structure_level] != 0)
                    occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                    occupied_locations = list(set(occupied_locations).intersection(correct_locations))
                    for b in environment.placed_blocks:
                        if b.is_seed and b.grid_position[2] == self.current_structure_level \
                                and (b.grid_position[1], b.grid_position[0]) in occupied_locations:
                            self.current_seed = b
                            self.current_path = None
                            self.current_task = Task.TRANSPORT_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_visited_sites = None
                            break
                else:
                    if self.current_structure_level >= self.target_map.shape[0] - 1:
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_visited_sites = None
                    else:
                        self.current_task = Task.MOVE_UP_LAYER
                        self.task_history.append(self.current_task)
                        self.current_structure_level += 1
                        self.current_visited_sites = None
                    self.current_path = None
                    self.current_component_marker = -1
                return

        # find an attachment site by finding the closest allowed/legal attachment site and following the path
        # over the blocks there (this is done for "realism", because the blocks are needed for orientation; it
        # could be worth considering just taking the shortest direct flight path, orientation oneself using a
        # grid on the construction site floor)
        if self.current_path is None:
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                      environment.occupancy_map[self.current_structure_level],
                                                      component_marker=self.current_component_marker)

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(environment.occupancy_map[self.current_structure_level])

            # convert to coordinates
            attachment_sites = np.where(attachment_sites == 1)
            attachment_sites = list(zip(attachment_sites[1], attachment_sites[0]))

            # find the closest one
            shortest_paths = []
            for x, y in attachment_sites:
                occupancy_map_copy[y, x] = 1
                sp = shortest_path(occupancy_map_copy, (self.current_grid_position[0],
                                                        self.current_grid_position[1]), (x, y))
                occupancy_map_copy[y, x] = 0
                shortest_paths.append(sp)
            shortest_paths = sorted(shortest_paths, key=lambda x: len(x))
            # aprint("SHORTEST PATHS: {}".format(shortest_paths))

            self.current_grid_position = np.array(
                [shortest_paths[0][-1][0], shortest_paths[0][-1][1], self.current_structure_level])
            self.current_path = Path()
            for x, y in shortest_paths[0]:
                current_coordinate = [environment.offset_origin[0] + Block.SIZE * x,
                                      environment.offset_origin[0] + Block.SIZE * y,
                                      self.geometry.position[2]]
                self.current_path.add_position(current_coordinate)

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

        # do (force field, not planned) collision avoidance
        if self.collision_possible:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    force_field_vector = np.array([0.0, 0.0, 0.0])
                    force_field_vector += (self.geometry.position - a.geometry.position)
                    force_field_vector /= sum(np.sqrt(force_field_vector ** 2))
                    # force_field_vector = rotation_2d(force_field_vector, np.pi / 4)
                    # force_field_vector = rotation_2d_experimental(force_field_vector, np.pi / 4)
                    force_field_vector *= 50 / simple_distance(self.geometry.position, a.geometry.position)
                    current_direction += 2 * force_field_vector

        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP

        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                # since the path is planned completely at the beginning, this signals arriving at the site itself
                # TODO: figure out whether this scheme allows holes
                # otherwise it would be pretty simple to restrict attachment
                # sites to the same region as with perimeter search

                self.current_task = Task.PLACE_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None

                # need a way to check whether the current level has been completed already
                tm = np.copy(self.target_map[self.current_structure_level])
                np.place(tm, tm == 2, 1)
                om = np.copy(environment.occupancy_map[self.current_structure_level])
                np.place(om, om == 2, 1)
                if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                    self.current_task = Task.MOVE_UP_LAYER
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_structure_level += 1
                    self.current_path = None
                    self.next_seed = self.current_block
            else:
                # also need to account for reaching a point on the shortest path and thereby moving the grid position
                # actually, might just be able to set it at the start?
                pass
        else:
            self.geometry.position = self.geometry.position + current_direction

    def advance(self, environment: env.map.Map):
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]
        if self.current_task == Task.MOVE_TO_PERIMETER:
            self.current_task = Task.FIND_ATTACHMENT_SITE

        self.agent_statistics.step(environment)

        if self.current_task == Task.AVOID_COLLISION:
            self.avoid_collision(environment)
        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.MOVE_UP_LAYER:
            self.move_up_layer(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    self.previous_task = self.current_task
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    break
