import numpy as np
import logging
import random
import env.map
from enum import Enum
from typing import List
from abc import ABCMeta, abstractmethod
from env.block import Block
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance

np.seterr(divide='ignore', invalid='ignore')


class AgentType(Enum):
    RANDOM_WALK_AGENT = 0


class Task(Enum):
    FETCH_BLOCK = 0
    TRANSPORT_BLOCK = 1
    MOVE_TO_PERIMETER = 2
    FIND_ATTACHMENT_SITE = 3
    PLACE_BLOCK = 4
    MOVE_UP_LAYER = 5
    FINISHED = 6


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
        self.target_map = target_map
        self.component_target_map = None
        self.required_spacing = required_spacing

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
        self.current_component_marker = -1
        self.backup_grid_position = None

        self.seed_on_perimeter = False

    @abstractmethod
    def advance(self, environment: env.map.Map):
        pass

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

    def check_target_map(self, position, comparator=lambda x: x != 0):
        if any(position < 0):
            return comparator(0)
        try:
            temp = self.target_map[tuple(np.flip(position, 0))]
        except IndexError:
            return comparator(0)
        else:
            val = comparator(temp)
            return val


class CollisionAvoidanceAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 10):
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
                 required_spacing: float = 10):
        super(PerimeterFollowingAgent, self).__init__(position, size, target_map, required_spacing)
        self.block_locations_known = True
        self.structure_location_known = True
        self.component_target_map = self.split_into_components()
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
                    self.current_task = Task.FINISHED
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
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reach (i.e. the block can be picked up)
            if not ret:
                self.geometry.attached_geometries.append(self.current_block.geometry)
                self.current_task = Task.TRANSPORT_BLOCK
                self.current_path = None
        else:
            self.geometry.position += current_direction

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

                    # since MOVE_TO_PERIMETER is used, the direction to go into is initialised randomly
                    # a better way of doing it would probably be to take the shortest path to the perimeter
                    # using the available knowledge about the current state of the structure (this could even
                    # include leading the agent to an area where it is likely to find an attachment site soon)
                    self.current_grid_direction = np.array(
                        random.sample([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], 1)[0])
                else:
                    self.current_task = Task.FIND_ATTACHMENT_SITE
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
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.current_grid_position += self.current_grid_direction
                if not environment.check_over_structure(self.geometry.position, self.current_structure_level):
                    # have reached perimeter
                    self.current_task = Task.FIND_ATTACHMENT_SITE
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

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                # corner of the current block reached, assess next action
                if self.check_target_map(self.current_grid_position) and \
                        (environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction) or
                         (self.current_row_started and (self.check_target_map(self.current_grid_position +
                                                                              self.current_grid_direction,
                                                                              lambda x: x == 0) or
                                                        environment.check_occupancy_map(
                                                            self.current_grid_position + self.current_grid_direction +
                                                            np.array([-self.current_grid_direction[1],
                                                                      self.current_grid_direction[0], 0],
                                                                     dtype="int32"), lambda x: x == 0)))):
                    # site should be occupied AND
                    # 1. site ahead has a block (inner corner) OR
                    # 2. the current "identified" row ends (i.e. no chance of obstructing oneself)
                    self.current_task = Task.PLACE_BLOCK
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
                environment.place_block(self.current_grid_position, self.current_block)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                self.current_block.placed = True
                self.current_block.grid_position = self.current_grid_position
                self.current_block = None
                self.current_path = None

                if self.current_component_marker != -1:
                    print("CASE 1 AFTER PLACING")
                    tm = np.zeros_like(self.target_map[self.current_structure_level])
                    np.place(tm, self.component_target_map[self.current_structure_level] ==
                             self.current_component_marker, 1)
                    om = np.copy(environment.occupancy_map[self.current_structure_level])
                    np.place(om, om > 0, 1)
                    np.place(om, self.component_target_map[self.current_structure_level] !=
                             self.current_component_marker, 0)
                    print("tm: {}".format(tm))
                    print("om: {}".format(om))
                    if np.array_equal(om, tm):
                        # current component completed, see whether there is a different one that should constructed
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
                            self.current_component_marker = random.sample(candidate_components, 1)
                            print("After placing block: unfinished components left, choosing {}".format(self.current_component_marker))
                            # getting the coordinates of those positions where the other component already has blocks
                            correct_locations = np.where(self.component_target_map[self.current_structure_level] == self.current_component_marker)
                            correct_locations = list(zip(correct_locations[0], correct_locations[1]))
                            occupied_locations = np.where(environment.occupancy_map[self.current_structure_level] != 0)
                            occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                            occupied_locations = list(set(occupied_locations).intersection(correct_locations))
                            for b in environment.placed_blocks:
                                if b.is_seed and (b.grid_position[1], b.grid_position[0]) in occupied_locations:
                                    self.current_seed = b
                                    print("New seed location: {}".format(self.current_seed))
                                    self.current_path = None
                                    self.current_task = Task.FETCH_BLOCK
                                    break
                        else:
                            if self.current_structure_level >= self.target_map.shape[0] - 1:
                                self.current_task = Task.FINISHED
                            self.current_component_marker = -1
                            self.current_task = Task.MOVE_UP_LAYER
                            self.current_structure_level += 1
                    else:
                        self.current_task = Task.FETCH_BLOCK
                else:
                    print("CASE 2 AFTER PLACING")
                    tm = np.copy(self.target_map[self.current_structure_level])
                    np.place(tm, tm == 2, 1)
                    om = np.copy(environment.occupancy_map[self.current_structure_level])
                    np.place(om, om == 2, 1)
                    if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                        self.current_task = Task.MOVE_UP_LAYER
                        self.current_structure_level += 1
                    elif np.array_equal(environment.occupancy_map[self.current_structure_level], tm):
                        self.current_task = Task.FINISHED
                        self.logger.info("Construction finished (3).")
                    else:
                        self.current_task = Task.FETCH_BLOCK
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
                            if allowed_position(i, j + 1) and self.target_map[self.current_structure_level, i, j + 1] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("up")
                            if allowed_position(i, j - 1) and self.target_map[self.current_structure_level, i, j - 1] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("down")
                            if allowed_position(i - 1, j) and self.target_map[self.current_structure_level, i - 1, j] > 0:
                                current_adjacent += 1
                                current_free_edges.remove("left")
                            if allowed_position(i + 1, j) and self.target_map[self.current_structure_level, i + 1, j] > 0:
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
                    if 0 in unique_values:
                        unique_values.remove(0)
                    self.current_component_marker = random.sample(unique_values, 1)
                    occupied_locations = np.where(
                        self.component_target_map[self.current_structure_level] == self.current_component_marker)
                    print(np.unique(self.component_target_map[self.current_structure_level]))
                    print(self.current_component_marker, occupied_locations)
                    occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                    # still using np.nonzero here because it does not matter what the supporting block belongs to
                    supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                    supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                    occupied_locations = [x for x in occupied_locations if x in supported_locations]
                    coordinates = random.sample(occupied_locations, 1)
                    min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]
                    print("MIN COORDINATES: {}".format(min_coordinates))

                min_block = None
                if self.current_block is None:
                    print("DECIDING ON NEW SEED BLOCK FOR NEXT LAYER")
                    min_distance = float("inf")
                    for b in environment.blocks:
                        temp = self.geometry.distance_2d(b.geometry)
                        if not (b.is_seed or b.placed or any(b is a.current_block for a in environment.agents)) \
                                and temp < min_distance:
                            min_block = b
                            min_distance = temp
                    print(min_block)
                    if min_block is None:
                        self.logger.info("Construction finished (2).")
                        self.current_task = Task.FINISHED
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
                self.current_path.add_position([self.next_seed.geometry.position[0], self.next_seed.geometry.position[1],
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

            print("self.next_seed:", self.next_seed)
            print("self.next_seed.grid_position:", self.next_seed.grid_position)
            print("environment.offset_origin:", environment.offset_origin)
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
                        print("next_seed = None bc self.seed_on_perimeter")
                        self.next_seed = None
                        self.current_task = Task.TRANSPORT_BLOCK
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
                if environment.check_over_structure(self.geometry.position) \
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
                        self.current_component_marker = random.sample(candidate_components, 1)
                        occupied_locations = np.where(
                            self.component_target_map[self.current_structure_level] == self.current_component_marker)
                        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                        supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                        occupied_locations = [x for x in occupied_locations if x in supported_locations]
                        coordinates = random.sample(occupied_locations, 1)
                        min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]

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
                        self.current_block.color = "green"
                        self.current_block.is_seed = False
                        self.current_path = None
                        self.current_seed = None
                        # this needs a-changing:
                        y, x = np.where(environment.occupancy_map[self.current_structure_level] != 0)
                        for b in environment.placed_blocks:
                            for c in zip(x, y):
                                if b.is_seed and all([b.grid_position[i] == c[i] for i in range(2)]):
                                    self.current_seed = b
                                    break
                        print("next_seed = None bc block should be placed rather than used as seed")
                        self.next_seed = None
                        self.current_task = Task.TRANSPORT_BLOCK
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
                    print("CASE 1")
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
                    print("CASE 2")
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
                    print("CASE 3")
                    # TODO: clean up this mess with the previous seed position
                    self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                    Block.SIZE * (self.current_structure_level + 1) +
                                                    self.geometry.size[2] / 2])
                else:
                    print("CASE 4")
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
                        self.current_component_marker = random.sample(candidate_components, 1)
                        occupied_locations = np.where(
                            self.component_target_map[self.current_structure_level] == self.current_component_marker)
                        occupied_locations = list(zip(occupied_locations[0], occupied_locations[1]))
                        supported_locations = np.nonzero(self.target_map[self.current_structure_level - 1])
                        supported_locations = list(zip(supported_locations[0], supported_locations[1]))
                        occupied_locations = [x for x in occupied_locations if x in supported_locations]
                        coordinates = random.sample(occupied_locations, 1)
                        min_coordinates = [coordinates[0][1], coordinates[0][0], self.current_structure_level]

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
                            self.current_task = Task.FINISHED
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
                            self.current_structure_level += 1
                            self.current_component_marker = -1
                        elif np.array_equal(om, tm):
                            self.current_task = Task.FINISHED
                            self.logger.info("Construction finished (1).")
                        else:
                            self.current_task = Task.FETCH_BLOCK
        else:
            self.geometry.position = self.geometry.position + current_direction

        # move up one level in the structure and place the seed block

        # continue normal operation
        pass

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

    def advance(self, environment: env.map.Map):
        # determine current task:
        # fetch block
        # bring block to structure (i.e. until it comes into sight)
        # find attachment site
        # place block
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]

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

