import numpy as np
import logging
import env.map
from enum import Enum
from typing import List
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
    FIND_ATTACHMENT_SITE = 2
    PLACE_BLOCK = 3
    FINISHED = 4


class Agent:

    MOVEMENT_PER_STEP = 5

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.geometry = GeomBox(position, size, 0.0)
        self.target_map = target_map
        self.required_spacing = required_spacing
        self.spacing_per_level = self.geometry.size[2] + Block.SIZE + 2 * self.required_spacing

        self.local_occupancy_map = None
        self.current_block = None
        self.current_trajectory = None
        self.current_task = Task.FETCH_BLOCK
        self.current_path = None
        self.current_structure_level = 0
        self.current_grid_position = None  # closest grid position if at structure
        self.current_grid_direction = None
        self.current_row_started = False

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

    def advance(self, environment: env.map.Map):
        pass

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


class RandomWalkAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 10):
        super(RandomWalkAgent, self).__init__(position, size, target_map, required_spacing)
        self.block_locations_known = True
        self.structure_location_known = True

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
                    self.logger.info("Construction finished.")
                    self.current_task = Task.FINISHED
                    return
                min_block.color = "green"

                # first add a point to get up to the level of movement for fetching blocks
                # which is one above the current construction level
                self.current_path = Path()
                fetch_level_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level + \
                                self.geometry.size[2] / 2 + self.required_spacing
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
                self.current_path.add_position(
                    [min_block.geometry.position[0], min_block.geometry.position[1], fetch_level_z])
                self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1],
                                                min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[
                                                    2] / 2])
                self.current_block = min_block
        else:
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
                transport_level_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level - \
                                    (self.required_spacing + self.geometry.size[2] / 2)
                seed_location = environment.seed_position()
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                transport_level_z])
                self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z])
        else:
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

            # if the final point on the path has been reach, search for attachment site should start
            if not ret:
                self.current_task = Task.FIND_ATTACHMENT_SITE
                self.current_grid_position = np.copy(environment.blocks[0].grid_position)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        # determine path to structure location

        # avoid collisions until structure/seed comes in sight
        pass

    def find_attachment_site(self, environment: env.map.Map):
        # structure should be in view at this point

        # if allowed attachment site can be determined from visible structure:
        #   determine allowed attachment site given knowledge of target structure
        # else:
        #   use current searching scheme to find legal attachment site

        seed_block = environment.blocks[0]

        # orientation happens counter-clockwise -> follow seed edge in that direction once its reached
        # can either follow the perimeter itself or just fly over blocks (do the latter for now)
        if self.current_path is None:
            # path only leads to next possible site (assumption for now is that only block below is known)
            # first go to actual perimeter of structure (correct side of seed block)
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

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                # print("Check: {}".format(self.check_target_map(self.current_grid_position)))
                # print("Target map: \n{}".format(self.target_map))
                # print("Position: {}".format(self.current_grid_position))
                # print("Row started: {}".format(self.current_row_started))
                # print("Position ahead: {}".format(self.current_grid_position + self.current_grid_direction))
                # print("Occupancy: {}".format(environment.check_occupancy_map(self.current_grid_position +
                #                                                              self.current_grid_direction,
                #                                                              lambda x: x == 0)))

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
                    self.logger.info(log_string)
                else:
                    # site should not be occupied -> determine whether to turn a corner or continue, options:
                    # 1. turn right (site ahead occupied)
                    # 2. turn left
                    # 3. continue straight ahead along perimeter
                    # print(environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction +
                    #                                      np.array([-self.current_grid_direction[1],
                    #                                                self.current_grid_direction[0], 0],
                    #                                               dtype="int32"),
                    #                                      lambda x: x == 0))
                    # print(self.current_grid_position + self.current_grid_direction +
                    #                                      np.array([-self.current_grid_direction[1],
                    #                                                self.current_grid_direction[0], 0],
                    #                                               dtype="int32"))
                    # print(np.array([-self.current_grid_direction[1],
                    #                                                self.current_grid_direction[0], 0],
                    #                                               dtype="int32"))
                    # print(environment.occupancy_map)
                    if environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction):
                        # turn right
                        # print(self.current_grid_position)
                        # print(self.current_grid_direction)
                        # print(environment.check_occupancy_map(self.current_grid_position + self.current_grid_direction))
                        self.current_grid_direction = np.array([self.current_grid_direction[1],
                                                                -self.current_grid_direction[0], 0],
                                                               dtype="int32")
                        # self.current_grid_position += self.current_grid_direction
                        self.logger.info("CASE 2: Position straight ahead occupied, turning clockwise.")
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
                        self.logger.info("CASE 3: Reached corner of structure, turning counter-clockwise.")
                        self.current_path.add_position(reference_position + Block.SIZE * self.current_grid_direction)
                    else:
                        # otherwise site "around the corner" occupied -> continue straight ahead
                        self.current_grid_position += self.current_grid_direction
                        self.current_row_started = True
                        self.logger.info("CASE 4: Adjacent positions ahead occupied, continuing to follow perimeter.")
                        self.current_path.add_position(self.geometry.position + Block.SIZE * self.current_grid_direction)

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
            placement_x = environment.offset_origin[0] + Block.SIZE * self.current_grid_position[0]
            placement_y = environment.offset_origin[1] + Block.SIZE * self.current_grid_position[1]
            placement_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level - \
                          (self.required_spacing + self.geometry.size[2] / 2)
            self.current_path = Path()
            self.current_path.add_position([placement_x, placement_y, placement_z])
            self.current_path.add_position([placement_x, placement_y, Block.SIZE + self.geometry.size[2] / 2])

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
                print(environment.occupancy_map)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                self.current_block.placed = True
                self.current_block.grid_position = self.current_grid_position
                self.current_block = None
                self.current_task = Task.FETCH_BLOCK
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        # if interrupted, search for new attachment site again
        pass

    def advance(self, environment: env.map.Map):
        # determine current task:
        # fetch block
        # bring block to structure (i.e. until it comes into sight)
        # find attachment site
        # place block

        if self.current_task == Task.FETCH_BLOCK:
            self.logger.debug("Current task: FETCH_BLOCK")
            self.fetch_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.logger.debug("Current task: TRANSPORT_BLOCK")
            self.transport_block(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.logger.debug("Current task: FIND_ATTACHMENT_SITE")
            self.find_attachment_site(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.logger.debug("Current task: PLACE_BLOCK")
            self.place_block(environment)

        # determine information known about the structure
        # self.logger.debug("Determining accessible information about structure.")

        # determine whether collision avoidance is necessary
        # self.logger.debug("Determining whether to avoid other agents.")

        # update local occupancy map
        # self.logger.debug("Updating local occupancy map if new information is available.")

        # decide whether to drop or pick up a block
        # self.logger.debug("Dropping or picking up block if possible.")

        # determine current trajectory
        # self.logger.debug("Determining trajectory.")
