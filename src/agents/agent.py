import numpy as np
import logging
import random
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
    MOVE_UP_LAYER = 4
    FINISHED = 5


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


class SimpleSeededAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 10):
        super(SimpleSeededAgent, self).__init__(position, size, target_map, required_spacing)
        self.block_locations_known = True
        self.structure_location_known = True
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
                seed_location = self.current_seed.geometry.position
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
                self.current_grid_position = np.copy(self.current_seed.grid_position)
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

        seed_block = self.current_seed

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
                        self.logger.debug("CASE 3: Reached corner of structure, turning counter-clockwise. {} {}".format(self.current_grid_position, self.current_grid_direction))
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
                    self.current_seed = self.current_block
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
            init_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level - \
                     (self.required_spacing + self.geometry.size[2] / 2)
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
            def allowed_position(y, x):
                return 0 <= y < self.target_map[self.current_structure_level].shape[0] and \
                       0 <= x < self.target_map[self.current_structure_level].shape[1]

            # determine one block to serve as seed (a position in the target map)
            min_adjacent = 4
            min_coords = [0, 0, self.current_structure_level]  # [x, y]
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
                        min_coords = [j, i, self.current_structure_level]
                        min_free_edges = current_free_edges

            # fetch a block and mark it as seed/fetch a seed block
            min_block = None
            if self.current_block is None:
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
            else:
                min_block = self.current_block

            min_block.is_seed = True
            min_block.color = "red"
            min_block.seed_marked_edge = random.choice(min_free_edges)

            # first add a point to get up to the level of movement for fetching blocks
            # which is one above the current construction level
            self.current_path = Path()
            fetch_level_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level + \
                            self.geometry.size[2] / 2 + self.required_spacing
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            if self.current_block is None:
                self.current_path.add_position(
                    [min_block.geometry.position[0], min_block.geometry.position[1], fetch_level_z])
                self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1],
                                                min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[
                                                    2] / 2])
            else:
                transport_level_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level - \
                                    (self.required_spacing + self.geometry.size[2] / 2)
                destination_x = self.current_seed.geometry.position[0]
                destination_y = self.current_seed.geometry.position[1]
                self.current_path.add_position(
                    [self.geometry.position[0], self.geometry.position[1], transport_level_z])
                self.current_path.add_position([destination_x, destination_y, transport_level_z])

            self.next_seed = min_block
            self.next_seed.grid_position = np.array(min_coords)
            self.current_grid_position = self.next_seed.grid_position
        else:
            pass

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            destination = [Block.SIZE * self.next_seed.grid_position[0] + environment.offset_origin[0],
                           Block.SIZE * self.next_seed.grid_position[1] + environment.offset_origin[1]]
            if self.current_block is not None and all([self.geometry.position[i] == destination[i] for i in range(2)]):
                # check whether seed block has already been placed
                print("In position: {}".format(self.next_seed.grid_position))
                print(environment.occupancy_map)
                if environment.check_occupancy_map(self.next_seed.grid_position):
                    print("Already occupied")
                    self.current_block.color = "green"
                    self.current_block.is_seed = False
                    self.current_block.seed_marked_edge = "down"
                    self.current_path = None
                    self.current_seed = None
                    for b in environment.placed_blocks:
                        if b.is_seed and all([b.grid_position[i] == self.next_seed.grid_position[i] for i in range(3)]):
                            self.current_seed = b
                            break
                    self.next_seed = None
                    self.current_task = Task.TRANSPORT_BLOCK
                    return
            else:
                print("Own position: {}".format(self.geometry.position))
                print("Destination: {}".format(destination))
                print("Seed grid: {}".format(self.next_seed.grid_position))

            # if the final point on the path has been reach, search for attachment site should start
            if not ret:
                # also need to check whether there is already a seed on that layer or nah
                # -> to simplify things, can assume that agents would all choose the same seed location
                if self.current_block is None:
                    print("CASE 1")
                    self.current_block = self.next_seed
                    self.geometry.attached_geometries.append(self.current_block.geometry)
                    transport_level_z = Block.SIZE + (self.current_structure_level + 1) * self.spacing_per_level - \
                                        (self.required_spacing + self.geometry.size[2] / 2)

                    # the following two basically assume perfect knowledge (could be done using unique blocks),
                    # but it makes more sense (here) to go to the seed on the previous level first
                    destination_x = Block.SIZE * self.current_grid_position[0] + environment.offset_origin[0]
                    destination_y = Block.SIZE * self.current_grid_position[1] + environment.offset_origin[1]

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
                else:
                    print("CASE 3")
                    self.geometry.attached_geometries.remove(self.current_block.geometry)
                    self.current_seed = self.next_seed
                    self.current_block.placed = True
                    self.current_block.grid_position = self.current_grid_position
                    environment.place_block(self.current_grid_position, self.current_block)
                    self.current_block = None
                    self.current_path = None
                    self.next_seed = None

                    tm = np.copy(self.target_map[self.current_structure_level])
                    np.place(tm, tm == 2, 1)
                    om = np.copy(environment.occupancy_map[self.current_structure_level])
                    np.place(om, om == 2, 1)
                    if np.array_equal(om, tm) and self.target_map.shape[0] > self.current_structure_level + 1:
                        self.current_task = Task.MOVE_UP_LAYER
                        self.current_structure_level += 1
                    elif np.array_equal(environment.occupancy_map[self.current_structure_level], tm):
                        self.current_task = Task.FINISHED
                        self.logger.info("Construction finished (1).")
                    else:
                        self.current_task = Task.FETCH_BLOCK
        else:
            self.geometry.position = self.geometry.position + current_direction

        # move up one level in the structure and place the seed block

        # continue normal operation
        pass

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
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.MOVE_UP_LAYER:
            self.move_up_layer(environment)
