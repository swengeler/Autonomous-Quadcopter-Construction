import random
from abc import abstractmethod

import env.map
from agents.agent import Task, Agent, check_map
from env.block import Block
from geom.path import Path
from geom.shape import *
from geom.util import simple_distance


class LocalKnowledgeAgent(Agent):
    """
    A super class encapsulating the information and functionality expected to be used by all agents with
    local knowledge of the environment.
    """

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(LocalKnowledgeAgent, self).__init__(position, size, target_map, required_spacing, printing_enabled)
        self.seed_if_possible_enabled = True
        self.seeding_strategy = "distance_self"  # others being: "distance_center", "agent_count"
        self.find_next_component_count = []

    def update_local_occupancy_map(self, environment: env.map.Map):
        """
        Update the agent's knowledge of the current state of the structure given its current position.

        Depending on additional blocks the agent could observe since the last update to the local occupancy
        map, update the occupancy map "around" the current grid position (the 9 blocks below the agent) and
        also extrapolate information about the state of the structure from that.

        :param environment: the environment the agent operates in
        """

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
        # this point, which would actually also be the case if there is a gap of more than 1:
        # any continuous row/column in the target map between two occupied locations in the local
        # occupancy map should be assumed to be filled out already
        current_occupancy_map = self.local_occupancy_map[self.current_grid_position[2]]
        current_target_map = self.target_map[self.current_grid_position[2]]
        for y in range(current_target_map.shape[0]):
            for x in range(current_target_map.shape[1]):
                if current_occupancy_map[y, x] != 0:
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

        # print out if the method of extrapolating information implemented above results in a mistake
        for z in range(self.local_occupancy_map.shape[0]):
            for y in range(self.local_occupancy_map.shape[1]):
                for x in range(self.local_occupancy_map.shape[2]):
                    if self.local_occupancy_map[z, y, x] != 0 and environment.occupancy_map[z, y, x] == 0:
                        self.aprint(
                            "Local occupancy map occupied at {} where environment not occupied.".format((x, y, z)))

    def check_stashes(self, environment: env.map.Map):
        """
        Move with the goal of checking whether block stashes still contain blocks.

        This method is called if the current task is CHECK_STASHES. If the agent has not planned a path yet, it
        determines the stashes which have previously been determined to be empty and plans a path visiting all of them.
        When a stash has been reached, the agent checks whether there are any blocks left there and if so switches to
        the task FETCH_BLOCK. When all stashes have been visited and none of them had blocks left, the agent switches
        to LAND. Checking supposedly empty stashes before landing is necessary, because the agent only has local
        information available and other agents might return blocks to previously empty stashes. If no checking is done,
        it can happen that all agents land even though the structure is not finished, because according to their
        knowledge there are no more blocks left.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # check all stashes in CCW order
            ordered_stash_positions = environment.ccw_seed_stash_locations() if self.current_block_type_seed \
                else environment.ccw_block_stash_locations()
            min_stash_position = None
            min_distance = float("inf")
            for p in ordered_stash_positions:
                temp = simple_distance(self.geometry.position, p)
                if temp < min_distance:
                    min_distance = temp
                    min_stash_position = p
            index = ordered_stash_positions.index(min_stash_position)
            self.current_stash_path = ordered_stash_positions[index:]
            self.current_stash_path.extend(ordered_stash_positions[:index])
            self.current_stash_path_index = 0
            self.current_stash_position = min_stash_position

            check_level_z = self.geometry.position[2]
            self.current_path = Path()
            self.current_path.add_position([min_stash_position[0], min_stash_position[1], check_level_z],
                                           optional_distance=20)

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                stash_position = self.current_stash_position

                # check whether there are actually blocks at this location
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                current_stash = stashes[stash_position]

                self.aprint("Reached stash {} at {} when checking for stashes.".format(current_stash, stash_position))

                min_block = None
                min_distance = float("inf")
                for b in current_stash:
                    temp = simple_distance(self.geometry.position, b.geometry.position)
                    if temp < min_distance and not any([b is a.current_block for a in environment.agents]):
                        min_distance = temp
                        min_block = b

                if min_block is None:
                    # the stash is empty
                    if stash_position not in self.known_empty_stashes:
                        self.known_empty_stashes.append(stash_position)

                    # move on to next stash
                    if self.current_stash_path_index + 1 < len(self.current_stash_path):
                        self.current_stash_path_index += 1
                        next_position = self.current_stash_path[self.current_stash_path_index]
                        self.current_stash_position = next_position
                        self.current_path.add_position([next_position[0], next_position[1], self.geometry.position[2]],
                                                       optional_distance=20)
                    else:
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.aprint("Landing because all stashes are empty.")
                else:
                    # stash is not empty, can now pick up block from there
                    if stash_position in self.known_empty_stashes:
                        self.known_empty_stashes.remove(stash_position)
                    self.current_task = Task.FETCH_BLOCK
                    self.task_history.append(self.current_task)
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.CHECK_STASHES] += simple_distance(position_before, self.geometry.position)

    def fetch_block(self, environment: env.map.Map):
        """
        Move with the goal of fetching a block.

        This method is called if the current task is FETCH_BLOCK. If the agent has not planned a path to fetch
        a block yet, it first determines the closest stash of normal/seed blocks which still has blocks left.
        It then plans a path to that location and follows the path until it reaches the end point. If this parameter
        is set and there are more than 3 other agents around that block, the agent may choose a different stash
        of blocks to go to if it there is one. Once the agent reaches a block stash and it is not empty yet,
        the task changes to PICK_UP_BLOCK. Otherwise, it will move on to a different stash it believes to still
        contain blocks.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            if len(self.known_empty_stashes) == len(environment.seed_stashes) + len(environment.block_stashes):
                # to the agent's knowledge only empty stashes are left, but since this is the local information
                # version of the algorithm it still has to check whether blocks have been returned
                self.current_task = Task.CHECK_STASHES
                self.task_history.append(self.current_task)
                self.current_path = None
                self.check_stashes(environment)
                return

            # an idea that I tried, but which did not yield any better performance, is to fly away from the structure
            # as quickly as possible to free up space and then move to the closest stash from that position
            # this general idea may be worth exploring further though, in the spirit of freeing up the construction area
            # as much as possible

            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            min_stash_location = None
            min_distance = float("inf")
            stash_list = []
            compared_location = self.geometry.position
            for p in list(stashes.keys()):
                if p not in self.known_empty_stashes:
                    distance = simple_distance(compared_location, p)
                    count = self.count_in_direction(environment,
                                                    [p - self.geometry.position[:2]],
                                                    angle=np.pi / 2,
                                                    max_vertical_distance=200)[0]
                    count_at_stash = 0
                    for a in environment.agents:
                        if a is not self and simple_distance(a.geometry.position[:2], p) <= self.stash_min_distance:
                            count_at_stash += 1
                    block_count = len(stashes[p])
                    stash_list.append((p, distance, count, count_at_stash, block_count))
                    if distance < min_distance:
                        min_distance = distance
                        min_stash_location = p

            if min_stash_location is None:
                if self.current_block_type_seed:
                    self.current_block_type_seed = False
                else:
                    self.current_block_type_seed = True
                self.next_seed_position = None
                self.current_path = None
                self.fetch_block(environment)
                return

            self.current_stash_position = min_stash_location

            # construct path to target location
            fetch_level_z = max(self.geometry.position[2], self.geometry.position[2] + Block.SIZE * 2)
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z],
                                           optional_distance=20)

        # if within a certain distance of the stash, check whether there are many other agents there
        # that should maybe be avoided (issue here might be that other it's is fairly likely due to
        # the low approach to the stashes that other agents push ones already there out of the way; in
        # that case this check might still do some good)
        if self.avoiding_crowded_stashes_enabled and \
                simple_distance(self.geometry.position[:2], self.current_path.positions[-1][:2]) < 50:
            # while performance seemed to improve in some informal tests, these parameters have not been tested
            # extensively and e.g. reacting earlier (at a longer distance to the stash) or not basing the decision
            # on a hard threshold might be more desirable
            count_at_stash = 0
            for a in environment.agents:
                if a is not self and simple_distance(
                        a.geometry.position[:2], self.current_path.positions[-1][:2]) <= self.stash_min_distance:
                    count_at_stash += 1
            if count_at_stash > 3:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                min_stash_location = None
                min_distance = float("inf")
                for p in list(stashes.keys()):
                    if p not in self.known_empty_stashes \
                            and any([p[i] != self.current_path.positions[-1][i] for i in range(2)]):
                        distance = simple_distance(self.geometry.position, p)
                        if distance < min_distance:
                            min_distance = distance
                            min_stash_location = p

                if min_stash_location is not None:
                    self.current_stash_position = min_stash_location
                    fetch_level_z = max(self.geometry.position[2], self.geometry.position[2] + Block.SIZE * 2)
                    self.current_path = Path()
                    self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, determine a block to pick up
            # this should also still work if blocks are not overlapping (as is the case currently), but are
            # actually stored physically separate in a block (although then the code to fetch a block should
            # probably be adapted anyway)
            if not ret:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stash_position = self.current_stash_position
                if stash_position not in list(stashes.keys()):
                    min_stash_distance = float("inf")
                    for p in list(stashes.keys()):
                        temp = simple_distance(self.geometry.position, p)
                        if temp < min_stash_distance:
                            min_stash_distance = temp
                            stash_position = p

                if len(stashes[stash_position]) == 0:
                    # remember that this stash is empty
                    self.known_empty_stashes.append(stash_position)

                    # need to go to other stash, i.e. go to start of fetch_block again
                    self.current_path = None
                    return

                self.current_task = Task.PICK_UP_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FETCH_BLOCK] += simple_distance(position_before, self.geometry.position)

    def pick_up_block(self, environment: env.map.Map):
        """
        Move with the goal of picking up a block.

        This method is called if the current task is PICK_UP_BLOCK. If the agent has not planned a path and
        selected a block to pick up, it first selects the closest block from the block stash it is at and plans
        a path of movement to pick up the block. Once it is at level low enough to pick up the block, it attaches
        it to itself and the task changes to TRANSPORT_BLOCK. Other agents are also notified updated about a
        block having been removed from the stash.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        # at this point it has been confirmed that there is indeed a block around that location
        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            stash_position = tuple(self.geometry.position[:2])
            min_block = None
            min_distance = float("inf")
            if stash_position not in list(stashes.keys()):
                min_stash_distance = float("inf")
                for p in list(stashes.keys()):
                    temp = simple_distance(self.geometry.position, p)
                    if temp < min_stash_distance:
                        min_stash_distance = temp
                        stash_position = p

            for b in stashes[stash_position]:
                temp = self.geometry.distance_2d(b.geometry)
                if (not b.is_seed or self.current_block_type_seed) and not b.placed \
                        and not any([b is a.current_block for a in environment.agents]) and temp < min_distance:
                    min_block = b
                    min_distance = temp

            if min_block is None:
                # no more blocks at that location, need to go elsewhere
                self.known_empty_stashes.append(stash_position)
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
                self.fetch_block(environment)
                return

            # otherwise, make the selected block the current block and pick it up
            min_block.color = "red"

            pickup_z = min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[2] / 2
            self.current_path = Path()
            self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1], pickup_z])
            self.current_block = min_block

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                # attach block and move on to the transport_block task
                if self.current_block is None:
                    self.aprint("Current block is None when trying to pick it up.")
                    self.current_path = None
                    self.pick_up_block(environment)
                    return

                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stashes[tuple(self.geometry.position[:2])].remove(self.current_block)
                self.geometry.attached_geometries.append(self.current_block.geometry)
                if self.current_block_type_seed:
                    self.current_block.color = "#f44295"
                else:
                    self.current_block.color = "green"

                if self.rejoining_swarm and not self.current_block_type_seed:
                    self.current_task = Task.REJOIN_SWARM
                else:
                    self.current_task = Task.TRANSPORT_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None

                if self.current_block_type_seed and self.next_seed_position is None:
                    if self.current_grid_positions_to_be_seeded is not None:
                        # DEPRECATED
                        # decide on which possible seed positions to go for if the position has not been determined yet
                        temp_locations = []
                        for l in self.current_grid_positions_to_be_seeded:
                            temp_locations.append(np.array([environment.offset_origin[0] + l[0] * Block.SIZE,
                                                            environment.offset_origin[1] + l[1] * Block.SIZE,
                                                            (l[2] + 0.5) * Block.SIZE]))
                        order = []
                        while len(order) != len(temp_locations):
                            min_index = None
                            min_distance = float("inf")
                            for l_idx, l in enumerate(temp_locations):
                                if l_idx not in order:
                                    temp = simple_distance(self.geometry.position if len(order) == 0
                                                           else temp_locations[order[-1]], l)
                                    if temp < min_distance:
                                        min_distance = temp
                                        min_index = l_idx
                            order.append(min_index)

                        # order the possible grid positions and, if carrying seed, go through each of them
                        # considering returning the seed (when they are invalidated by observing that they are filled)
                        self.current_grid_positions_to_be_seeded = [np.array(
                            self.current_grid_positions_to_be_seeded[i]) for i in order]
                        self.current_block.grid_position = self.current_grid_positions_to_be_seeded[0]
                        self.next_seed_position = self.current_grid_positions_to_be_seeded[0]
                    else:
                        self.next_seed_position = self.current_seed.grid_position
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.PICK_UP_BLOCK] += simple_distance(position_before, self.geometry.position)

    def wait_on_perimeter(self, environment: env.map.Map):
        """
        Move with the goal of waiting on the structure perimeter (circling).

        This method is called if the current task is WAIT_ON_PERIMETER. If the agent has not planned a path yet,
        it first determines the closest corner of the construction area in a counter-clockwise direction and then
        plans a path there. Once it reaches the corner it plans a path to the next corner and so on until the
        condition is met to enter the construction area and perform the original task (TRANSPORT_BLOCK).

        :param environment: the environment the agent operates in
        """

        # instead of switching to the original start whenever there is an opportunity,
        # a more sophisticated queueing system may work better

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # ascend to a level a bit higher than that of other agents to further avoid collisions
            self.current_waiting_height = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + \
                                          self.required_distance + self.geometry.size[2] * 1.5 + \
                                          self.geometry.size[2] + Block.SIZE * 2

            # first calculate the next counter-clockwise corner point of the construction area and go there
            ordered_corner_points = environment.ccw_corner_locations(self.geometry.position[:2],
                                                                     self.geometry.size[2] * 1.5)

            # plan path to the first of these points (when arriving there, simply recalculate, I think)
            corner_point = ordered_corner_points[0]

            self.current_path = Path()
            self.current_path.add_position([corner_point[0], corner_point[1], self.current_waiting_height],
                                           optional_distance=(20, 20, 20))

        # re-check whether the condition for waiting on the perimeter still holds
        if self.area_density_restricted and environment.density_over_construction_area() <= 1:
            self.current_path = self.previous_path
            self.current_task = Task.TRANSPORT_BLOCK
            self.task_history.append(self.current_task)
            self.transport_block(environment)
            return

        next_position, current_direction = self.move(environment)
        reached_end_zone = False
        if self.current_path.optional_area_reached(self.geometry.position) \
                and not simple_distance(self.geometry.position + current_direction, next_position) \
                        < simple_distance(self.geometry.position, next_position):
            ret = self.current_path.advance()
            next_position, current_direction = self.move(environment)
            if not ret:
                reached_end_zone = True

        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP or reached_end_zone:
            if not reached_end_zone:
                self.geometry.position = next_position
                ret = self.current_path.advance()

            if reached_end_zone or not ret:
                if self.area_density_restricted and environment.density_over_construction_area() <= 1:
                    self.current_path = self.previous_path
                    self.current_task = Task.TRANSPORT_BLOCK
                    self.task_history.append(self.current_task)
                    return

                # calculate next most CCW site
                ordered_corner_points = environment.ccw_corner_locations(self.geometry.position[:2],
                                                                         self.geometry.size[2] * 1.5)
                corner_point = ordered_corner_points[0]
                self.current_path.add_position([corner_point[0], corner_point[1], self.current_waiting_height],
                                               optional_distance=(20, 20, 20))
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.WAIT_ON_PERIMETER] += simple_distance(position_before, self.geometry.position)

    def transport_block(self, environment: env.map.Map):
        """
        Move with the goal of transporting a picked-up block to a seed in the structure for further orientation.

        This method is called if the current task is TRANSPORT_BLOCK. If the agent has not planned a path yet,
        it determines a path to the position of the seed it is currently using for localisation in the structure.
        Then the agent moves to that position and if it is carrying a normal block switches the task to
        FIND_ATTACHMENT_SITE. If it is carrying a seed block, it moves to the intended location for that seed
        instead and upon reaching it (if it is not occupied), switches to the task PLACE_BLOCK.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            self.current_visited_sites = None

            seed_location = self.current_seed.geometry.position
            # gain height, fly to seed location and then start search for attachment site
            self.current_path = Path()
            transport_level_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + \
                                self.required_distance + self.geometry.size[2] * 1.5
            other_transport_level_z = (self.current_seed.grid_position[2] + 2) * Block.SIZE + self.geometry.size[2] * 2
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], transport_level_z],
                                           optional_distance=(70, 70, 20))
            self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z],
                                           optional_distance=30)
            self.current_path.add_position([seed_location[0], seed_location[1], other_transport_level_z])

        # check whether the construction area is overcrowded and the agent should wait on the perimeter
        if self.waiting_on_perimeter_enabled and not self.current_block_type_seed \
                and not environment.check_over_construction_area(self.geometry.position):
            # not over construction area yet
            if environment.distance_to_construction_area(self.geometry.position) <= self.geometry.size[0] * 2:
                if self.area_density_restricted and environment.density_over_construction_area() > 1:
                    # in this case it's too crowded -> don't move in yet
                    self.current_static_location = np.copy(self.geometry.position)
                    self.current_task = Task.WAIT_ON_PERIMETER
                    self.task_history.append(self.current_task)
                    self.previous_path = self.current_path
                    self.current_path = None
                    self.wait_on_perimeter(environment)
                    return

        if simple_distance(self.current_seed.geometry.position[:2], self.geometry.position[:2]) < 50.0:
            self.close_to_seed_count += 1

        next_position, current_direction = self.move(environment)
        if self.current_path.optional_area_reached(self.geometry.position):
            ret = self.current_path.advance()
            if ret:
                next_position, current_direction = self.move(environment)
            else:
                self.current_path.retreat()

        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.seed_arrival_delay_queue.append(self.close_to_seed_count)
                self.close_to_seed_count = 0

                self.current_path = None
                self.current_grid_position = np.copy(self.current_seed.grid_position)
                self.update_local_occupancy_map(environment)

                # since this method is also used to move to a seed site with a carried seed after already having
                # found the current seed for localisation, need to check whether we have arrived and should
                # drop off the carried seed
                if self.current_block_type_seed and self.transporting_to_seed_site:
                    # this means that we have arrived at the intended site for the seed, it should
                    # now be placed or, alternatively, a different site for it should be found
                    self.current_grid_position = np.array(self.next_seed_position)
                    if not environment.check_occupancy_map(self.next_seed_position):
                        # can place the carried seed
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_component_marker = self.component_target_map[self.current_grid_position[2],
                                                                                  self.current_grid_position[1],
                                                                                  self.current_grid_position[0]]
                    else:
                        # the position is already occupied, need to move to different site
                        # check whether there are any unseeded sites
                        unseeded = self.unseeded_component_markers(self.local_occupancy_map)
                        unfinished = self.unfinished_component_markers(self.local_occupancy_map)
                        if len(unseeded) == 0 and len(unfinished) > 0:
                            # should only count block (not seed) stashes here:
                            counter = 0
                            for s in self.known_empty_stashes:
                                if s in environment.block_stashes.keys():
                                    counter += 1
                            if counter >= len(environment.block_stashes):
                                self.current_task = Task.MOVE_TO_PERIMETER
                            else:
                                self.current_task = Task.RETURN_BLOCK
                        else:
                            if len(unseeded) == 0:
                                self.current_structure_level += 1
                            self.current_task = Task.FIND_NEXT_COMPONENT
                        self.task_history.append(self.current_task)
                    self.transporting_to_seed_site = False
                    return

                position_above = [self.current_grid_position[0],
                                  self.current_grid_position[1],
                                  self.current_grid_position[2] + 1]
                if not environment.check_occupancy_map(position_above):
                    # the current seed is not covered by anything
                    if not self.current_block_type_seed:
                        # should check whether the component is finished
                        if self.check_component_finished(self.local_occupancy_map,
                                                         self.component_target_map[self.current_seed.grid_position[2],
                                                                                   self.current_seed.grid_position[1],
                                                                                   self.current_seed.grid_position[0]]):
                            self.current_task = Task.FIND_NEXT_COMPONENT
                        else:
                            self.current_grid_direction = [0, -1, 0]
                            self.current_task = Task.MOVE_TO_PERIMETER
                        self.task_history.append(self.current_task)
                    else:
                        # if a seed is being carried, the transport phase continues to the designated seed position
                        seed_x = environment.offset_origin[0] + self.next_seed_position[0] * Block.SIZE
                        seed_y = environment.offset_origin[1] + self.next_seed_position[1] * Block.SIZE

                        directions = [np.array([self.next_seed_position[0],
                                                self.current_grid_position[1]]) - self.current_grid_position[:2],
                                      np.array([self.current_grid_position[0],
                                                self.next_seed_position[1]]) - self.current_grid_position[:2]]
                        counts = self.count_in_direction(environment, directions, angle=np.pi / 2)
                        if self.transport_avoid_others_enabled and counts[0] < counts[1]:
                            first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                        elif self.transport_avoid_others_enabled and counts[1] < counts[0]:
                            first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]
                        else:
                            if random.random() < 0.5:
                                first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                            else:
                                first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]

                        self.current_path = Path()
                        self.current_path.add_position(first_location)
                        self.current_path.add_position([seed_x, seed_y, self.geometry.position[2]])
                        self.transporting_to_seed_site = True
                else:
                    # the position is unexpectedly occupied, therefore the local map should be updated
                    self.current_visited_sites = None
                    block_above_seed = environment.block_at_position(position_above)
                    self.local_occupancy_map[block_above_seed.grid_position[2],
                                             block_above_seed.grid_position[1],
                                             block_above_seed.grid_position[0]] = 1

                    if block_above_seed.is_seed:
                        # the blocking block is a seed and can therefore also be used for orientation
                        self.current_seed = block_above_seed
                        self.current_structure_level = self.current_seed.grid_position[2]
                        for layer in range(self.current_structure_level):
                            self.local_occupancy_map[layer][self.target_map[layer] != 0] = 1
                        if not self.current_block_type_seed:
                            # simply use this for attachment
                            self.current_component_marker = self.component_target_map[block_above_seed.grid_position[2],
                                                                                      block_above_seed.grid_position[1],
                                                                                      block_above_seed.grid_position[0]]
                            self.current_grid_direction = [0, -1, 0]
                            self.current_grid_position = np.array(self.current_seed.grid_position)
                            self.current_task = Task.MOVE_TO_PERIMETER
                            self.task_history.append(self.current_task)
                        else:
                            if self.next_seed_position[2] == block_above_seed.grid_position[2]:
                                # use as orientation if at same level as carried seed
                                # simply move to the intended position for the carried seed
                                seed_x = environment.offset_origin[0] + self.next_seed_position[0] * Block.SIZE
                                seed_y = environment.offset_origin[1] + self.next_seed_position[1] * Block.SIZE

                                directions = [np.array([self.next_seed_position[0],
                                                        self.current_grid_position[1]]) - self.current_grid_position[
                                                                                          :2],
                                              np.array([self.current_grid_position[0],
                                                        self.next_seed_position[1]]) - self.current_grid_position[:2]]
                                counts = self.count_in_direction(environment, directions, angle=np.pi / 2)
                                if self.transport_avoid_others_enabled and counts[0] < counts[1]:
                                    first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                                elif self.transport_avoid_others_enabled and counts[1] < counts[0]:
                                    first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]
                                else:
                                    if random.random() < 0.5:
                                        first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                                    else:
                                        first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]

                                self.current_path = Path()
                                self.current_path.add_position(first_location)
                                self.current_path.add_position([seed_x, seed_y, self.geometry.position[2]])
                                self.transporting_to_seed_site = True
                            else:
                                # otherwise, need to try and find a site to be seeded
                                self.current_task = Task.FIND_NEXT_COMPONENT
                                self.task_history.append(self.current_task)
                                self.next_seed_position = None
                    else:
                        # the position is not a seed, therefore need to move to the seed of that component
                        self.aprint("Current seed {} covered by block, trying to "
                                    "find seed of the covering component at {}"
                                    .format(self.current_seed.grid_position, block_above_seed.grid_position))

                        seed_grid_location = self.component_seed_location(
                            self.component_target_map[block_above_seed.grid_position[2],
                                                      block_above_seed.grid_position[1],
                                                      block_above_seed.grid_position[0]],
                            self.current_grid_position[2] + 1)
                        self.current_seed = environment.block_at_position(seed_grid_location)

                        seed_x = environment.offset_origin[0] + seed_grid_location[0] * Block.SIZE
                        seed_y = environment.offset_origin[1] + seed_grid_location[1] * Block.SIZE

                        directions = [np.array([seed_grid_location[0],
                                                self.current_grid_position[1]]) - self.current_grid_position[:2],
                                      np.array([self.current_grid_position[0],
                                                seed_grid_location[1]]) - self.current_grid_position[:2]]
                        counts = self.count_in_direction(environment, directions, angle=np.pi / 2)
                        if self.transport_avoid_others_enabled and counts[0] < counts[1]:
                            first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                        elif self.transport_avoid_others_enabled and counts[1] < counts[0]:
                            first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]
                        else:
                            if random.random() < 0.5:
                                first_location = [seed_x, self.geometry.position[1], self.geometry.position[2]]
                            else:
                                first_location = [self.geometry.position[0], seed_y, self.geometry.position[2]]

                        self.current_path = Path()
                        self.current_path.add_position(first_location)
                        self.current_path.add_position([seed_x, seed_y, self.geometry.position[2]])

                if self.check_component_finished(self.local_occupancy_map):
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.TRANSPORT_BLOCK] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def move_to_perimeter(self, environment: env.map.Map):
        """
        Move with the goal of reaching the perimeter of the structure (more specifically the current component).

        This method is called if the current task is MOVE_TO_PERIMETER. If the agent has not planned a path yet,
        it determines a direction to move into and then proceeds to move into that direction until it has reached
        the structure/component perimeter. When it has reached it, the task changes FIND_ATTACHMENT_SITE or
        SURVEY_COMPONENT depending on whether the agent is carrying a normal block or a seed block.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # move to next block position in designated direction (which could be the shortest path or
            # just some direction chosen e.g. at the start, which is assumed here)
            directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])]
            counts = self.count_in_direction(environment)

            if any([c != 0 for c in counts]):
                sorter = sorted(range(len(directions)), key=lambda i: counts[i])
                directions = [directions[i] for i in sorter]
                self.current_grid_direction = directions[0]
            else:
                self.current_grid_direction = random.sample(directions, 1)[0]

            self.current_path = Path()
            destination_x = (self.current_grid_position + self.current_grid_direction)[0] * Block.SIZE + \
                            environment.offset_origin[0]
            destination_y = (self.current_grid_position + self.current_grid_direction)[1] * Block.SIZE + \
                            environment.offset_origin[1]
            self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.current_grid_position += self.current_grid_direction

                self.current_blocks_per_attachment += 1

                self.update_local_occupancy_map(environment)

                try:
                    # this is basically getting all values of the occupancy map at the locations where the hole map
                    # has the value of the hole which we are currently over
                    result = all(self.local_occupancy_map[self.hole_boundaries[self.hole_map[
                        self.current_grid_position[2], self.current_grid_position[1],
                        self.current_grid_position[0]]]] != 0)
                    # if the result is True, then we know that there is a hole and it is closed already
                except (IndexError, KeyError):
                    result = False

                if not self.current_block_type_seed:
                    self.current_blocks_per_attachment += 1

                if environment.block_below(self.geometry.position, self.current_structure_level) is None and \
                        (check_map(self.hole_map, self.current_grid_position, lambda x: x < 2) or not result):
                    # have reached perimeter, two possibilities:
                    # - carrying normal block -> should find an attachment site
                    # - carrying seed -> should do a survey of the current component

                    if not self.current_block_type_seed:
                        self.current_task = Task.FIND_ATTACHMENT_SITE
                    else:
                        self.current_task = Task.SURVEY_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_grid_direction = np.array(
                        [-self.current_grid_direction[1], self.current_grid_direction[0], 0], dtype="int64")
                else:
                    destination_x = (self.current_grid_position + self.current_grid_direction)[0] * Block.SIZE + \
                                    environment.offset_origin[0]
                    destination_y = (self.current_grid_position + self.current_grid_direction)[1] * Block.SIZE + \
                                    environment.offset_origin[1]
                    self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.MOVE_TO_PERIMETER] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def survey_component(self, environment: env.map.Map):
        """
        Move with the goal of surveying the current component to find out whether it is finished.

        This method is called if the current task is SURVEY_COMPONENT. If this method is called, the agent is already
        on the structure/component perimeter. The agent then moves around the perimeter in the same way as it would
        for finding an attachment site using the perimeter following algorithm. If it revisits a site, it has circled
        the entire component and either knows that the component is complete and it should bring the currently
        carried seed elsewhere or that there are still blocks missing and that it should return the seed block and
        fetch a normal block. In the former case the agent switches to task TRANSPORT_BLOCK, and in the latter it
        switches to task RETURN_BLOCK.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            self.current_path = Path()
            self.current_path.add_position(self.geometry.position)

        if self.current_visited_sites is None:
            self.current_visited_sites = []

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.update_local_occupancy_map(environment)

                # if the component was considered to be unfinished but is now confirmed to be, switch to next another
                if self.check_component_finished(self.local_occupancy_map):
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_path = None
                    self.current_visited_sites = None
                    return

                current_site_tuple = (tuple(self.current_grid_position), tuple(self.current_grid_direction))
                if current_site_tuple in self.current_visited_sites:
                    if self.check_component_finished(self.local_occupancy_map):
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.current_visited_sites = None
                    else:
                        self.current_task = Task.RETURN_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.current_visited_sites = None
                    return

                # adding location and direction here to check for revisiting
                self.current_visited_sites.append(current_site_tuple)

                # the checks need to determine whether the current position is a valid attachment site
                position_ahead_occupied = environment.check_occupancy_map(
                    self.current_grid_position + self.current_grid_direction)
                position_around_corner_empty = environment.check_occupancy_map(
                    self.current_grid_position + self.current_grid_direction +
                    np.array([-self.current_grid_direction[1], self.current_grid_direction[0], 0], dtype="int64"),
                    lambda x: x == 0)

                # if block ahead, turn right
                # if position around corner empty, turn left
                # if neither of these, continue straight
                if position_ahead_occupied:
                    # turn right
                    self.current_grid_direction = np.array([self.current_grid_direction[1],
                                                            -self.current_grid_direction[0], 0],
                                                           dtype="int64")
                elif position_around_corner_empty:
                    # first move forward (to the corner)
                    self.current_path.add_position(self.geometry.position + Block.SIZE * self.current_grid_direction)
                    reference_position = self.current_path.positions[-1]

                    # then turn left
                    self.current_grid_position += self.current_grid_direction
                    self.current_grid_direction = np.array([-self.current_grid_direction[1],
                                                            self.current_grid_direction[0], 0],
                                                           dtype="int64")
                    self.current_grid_position += self.current_grid_direction
                    self.current_path.add_position(reference_position + Block.SIZE * self.current_grid_direction)
                else:
                    # otherwise site "around the corner" occupied -> continue straight ahead
                    self.current_grid_position += self.current_grid_direction
                    self.current_path.add_position(self.geometry.position + Block.SIZE * self.current_grid_direction)
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.SURVEY_COMPONENT] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def find_next_component(self, environment: env.map.Map):
        """
        Move with the goal of finding a different component after the current one has been completed.

        This method is called if the current task is FIND_NEXT_COMPONENT. If the agent has not planned a path yet,
        it determines the components which have not been seeded and those which have not been finished yet. Unless
        it already knows about one of these components being seeded and is carrying a normal block, it determines
        an order in which to visit the seed positions of the determined components to see whether they are seeded.
        If they are not seeded and the agent is carrying a seed, it can seed the component. If the agent is carrying
        a normal block, it can continue searching for seeded components or return the block and fetch a seed as soon
        as it finds an unseeded component. If the agent is not carrying any block, it should also look for the first
        unseeded component it can find and then fetch a seed for it. If there are none but there are unfinished
        components, it should fetch a normal block instead.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # first check whether we know that the current layer is already completed
            if self.check_layer_finished(self.local_occupancy_map):
                self.current_structure_level += 1

            candidate_components_placement = self.unfinished_component_markers(self.local_occupancy_map)
            candidate_components_seeding = self.unseeded_component_markers(self.local_occupancy_map)

            if len(candidate_components_seeding) == 0 and self.current_block_type_seed \
                    and self.check_layer_finished(self.local_occupancy_map):
                self.current_structure_level += 1
                self.find_next_component(environment)
                return
            elif len(candidate_components_seeding) == 0 and self.current_block_type_seed:
                self.current_task = Task.RETURN_BLOCK
                self.task_history.append(self.current_task)
                self.return_block(environment)
                return

            # check if any of these components have been seeded (and are unfinished) and the agent knows this
            if not self.current_block_type_seed:
                for m in candidate_components_placement:
                    if any(self.local_occupancy_map[self.current_structure_level][
                               self.component_target_map[self.current_structure_level] == m]):
                        self.current_component_marker = m
                        self.current_seed = environment.block_at_position(self.component_seed_location(m))
                        self.current_task = Task.FETCH_BLOCK if self.current_block is None else Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        return

            candidate_components = candidate_components_seeding if self.current_block_type_seed \
                else candidate_components_placement

            # first, need to determine seed locations
            seed_grid_locations = []
            seed_locations = []
            for m in candidate_components:
                temp = tuple(self.component_seed_location(m))
                seed_grid_locations.append(temp)
                seed_locations.append(np.array([environment.offset_origin[0] + temp[0] * Block.SIZE,
                                                environment.offset_origin[1] + temp[1] * Block.SIZE,
                                                (temp[2] + 0.5) * Block.SIZE]))
            order = []
            while len(order) != len(seed_locations):
                min_index = None
                min_distance = float("inf")
                for l_idx, l in enumerate(seed_locations):
                    if l_idx not in order:
                        temp = simple_distance(self.geometry.position if len(order) == 0
                                               else seed_locations[order[-1]], l)
                        if temp < min_distance:
                            min_distance = temp
                            min_index = l_idx
                order.append(min_index)

            if self.seeding_strategy == "distance_self":
                if self.order_only_one_metric:
                    order = sorted(range(len(seed_locations)),
                                   key=lambda x: (simple_distance(seed_locations[x], self.geometry.position), random.random()))
                else:
                    order = sorted(range(len(seed_locations)),
                                   key=lambda x: (simple_distance(seed_locations[x], self.geometry.position),
                                                  simple_distance(seed_locations[x], environment.center)))
            elif self.seeding_strategy == "distance_center":
                if self.order_only_one_metric:
                    order = sorted(range(len(seed_locations)),
                                   key=lambda x: (simple_distance(seed_locations[x], environment.center), random.random()))
                else:
                    order = sorted(range(len(seed_locations)),
                                   key=lambda x: (simple_distance(seed_locations[x], environment.center),
                                                  simple_distance(seed_locations[x], self.geometry.position)))
            elif self.seeding_strategy == "agent_count":
                candidate_component_count = [0] * len(candidate_components)
                for a in environment.agents:
                    for m_idx, m in enumerate(candidate_components):
                        closest_x = int((a.geometry.position[0] - environment.offset_origin[0]) / env.block.Block.SIZE)
                        closest_y = int((a.geometry.position[1] - environment.offset_origin[1]) / env.block.Block.SIZE)
                        if 0 <= closest_x < self.target_map.shape[2] and 0 <= closest_y < self.target_map.shape[1] \
                                and any([self.component_target_map[z, closest_y, closest_x] == m
                                         for z in range(self.target_map.shape[0])]):
                            candidate_component_count[m_idx] += 1
                if self.order_only_one_metric:
                    order = sorted(range(len(seed_locations)), key=lambda x: (candidate_component_count[x], random.random()))
                else:
                    order = sorted(range(len(seed_locations)),
                                   key=lambda x: (candidate_component_count[x],
                                                  simple_distance(seed_locations[x], environment.center)))

            # sort the locations
            seed_grid_locations = [np.array(seed_grid_locations[i]) for i in order]
            seed_locations = [seed_locations[i] for i in order]

            search_z = (self.current_seed.grid_position[2] + 2) * Block.SIZE + self.geometry.size[2] * 2
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], search_z])
            self.current_path.inserted_sequentially[self.current_path.current_index] = False

            directions = [np.array([seed_locations[0][0],
                                    self.geometry.position[1]]) - self.geometry.position[:2],
                          np.array([self.geometry.position[0],
                                    seed_locations[0][1]]) - self.geometry.position[:2]]
            counts = self.count_in_direction(environment, directions, angle=np.pi / 2)
            if self.transport_avoid_others_enabled and counts[0] < counts[1]:
                first_site = [seed_locations[0][0], self.geometry.position[1], self.geometry.position[2]]
            elif self.transport_avoid_others_enabled and counts[1] < counts[0]:
                first_site = [self.geometry.position[0], seed_locations[0][1], self.geometry.position[2]]
            else:
                if random.random() < 0.5:
                    first_site = [seed_locations[0][0], self.geometry.position[1], self.geometry.position[2]]
                else:
                    first_site = [self.geometry.position[0], seed_locations[0][1], self.geometry.position[2]]

            second_site = np.array([seed_locations[0][0], seed_locations[0][1], search_z])
            self.current_path.add_position(first_site)
            self.current_path.add_position(second_site)

            self.current_seed_grid_positions = seed_grid_locations
            self.current_seed_grid_position_index = 0

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position

            can_skip = not self.current_path.inserted_sequentially[self.current_path.current_index]

            ret = self.current_path.advance()
            if ret and np.array_equal(self.geometry.position, self.current_path.positions[-1]):
                ret = False

            if self.current_path.current_index != 0 and not can_skip and not ret:
                self.current_seed_grid_position_index += 1

            if not can_skip and not ret:
                # if at a location where it can be seen whether the block location has been seeded,
                # check whether the position below has been seeded
                current_seed_position = self.current_seed_grid_positions[self.current_seed_grid_position_index - 1]
                self.current_grid_position = np.array(current_seed_position)
                self.update_local_occupancy_map(environment)
                if environment.check_occupancy_map(current_seed_position):
                    # there is already a seed at the position
                    self.current_seed = environment.block_at_position(current_seed_position)
                    if self.current_block is not None:
                        # check again whether there has been any change
                        if not self.check_component_finished(self.local_occupancy_map,
                                                             self.component_target_map[current_seed_position[2],
                                                                                       current_seed_position[1],
                                                                                       current_seed_position[0]]):
                            if not self.current_block_type_seed:
                                # if it has, simply switch to that component and try to attach there
                                self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                          current_seed_position[1],
                                                                                          current_seed_position[0]]
                                self.current_seed = environment.block_at_position(current_seed_position)
                                self.current_task = Task.MOVE_TO_PERIMETER
                                self.current_grid_direction = [1, 0, 0]
                                self.task_history.append(self.current_task)
                                self.current_grid_positions_to_be_seeded = None
                                self.current_path = None
                            else:
                                # need to move on to next location
                                if self.current_seed_grid_position_index + 1 > len(self.current_seed_grid_positions):
                                    # have not found a single location without seed, therefore return it and fetch block
                                    self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                              current_seed_position[1],
                                                                                              current_seed_position[0]]
                                    self.current_task = Task.RETURN_BLOCK
                                    self.task_history.append(self.current_task)
                                    self.current_path = None
                                else:
                                    next_x = environment.offset_origin[0] + Block.SIZE * \
                                             self.current_seed_grid_positions[self.current_seed_grid_position_index][0]
                                    next_y = environment.offset_origin[1] + Block.SIZE * \
                                             self.current_seed_grid_positions[self.current_seed_grid_position_index][1]
                                    first_site = np.array(
                                        [self.geometry.position[0], next_y, self.geometry.position[2]])
                                    second_site = np.array([next_x, next_y, self.geometry.position[2]])
                                    self.current_path.add_position(first_site)
                                    self.current_path.add_position(second_site)
                        else:
                            # simply try again
                            self.current_path = None
                    else:
                        if self.current_seeded_positions is None:
                            # DEPRECATED
                            self.current_seeded_positions = []

                        # check whether that component is even in question
                        if not self.check_component_finished(self.local_occupancy_map,
                                                             self.component_target_map[current_seed_position[2],
                                                                                       current_seed_position[1],
                                                                                       current_seed_position[0]]):
                            self.current_task = Task.FETCH_BLOCK
                            self.task_history.append(self.current_task)
                            self.next_seed_position = current_seed_position
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                        else:
                            if self.check_structure_finished(self.local_occupancy_map):
                                self.current_task = Task.LAND
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                self.current_path = None
                else:
                    # no seed at the position yet
                    if self.current_block is not None:
                        if not self.current_block_type_seed:
                            if self.seed_if_possible_enabled:
                                # return the current normal block to fetch a seed
                                self.current_block_type_seed = False
                                self.current_task = Task.RETURN_BLOCK
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                if self.current_seed_grid_position_index + 1 > len(self.current_seed_grid_positions):
                                    # if there are no other options left, also return and fetch seed
                                    if self.current_block is None:
                                        self.current_block_type_seed = True
                                    self.current_task = Task.RETURN_BLOCK
                                    self.task_history.append(self.current_task)
                                    self.current_path = None
                                else:
                                    # otherwise, move on to next potential seed
                                    next_x = environment.offset_origin[0] + Block.SIZE * \
                                        self.current_seed_grid_positions[self.current_seed_grid_position_index][0]
                                    next_y = environment.offset_origin[1] + Block.SIZE * \
                                        self.current_seed_grid_positions[self.current_seed_grid_position_index][1]

                                    directions = [np.array([next_x, self.geometry.position[1]]) -
                                                  self.geometry.position[:2],
                                                  np.array([self.geometry.position[0], next_y]) -
                                                  self.geometry.position[:2]]
                                    counts = self.count_in_direction(environment, directions, angle=np.pi / 2)
                                    if self.transport_avoid_others_enabled and counts[0] < counts[1]:
                                        first_site = [next_x, self.geometry.position[1], self.geometry.position[2]]
                                    elif self.transport_avoid_others_enabled and counts[1] < counts[0]:
                                        first_site = [self.geometry.position[0], next_y, self.geometry.position[2]]
                                    else:
                                        if random.random() < 0.5:
                                            first_site = [next_x, self.geometry.position[1], self.geometry.position[2]]
                                        else:
                                            first_site = [self.geometry.position[0], next_y, self.geometry.position[2]]

                                    second_site = np.array([next_x, next_y, self.geometry.position[2]])
                                    self.current_path.add_position(first_site)
                                    self.current_path.add_position(second_site)
                        else:
                            # can place the seed here
                            self.current_task = Task.PLACE_BLOCK
                            self.task_history.append(self.current_task)
                            self.next_seed_position = current_seed_position
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                    else:
                        # DEPRECATED
                        # if we do not have a block currently, remember this site as the seed location and fetch seed
                        self.current_task = Task.FETCH_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_block_type_seed = True
                        self.next_seed_position = current_seed_position
                        self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                  current_seed_position[1],
                                                                                  current_seed_position[0]]
                        self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FIND_NEXT_COMPONENT] += simple_distance(position_before,
                                                                                      self.geometry.position)

    def return_block(self, environment: env.map.Map):
        """
        Move with the goal of returning the current block and then picking one of the 'opposite' type.

        This method is called if the current task is RETURN_BLOCK. If the agent has not planned a path yet, it plans
        one to the closest stash matching the type of block it is carrying and proceeds to move there. Once it has
        reached the stash, it descends and places the block. Note that stashes are not modelled realistically and
        blocks simply overlap each other in them, this process does not involve searching for a place in the stash
        to bring the block to, which may be desirable in a later version of the implementation. Once the block has
        been placed in the stash, the task switches to FETCH_BLOCK.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # select the closest block stash
            min_stash_location = None
            min_distance = float("inf")
            compared_location = self.geometry.position
            stash_list = []
            for key, value in stashes.items():
                distance = simple_distance(compared_location, key)
                count = self.count_in_direction(environment,
                                                [key - self.geometry.position[:2]],
                                                angle=np.pi / 2)[0]
                count_at_stash = 0
                for a in environment.agents:
                    if a is not self and simple_distance(a.geometry.position[:2], key) <= self.stash_min_distance:
                        count_at_stash += 1
                stash_list.append((key, distance, count, count_at_stash))
                if distance < min_distance:
                    min_stash_location = key
                    min_distance = distance

            self.current_stash_position = min_stash_location

            # plan a path there
            return_z = self.geometry.position[2]
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], return_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], return_z],
                                           optional_distance=30)
            self.current_path.add_position([min_stash_location[0], min_stash_location[1],
                                            Block.SIZE + self.geometry.size[2] / 2])

        # see fetch_block for comments
        if self.avoiding_crowded_stashes_enabled \
                and simple_distance(self.geometry.position[:2], self.current_path.positions[-1][:2]) < 50:
            count_at_stash = 0
            for a in environment.agents:
                if a is not self and simple_distance(
                        a.geometry.position[:2], self.current_path.positions[-1][:2]) <= self.stash_min_distance:
                    count_at_stash += 1
            if count_at_stash > 3:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                min_stash_location = None
                min_distance = float("inf")
                for p in list(stashes.keys()):
                    if p not in self.known_empty_stashes \
                            and any([p[i] != self.current_path.positions[-1][i] for i in range(2)]):
                        distance = simple_distance(self.geometry.position, p)
                        if distance < min_distance:
                            min_distance = distance
                            min_stash_location = p

                if min_stash_location is not None:
                    self.current_stash_position = min_stash_location
                    return_z = self.geometry.position[2]
                    self.current_path = Path()
                    self.current_path.add_position([min_stash_location[0], min_stash_location[1], return_z],
                                                   optional_distance=30)
                    self.current_path.add_position([min_stash_location[0], min_stash_location[1],
                                                    Block.SIZE + self.geometry.size[2] / 2])

        next_position, current_direction = self.move(environment)
        if self.current_path.optional_area_reached(self.geometry.position):
            ret = self.current_path.advance()
            if ret:
                next_position, current_direction = self.move(environment)
            else:
                self.current_path.retreat()

        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if arrived at the stash, release the block and go to fetch a seed currently there is not other task
            # that needs to be performed in that case, therefore we can be sure that a seed should be fetched
            if not ret:
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                backup = []
                for s in self.known_empty_stashes:
                    if not all(s[i] == self.current_block.geometry.position[i] for i in range(2)):
                        backup.append(s)

                if self.current_block_type_seed:
                    self.current_block.color = Block.COLORS_SEEDS[0]
                    environment.seed_stashes[self.current_stash_position].append(self.current_block)
                else:
                    self.current_block.color = "#FFFFFF"
                    environment.block_stashes[self.current_stash_position].append(
                        self.current_block)

                if self.current_block.color == "green":
                    self.current_block.color = "#FFFFFF"

                self.known_empty_stashes = backup
                self.current_block = None
                self.current_path = None
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)

                self.current_block_type_seed = not self.current_block_type_seed
                self.returned_blocks += 1
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.RETURN_BLOCK] += simple_distance(position_before,
                                                                               self.geometry.position)

    @abstractmethod
    def advance(self, environment: env.map.Map):
        """
        Abstract method to be overridden by subclasses.

        :param environment: the environment the agent operates in
        """

        pass
