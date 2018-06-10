import numpy as np
import random
import env.map
from abc import abstractmethod
from agents.agent import Task, Agent, check_map
from env.block import Block
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance, rotation_2d


class LocalKnowledgeAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(LocalKnowledgeAgent, self).__init__(position, size, target_map, required_spacing, printing_enabled)
        self.seed_if_possible_enabled = False
        self.seeding_strategy = "distance_self"  # others being: "distance_center", "agent_count"
        self.find_next_component_count = []

    def check_stashes(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)
        # TODO?: do the thing

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
            self.aprint("ORDERED STASH POSITIONS: {}".format(ordered_stash_positions))
            index = ordered_stash_positions.index(min_stash_position)
            self.current_stash_path = ordered_stash_positions[index:]
            self.current_stash_path.extend(ordered_stash_positions[:index])
            self.current_stash_path_index = 0
            self.current_stash_position = min_stash_position

            check_level_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                            self.required_spacing
            check_level_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + self.required_distance + \
                            self.geometry.size[2] * 1.5
            check_level_z = self.geometry.position[2]
            self.current_path = Path()
            self.current_path.add_position([min_stash_position[0], min_stash_position[1], check_level_z],
                                           optional_distance=20)
            # TODO: potentially make this work with "approximate goals"?

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                stash_position = self.current_stash_position

                # check whether there are actually blocks at this location
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                current_stash = stashes[stash_position]

                self.aprint("REACHED STASH {} (OWN POSITION {})"
                            .format(current_stash, stash_position))
                self.aprint("STASHES: {}".format(stashes))

                min_block = None
                min_distance = float("inf")
                for b in current_stash:
                    temp = simple_distance(self.geometry.position, b.geometry.position)
                    if temp < min_distance and not any([b is a.current_block for a in environment.agents]):
                        min_distance = temp
                        min_block = b

                if min_block is None:
                    if stash_position not in self.known_empty_stashes:
                        self.known_empty_stashes.append(stash_position)
                        self.aprint("EMPTY STASH APPENDED IN CHECK_STASHES")
                    # move on to next stash
                    if self.current_stash_path_index + 1 < len(self.current_stash_path):
                        self.current_stash_path_index += 1
                        next_position = self.current_stash_path[self.current_stash_path_index]
                        self.current_stash_position = next_position
                        self.aprint("MOVING ON TO NEXT STASH {}".format(next_position))
                        self.aprint("(index: {}, stash path: {})"
                                    .format(self.current_stash_path_index, self.current_stash_path))
                        self.current_path.add_position([next_position[0], next_position[1], self.geometry.position[2]],
                                                       optional_distance=20)
                    else:
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.aprint("LANDING BC STASHES EMPTY")
                else:
                    self.aprint("BEFORE: {}".format(self.known_empty_stashes))
                    if stash_position in self.known_empty_stashes:
                        self.known_empty_stashes.remove(stash_position)
                    self.aprint("AFTER: {}".format(self.known_empty_stashes))
                    self.current_task = Task.FETCH_BLOCK
                    self.task_history.append(self.current_task)
                    self.current_path = None
                    self.aprint("TRYING TO SWITCH TO FETCHING BLOCK FROM STASH")
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.CHECK_STASHES] += simple_distance(position_before, self.geometry.position)

    def fetch_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)
        # locate block, locations may be:
        # 1. known from the start (including block type)
        # 2. known roughly (in the case of block deposits/clusters)
        # 3. completely unknown; then the search is an important part
        # whether the own location relative to all this is known is also a question

        if self.current_path is None:
            # if (self.current_block_type_seed
            #     and all([s in self.known_empty_stashes for s in environment.seed_stashes.keys()])) \
            #         or (not self.current_block_type_seed
            #             and all([s in self.known_empty_stashes for s in environment.block_stashes.keys()])):
            if len(self.known_empty_stashes) == len(environment.seed_stashes) + len(environment.block_stashes):
                # i.e. the known-stashes thing would be reset at some point, e.g. each time returning to structure?
                self.current_task = Task.CHECK_STASHES
                self.task_history.append(self.current_task)
                self.current_path = None
                self.aprint("CHECKING STASHES ({})".format(self.known_empty_stashes))
                self.aprint("l1: {}, l2: {}, l3: {}".format(len(self.known_empty_stashes),
                                                            len(environment.seed_stashes),
                                                            len(environment.block_stashes)))
                self.check_stashes(environment)
                # self.aprint("LANDING (1)")
                return

            # first find the closest position that is not in the construction area anymore
            off_construction_locations = [
                (self.geometry.position[0], environment.offset_origin[1], self.geometry.position[2]),
                (self.geometry.position[0], environment.offset_origin[1] + Block.SIZE * self.target_map.shape[1],
                 self.geometry.position[2]),
                (environment.offset_origin[0], self.geometry.position[1], self.geometry.position[2]),
                (environment.offset_origin[0] + Block.SIZE * self.target_map.shape[2], self.geometry.position[1],
                 self.geometry.position[2])]

            off_construction_locations = sorted(off_construction_locations,
                                                key=lambda x: simple_distance(self.geometry.position, x))

            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # given an approximate location for blocks, go there to pick on eup
            min_stash_location = None
            min_distance = float("inf")
            stash_list = []
            compared_location = np.array(off_construction_locations[0])
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

            # TODO: check whether path has to be altered to avoid structure

            # stash locations sorted by some other measure but distance
            # stash_list = sorted(stash_list, key=lambda e: (-e[4], e[3], e[1], e[2]))
            # min_stash_location = stash_list[0][0]
            # min_stash_location = random.sample(stash_list, 1)[0][0]
            # self.aprint("Choosing stash with minimum number of agents there: {} (all stashes: {})"
            #             .format(stash_list[0][3], [e[3] for e in stash_list]))

            # construct path to that location
            # first add a point to get up to the level of movement for fetching blocks
            # which is one above the current construction level
            fetch_level_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                            self.required_spacing
            fetch_level_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + \
                            self.required_distance + self.geometry.size[2] * 1.5
            fetch_level_z = max(self.geometry.position[2] + self.geometry.size[2] * 2,
                                Block.SIZE * 2 + self.geometry.size[2] * 1.5)
            fetch_level_z = max(self.geometry.position[2], self.geometry.position[2] + Block.SIZE * 2)
            self.current_path = Path()
            # self.current_path.add_position([compared_location[0], compared_location[1], fetch_level_z])
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z],
                                           optional_distance=20)

            # TODO: need to find the direction with the fewest other agents (and simultaneously closest to perimeter)
            # either leave the structure in that direction only or constantly check whether there is free space
            # wherever the closest perimeter thing is
            # should the movement happen over the grid only (?)

        # if within a certain distance of the stash, check whether there are many other agents there
        # that should maybe be avoided (issue here might be that other it's is fairly likely due to
        # the low approach to the stashes that other agents push ones already there out of the way; in
        # that case it would this check might still do some good (?))
        if self.avoiding_crowded_stashes_enabled and \
                simple_distance(self.geometry.position[:2], self.current_path.positions[-1][:2]) < 50:
            count_at_stash = 0
            for a in environment.agents:
                if a is not self and simple_distance(
                        a.geometry.position[:2], self.current_path.positions[-1][:2]) <= self.stash_min_distance:
                    count_at_stash += 1
            # maybe don't make this a hard threshold though
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

        # assuming that the if-statement above takes care of setting the path:
        # collision detection should intervene here if necessary
        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, determine a block to pick up
            if not ret:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stash_position = self.current_stash_position
                if stash_position not in list(stashes.keys()):
                    # find closest
                    min_stash_distance = float("inf")
                    for p in list(stashes.keys()):
                        temp = simple_distance(self.geometry.position, p)
                        if temp < min_stash_distance:
                            min_stash_distance = temp
                            stash_position = p

                if len(stashes[stash_position]) == 0:
                    # remember that this stash is empty
                    self.known_empty_stashes.append(stash_position)
                    self.aprint("EMPTY STASH APPENDED IN FETCH_BLOCK")

                    # need to go to other stash, i.e. go to start of fetch_block again
                    self.current_path = None
                    self.aprint("STASH EMPTY AT {}".format(stash_position))
                    self.aprint("STASHES: {}".format(stashes))
                    return

                self.current_task = Task.PICK_UP_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FETCH_BLOCK] += simple_distance(position_before, self.geometry.position)

    def pick_up_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        # at this point it has been confirmed that there is indeed a block around that location
        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # determine the closest block in that stash (instead of using location, might want to use current_stash?)
            stash_position = tuple(self.geometry.position[:2])
            min_block = None
            min_distance = float("inf")
            if stash_position not in list(stashes.keys()):
                # find closest
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
                if not len(stashes[stash_position]) == 0:
                    for a in environment.agents:
                        if stashes[stash_position][0] is a.current_block:
                            self.aprint("AGENT {} OCCUPYING BLOCK".format(a.id))
                self.aprint("EMPTY STASH APPENDED IN PICK_UP_BLOCK")
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
                self.fetch_block(environment)
                return

            # stashes[stash_position].remove(min_block)

            # self.aprint("PICKING UP BLOCK: {}".format(min_block))

            # otherwise, make the selected block the current block and pick it up
            min_block.color = "red"
            self.current_path = Path()
            pickup_z = min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[2] / 2
            self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1], pickup_z])
            self.current_block = min_block
            # if self.current_block_type_seed:
            #     self.current_seed = self.current_block

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # attach block and move on to the transport_block task
            if not ret:
                if self.current_block is None:
                    self.aprint("AQUI ES EL PROBLEMO")
                    self.current_path = None
                    self.pick_up_block(environment)
                    return

                for a in environment.agents:
                    if a is not self and self.current_block is a.current_block:
                        self.aprint("CURRENT TARGET BLOCK PICKED UP BY OTHER AGENT {}".format(a.id))
                        self.aprint("")

                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stashes[tuple(self.geometry.position[:2])].remove(self.current_block)
                self.geometry.attached_geometries.append(self.current_block.geometry)
                if self.current_block_type_seed:
                    self.current_block.color = "#f44295"
                else:
                    self.current_block.color = "green"

                if self.rejoining_swarm and not self.current_block_type_seed:
                    self.current_task = Task.REJOIN_SWARM
                    self.aprint("SWITCHING TO REJOINING SWARM WITH CURRENT BLOCK {}".format(self.current_block))
                    self.aprint("")
                else:
                    self.current_task = Task.TRANSPORT_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None

                if self.current_block_type_seed and self.next_seed_position is None:
                    if self.current_grid_positions_to_be_seeded is not None:
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
                        # self.aprint("(4) self.next_seed_position = {}".format(self.next_seed_position))
                    else:
                        # try this stuff?
                        self.next_seed_position = self.current_seed.grid_position
                        # self.aprint("(5) self.next_seed_position = {}".format(self.next_seed_position))
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.PICK_UP_BLOCK] += simple_distance(position_before, self.geometry.position)

    def wait_on_perimeter(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # from the current position, start moving counter-clockwise around the perimeter of the construction
            # zone for now, maybe change this to the perimeter of the existing structure depending on its size
            # -> might also have to account for components, maybe using some maximum distance allowed (?)
            # -> could also consider ascending to a level a bit higher as to not disturb the movement of
            #    quadcopters below (although they should almost always be at construction level anyway)

            self.current_waiting_height = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + \
                                          self.required_distance + self.geometry.size[2] * 1.5 + \
                                          self.geometry.size[2] + Block.SIZE * 2

            # first calculate the next counter-clockwise corner point of the construction area and go there
            ordered_corner_points = environment.ccw_corner_locations(self.geometry.position[:2],
                                                                     self.geometry.size[2] * 1.5)

            # plan path to the first of these points (when arriving there, simply recalculate, I think)
            corner_point = ordered_corner_points[0]

            self.current_path = Path()
            self.current_path.add_position([corner_point[0], corner_point[1], self.current_waiting_height])

        # need to re-check the condition on staying outside the construction zone and if it is not fulfilled anymore,
        # somehow decide whether to enter or not (which might be problematic with multiple agents trying to do it)

        # agent_count = 0
        # for a in environment.agents:
        #     if environment.check_over_construction_area(a.geometry.position):
        #         agent_count += 1

        # if agent_count < self.max_agent_count:
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
                # check again whether the zone can be entered, if not move on
                # agent_count = 0
                # for a in environment.agents:
                #     if environment.check_over_construction_area(a.geometry.position):
                #         agent_count += 1

                # if agent_count < self.max_agent_count:
                if self.area_density_restricted and environment.density_over_construction_area() <= 1:
                    self.current_path = self.previous_path
                    self.current_task = Task.TRANSPORT_BLOCK
                    self.task_history.append(self.current_task)
                    return

                # calculate next most CCW site
                ordered_corner_points = environment.ccw_corner_locations(self.geometry.position[:2],
                                                                         self.geometry.size[2] * 1.5)
                corner_point = ordered_corner_points[0]
                self.current_path.add_position([corner_point[0], corner_point[1], self.current_waiting_height])
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.WAIT_ON_PERIMETER] += simple_distance(position_before, self.geometry.position)

    def rejoin_swarm(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        # while this is the task, should probably assign small probability to rejoining
        # 1. fly towards the structure (towards the remembered seed?)
        # 2. if the structure is higher than the current level, rise up as long as up as needed
        #    (do the same when looking for new seed)
        # 3. if the seed is still uncovered and no block at higher location was seen, assume that this is
        #    still the same level and search for new component (?)
        #    -> issue might be that there might not be anything at that point in the structure and it would
        #       take a very long time to find out that it is required to go up a level (or even multiple)
        #    -> best to survey the structure somehow? depending on what it looks like this could take very long
        # 3. if some other block has been spotted, go to that component's approximate location for orientation
        # => for now, we just cheat and select some seed on the latest component

        # if self.current_path is None:
        # if not self.rejoining_swarm and not ((self.collision_count / self.step_count) < 0.25 and random.random() < 0.01):
        # if not self.rejoining_swarm and random.random() > 0.0005:
        # if not self.rejoining_swarm and not ((self.collision_count / self.step_count) < 0.3 and random.random() < 1.0):
        if not self.rejoining_swarm and not ((self.collision_count / self.step_count) < 0.34 and random.random() < 0.9):
            return
        # best performing so far: was simply proportion < 0.34 and random < 0.9 and 0.1 probability in the other place
        # < 0.2 for other works fairly well too

        self.aprint("REALLY THO")

        self.rejoining_swarm = True
        self.wait_for_rejoining = False
        self.drop_out_of_swarm = False

        if self.current_block is None:
            self.current_task = Task.FETCH_BLOCK
            self.task_history.append(self.current_task)
            self.current_block_type_seed = False
            return

        self.aprint("WE GETTING SOMEWHERE")

        # for now cheating: find the highest layer
        highest_layer = self.current_structure_level
        for z in range(self.current_structure_level, self.target_map.shape[0]):
            if np.count_nonzero(environment.occupancy_map[z]) > 0:
                highest_layer = z

        for z in range(highest_layer + 1):
            for y in range(self.target_map.shape[1]):
                for x in range(self.target_map.shape[2]):
                    self.local_occupancy_map[z, y, x] = environment.occupancy_map[z, y, x]
        self.current_structure_level = highest_layer

        if self.check_structure_finished(self.local_occupancy_map):
            self.current_task = Task.LAND
            self.task_history.append(self.current_task)
            self.current_path = None
            return

        # now find some component on that layer
        candidate_components = []
        cc_grid_positions = []
        cc_locations = []
        for m in self.unfinished_component_markers(environment.occupancy_map, highest_layer):
            # for m in [cm for cm in np.unique(self.component_target_map[highest_layer]) if cm != 0]:
            #     if np.count_nonzero(self.local_occupancy_map[highest_layer][
            #                             self.component_target_map[highest_layer] == m] != 0) > 0:
            candidate_components.append(m)
            grid_position = self.component_seed_location(m)
            cc_grid_positions.append(grid_position)
            cc_locations.append([environment.offset_origin[0] + Block.SIZE * grid_position[0],
                                 environment.offset_origin[1] + Block.SIZE * grid_position[1],
                                 Block.SIZE * (0.5 + grid_position[2])])

        if len(candidate_components) == 0:
            highest_layer += 1
            for m in self.unfinished_component_markers(environment.occupancy_map, highest_layer):
                candidate_components.append(m)
                grid_position = self.component_seed_location(m)
                cc_grid_positions.append(grid_position)
                cc_locations.append([environment.offset_origin[0] + Block.SIZE * grid_position[0],
                                     environment.offset_origin[1] + Block.SIZE * grid_position[1],
                                     Block.SIZE * (0.5 + grid_position[2])])

        # what if the components are unseeded? need a different seed for orientation then

        self.aprint("Highest layer: {}".format(highest_layer))
        self.aprint("Markers: {}".format(self.unfinished_component_markers(environment.occupancy_map, highest_layer)))
        self.aprint("Local map: {}".format(environment.occupancy_map))

        order = sorted(range(len(candidate_components)),
                       key=lambda x: simple_distance(self.geometry.position, cc_locations[x]))

        order = sorted(order, key=lambda x: int(
            candidate_components[x] in self.unseeded_component_markers(environment.occupancy_map, highest_layer)))

        # just pick closest bc lazy
        candidate_components = [candidate_components[i] for i in order]
        cc_grid_positions = [cc_grid_positions[i] for i in order]
        cc_locations = [cc_locations[i] for i in order]

        # this is only possible because of the cheating (or if global information is used, although
        # in the latter case it would also be good to check whether a seed should be fetched instead
        self.current_component_marker = candidate_components[0]

        if self.current_component_marker in self.unseeded_component_markers(
                environment.occupancy_map, highest_layer):
            # the component is unseeded, therefore the current seed should be a different component's seed
            candidate_seed_components = []
            for m in [cm for cm in np.unique(self.component_target_map[highest_layer])
                      if cm != 0 and cm != self.current_component_marker]:
                if np.count_nonzero(environment.occupancy_map[highest_layer][
                                        self.component_target_map[highest_layer] == m] != 0) > 0:
                    candidate_seed_components.append(m)
            if len(candidate_seed_components) > 0:
                self.current_seed = environment.block_at_position(
                    self.component_seed_location(candidate_seed_components[0]))
                if self.current_seed is None:
                    self.aprint("(1) Seed at grid position {} for marker {} is None"
                                .format(candidate_seed_components[0], self.current_component_marker))
            elif highest_layer > 0:
                # do the same thing on a lower layer
                for m in [cm for cm in np.unique(self.component_target_map[highest_layer - 1]) if cm != 0]:
                    if np.count_nonzero(environment.occupancy_map[highest_layer - 1][
                                            self.component_target_map[highest_layer - 1] == m] != 0) > 0:
                        candidate_seed_components.append(m)
                self.current_seed = environment.block_at_position(
                    self.component_seed_location(candidate_seed_components[0]))
                if self.current_seed is None:
                    self.aprint("(2) Seed at grid position {} for marker {} is None"
                                .format(candidate_seed_components[0], self.current_component_marker))
            self.current_block_type_seed = False
            self.current_task = Task.RETURN_BLOCK
        else:
            self.current_seed = environment.block_at_position(cc_grid_positions[0])
            if self.current_seed is None:
                self.aprint("(3) Seed at grid position {} for marker {} is None"
                            .format(cc_grid_positions[0], self.current_component_marker))
            self.current_task = Task.TRANSPORT_BLOCK
        self.aprint("Current task: {}".format(self.current_task))
        self.aprint("Current seed: {}".format(self.current_seed.grid_position))
        self.aprint("Current proportion: {}".format((self.collision_count / self.step_count)))
        self.aprint("Known empty stashes: {}".format(self.known_empty_stashes))
        self.current_path = None
        self.task_history.append(self.current_task)

        self.per_task_distance_travelled[Task.REJOIN_SWARM] += simple_distance(position_before, self.geometry.position)

    def transport_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        # gain height to the "transport-to-structure" level

        # locate structure (and seed in particular), different ways of doing this:
        # 1. seed location is known (beacon)
        # 2. location has to be searched for

        # in this case the seed location is taken as the structure location,
        # since that is where the search for attachment sites would start anyway
        if self.current_path is None:
            self.current_visited_sites = None

            # gain height, fly to seed location and then start search for attachment site
            self.current_path = Path()
            transport_level_z = Block.SIZE * (self.current_structure_level + 3) + \
                                (self.geometry.size[2] / 2 + self.required_spacing) * 2
            transport_level_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + self.required_distance + \
                                self.geometry.size[2] * 1.5
            # take the first possible location to be seeded as that which should be seeded
            # seed_candidate_location = None
            # if self.current_grid_positions_to_be_seeded is not None:
            #     seed_candidate = self.current_grid_positions_to_be_seeded[0]
            #     seed_candidate_location = [environment.offset_origin[0] + seed_candidate[0] * Block.SIZE,
            #                                environment.offset_origin[1] + seed_candidate[1] * Block.SIZE,
            #                                (seed_candidate[2] + 0.5) * Block.SIZE]
            # seed_location = seed_candidate_location if self.current_block_type_seed \
            #     else self.current_seed.geometry.position
            seed_location = self.current_seed.geometry.position

            other_transport_level_z = (self.current_seed.grid_position[2] + 2) * Block.SIZE + self.geometry.size[2] * 2

            # self.aprint("Current seed at {}".format(self.current_seed.grid_position))
            # self.aprint("Next seed intended for {}".format(self.next_seed_position))
            # self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], transport_level_z],
            #                                optional_distance=70, axes=(0, 1))
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], transport_level_z],
                                           optional_distance=(70, 70, 20))
            self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z],
                                           optional_distance=30)
            self.current_path.add_position([seed_location[0], seed_location[1], other_transport_level_z])

        # TODO: if not over structure already, check whether close enough to wait
        if self.waiting_on_perimeter_enabled and not self.current_block_type_seed \
                and not environment.check_over_construction_area(self.geometry.position):
            # not over construction area yet
            if environment.distance_to_construction_area(self.geometry.position) <= self.geometry.size[0] * 2:
                if self.area_density_restricted and environment.density_over_construction_area() > 1:
                    # in this case it's too crowded -> don't move in yet
                    self.current_static_location = np.copy(self.geometry.position)
                    self.current_task = Task.WAIT_ON_PERIMETER
                    # self.aprint("Waiting on perimeter with {} agents in construction area"
                    #             .format(environment.count_over_construction_area()))
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

            # if the final point on the path has been reached, search for attachment site should start
            if not ret:
                self.seed_arrival_delay_queue.append(self.close_to_seed_count)
                self.close_to_seed_count = 0

                # since this method is also used to move to a seed site with a carried seed after already having
                # found the current seed for localisation, need to check whether we have arrived and should
                # drop off the carried seed
                self.current_path = None
                self.current_grid_position = np.copy(self.current_seed.grid_position)
                self.update_local_occupancy_map(environment)

                if self.current_block_type_seed and self.transporting_to_seed_site:
                    self.aprint("HAVE REACHED SITE FOR CARRIED SEED (DESTINATION: {})"
                                .format(self.next_seed_position))
                    # this means that we have arrived at the intended site for the seed, it should
                    # now be placed or, alternatively, a different site for it should be found
                    self.current_grid_position = np.array(self.next_seed_position)
                    if not environment.check_occupancy_map(self.next_seed_position):
                        self.aprint("GOING TO PLACE THE SEED (AT {})".format(self.current_grid_position))
                        # can place the carried seed
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_component_marker = self.component_target_map[self.current_grid_position[2],
                                                                                  self.current_grid_position[1],
                                                                                  self.current_grid_position[0]]
                    else:
                        self.aprint("NEED TO FIND DIFFERENT SEED SITE (ON LEVEL {})"
                                    .format(self.current_structure_level))
                        # self.aprint(True, self.local_occupancy_map)
                        # the position is already occupied, need to move to different site
                        # check whether there are even any unseeded sites
                        unseeded = self.unseeded_component_markers(self.local_occupancy_map)
                        unfinished = self.unfinished_component_markers(self.local_occupancy_map)
                        if len(unseeded) == 0 and len(unfinished) > 0:
                            # this might only be applicable if we know that the block stashes are all exhausted?
                            # should only count block stashes here:
                            counter = 0
                            for s in self.known_empty_stashes:
                                if s in environment.block_stashes.keys():
                                    counter += 1
                            if counter >= len(environment.block_stashes):
                                self.current_task = Task.MOVE_TO_PERIMETER
                                # self.current_grid_direction = [1, 0, 0]
                            else:
                                self.current_task = Task.RETURN_BLOCK
                        else:
                            if len(unseeded) == 0:
                                self.current_structure_level += 1
                            self.current_task = Task.FIND_NEXT_COMPONENT
                        self.task_history.append(self.current_task)
                    self.transporting_to_seed_site = False
                    return

                # self.aprint("HAVE REACHED END OF TRANSPORT PATH")

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
                        # here this is done assuming "perfect" knowledge, in reality more complicated
                        # search would probably have to be implemented
                        # for both cases, first determine the grid position for that component

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
                    self.aprint("FINISHED COMPONENT {} AFTER TRANSPORTING".format(self.current_component_marker))
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.TRANSPORT_BLOCK] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def move_to_perimeter(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # move to next block position in designated direction (which could be the shortest path or
            # just some direction chosen e.g. at the start, which is assumed here)
            directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])]
            counts = self.count_in_direction(environment)
            # distances would probably be pretty good as well (?)

            if any([c != 0 for c in counts]):
                sorter = sorted(range(len(directions)), key=lambda i: counts[i])
                directions = [directions[i] for i in sorter]
                self.current_grid_direction = directions[0]
            else:
                self.current_grid_direction = random.sample(directions, 1)[0]

            # self.current_grid_direction = random.sample(directions, 1)[0]

            self.current_path = Path()
            # self.current_grid_direction = shortest_direction_to_perimeter(
            #     self.local_occupancy_map[self.current_structure_level], self.current_grid_position[:2])
            destination_x = (self.current_grid_position + self.current_grid_direction)[0] * Block.SIZE + \
                            environment.offset_origin[0]
            destination_y = (self.current_grid_position + self.current_grid_direction)[1] * Block.SIZE + \
                            environment.offset_origin[1]
            self.current_path.add_position([destination_x, destination_y, self.geometry.position[2]])

            # TODO: move in direction with:
            # a) fewest agents
            # b) shortest (assumed) distance to perimeter
            # c) a mixture of the two
            # (d) soonest chance for attachment site finding (?) -> kinda difficult)
            # ACTUALLY, shortest path might be a bad idea (at least if seeds are in the middle of the structure),
            # because there is a good chance that there structure can only be extended in that direction, therefore
            # necessitating going around again (wait, is that even the case?), ACTUALLY NVM, PROBABLY FINE
            # but it might still not be desirable, option d) is probably the best

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.current_grid_position += self.current_grid_direction

                self.current_blocks_per_attachment += 1

                self.update_local_occupancy_map(environment)

                # TODO: need to update local_occupancy_map during find_attachment_site and recognise when sites
                # are being revisited, and in that case go back to the perimeter (which I think is already being done?)
                # could e.g. try to go to perimeter and when revisiting the same site on attachment site finding,
                # you know that you must be either in a hole or might have to move up a layer
                try:
                    # this is basically getting all values of the occupancy map at the locations where the hole map
                    # has the value of the hole which we are currently over
                    # checking only on the local occupancy map, because that is the only information actually available
                    # if this is successful
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
                    # have reached perimeter
                    # two possibilites:
                    # - carrying normal block -> should find an attachment site
                    # - carrying seed -> should do a survey of the current component
                    if not self.current_block_type_seed:
                        self.current_task = Task.FIND_ATTACHMENT_SITE
                    else:
                        self.current_task = Task.SURVEY_COMPONENT
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

        self.per_task_distance_travelled[Task.MOVE_TO_PERIMETER] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def survey_component(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_visited_sites is None:
            self.current_visited_sites = []

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.update_local_occupancy_map(environment)

                # if the component was considered to be unfinished but is not confirmed to be, go back to transport
                if self.check_component_finished(self.local_occupancy_map):
                    # TODO: should this be Task.FIND_NEXT_COMPONENT?
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
                    np.array([-self.current_grid_direction[1], self.current_grid_direction[0], 0], dtype="int32"),
                    lambda x: x == 0)

                # if block ahead, turn right
                # if position around corner empty, turn left
                # if neither of these, continue straight
                if position_ahead_occupied:
                    # turn right
                    self.current_grid_direction = np.array([self.current_grid_direction[1],
                                                            -self.current_grid_direction[0], 0],
                                                           dtype="int32")
                elif position_around_corner_empty:
                    # first move forward (to the corner)
                    self.current_path.add_position(self.geometry.position + Block.SIZE * self.current_grid_direction)
                    reference_position = self.current_path.positions[-1]

                    # then turn left
                    self.current_grid_position += self.current_grid_direction
                    self.current_grid_direction = np.array([-self.current_grid_direction[1],
                                                            self.current_grid_direction[0], 0],
                                                           dtype="int32")
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
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # need two different things depending on whether the agent has a block or not
            # first check whether we know that the current layer is already completed

            if self.check_layer_finished(self.local_occupancy_map):
                self.current_structure_level += 1

            # at this point the agent still has a block, and should therefore look for an attachment site
            # since we still have a block, we would like to place it, if possible somewhere with a seed
            candidate_components_placement = self.unfinished_component_markers(self.local_occupancy_map)
            candidate_components_seeding = self.unseeded_component_markers(self.local_occupancy_map)

            self.aprint("INFO: {}, {}, {}, {}\n{}".format(len(candidate_components_seeding),
                                                          self.current_block_type_seed,
                                                          self.check_layer_finished(self.local_occupancy_map),
                                                          self.current_structure_level,
                                                          self.local_occupancy_map))
            if len(candidate_components_seeding) == 0 and self.current_block_type_seed \
                    and self.check_layer_finished(self.local_occupancy_map):
                # if environment.global_information:
                #     self.current_task = Task.RETURN_BLOCK
                #     self.task_history.append(self.current_task)
                # else:
                # and self.check_layer_finished(self.local_occupancy_map):
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
                        # instead of the first that is available, might want the closest one instead
                        self.current_component_marker = m
                        self.current_seed = environment.block_at_position(self.component_seed_location(m))
                        self.current_task = Task.FETCH_BLOCK if self.current_block is None else Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.aprint("find_next_component: FOUND NEW COMPONENT ({}) IMMEDIATELY".format(m))
                        # TODO: even if this is the case, should probably switch to seeding other component after
                        return

            candidate_components = candidate_components_seeding if self.current_block_type_seed \
                else candidate_components_placement
            self.aprint("UNFINISHED COMPONENT MARKERS: {} ({})".format(candidate_components,
                                                                       self.current_structure_level))

            # first, need to figure out seed locations
            seed_grid_locations = []
            seed_locations = []
            for m in candidate_components:
                # if not self.current_block_type_seed and np.count_nonzero(self.component_target_map == m) == 1:
                #     continue
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
                # order this by number of agents over each component (?)
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

            # then plan a path to visit all seed locations as quickly as possible
            # while this may not be the best solution (NP-hardness, yay) it should not be terrible
            seed_grid_locations = [np.array(seed_grid_locations[i]) for i in order]
            seed_locations = [seed_locations[i] for i in order]

            search_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                       self.required_spacing
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

            # first_site = np.array([self.geometry.position[0], seed_locations[0][1], search_z])
            second_site = np.array([seed_locations[0][0], seed_locations[0][1], search_z])
            self.current_path.add_position(first_site)
            self.current_path.add_position(second_site)
            # for l in seed_locations:
            #     self.current_path.add_position([l[0], l[1], search_z])

            # TODO: make these paths follow the grid (i.e. always perpendicular)
            # also, since this is a good opportunity to avoid other agents, maybe should do this one by one?

            self.aprint("SEED PATH: {} ({})".format(self.current_path.positions, seed_grid_locations))
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
                self.aprint("self.current_seed_grid_position_index increased")
                self.current_seed_grid_position_index += 1

            if not can_skip and not ret:
                # if at a location where it can be seen whether the block location has been seeded,
                # check whether the position below has been seeded
                # current_seed_position = self.current_seed_grid_positions[
                #     self.current_path.current_index - (2 if ret else 1)]
                # self.aprint("AT LOCATION OF BLOCK THING: {}".format(self.current_seed_grid_position_index))
                # self.aprint("POSITION: {}".format(self.geometry.position))
                # self.aprint("PATH: {}".format(self.current_path.positions))
                # self.aprint("index: {}, length: {}".format(self.current_path.current_index, len(self.current_path.positions)))
                # self.aprint("ret: {}".format(ret))
                current_seed_position = self.current_seed_grid_positions[self.current_seed_grid_position_index - 1]
                self.current_grid_position = np.array(current_seed_position)
                self.update_local_occupancy_map(environment)
                if environment.check_occupancy_map(current_seed_position):
                    self.aprint("find_next_component: SEED AT {}".format(current_seed_position))
                    self.aprint("find_next_component: own location is {}".format(self.geometry.position))
                    self.local_occupancy_map[current_seed_position[2],
                                             current_seed_position[1],
                                             current_seed_position[0]] = 1
                    self.current_seed = environment.block_at_position(current_seed_position)
                    if self.current_block is not None:
                        self.aprint("find_next_component: still carrying block")
                        # check again whether there has been any change
                        if not self.check_component_finished(self.local_occupancy_map,
                                                             self.component_target_map[current_seed_position[2],
                                                                                       current_seed_position[1],
                                                                                       current_seed_position[0]]):
                            self.aprint("find_next_component: component not finished yet")
                            if not self.current_block_type_seed:
                                self.aprint("find_next_component: try to attach there")
                                # if it has, simply switch to that component and try to attach there
                                previous = self.current_component_marker
                                self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                          current_seed_position[1],
                                                                                          current_seed_position[0]]
                                self.current_seed = environment.block_at_position(current_seed_position)
                                # self.current_task = Task.FIND_ATTACHMENT_SITE
                                self.current_task = Task.MOVE_TO_PERIMETER
                                self.current_grid_direction = [1, 0, 0]
                                self.task_history.append(self.current_task)
                                self.current_grid_positions_to_be_seeded = None
                                self.current_path = None
                            else:
                                self.aprint("find_next_component: move on to next location (with seed)")
                                # need to move on to next location
                                if self.current_seed_grid_position_index + 1 > len(self.current_seed_grid_positions):
                                    # have not found a single location without seed, therefore return it and fetch block
                                    # might want to attach at the original component instead (?)
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
                            self.aprint("find_next_component: component is finished")
                            # would need to update going to the next position
                            self.current_path = None
                    else:
                        # otherwise, register as possible seed if all positions happen to be seeded (?)
                        if self.current_seeded_positions is None:
                            self.current_seeded_positions = []
                        # check whether that component is even in question
                        if not self.check_component_finished(self.local_occupancy_map,
                                                             self.component_target_map[current_seed_position[2],
                                                                                       current_seed_position[1],
                                                                                       current_seed_position[0]]):
                            # self.current_seeded_positions.append(current_seed_position)
                            self.current_task = Task.FETCH_BLOCK
                            self.task_history.append(self.current_task)
                            self.next_seed_position = current_seed_position
                            previous = self.current_component_marker
                            # self.aprint("(1) self.next_seed_position = {}".format(self.next_seed_position))
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                        else:
                            if self.check_structure_finished(self.local_occupancy_map):
                                self.current_task = Task.LAND
                                self.aprint("LANDING (7)")
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                # would need to update going to the next position
                                self.current_path = None
                else:
                    self.aprint("find_next_component: NO SEED AT {}".format(current_seed_position))
                    if self.current_block is not None:
                        if not self.current_block_type_seed:
                            # TODO: if every block is supposed to be returned once there is an opportunity for
                            # getting a seed, then this has to be changed here
                            if self.seed_if_possible_enabled:
                                self.aprint("RETURNING CURRENT (NORMAL) BLOCK TO GET SEED")
                                self.current_block_type_seed = False
                                self.current_task = Task.RETURN_BLOCK
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                if self.current_seed_grid_position_index + 1 > len(self.current_seed_grid_positions):
                                    self.aprint("RETURNING CURRENT (NORMAL) BLOCK SINCE THERE ARE NO SEEDS YET")
                                    if self.current_block is None:
                                        self.current_block_type_seed = True
                                    self.current_task = Task.RETURN_BLOCK
                                    self.task_history.append(self.current_task)
                                    self.current_path = None
                                else:
                                    self.aprint("MORE SEED POSITIONS TO CHECK OUT: {}"
                                           .format(self.current_seed_grid_positions))

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
                            previous = self.current_component_marker
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                    else:
                        self.aprint("REMEMBERING SEED LOCATION AND FETCHING SEED FOR ATTACHMENT")
                        # if we do not have a block currently, remember this site as the seed location and fetch seed
                        self.current_task = Task.FETCH_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_block_type_seed = True
                        self.next_seed_position = current_seed_position
                        previous = self.current_component_marker
                        self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                  current_seed_position[1],
                                                                                  current_seed_position[0]]
                        if self.current_component_marker == 11 and self.id == 8:
                            self.aprint("Current position: {}, previous: {}"
                                        .format(self.current_grid_position, previous))
                            self.aprint("")
                        self.current_path = None
                # in case we currently have a block:
                # if all locations turn out not to be seeded, initiate returning the current block to the
                # nearest stash and then go to the nearest seed stash and then try each registered location
                # for seeds based on distance

                # three possible scenarios:
                # 1. have a  normal construction block
                # 2. have a seed
                # 3. don't have any block
        else:
            # block_below = environment.block_below(self.geometry.position)
            # if block_below is not None and block_below.grid_position[2] == self.current_grid_position[2]:
            #     self.current_grid_position = np.copy(block_below.grid_position)
            #     self.update_local_occupancy_map(environment)
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FIND_NEXT_COMPONENT] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def return_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            off_construction_locations = [
                (self.geometry.position[0], environment.offset_origin[1], self.geometry.position[2]),
                (self.geometry.position[0], environment.offset_origin[1] + Block.SIZE * self.target_map.shape[1],
                 self.geometry.position[2]),
                (environment.offset_origin[0], self.geometry.position[1], self.geometry.position[2]),
                (environment.offset_origin[0] + Block.SIZE * self.target_map.shape[2], self.geometry.position[1],
                 self.geometry.position[2])]

            off_construction_locations = sorted(off_construction_locations,
                                                key=lambda x: simple_distance(self.geometry.position, x))

            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # select the closest block stash
            min_stash_location = None
            min_distance = float("inf")
            compared_location = np.array(off_construction_locations[0])
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

            # stash locations sorted by some other measure but distance
            # stash_list = sorted(stash_list, key=lambda e: (e[3], e[1], e[2]))
            # min_stash_location = stash_list[0][0]
            # min_stash_location = random.sample(stash_list, 1)[0][0]

            # plan a path there
            return_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                       self.required_spacing
            return_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + self.required_distance + \
                       self.geometry.size[2] * 1.5
            return_z = self.geometry.position[2]
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], return_z])
            # self.current_path.add_position([compared_location[0], compared_location[1], return_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], return_z],
                                           optional_distance=30)
            self.current_path.add_position([min_stash_location[0], min_stash_location[1],
                                            Block.SIZE + self.geometry.size[2] / 2])

        if self.avoiding_crowded_stashes_enabled \
                and simple_distance(self.geometry.position[:2], self.current_path.positions[-1][:2]) < 50:
            count_at_stash = 0
            for a in environment.agents:
                if a is not self and simple_distance(
                        a.geometry.position[:2], self.current_path.positions[-1][:2]) <= self.stash_min_distance:
                    count_at_stash += 1
            # maybe don't make this a hard threshold though
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
                self.aprint("Trying to return block {} (drop_out_of_swarm: {})"
                            .format(self.current_block, self.drop_out_of_swarm))
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                backup = []
                for s in self.known_empty_stashes:
                    if not all(s[i] == self.current_block.geometry.position[i] for i in range(2)):
                        backup.append(s)
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                if self.current_block_type_seed:
                    self.aprint("Block color should be set to {}".format(Block.COLORS_SEEDS[0]))
                    self.current_block.color = Block.COLORS_SEEDS[0]
                    environment.seed_stashes[self.current_stash_position].append(self.current_block)
                else:
                    self.aprint("Block color should be set to white")
                    self.current_block.color = "#FFFFFF"
                    environment.block_stashes[self.current_stash_position].append(
                        self.current_block)
                # self.aprint("RETURNING BLOCK TO STASH AT {}".format(tuple(self.current_block.geometry.position[:2])))
                # self.aprint("STASHES: {}".format(stashes))
                for key in environment.block_stashes:
                    for block in environment.block_stashes[key]:
                        if block.color == "green":
                            self.aprint("BLOCK RETURNED TO BLOCK STASH WITHOUT TURNING HWHITE")
                            self.aprint("")
                if self.current_block.color == "green":
                    self.aprint("BLOCK RETURNED STILL GREEN")
                    self.current_block.color = "#FFFFFF"
                    self.aprint("")
                self.known_empty_stashes = backup
                self.current_block = None
                self.current_path = None
                if self.drop_out_of_swarm:
                    self.current_task = Task.LAND
                else:
                    self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)

                self.current_block_type_seed = not self.current_block_type_seed
                self.returned_blocks += 1
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.RETURN_BLOCK] += simple_distance(position_before,
                                                                                    self.geometry.position)

    def land(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

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

            candidate_x = environment.environment_extent[0] + self.geometry.size[0] + self.required_distance
            candidate_y = environment.environment_extent[1] + self.geometry.size[0] + self.required_distance

            candidate_x = self.initial_position[0]
            candidate_y = self.initial_position[1]

            land_level_z = Block.SIZE * (self.current_structure_level + 2) + \
                           self.geometry.size[2] / 2 + self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, self.geometry.size[2] / 2])
            # self.aprint("SETTING PATH TO LAND: {}".format(self.current_path.positions))

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                if abs(self.geometry.position[2] - Block.SIZE / 2) > Block.SIZE / 2:
                    self.aprint("FINISHED WITHOUT LANDING")
                    self.aprint("PATH POSITIONS: {}\nPATH INDEX: {}".format(self.current_path.positions,
                                                                            self.current_path.current_index))
                    self.aprint("POSITION IN QUESTION: {}".format(
                        self.current_path.positions[self.current_path.current_index]))
                    self.aprint("LAST 10 TASKS: {}".format(self.task_history[-10:]))
                    self.aprint("HAPPENING IN AGENT: {}".format(self))
                    self.aprint("placeholder")
                    self.current_path = None
                if self.current_block is not None:
                    self.aprint("LANDING WITH BLOCK STILL ATTACHED")
                    self.aprint("LAST 20 TASKS: {}".format(self.task_history[-10:]))
                    self.aprint("what")
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

    @abstractmethod
    def advance(self, environment: env.map.Map):
        pass
