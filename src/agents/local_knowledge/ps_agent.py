import numpy as np
import random
import env.map
from agents.agent import Task, Agent, aprint, check_map
from env.block import Block
from env.util import print_map
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance, rotation_2d


class PerimeterFollowingAgentLocal(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(PerimeterFollowingAgentLocal, self).__init__(position, size, target_map, required_spacing)

    def check_stashes(self, environment: env.map.Map):
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
            aprint(self.id, "ORDERED STASH POSITIONS: {}".format(ordered_stash_positions))
            index = ordered_stash_positions.index(min_stash_position)
            self.current_stash_path = ordered_stash_positions[index:]
            self.current_stash_path.extend(ordered_stash_positions[:index])
            self.current_stash_path_index = 0

            check_level_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([min_stash_position[0], min_stash_position[1], check_level_z])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                stash_position = tuple(self.geometry.position[:2])

                # check whether there are actually blocks at this location
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                current_stash = stashes[stash_position]

                aprint(self.id, "REACHED STASH {} (OWN POSITION {})"
                       .format(current_stash, stash_position))
                aprint(self.id, "STASHES: {}".format(stashes))

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
                        aprint(self.id, "EMPTY STASH APPENDED IN CHECK_STASHES")
                    # move on to next stash
                    if self.current_stash_path_index + 1 < len(self.current_stash_path):
                        self.current_stash_path_index += 1
                        next_position = self.current_stash_path[self.current_stash_path_index]
                        aprint(self.id, "MOVING ON TO NEXT STASH {}".format(next_position))
                        aprint(self.id, "(index: {}, stash path: {})"
                               .format(self.current_stash_path_index, self.current_stash_path))
                        self.current_path.add_position([next_position[0], next_position[1], self.geometry.position[2]])
                    else:
                        self.current_task = Task.LAND
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        aprint(self.id, "LANDING BC STASHES EMPTY")
                else:
                    aprint(self.id, "BEFORE: {}".format(self.known_empty_stashes))
                    if stash_position in self.known_empty_stashes:
                        self.known_empty_stashes.remove(stash_position)
                    aprint(self.id, "AFTER: {}".format(self.known_empty_stashes))
                    self.current_task = Task.FETCH_BLOCK
                    self.task_history.append(self.current_task)
                    self.current_path = None
                    aprint(self.id, "TRYING TO SWITCH TO FETCHING BLOCK FROM STASH")
        else:
            self.geometry.position = self.geometry.position + current_direction

    def fetch_block(self, environment: env.map.Map):
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
                # TODO: should maybe re-check stashes because blocks might be brought back
                # i.e. the known-stashes thing would be reset at some point, e.g. each time returning to structure?
                self.current_task = Task.CHECK_STASHES
                self.task_history.append(self.current_task)
                self.current_path = None
                aprint(self.id, "CHECKING STASHES ({})".format(self.known_empty_stashes))
                aprint(self.id, "l1: {}, l2: {}, l3: {}".format(len(self.known_empty_stashes),
                                                                len(environment.seed_stashes),
                                                                len(environment.block_stashes)))
                self.check_stashes(environment)
                # aprint(self.id, "LANDING (1)")
                return

            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # given an approximate location for blocks, go there to pick on eup
            min_stash_location = None
            min_distance = float("inf")
            for p in list(stashes.keys()):
                if p not in self.known_empty_stashes:
                    temp = simple_distance(self.geometry.position, p)
                    if temp < min_distance:
                        min_distance = temp
                        min_stash_location = p

            if min_stash_location is None:
                if self.current_block_type_seed:
                    self.current_block_type_seed = False
                else:
                    self.current_block_type_seed = True
                self.next_seed_position = None
                self.current_path = None
                # TODO: instead of landing, should check whether there are still seeds
                return

            # aprint(self.id, "FETCHING BLOCK FROM {}".format(min_block_location))

            # construct path to that location
            # first add a point to get up to the level of movement for fetching blocks
            # which is one above the current construction level
            fetch_level_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                            self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z])

        # assuming that the if-statement above takes care of setting the path:
        # collision detection should intervene here if necessary
        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, determine a block to pick up
            if not ret:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stash_position = tuple(self.geometry.position[:2])
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
                    aprint(self.id, "EMPTY STASH APPENDED IN FETCH_BLOCK")

                    # need to go to other stash, i.e. go to start of fetch_block again
                    self.current_path = None
                    aprint(self.id, "STASH EMPTY AT {}".format(stash_position))
                    aprint(self.id, "STASHES: {}".format(stashes))
                    return

                self.current_task = Task.PICK_UP_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

    def pick_up_block(self, environment: env.map.Map):
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
                            aprint(self.id, "AGENT {} OCCUPYING BLOCK".format(a.id))
                aprint(self.id, "EMPTY STASH APPENDED IN PICK_UP_BLOCK")
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
                return

            # stashes[stash_position].remove(min_block)

            # aprint(self.id, "PICKING UP BLOCK: {}".format(min_block))

            # otherwise, make the selected block the current block and pick it up
            min_block.color = "red"
            self.current_path = Path()
            pickup_z = min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[2] / 2
            self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1], pickup_z])
            self.current_block = min_block
            # if self.current_block_type_seed:
            #     self.current_seed = self.current_block

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # attach block and move on to the transport_block task
            if not ret:
                if self.current_block is None:
                    aprint(self.id, "AQUI ES EL PROBLEMO")
                    self.current_path = None
                    self.pick_up_block(environment)
                    return
                for a in environment.agents:
                    if a is not self and self.current_block is a.current_block:
                        aprint(self.id, "CURRENT TARGET BLOCK PICKED UP BY OTHER AGENT {}".format(a.id))
                        print()
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stashes[tuple(self.geometry.position[:2])].remove(self.current_block)
                self.geometry.attached_geometries.append(self.current_block.geometry)
                if self.current_block_type_seed:
                    self.current_block.color = "#f44295"
                else:
                    self.current_block.color = "green"
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
                        # aprint(self.id, "(4) self.next_seed_position = {}".format(self.next_seed_position))
                    else:
                        # try this stuff?
                        self.next_seed_position = self.current_seed.grid_position
                        # aprint(self.id, "(5) self.next_seed_position = {}".format(self.next_seed_position))
        else:
            self.geometry.position = self.geometry.position + current_direction

    def transport_block(self, environment: env.map.Map):
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

            # aprint(self.id, "Current seed at {}".format(self.current_seed.grid_position))
            # aprint(self.id, "Next seed intended for {}".format(self.next_seed_position))
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                            transport_level_z])
            self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, search for attachment site should start
            if not ret:
                # since this method is also used to move to a seed site with a carried seed after already having
                # found the current seed for localisation, need to check whether we have arrived and should
                # drop off the carried seed
                self.current_path = None
                self.current_grid_position = np.copy(self.current_seed.grid_position)
                self.update_local_occupancy_map(environment)

                if self.current_block_type_seed and self.transporting_to_seed_site:
                    aprint(self.id, "HAVE REACHED SITE FOR CARRIED SEED (DESTINATION: {})"
                           .format(self.next_seed_position))
                    # this means that we have arrived at the intended site for the seed, it should
                    # now be placed or, alternatively, a different site for it should be found
                    self.current_grid_position = np.array(self.next_seed_position)
                    if not environment.check_occupancy_map(self.next_seed_position):
                        aprint(self.id, "GOING TO PLACE THE SEED (AT {})".format(self.current_grid_position))
                        # can place the carried seed
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_component_marker = self.component_target_map[self.current_grid_position[2],
                                                                                  self.current_grid_position[1],
                                                                                  self.current_grid_position[0]]
                    else:
                        aprint(self.id, "NEED TO FIND DIFFERENT SEED SITE (ON LEVEL {})"
                               .format(self.current_structure_level))
                        # print_map(self.local_occupancy_map)
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

                # aprint(self.id, "HAVE REACHED END OF TRANSPORT PATH")

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
                        self.current_path = Path()
                        self.current_path.add_position([seed_x, seed_y, self.geometry.position[2]])
                        self.transporting_to_seed_site = True
                else:
                    # the position is unexpectedly occupied, therefore the local map should be updated
                    self.current_visited_sites = None
                    block_above_seed = environment.block_at_position(position_above)
                    # aprint(self.id, "BLOCK_ABOVE_SEED AT POSITION {}, THEREFORE FILLING OUT".format(position_above))
                    # aprint(self.id, "BBFORE")
                    # print_map(environment.occupancy_map)
                    # aprint(self.id, "BEFORE")
                    # print_map(self.local_occupancy_map)
                    # for layer in range(block_above_seed.grid_position[2]):
                    #     self.local_occupancy_map[layer][self.target_map[layer] != 0] = 1
                    # aprint(self.id, "AFTER")
                    # print_map(self.local_occupancy_map)
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
                                seed_z = Block.SIZE * (self.current_structure_level + 1) + \
                                         self.geometry.size[2] / 2 + self.required_spacing
                                self.current_path = Path()
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
                        if self.current_structure_level == 9:
                            aprint(self.id, "WHAT THE FUCK IS GOING ON HERE")
                            aprint(self.id, "STRUCTURE LEVEL {}".format(self.current_structure_level))
                            aprint(self.id, "SEED POSITION {}".format(self.current_seed.grid_position))
                            aprint(self.id, "GRID POSITION {}".format(self.current_grid_position))
                        seed_grid_location = self.component_seed_location(
                            self.component_target_map[block_above_seed.grid_position[2],
                                                      block_above_seed.grid_position[1],
                                                      block_above_seed.grid_position[0]],
                            self.current_grid_position[2] + 1)
                        self.current_seed = environment.block_at_position(seed_grid_location)

                        seed_x = environment.offset_origin[0] + seed_grid_location[0] * Block.SIZE
                        seed_y = environment.offset_origin[1] + seed_grid_location[1] * Block.SIZE
                        seed_z = Block.SIZE * (self.current_structure_level + 1) + \
                                 self.geometry.size[2] / 2 + self.required_spacing
                        self.current_path = Path()
                        self.current_path.add_position([seed_x, seed_y, self.geometry.position[2]])

                if self.check_component_finished(self.local_occupancy_map):
                    aprint(self.id, "FINISHED COMPONENT {} AFTER TRANSPORTING".format(self.current_component_marker))
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

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

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                self.current_grid_position += self.current_grid_direction

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

    def find_attachment_site(self, environment: env.map.Map):
        # orientation happens counter-clockwise -> follow seed edge in that direction once its reached
        # can either follow the perimeter itself or just fly over blocks (do the latter for now)
        if self.current_path is None:
            # might consider putting something here
            aprint(self.id, "current_path is None in find_attachment_site")
            self.current_path = Path()
            self.current_path.add_position(self.geometry.position)

        # use the following list to keep track of the combination of visited attachment sites and the direction of
        # movement at that point in time; if, during the same "attachment search" one such "site" is revisited, then
        # the agent is stuck in a loop, most likely caused by being trapped in a hole
        if self.current_visited_sites is None:
            self.current_visited_sites = []

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # aprint(self.id, "Current local occupancy map:")
            # print_map(self.local_occupancy_map)

            if not ret:
                self.update_local_occupancy_map(environment)

                # corner of the current block reached, assess next action
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(environment, self.current_grid_position)

                # check whether location is somewhere NORTH-EAST of any closing corner, i.e. the block should not be
                # placed there before closing that loop (NE because currently all closing corners are located there)
                allowable_region_attachable = True
                if not at_loop_corner:
                    closing_corners = self.closing_corners[self.current_structure_level][self.current_component_marker]
                    for i in range(len(closing_corners)):
                        x, y, z = closing_corners[i]
                        orientation = self.closing_corner_orientations[self.current_structure_level][
                            self.current_component_marker][i]
                        if not environment.check_occupancy_map(np.array([x, y, z])):
                            if orientation == "NW":
                                if x >= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "NE":
                                if x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SW":
                                if x >= self.current_grid_position[0] and y >= self.current_grid_position[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SE":
                                if x <= self.current_grid_position[0] and y >= self.current_grid_position[1]:
                                    allowable_region_attachable = False
                                    break

                current_site_tuple = (tuple(self.current_grid_position), tuple(self.current_grid_direction))
                if current_site_tuple in self.current_visited_sites:
                    # there are two options here: either the current component is finished, or we are trapped in a hole
                    # check if in hole
                    if check_map(self.hole_map, self.current_grid_position, lambda x: x > 1):
                        # aprint(self.id, "REVISITED SITE (IN HOLE)")
                        # aprint(self.id, "self.current_grid_position = {}".format(self.current_grid_position))
                        # aprint(self.id, "HOLE MAP:\n{}".format(self.hole_map))
                        self.current_task = Task.MOVE_TO_PERIMETER
                        self.task_history.append(self.current_task)
                        self.current_grid_direction = [1, 0, 0]  # should probably use expected SP here
                        self.local_occupancy_map[self.hole_boundaries[self.hole_map[self.current_grid_position[2],
                                                                                    self.current_grid_position[1],
                                                                                    self.current_grid_position[0]]]] = 1
                    else:
                        # aprint(self.id, "REVISITED SITE (ON PERIMETER), MOVING COMPONENTS")
                        # aprint(self.id, "self.current_component_marker = {}".format(self.current_component_marker))
                        # aprint(self.id, "current_site_tuple = {}".format(current_site_tuple))
                        # aprint(self.id, "self.current_seed.grid_position = {}".format(self.current_seed.grid_position))
                        # aprint(self.id, "self.current_visited_sites = {}".format(self.current_visited_sites))
                        self.local_occupancy_map[self.component_target_map == self.current_component_marker] = 1
                        self.current_task = Task.FIND_NEXT_COMPONENT
                        self.task_history.append(self.current_task)
                    self.current_path = None
                    self.current_visited_sites = None
                    return

                # adding location and direction here to check for revisiting
                if self.current_row_started:
                    self.current_visited_sites.append(current_site_tuple)

                # the checks need to determine whether the current position is a valid attachment site
                position_ahead_occupied = environment.check_occupancy_map(
                    self.current_grid_position + self.current_grid_direction)
                position_ahead_to_be_empty = check_map(
                    self.target_map, self.current_grid_position + self.current_grid_direction, lambda x: x == 0)
                position_around_corner_empty = environment.check_occupancy_map(
                    self.current_grid_position + self.current_grid_direction +
                    np.array([-self.current_grid_direction[1], self.current_grid_direction[0], 0], dtype="int32"),
                    lambda x: x == 0)
                row_ending = self.current_row_started and (position_ahead_to_be_empty or position_around_corner_empty)

                if loop_corner_attachable and allowable_region_attachable and \
                        check_map(self.target_map, self.current_grid_position) and \
                        (position_ahead_occupied or row_ending):
                    if ((environment.check_occupancy_map(self.current_grid_position + np.array([1, 0, 0])) and
                         environment.check_occupancy_map(self.current_grid_position + np.array([-1, 0, 0]))) or
                        (environment.check_occupancy_map(self.current_grid_position + np.array([0, 1, 0])) and
                         environment.check_occupancy_map(self.current_grid_position + np.array([0, -1, 0])))) and \
                            not environment.check_occupancy_map(self.current_grid_position, lambda x: x > 0):
                        self.current_task = Task.LAND
                        self.current_visited_sites = None
                        self.current_path = None
                        self.logger.error("CASE 1-3: Attachment site found, but block cannot be placed at {}."
                                          .format(self.current_grid_position))
                        aprint(self.id, "LANDING (2)")
                        print_map(self.target_map)
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
                    if position_ahead_occupied:
                        # turn right
                        self.current_grid_direction = np.array([self.current_grid_direction[1],
                                                                -self.current_grid_direction[0], 0],
                                                               dtype="int32")
                        self.logger.debug("CASE 2: Position straight ahead occupied, turning clockwise.")
                    elif position_around_corner_empty:
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
                        self.logger.debug(
                            "CASE 3: Reached corner of structure, turning counter-clockwise. {} {}".format(
                                self.current_grid_position, self.current_grid_direction))
                        self.current_path.add_position(reference_position + Block.SIZE * self.current_grid_direction)
                        self.current_row_started = True
                    else:
                        # otherwise site "around the corner" occupied -> continue straight ahead
                        self.current_grid_position += self.current_grid_direction
                        self.logger.debug("CASE 4: Adjacent positions ahead occupied, continuing to follow perimeter.")
                        self.current_path.add_position(
                            self.geometry.position + Block.SIZE * self.current_grid_direction)
                        self.current_row_started = True

                if self.check_component_finished(self.local_occupancy_map):
                    aprint(self.id, "FINISHED COMPONENT {} AFTER MOVING TO NEXT BLOCK IN ATTACHMENT SITE"
                           .format(self.current_component_marker))
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

    def survey_component(self, environment: env.map.Map):
        if self.current_visited_sites is None:
            self.current_visited_sites = []

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
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
            self.local_occupancy_map[self.current_grid_position[2],
                                     self.current_grid_position[1],
                                     self.current_grid_position[0]] = 1
            # a different agent has already placed the block in the meantime
            # TODO: should probably change this to something that is faster depending on what block you're carrying
            self.current_path = None
            # TODO: figure out whether the below works or if it should be changed back to TRANSPORT_BLOCK
            self.current_task = Task.TRANSPORT_BLOCK  # should maybe be find_attachment_site?
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
                if self.current_block.is_seed:
                    self.current_seed = self.current_block
                    self.next_seed_position = None

                if self.current_block.geometry.position[2] > (self.current_grid_position[2] + 0.5) * Block.SIZE:
                    self.logger.error("BLOCK PLACED IN AIR ({})".format(self.current_grid_position[2]))
                    self.current_path.add_position(
                        np.array([self.geometry.position[0], self.geometry.position[1],
                                  (self.current_grid_position[2] + 1) * Block.SIZE + self.geometry.size[2] / 2]))
                    return

                environment.place_block(self.current_grid_position, self.current_block)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                self.local_occupancy_map[self.current_grid_position[2],
                                         self.current_grid_position[1],
                                         self.current_grid_position[0]] = 1
                self.current_block.placed = True
                self.current_block.grid_position = self.current_grid_position
                self.current_block = None
                self.current_path = None
                self.current_visited_sites = None
                self.transporting_to_seed_site = False

                self.update_local_occupancy_map(environment)
                if self.check_structure_finished(self.local_occupancy_map) \
                        or (self.check_layer_finished(self.local_occupancy_map)
                            and self.current_structure_level >= self.target_map.shape[0] - 1):
                    aprint(self.id, "AFTER PLACING BLOCK: FINISHED")
                    self.current_task = Task.LAND
                    aprint(self.id, "LANDING (3)")
                elif self.check_component_finished(self.local_occupancy_map):
                    aprint(self.id, "AFTER PLACING BLOCK: FINDING NEXT COMPONENT")
                    print_map(self.local_occupancy_map)
                    self.current_task = Task.FIND_NEXT_COMPONENT
                else:
                    aprint(self.id, "AFTER PLACING BLOCK: FETCHING BLOCK FOR COMPONENT {} (PREVIOUS WAS SEED: {})"
                           .format(self.current_component_marker, self.current_block_type_seed))
                    print_map(self.local_occupancy_map)
                    self.current_task = Task.FETCH_BLOCK
                    if self.current_block_type_seed:
                        self.current_block_type_seed = False
                self.task_history.append(self.current_task)

                for y in range(self.local_occupancy_map.shape[1]):
                    for x in range(self.local_occupancy_map.shape[2]):
                        if self.local_occupancy_map[self.current_grid_position[2], y, x] != 0 \
                                and self.target_map[self.current_grid_position[2], y, x] == 0:
                            aprint(self.id, "LOCAL OCCUPANCY MAP IS WRONG")
                            print()
        else:
            self.geometry.position = self.geometry.position + current_direction

    def find_next_component(self, environment: env.map.Map):
        if self.current_path is None:
            # need two different things depending on whether the agent has a block or not
            # first check whether we know that the current layer is already completed

            if self.check_layer_finished(self.local_occupancy_map):
                self.current_structure_level += 1

            # at this point the agent still has a block, and should therefore look for an attachment site
            # since we still have a block, we would like to place it, if possible somewhere with a seed
            candidate_components_placement = self.unfinished_component_markers(self.local_occupancy_map)
            candidate_components_seeding = self.unseeded_component_markers(self.local_occupancy_map)

            aprint(self.id, "INFO: {}, {}, {}, {}\n{}".format(len(candidate_components_seeding),
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
                        aprint(self.id, "find_next_component: FOUND NEW COMPONENT ({}) IMMEDIATELY".format(m))
                        return

            candidate_components = candidate_components_seeding if self.current_block_type_seed \
                else candidate_components_placement
            aprint(self.id, "UNFINISHED COMPONENT MARKERS: {} ({})".format(candidate_components,
                                                                           self.current_structure_level))

            if len(candidate_components) == 0:
                # if carrying block: there are no unfinished components left, have to move up layer?
                # if carrying seed: there are no unseeded components left, need to check whether component is finished
                pass

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

            # then plan a path to visit all seed locations as quickly as possible
            # while this may not be the best solution (NP-hardness, yay) it should not be terrible
            seed_grid_locations = [np.array(seed_grid_locations[i]) for i in order]
            seed_locations = [seed_locations[i] for i in order]

            search_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], search_z])
            self.current_path.inserted_sequentially[self.current_path.current_index] = False
            for l in seed_locations:
                self.current_path.add_position([l[0], l[1], search_z])

            aprint(self.id, "SEED PATH: {} ({})".format(self.current_path.positions, seed_grid_locations))
            self.current_seed_grid_positions = seed_grid_locations
            self.current_seed_grid_position_index = 0

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position

            can_skip = not self.current_path.inserted_sequentially[self.current_path.current_index]
            if self.current_path.current_index != 0 and not can_skip:
                self.current_seed_grid_position_index += 1

            ret = self.current_path.advance()
            if ret and np.array_equal(self.geometry.position, self.current_path.positions[-1]):
                ret = False

            if not can_skip and (self.current_seed_grid_position_index > 0 or not ret):
                # if at a location where it can be seen whether the block location has been seeded,
                # check whether the position below has been seeded
                # current_seed_position = self.current_seed_grid_positions[
                #     self.current_path.current_index - (2 if ret else 1)]
                aprint(self.id, "AT LOCATION OF BLOCK THING: {}".format(self.current_seed_grid_position_index))
                aprint(self.id, "POSITION: {}".format(self.geometry.position))
                aprint(self.id, "PATH: {}".format(self.current_path.positions))
                aprint(self.id, "index: {}, length: {}".format(self.current_path.current_index, len(self.current_path.positions)))
                aprint(self.id, "ret: {}".format(ret))
                current_seed_position = self.current_seed_grid_positions[self.current_seed_grid_position_index - 1]
                self.current_grid_position = np.array(current_seed_position)
                self.update_local_occupancy_map(environment)
                if environment.check_occupancy_map(current_seed_position):
                    aprint(self.id, "find_next_component: SEED AT {}".format(current_seed_position))
                    # print_map(self.local_occupancy_map)
                    self.local_occupancy_map[current_seed_position[2],
                                             current_seed_position[1],
                                             current_seed_position[0]] = 1
                    self.current_seed = environment.block_at_position(current_seed_position)
                    if self.current_block is not None:
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
                                # self.current_task = Task.FIND_ATTACHMENT_SITE
                                self.current_task = Task.MOVE_TO_PERIMETER
                                self.current_grid_direction = [1, 0, 0]
                                self.task_history.append(self.current_task)
                                self.current_grid_positions_to_be_seeded = None
                                self.current_path = None
                            else:
                                # need to move on to next location
                                if not ret:
                                    # have not found a single location without seed, therefore return it and fetch block
                                    self.current_task = Task.RETURN_BLOCK
                                    self.task_history.append(self.current_task)
                                    self.current_path = None
                        else:
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
                            # aprint(self.id, "(1) self.next_seed_position = {}".format(self.next_seed_position))
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                        else:
                            if self.check_structure_finished(self.local_occupancy_map):
                                self.current_task = Task.LAND
                                aprint(self.id, "LANDING (7)")
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                self.current_path = None
                else:
                    aprint(self.id, "find_next_component: NO SEED AT {}".format(current_seed_position))
                    aprint(self.id, "occupancy map:\n{}".format(environment.occupancy_map))
                    # if self.current_grid_positions_to_be_seeded is None:
                    #     self.current_grid_positions_to_be_seeded = []
                    if self.current_block is not None:
                        if not self.current_block_type_seed:
                            # remember that location as to-be-seeded and continue on path until reaching the end
                            # in_list = False
                            # for p in self.current_grid_positions_to_be_seeded:
                            #     if all(p[i] == current_seed_position[i] for i in range(3)):
                            #         in_list = True
                            # if not in_list:
                            #     self.current_grid_positions_to_be_seeded.append(current_seed_position)
                            if not ret:
                                aprint(self.id, "RETURNING CURRENT (NORMAL) BLOCK SINCE THERE ARE NO SEEDS YET")
                                if self.current_block is None:
                                    self.current_block_type_seed = True
                                self.current_task = Task.RETURN_BLOCK
                                self.task_history.append(self.current_task)
                                self.current_path = None
                            else:
                                aprint(self.id, "MORE SEED POSITIONS TO CHECK OUT: {}"
                                       .format(self.current_seed_grid_positions))
                                aprint(self.id, "PATH (with index {} at location {}): {}".format(
                                    self.current_path.current_index, self.geometry.position, self.current_path.positions))
                        else:
                            # can place the seed here
                            self.current_task = Task.PLACE_BLOCK
                            self.task_history.append(self.current_task)
                            self.next_seed_position = current_seed_position
                            # aprint(self.id, "(2) self.next_seed_position = {}".format(self.next_seed_position))
                            self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                      current_seed_position[1],
                                                                                      current_seed_position[0]]
                            self.current_path = None
                    else:
                        aprint(self.id, "REMEMBERING SEED LOCATION AND FETCHING SEED FOR ATTACHMENT")
                        # if we do not have a block currently, remember this site as the seed location and fetch seed
                        self.current_task = Task.FETCH_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_block_type_seed = True
                        self.next_seed_position = current_seed_position
                        # aprint(self.id, "(3) self.next_seed_position = {}".format(self.next_seed_position))
                        self.current_component_marker = self.component_target_map[current_seed_position[2],
                                                                                  current_seed_position[1],
                                                                                  current_seed_position[0]]
                        # self.current_grid_positions_to_be_seeded.append(current_seed_position)
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
            self.geometry.position = self.geometry.position + current_direction

    def return_block(self, environment: env.map.Map):
        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # select the closest block stash
            min_stash_location = None
            min_distance = float("inf")
            for key, value in stashes.items():
                temp = simple_distance(self.geometry.position, key)
                if temp < min_distance:
                    min_stash_location = key
                    min_distance = temp

            # plan a path there
            return_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + \
                self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], return_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], return_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1],
                                            Block.SIZE + self.geometry.size[2] / 2])

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
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
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                if self.current_block_type_seed:
                    self.current_block.color = Block.COLORS_SEEDS[0]
                    environment.seed_stashes[tuple(self.current_block.geometry.position[:2])].append(self.current_block)
                else:
                    self.current_block.color = "#FFFFFF"
                    environment.block_stashes[tuple(self.current_block.geometry.position[:2])].append(self.current_block)
                # aprint(self.id, "RETURNING BLOCK TO STASH AT {}".format(tuple(self.current_block.geometry.position[:2])))
                # aprint(self.id, "STASHES: {}".format(stashes))
                for key in environment.block_stashes:
                    for block in environment.block_stashes[key]:
                        if block.color == "green":
                            aprint(self.id, "BLOCK RETURNED TO BLOCK STASH WITHOUT TURNING HWHITE")
                            print()
                if self.current_block.color == "green":
                    aprint(self.id, "BLOCK RETURNED STILL GREEN")
                    print()
                self.known_empty_stashes = backup
                self.current_block = None
                self.current_path = None
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)
                if not self.current_block_type_seed:
                    # decide on where to place the seed that is to be fetched
                    # order grid positions based on distance from here
                    pass

                self.current_block_type_seed = not self.current_block_type_seed
        else:
            self.geometry.position = self.geometry.position + current_direction

    def new_avoid_collision(self, environment: env.map.Map):
        # get list of agents for which there is collision potential/danger and which are below/visible
        # could/should probably make detection of possible colliding agents more realistic
        collision_danger_agents = []
        for a in environment.agents:
            if self is not a and self.collision_potential(a) \
                    and a.geometry.position[2] <= self.geometry.position[2] - self.required_vertical_distance:
                collision_danger_agents.append(a)

        not_dodging = True
        for a in collision_danger_agents:
            not_dodging = not_dodging and \
                          simple_distance(a.geometry.position, self.current_seed.geometry.position) > \
                          simple_distance(self.geometry.position, self.current_seed.geometry.position)

        # TODO PROBLEM: DEADLOCK WHEN TWO AGENTS DECIDE THEY SHOULD DODGE -> RANDOMNESS FOR TIEBREAKER?

        # self does not even move towards the structure, evade
        # if self.current_task in [Task.LAND, Task.FETCH_BLOCK, Task.FINISHED]:
        #     not_dodging = True

        # stuff that could be done if heading was known:
        # - if QC below rising -> dodge sideways
        # - if QC below lowering -> dodge upwards as needed (slightly)
        # - if QC below moving sideways -> dodge upwards as needed and if possible go in different sideways direction
        # - if QC same level and moving in same direction (roughly), don't change anything, slow down a bit (?)
        # - if QC same level and moving in opposite direction (roughly), whichever is <some condition> rises to avoid
        # - if QC same level and rising -> sideways/lower avoidance
        # - if QC same level and lowering -> upwards

        # if self carries no block and some of the others carry one, then simply evade
        if (self.current_block is None or self.current_block.geometry not in self.geometry.following_geometries) and \
                any([a.current_block is not None and a.current_block.geometry in a.geometry.following_geometries
                     for a in collision_danger_agents]):
            not_dodging = False

        if not_dodging:
            self.current_task = self.previous_task
            self.task_history.append(self.current_task)
            # self.current_path = None if self.previous_path is None else self.previous_path
            # maybe insert this short path into the current path and then remove it once dodging is completed
            if self.current_path is not None and self.current_path.dodging_path:
                self.current_path = None
            if self.current_path is not None and len(self.current_path.inserted_paths) > 0:
                self.current_path.remove_path(self.current_path.current_index)
            return
        else:
            # decide on direction to go into
            if self.current_path is None:
                self.current_path = Path(dodging_path=True)
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                                               self.current_path.current_index)
            else:
                next_position = self.current_path.next()
                dodging_path = Path(dodging_path=True)
                dodging_path.add_position([next_position[0], next_position[1],
                                           self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5])
                self.current_path.insert_path(dodging_path)
                # self.current_path.add_position([next_position[0], next_position[1],
                #                                 self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                #                                self.current_path.current_index)
            # TODO: instead of simply going up, also move in direction of next goal

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

    def avoid_collision(self, environment: env.map.Map):
        # self.new_avoid_collision(environment)
        # return

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
        if self.current_task in [Task.LAND, Task.FETCH_BLOCK, Task.FINISHED]:
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
            if self.path_before_collision_avoidance_none:
                self.current_path = None
            self.current_collision_avoidance_counter = 0
            return
        else:
            # decide on direction to go into
            if self.current_path is None:
                if self.current_collision_avoidance_counter == 0:
                    self.path_before_collision_avoidance_none = True
                self.current_path = Path()
                self.current_path.add_position([self.geometry.position[0], self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                                               self.current_path.current_index)
                self.current_path.inserted_indices.append(self.current_path.current_index)
                self.current_path.number_inserted_positions = 1
            else:
                if self.current_collision_avoidance_counter == 0:
                    self.path_before_collision_avoidance_none = False
                next_position = self.current_path.next()
                self.current_path.add_position([next_position[0], next_position[1],
                                                self.geometry.position[2] + self.geometry.size[2] + Block.SIZE + 5],
                                               self.current_path.current_index)
                self.current_path.inserted_indices.append(self.current_path.current_index)
                self.current_path.number_inserted_positions += 1
            # TODO: instead of simply going up, also move in direction of next goal

        self.current_collision_avoidance_counter += 1

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()
            if not ret:
                self.current_collision_avoidance_counter = 0
        else:
            self.geometry.position = self.geometry.position + current_direction

    def land(self, environment: env.map.Map):
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

            land_level_z = Block.SIZE * (self.current_structure_level + 2) + \
                           self.geometry.size[2] / 2 + self.required_spacing
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, land_level_z])
            self.current_path.add_position([candidate_x, candidate_y, self.geometry.size[2] / 2])
            # aprint(self.id, "SETTING PATH TO LAND: {}".format(self.current_path.positions))

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))

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
                    self.current_path = None
                if self.current_block is not None:
                    aprint(self.id, "LANDING WITH BLOCK STILL ATTACHED")
                    aprint(self.id, "LAST 20 TASKS: {}".format(self.task_history[-10:]))
                    aprint(self.id, "what")
                self.current_task = Task.FINISHED
                self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

    def advance(self, environment: env.map.Map):
        # determine current task:
        # fetch block
        # bring block to structure (i.e. until it comes into sight)
        # find attachment site
        # place block
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]

        if self.id == 3:
            self.logger.warning("Agent 6 still being called (task: {}).".format(self.current_task))

        if self.check_structure_finished(self.local_occupancy_map):
            self.current_task = Task.LAND
            aprint(self.id, "LANDING (8)")
            self.task_history.append(self.current_task)

        self.agent_statistics.step(environment)

        for b in environment.blocks:
            if not b.placed and all([b.geometry.position[i] == (350.0, 0.0)[i] for i in range(2)]) \
                    and b.color == "red" and b not in environment.block_stashes[(350.0, 0.0)]:
                aprint(self.id, "TARGETED BLOCK NOT IN STASH ANYMORE")
                for a in environment.agents:
                    if b is a.current_block:
                        aprint(self.id, "BLOCK BELONGING TO: {}".format(a.id))
                print()

        if self.current_task == Task.AVOID_COLLISION:
            self.avoid_collision(environment)
        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.PICK_UP_BLOCK:
            self.pick_up_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.MOVE_TO_PERIMETER:
            self.move_to_perimeter(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.FIND_NEXT_COMPONENT:
            self.find_next_component(environment)
        elif self.current_task == Task.SURVEY_COMPONENT:
            self.survey_component(environment)
        elif self.current_task == Task.RETURN_BLOCK:
            self.return_block(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.CHECK_STASHES:
            self.check_stashes(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a) \
                        and a.geometry.position[2] <= self.geometry.position[2] - self.required_vertical_distance:
                    # aprint(self.id, "INITIATING HIGH-LEVEL COLLISION AVOIDANCE")
                    self.previous_task = self.current_task
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    if self.current_path is None:
                        self.path_before_collision_avoidance_none = True
                    break
