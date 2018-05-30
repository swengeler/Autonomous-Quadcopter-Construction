import numpy as np
import random
import env.map
from agents.agent import Agent, Task, check_map
from agents.local_knowledge.lk_agent import LocalKnowledgeAgent
from env.block import Block
from env.util import shortest_path, legal_attachment_sites
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance


class ShortestPathAgentLocal(LocalKnowledgeAgent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(ShortestPathAgentLocal, self).__init__(position, size, target_map, required_spacing, printing_enabled)
        self.current_shortest_path = None
        self.current_sp_index = 0
        self.illegal_sites = []
        self.current_attachment_info = None

    def find_attachment_site(self, environment: env.map.Map):
        # self.aprint("Calling find_attachment_site (path is None: {})".format(self.current_path is None))
        # if self.id == 4 and self.current_component_marker == 15:
        #     self.aprint("Current path when calling attachment site search is {}".format(self.current_path))
        #     self.aprint("self.current_shortest_path = {}".format(self.current_shortest_path))
        #     self.aprint("")

        # hacky solution for the None problem, but here goes:
        if self.current_path is None or self.current_shortest_path is None:
            self.update_local_occupancy_map(environment)

            if self.check_component_finished(self.local_occupancy_map, self.current_component_marker):
                self.aprint("Component {} finished, moving on".format(self.current_component_marker))
                self.find_next_component(environment)
                return

            # TODO: if coming back with same state of map (?),
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites, corner_sites, protruding_sites, most_ccw_sites = \
                legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                       self.local_occupancy_map[self.current_structure_level],
                                       component_marker=self.current_component_marker, local_info=True)

            self.aprint("DETERMINING ATTACHMENT SITES ON LEVEL {} WITH MARKER {} AND SEED AT {}:"
                        .format(self.current_structure_level,
                                self.current_component_marker,
                                self.current_seed.grid_position))
            # self.aprint(True, attachment_sites)
            # self.aprint("CURRENT MAP")
            # self.aprint(True, self.local_occupancy_map[self.current_structure_level])
            # self.aprint("CURRENT COMPONENT MAP")
            # self.aprint(True, self.component_target_map[self.current_structure_level])
            # print(self.component_target_map[self.current_structure_level])
            self.aprint("corner_sites = {}, protruding_sites = {}, most_ccw_sites = {}"
                        .format(corner_sites, protruding_sites, most_ccw_sites))

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(self.local_occupancy_map[self.current_structure_level])

            backup = []
            for site in corner_sites:
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                if not at_loop_corner:
                    closing_corners = self.closing_corners[self.current_structure_level][self.current_component_marker]
                    for i in range(len(closing_corners)):
                        x, y, z = closing_corners[i]
                        orientation = self.closing_corner_orientations[self.current_structure_level][
                            self.current_component_marker][i]
                        if not environment.check_occupancy_map(np.array([x, y, z])):
                            if orientation == "NW":
                                if x >= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "NE":
                                if x <= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SW":
                                if x >= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SE":
                                if x <= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    break
                if loop_corner_attachable and allowable_region_attachable:
                    backup.append(site)
                else:
                    self.aprint("CORNER SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                                .format(site, loop_corner_attachable, allowable_region_attachable))
            corner_sites = backup

            backup = []
            for site in protruding_sites:
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                if not at_loop_corner:
                    closing_corners = self.closing_corners[self.current_structure_level][self.current_component_marker]
                    for i in range(len(closing_corners)):
                        x, y, z = closing_corners[i]
                        orientation = self.closing_corner_orientations[self.current_structure_level][
                            self.current_component_marker][i]
                        if not environment.check_occupancy_map(np.array([x, y, z])):
                            if orientation == "NW":
                                if x >= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "NE":
                                if x <= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SW":
                                if x >= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    break
                            elif orientation == "SE":
                                if x <= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    break
                # NOTE: actually not sure if the allowable region even plays into this because by their very nature
                # these sites could pretty much only be corners themselves and not obstruct anything (?)
                if loop_corner_attachable and allowable_region_attachable:
                    backup.append(site)
                else:
                    self.aprint("PROTRUDING SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                                .format(site, loop_corner_attachable, allowable_region_attachable))
                    self.aprint("")
            protruding_sites = backup

            backup = []
            for site, direction, expected_length in most_ccw_sites:
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                thing = None
                if not at_loop_corner:
                    closing_corners = self.closing_corners[self.current_structure_level][self.current_component_marker]
                    for i in range(len(closing_corners)):
                        x, y, z = closing_corners[i]
                        orientation = self.closing_corner_orientations[self.current_structure_level][
                            self.current_component_marker][i]
                        if not environment.check_occupancy_map(np.array([x, y, z])):
                            if orientation == "NW":
                                if x >= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    thing = ((x, y, z), orientation)
                                    break
                            elif orientation == "NE":
                                if x <= site[0] and y <= site[1]:
                                    allowable_region_attachable = False
                                    thing = ((x, y, z), orientation)
                                    break
                            elif orientation == "SW":
                                if x >= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    thing = ((x, y, z), orientation)
                                    break
                            elif orientation == "SE":
                                if x <= site[0] and y >= site[1]:
                                    allowable_region_attachable = False
                                    thing = ((x, y, z), orientation)
                                    break
                # in this case, might have to block attachment if row reaches into either of these
                # regions, but I'm not sure that this can happen
                if loop_corner_attachable and allowable_region_attachable:
                    backup.append((site, direction, expected_length))
                else:
                    self.aprint("CCW SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                                .format(site, loop_corner_attachable, allowable_region_attachable))
                    self.aprint("THING: {}".format(thing))
            most_ccw_sites = backup

            # find the closest corner or protruding site
            # if there are none then take the closest most CCW site
            most_ccw_sites_used = False
            if len(corner_sites) != 0:
                attachment_sites = corner_sites
                self.aprint("USING CORNER SITES")
            elif len(protruding_sites) != 0:
                attachment_sites = protruding_sites
                self.aprint("USING PROTRUDING SITES")
            else:
                most_ccw_sites_used = True
                attachment_sites = []
                for site, _, _ in most_ccw_sites:
                    attachment_sites.append(site)
                self.aprint("USING END-OF-ROW SITES")

            attachment_sites = []
            attachment_sites.extend([(s,) for s in corner_sites])
            attachment_sites.extend([(s,) for s in protruding_sites])
            attachment_sites.extend(most_ccw_sites)

            most_ccw_sites_used = False

            if len(attachment_sites) == 0:
                self.aprint("NO LEGAL ATTACHMENT SITES AT LEVEL {} WITH MARKER {}"
                            .format(self.current_structure_level, self.current_component_marker))
                self.aprint("LOCAL MAP:\n{}".format(self.local_occupancy_map))

            # for now just take the shortest distance to CCW sites as well, to improve efficiency could include
            # the expected time to find an attachment sites, i.e. the expected length of the row

            # find the closest one
            shortest_paths = []
            number_adjacent_blocks = []
            for tpl in attachment_sites:
                x, y = tpl[0]
                occupancy_map_copy[y, x] = 1
                sp = shortest_path(occupancy_map_copy, (self.current_grid_position[0],
                                                        self.current_grid_position[1]), (x, y))
                occupancy_map_copy[y, x] = 0
                shortest_paths.append(sp)
                counter = 0
                for y2 in (y - 1, y + 1):
                    if 0 <= y2 < occupancy_map_copy.shape[0] and occupancy_map_copy[y2, x] != 0:
                        counter += 1
                for x2 in (x - 1, x + 1):
                    if 0 <= x2 < occupancy_map_copy.shape[1] and occupancy_map_copy[y, x2] != 0:
                        counter += 1
                number_adjacent_blocks.append(counter)
            if most_ccw_sites_used:
                sorted_indices = sorted(range(len(attachment_sites)),
                                        key=lambda i: len(shortest_paths[i]) + most_ccw_sites[i][2])
            else:
                sorted_indices = sorted(range(len(attachment_sites)), key=lambda i: len(shortest_paths[i]))

            directions = [np.array([s[0][0] - self.current_grid_position[0], s[0][1] - self.current_grid_position[1]])
                          for s in attachment_sites]
            counts = self.direction_agent_count(environment, directions=directions, angle=np.pi / 2)
            self.aprint("Directions: {}, Counts: {}".format(directions, counts))
            sorted_indices = sorted(sorted_indices, key=lambda i: counts[i])

            # sorted_indices = random.sample(sorted_indices, len(sorted_indices))

            sorted_indices = sorted(range(len(attachment_sites)),
                                    key=lambda i: abs(attachment_sites[i][0][0] - self.current_grid_position[0]) +
                                                  abs(attachment_sites[i][0][1] - self.current_grid_position[1]))

            attachment_sites = [attachment_sites[i] for i in sorted_indices]
            shortest_paths = [shortest_paths[i] for i in sorted_indices]

            new_sp = [(self.current_grid_position[0], attachment_sites[0][0][1]),
                      (attachment_sites[0][0][0], attachment_sites[0][0][1])]
            # the initial direction of this shortest path would maybe be good to decide based on

            sp = shortest_paths[0]
            # sp = new_sp

            # other option: find "bends" in the path and only require going there
            if len(sp) > 1:
                diffs = [(sp[1][0] - sp[0][0], sp[1][1] - sp[0][1])]
                new_sp = [sp[0]]
                for i in range(1, len(sp)):
                    if i < len(sp) - 1:
                        diff = (sp[i + 1][0] - sp[i][0], sp[i + 1][1] - sp[i][1])
                        if diff[0] != diffs[-1][0] or diff[1] != diffs[-1][1]:
                            # a "bend" is happening
                            new_sp.append((sp[i]))
                            self.aprint("'Bend' at: {}".format(sp[i]))
                        diffs.append(diff)
                new_sp.append(sp[-1])
                self.aprint("Former SP: {}".format(sp))

                sp = new_sp

            self.aprint("Shortest path to attachment site: {}".format(sp))

            if not all(sp[-1][i] == attachment_sites[0][0][i] for i in range(2)):
                self.aprint("SHORTEST PATH DOESN'T LEAD TO INTENDED ATTACHMENT SITE")
                self.aprint("Local map:\n{}".format(self.local_occupancy_map))
                self.aprint("")

            # if the CCW sites are used, then store the additional required information
            # also need to make sure to reset this to None because it will be used for checks
            self.current_attachment_info = None
            if most_ccw_sites_used:
                most_ccw_sites = [most_ccw_sites[i] for i in sorted_indices]
                self.current_attachment_info = most_ccw_sites[0]
            if len(attachment_sites[0]) > 1:
                self.current_attachment_info = attachment_sites[0]

            # construct the path to that site (might want to consider doing this step-wise instead)
            self.current_grid_position = np.array([sp[0][0], sp[0][1], self.current_structure_level])
            self.current_path = Path()
            position = [environment.offset_origin[0] + Block.SIZE * self.current_grid_position[0],
                        environment.offset_origin[0] + Block.SIZE * self.current_grid_position[1],
                        self.geometry.position[2]]
            self.current_path.add_position(position)

            # storing information about the path
            self.current_shortest_path = sp
            self.current_sp_index = 0

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                block_below = environment.block_below(self.geometry.position)
                if block_below is not None and block_below.grid_position[2] > self.current_structure_level:
                    # there is a block below that is higher than the current layer
                    # since layers are built sequentially this means that the current layer must be completed
                    # therefore, this is noted in the local map and we move up to the layer of that block
                    # and start looking for some component
                    for layer in range(block_below.grid_position[2]):
                        self.local_occupancy_map[layer][self.target_map[layer] != 0] = 1
                    self.current_structure_level = block_below.grid_position[2]

                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_grid_position = block_below.grid_position
                    self.current_path = None

                    return

                # have reached next point on shortest path to attachment site
                if self.current_shortest_path is None:
                    self.aprint("shortest path None (previous path None: {})".format(
                        self.path_before_collision_avoidance_none))
                if self.current_attachment_info is not None and self.current_sp_index >= len(
                        self.current_shortest_path):
                    self.current_grid_position = self.current_grid_position + self.current_grid_direction
                else:
                    current_spc = self.current_shortest_path[self.current_sp_index]
                    self.current_grid_position = np.array(
                        [current_spc[0], current_spc[1], self.current_structure_level])
                    self.aprint("REACHED {} ON SHORTEST PATH ({})".format(current_spc, self.current_shortest_path))
                    self.aprint("Own position: {}".format(self.geometry.position))
                    if self.id == 4 and self.current_component_marker == 15:
                        self.aprint("WHAT THE ACTUAL FUCK IS GOING ON HERE")
                        self.aprint("")
                self.update_local_occupancy_map(environment)
                self.aprint("GRID POSITION: {}".format(self.current_grid_position))
                if self.current_sp_index >= len(self.current_shortest_path) - 1:
                    self.aprint("TEST")
                    if self.current_attachment_info is None:
                        self.aprint("TEST 1")
                        # if the attachment site was determined definitively (corner or protruding), have reached
                        # intended attachment site and should assess whether block can be placed or not
                        if not environment.check_occupancy_map(self.current_grid_position):
                            # if yes, just place it
                            self.current_task = Task.PLACE_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            self.current_shortest_path = None
                            self.aprint("GOING TO PLACE BLOCK AT CORNER OR PROTRUDING SITE")
                        else:
                            # if no, need to find new attachment site (might just be able to restart this method)
                            self.current_path = None
                            self.current_shortest_path = None
                            self.aprint("SITE ALREADY OCCUPIED, SEARCH FOR NEW ONE")
                    else:
                        self.aprint("TEST 2")
                        # otherwise, if at starting position for the row/column/perimeter search, need to check
                        # whether attachment is already possible and if not, plan next path movement
                        # note that due to the code above, the local map should already be updated, making it
                        # possible to decide whether we are at an outer corner globally
                        self.current_sp_index += 1
                        self.current_grid_direction = self.current_attachment_info[1]
                        if environment.check_occupancy_map(self.current_grid_position):
                            self.aprint("TEST 3")
                            self.aprint("Map before:\n{}".format(self.local_occupancy_map))
                            # if there is a block at the position, it means that we are still at the end of the
                            # shortest path and all the sites should have been filled already, meaning that the
                            # local occupancy matrix for that row/column can be filled out as well
                            # while within map bounds:
                            #     check whether next site in row/column (according to direction) is still
                            #     "connected" in the target map and if so fill it, else break
                            position = self.current_grid_position.copy() + self.current_grid_direction
                            # site actually also needs to be connected
                            # could either do this according to local occupancy map or (if that results in problems)
                            # with checking component completeness, could "physically" explore this row to be sure
                            # that it is finished/to update the local occupancy map up until the next intended gap
                            while 0 <= position[0] < self.target_map.shape[2] \
                                    and 0 <= position[1] < self.target_map.shape[1] \
                                    and check_map(self.target_map, position) \
                                    and check_map(self.local_occupancy_map, position):
                                self.local_occupancy_map[position[2], position[1], position[0]] = 1
                                position = position + self.current_grid_direction
                            # afterwards, continue search for an attachment site
                            self.current_path = None
                            # self.current_shortest_path = None
                            self.aprint("Map after:\n{}".format(self.local_occupancy_map))
                        else:
                            self.aprint("TEST 4")
                            # check whether attachment possible right now
                            position_ahead_occupied = environment.check_occupancy_map(
                                self.current_grid_position + self.current_grid_direction)
                            position_ahead_to_be_empty = check_map(
                                self.target_map, self.current_grid_position + self.current_grid_direction,
                                lambda x: x == 0)
                            position_around_corner_empty = environment.check_occupancy_map(
                                self.current_grid_position + self.current_grid_direction +
                                np.array([-self.current_grid_direction[1], self.current_grid_direction[0], 0],
                                         dtype="int32"),
                                lambda x: x == 0)

                            if position_ahead_occupied or position_around_corner_empty or position_ahead_to_be_empty:
                                # attachment possible
                                self.current_task = Task.PLACE_BLOCK
                                self.task_history.append(self.current_task)
                                self.current_path = None
                                self.current_shortest_path = None
                                self.current_attachment_info = None
                                self.aprint("GOING TO PLACE BLOCK AT END OF ROW")
                            else:
                                # cannot place block yet, move on to next location
                                next_grid_position = self.current_grid_position + self.current_grid_direction
                                next_position = [environment.offset_origin[0] + Block.SIZE * next_grid_position[0],
                                                 environment.offset_origin[0] + Block.SIZE * next_grid_position[1],
                                                 self.geometry.position[2]]
                                self.current_path.add_position(next_position)
                else:
                    if self.current_attachment_info is None:
                        # if the attachment site was determined definitively (corner or protruding), have reached
                        # intended attachment site and should assess whether block can be placed or not
                        if check_map(self.local_occupancy_map, np.array([self.current_shortest_path[-1][0],
                                                                         self.current_shortest_path[-1][1],
                                                                         self.current_grid_position[2]])):
                            # if no, need to find new attachment site (might just be able to restart this method)
                            self.current_path = None
                            self.current_shortest_path = None
                            return

                            # still have to go on, therefore update the current path
                    self.current_sp_index += 1
                    next_spc = self.current_shortest_path[self.current_sp_index]
                    next_position = [environment.offset_origin[0] + Block.SIZE * next_spc[0],
                                     environment.offset_origin[0] + Block.SIZE * next_spc[1],
                                     self.geometry.position[2]]
                    self.current_path.add_position(next_position)

                if self.check_component_finished(self.local_occupancy_map):
                    self.aprint("FINISHED COMPONENT {} FINISHED AFTER MOVING TO NEXT BLOCK IN ATTACHMENT SITE"
                                .format(self.current_component_marker))
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            block_below = environment.block_below(self.geometry.position)
            # also need to check whether block is in shortest path
            if block_below is not None and block_below.grid_position[2] == self.current_grid_position[2] \
                    and self.component_target_map[block_below.grid_position[2],
                                                  block_below.grid_position[1],
                                                  block_below.grid_position[0]] == self.current_component_marker:
                # the following might not be enough, might have to check whether block was in the original SP
                if self.current_sp_index < len(self.current_shortest_path) - 1:
                    self.current_grid_position = np.copy(block_below.grid_position)
                    self.update_local_occupancy_map(environment)
            self.geometry.position = self.geometry.position + current_direction

    def place_block(self, environment: env.map.Map):
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
            self.current_path = None
            self.current_task = Task.TRANSPORT_BLOCK  # should maybe be find_attachment_site?
            self.task_history.append(self.current_task)
            return

        # check again whether attachment is allowed since other agents placing blocks there may have made it illegal
        # NOTE that the assumption here is that it can fairly easily be verified whether the surrounding sites
        # are making
        attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                  self.local_occupancy_map[self.current_structure_level],
                                                  component_marker=self.current_component_marker)
        if not self.current_block_type_seed \
                and attachment_sites[self.current_grid_position[1], self.current_grid_position[0]] == 0:
            self.current_path = None
            self.current_task = Task.TRANSPORT_BLOCK
            self.task_history.append(self.current_task)
            self.aprint("SITE FOR PLACEMENT HAS BECOME ILLEGAL")
            # self.illegal_sites.append(
            #     (self.current_grid_position[0], self.current_grid_position[1], self.current_structure_level))
            # MAJOR CHEATING HERE
            return

        next_position = self.current_path.next()
        current_direction = self.current_path.direction_to_next(self.geometry.position)
        current_direction /= sum(np.sqrt(current_direction ** 2))
        current_direction *= Agent.MOVEMENT_PER_STEP
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # block should now be placed in the environment's occupancy matrix
            if not ret:
                if self.current_block.is_seed:
                    self.current_seed = self.current_block
                    self.next_seed_position = None

                if self.current_block.geometry.position[2] > (self.current_grid_position[2] + 1.0) * Block.SIZE:
                    self.logger.error("BLOCK PLACED IN AIR ({}, {}, {})".format(
                        self.current_grid_position, self.id, self.current_block.geometry.position))
                    self.current_path.add_position(
                        np.array([self.geometry.position[0], self.geometry.position[1],
                                  (self.current_grid_position[2] + 1) * Block.SIZE + self.geometry.size[2] / 2]))
                    return

                if self.rejoining_swarm:
                    self.rejoining_swarm = False

                self.agent_statistics.attachment_interval.append(self.count_since_last_attachment)
                self.count_since_last_attachment = 0

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

                if self.check_structure_finished(self.local_occupancy_map) \
                        or (self.check_layer_finished(self.local_occupancy_map)
                            and self.current_structure_level >= self.target_map.shape[0] - 1):
                    self.aprint("AFTER PLACING BLOCK: FINISHED")
                    self.current_task = Task.LAND
                    self.aprint("LANDING (3)")
                elif self.check_component_finished(self.local_occupancy_map):
                    self.aprint("AFTER PLACING BLOCK: FINDING NEXT COMPONENT")
                    self.aprint(True, self.local_occupancy_map)
                    self.current_task = Task.FIND_NEXT_COMPONENT
                else:
                    self.aprint("AFTER PLACING BLOCK: FETCHING BLOCK (PREVIOUS WAS SEED: {})"
                                .format(self.current_block_type_seed))
                    self.current_task = Task.FETCH_BLOCK
                    if self.current_block_type_seed:
                        self.current_block_type_seed = False
                self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

    def advance(self, environment: env.map.Map):
        if self.current_task == Task.FINISHED:
            return

        # if self.step_count > 0:
        #     self.aprint("Current proportion: {}".format((self.collision_count / self.step_count)))

        # this is only for testing obviously
        # if not self.rejoining_swarm and not self.drop_out_of_swarm and self.step_count > 1000 \
        #         and (self.collision_count / self.step_count) > 0.35 \
        #         and random.random() < 0.1 \
        #         and self.current_task in [Task.RETURN_BLOCK,
        #                                   Task.FETCH_BLOCK,
        #                                   Task.TRANSPORT_BLOCK,
        #                                   Task.CHECK_STASHES,
        #                                   Task.WAIT_ON_PERIMETER]:
        #     if self.current_block is not None:
        #         self.current_task = Task.RETURN_BLOCK
        #     else:
        #         self.current_task = Task.LAND
        #     self.drop_out_of_swarm = True
        #     self.wait_for_rejoining = False
        #     self.rejoining_swarm = False
        #     self.current_path = None
        #     self.aprint("SETTING VARIABLES FOR LEAVING")
        # 
        # if self.wait_for_rejoining or self.current_task == Task.REJOIN_SWARM:
        #     self.rejoin_swarm(environment)

        if self.current_seed is None:
            self.current_seed = environment.blocks[0]
            self.local_occupancy_map[self.current_seed.grid_position[2],
                                     self.current_seed.grid_position[1],
                                     self.current_seed.grid_position[0]] = 1
            self.current_component_marker = self.component_target_map[self.current_seed.grid_position[2],
                                                                      self.current_seed.grid_position[1],
                                                                      self.current_seed.grid_position[0]]

        if self.initial_position is None:
            self.initial_position = np.copy(self.geometry.position)

        if self.current_task == Task.MOVE_TO_PERIMETER and not self.current_block_type_seed:
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.current_path = None  # path is not reset before MOVE_TO_PERIMETER, thus this has to be reset
            self.current_shortest_path = None
            self.aprint("Changing task from MOVE_TO_PERIMETER to FIND_ATTACHMENT_SITE")
            self.aprint("component marker is: {}".format(self.current_component_marker))

        if self.current_task != Task.LAND and self.check_structure_finished(self.local_occupancy_map):
            self.current_task = Task.LAND
            self.aprint("LANDING (8)")
            self.task_history.append(self.current_task)

        self.agent_statistics.step(environment)

        if self.current_task == Task.AVOID_COLLISION:
            self.avoid_collision(environment)

        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.PICK_UP_BLOCK:
            self.pick_up_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.WAIT_ON_PERIMETER:
            self.wait_on_perimeter(environment)
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
        elif self.current_task == Task.CHECK_STASHES:
            self.check_stashes(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.current_task == Task.MOVE_TO_PERIMETER and not self.current_block_type_seed:
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.current_path = None
            self.current_shortest_path = None

        if self.current_task != Task.FINISHED:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70 \
                    and self.current_path is not None and not self.wait_for_rejoining:
                self.aprint("STUCK")
                self.current_path.add_position([self.geometry.position[0],
                                                self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] * random.random()],
                                               self.current_path.current_index)

        self.position_queue.append(self.geometry.position.copy())

        collision_danger = False
        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    self.previous_task = self.current_task
                    # self.current_task = Task.AVOID_COLLISION
                    # self.task_history.append(self.current_task)
                    collision_danger = True
                    # self.collision_count += 1
                    break

        self.step_count += 1
        self.count_since_last_attachment += 1

        # self.collision_queue.append(collision_danger)
        if len(self.collision_queue) == self.collision_queue.maxlen:
            avg = sum(self.collision_queue) / self.collision_queue.maxlen
            # self.aprint("Proportion of collision danger to other movement: {}".format(avg))
            # if len(self.collision_average_queue) == self.collision_average_queue.maxlen:
            #     self.aprint("Moving average above 0.8: {}".format(sum(e > 0.8 for e in self.collision_average_queue)))
            self.collision_average_queue.append(avg)

        # if sum(self.seed_arrival_delay_queue) / self.seed_arrival_delay_queue.maxlen > 150:
        #     if random.random() < 0.1:
        #         if self.current_block is None:
        #             self.current_task = Task.LAND
        #         else:
        #             self.current_task = Task.RETURN_BLOCK
        #         self.drop_out_of_swarm = True
        #         self.current_path = None
        # else:
        #     self.aprint("Length of collision queue: {}".format(len(self.collision_queue)))

        # if not self.drop_out_of_swarm and self.step_count >= 1000 \
        #         and sum(e > 0.8 for e in self.collision_average_queue) > 90:
        #     if random.random() < 0.25:
        #         if self.current_block is None:
        #             self.current_task = Task.LAND
        #         else:
        #             self.current_task = Task.RETURN_BLOCK
        #         self.drop_out_of_swarm = True
        #         self.current_path = None

        # if len(self.collision_avoidance_contribution_queue) == self.collision_avoidance_contribution_queue.maxlen:
        #     self.aprint("Collision avoidance contribution average: {}".format(
        #         sum(self.collision_avoidance_contribution_queue) / self.collision_avoidance_contribution_queue.maxlen))
        #     # maybe if the moving average of this stays above 1 for too long...

        # if self.step_count % 20 == 0:
        #     # record the amount of movement every x actions?
        #     self.aprint("Movement away from reference in the last 50 steps: {}"
        #            .format(simple_distance(self.reference_position, self.geometry.position)))
        #     self.reference_position[:] = self.geometry.position[:]

        # self.aprint("Count since last attachment: {}".format(self.count_since_last_attachment))
        # self.aprint("Time close to seed but not quite there: {}".format(self.close_to_seed_count))
        # if count since last attachment is larger than expected, also likely to be "stuck"

        # self.aprint("Current collision danger proportion: {}".format(self.collision_count / self.step_count))
