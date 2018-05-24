import numpy as np
import random
import env.map
from agents.agent import Agent, Task, aprint, check_map
from agents.local_knowledge.ps_agent import PerimeterFollowingAgentLocal
from env.block import Block
from env.util import print_map, shortest_path, shortest_path_3d_in_2d, legal_attachment_sites, legal_attachment_sites_3d
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance


class ShortestPathAgentLocal(PerimeterFollowingAgentLocal):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(ShortestPathAgentLocal, self).__init__(position, size, target_map, required_spacing)
        self.current_shortest_path = None
        self.current_sp_index = 0
        self.illegal_sites = []
        self.current_attachment_info = None

    # need to change the following:
    # - find_attachment_site
    # - might be something else that might not work then?

    def find_attachment_site(self, environment: env.map.Map):
        # hacky solution for the None problem, but here goes:
        if self.current_path is None or self.current_shortest_path is None:
            self.update_local_occupancy_map(environment)

            if self.check_component_finished(self.local_occupancy_map, self.current_component_marker):
                self.find_next_component(environment)
                return

            # TODO: if coming back with same state of map (?),
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites, corner_sites, protruding_sites, most_ccw_sites = \
                legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                       self.local_occupancy_map[self.current_structure_level],
                                       component_marker=self.current_component_marker, local_info=True)

            aprint(self.id, "DETERMINING ATTACHMENT SITES ON LEVEL {} WITH MARKER {}:"
                   .format(self.current_structure_level, self.current_component_marker))
            # print_map(attachment_sites)
            # aprint(self.id, "CURRENT MAP")
            # print_map(self.local_occupancy_map[self.current_structure_level])
            # aprint(self.id, "CURRENT COMPONENT MAP")
            # print_map(self.component_target_map[self.current_structure_level])
            aprint(self.id, "corner_sites = {}, protruding_sites = {}, most_ccw_sites = {}"
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
                    aprint(self.id, "CORNER SITE {} REMOVED BECAUSE lca = {}, ara = {}"
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
                    aprint(self.id, "PROTRUDING SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                           .format(site, loop_corner_attachable, allowable_region_attachable))
            protruding_sites = backup

            backup = []
            for site, direction, expected_length in most_ccw_sites:
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
                # in this case, might have to block attachment if row reaches into either of these
                # regions, but I'm not sure that this can happen
                if loop_corner_attachable and allowable_region_attachable:
                    backup.append((site, direction, expected_length))
                else:
                    aprint(self.id, "CCW SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                           .format(site, loop_corner_attachable, allowable_region_attachable))
            most_ccw_sites = backup

            # find the closest corner or protruding site
            # if there are none then take the closest most CCW site
            most_ccw_sites_used = False
            if len(corner_sites) != 0:
                attachment_sites = corner_sites
                aprint(self.id, "USING CORNER SITES")
            elif len(protruding_sites) != 0:
                attachment_sites = protruding_sites
                aprint(self.id, "USING PROTRUDING SITES")
            else:
                most_ccw_sites_used = True
                attachment_sites = []
                for site, _, _ in most_ccw_sites:
                    attachment_sites.append(site)
                aprint(self.id, "USING END-OF-ROW SITES")

            attachment_sites = []
            attachment_sites.extend([(s,) for s in corner_sites])
            attachment_sites.extend([(s,) for s in protruding_sites])
            attachment_sites.extend(most_ccw_sites)

            most_ccw_sites_used = False

            if len(attachment_sites) == 0:
                aprint(self.id, "NO LEGAL ATTACHMENT SITES AT LEVEL {} WITH MARKER {}"
                       .format(self.current_structure_level, self.current_component_marker))
                aprint(self.id, "LOCAL MAP:\n{}".format(self.local_occupancy_map))

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
            aprint(self.id, "Directions: {}, Counts: {}".format(directions, counts))
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
            diffs = [(sp[1][0] - sp[0][0], sp[1][1] - sp[0][1])]
            new_sp = [sp[0]]
            for i in range(1, len(sp)):
                if i < len(sp) - 1:
                    diff = (sp[i + 1][0] - sp[i][0], sp[i + 1][1] - sp[i][1])
                    if diff[0] != diffs[-1][0] or diff[1] != diffs[-1][1]:
                        # a "bend" is happening
                        new_sp.append((sp[i]))
                        aprint(self.id, "'Bend' at: {}".format(sp[i]))
                    diffs.append(diff)
            new_sp.append(sp[-1])
            aprint(self.id, "Former SP: {}".format(sp))

            sp = new_sp

            aprint(self.id, "Shortest path to attachment site: {}".format(sp))

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
        if simple_distance(self.geometry.position, next_position) < Agent.MOVEMENT_PER_STEP:
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
                    aprint(self.id, "shortest path None (previous path None: {})".format(
                        self.path_before_collision_avoidance_none))
                if self.current_attachment_info is not None and self.current_sp_index >= len(
                        self.current_shortest_path):
                    self.current_grid_position = self.current_grid_position + self.current_grid_direction
                else:
                    current_spc = self.current_shortest_path[self.current_sp_index]
                    self.current_grid_position = np.array(
                        [current_spc[0], current_spc[1], self.current_structure_level])
                    aprint(self.id, "REACHED {} ON SHORTEST PATH ({})".format(current_spc, self.current_shortest_path))
                self.update_local_occupancy_map(environment)
                aprint(self.id, "GRID POSITION: {}".format(self.current_grid_position))
                if self.current_sp_index >= len(self.current_shortest_path) - 1:
                    aprint(self.id, "TEST")
                    if self.current_attachment_info is None:
                        aprint(self.id, "TEST 1")
                        # if the attachment site was determined definitively (corner or protruding), have reached
                        # intended attachment site and should assess whether block can be placed or not
                        if not environment.check_occupancy_map(self.current_grid_position):
                            # if yes, just place it
                            self.current_task = Task.PLACE_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            self.current_shortest_path = None
                            aprint(self.id, "GOING TO PLACE BLOCK AT CORNER OR PROTRUDING SITE")
                        else:
                            # if no, need to find new attachment site (might just be able to restart this method)
                            self.current_path = None
                            self.current_shortest_path = None
                            aprint(self.id, "SITE ALREADY OCCUPIED, SEARCH FOR NEW ONE")
                    else:
                        aprint(self.id, "TEST 2")
                        # otherwise, if at starting position for the row/column/perimeter search, need to check
                        # whether attachment is already possible and if not, plan next path movement
                        # note that due to the code above, the local map should already be updated, making it
                        # possible to decide whether we are at an outer corner globally
                        self.current_sp_index += 1
                        self.current_grid_direction = self.current_attachment_info[1]
                        if environment.check_occupancy_map(self.current_grid_position):
                            aprint(self.id, "TEST 3")
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
                            self.current_shortest_path = None
                        else:
                            aprint(self.id, "TEST 4")
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
                                aprint(self.id, "GOING TO PLACE BLOCK AT END OF ROW")
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
                    # TODO: check whether we have been pushed out of our way and it would be better to go for
                    # the next point on the path altogether
                    self.current_sp_index += 1
                    next_spc = self.current_shortest_path[self.current_sp_index]
                    next_position = [environment.offset_origin[0] + Block.SIZE * next_spc[0],
                                     environment.offset_origin[0] + Block.SIZE * next_spc[1],
                                     self.geometry.position[2]]
                    self.current_path.add_position(next_position)

                if self.check_component_finished(self.local_occupancy_map):
                    aprint(self.id, "FINISHED COMPONENT {} FINISHED AFTER MOVING TO NEXT BLOCK IN ATTACHMENT SITE"
                           .format(self.current_component_marker))
                    self.current_task = Task.FIND_NEXT_COMPONENT
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            # block_below = environment.block_below(self.geometry.position)
            # if block_below is not None and block_below.grid_position[2] == self.current_grid_position[2]:
            #     self.current_grid_position = np.copy(block_below.grid_position)
            #     self.update_local_occupancy_map(environment)
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
            aprint(self.id, "SITE FOR PLACEMENT HAS BECOME ILLEGAL")
            # self.illegal_sites.append(
            #     (self.current_grid_position[0], self.current_grid_position[1], self.current_structure_level))
            # MAJOR CHEATING HERE
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
                    aprint(self.id, "AFTER PLACING BLOCK: FETCHING BLOCK (PREVIOUS WAS SEED: {})"
                           .format(self.current_block_type_seed))
                    self.current_task = Task.FETCH_BLOCK
                    if self.current_block_type_seed:
                        self.current_block_type_seed = False
                self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

    def advance(self, environment: env.map.Map):
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]
        if self.initial_position is None:
            self.initial_position = np.copy(self.geometry.position)

        if self.current_task == Task.MOVE_TO_PERIMETER and not self.current_block_type_seed:
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.current_path = None  # path is not reset before MOVE_TO_PERIMETER, thus this has to be reset

        if self.current_task != Task.LAND and self.check_structure_finished(self.local_occupancy_map):
            self.current_task = Task.LAND
            aprint(self.id, "LANDING (8)")
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

        if self.current_task != Task.FINISHED:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70:
                aprint(self.id, "STUCK")
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
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    collision_danger = True
                    self.collision_count += 1
                    break
        self.step_count += 1

        self.collision_queue.append(collision_danger)
        if len(self.collision_queue) == self.collision_queue.maxlen:
            aprint(self.id, "Proportion of collision danger to other movement: {}"
                   .format(sum(self.collision_queue) / self.collision_queue.maxlen))

        # aprint(self.id, "Current collision danger proportion: {}".format(self.collision_count / self.step_count))
