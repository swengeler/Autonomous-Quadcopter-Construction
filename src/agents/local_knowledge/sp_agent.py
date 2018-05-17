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

    # need to change the following:
    # - find_attachment_site
    # - might be something else that might not work then?

    """
    def find_attachment_site(self, environment: env.map.Map):
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

            # aprint(self.id, self.current_component_marker)
            # aprint(self.id, "\nOCCUPANCY MATRIX:")
            # print_map(occupancy_map_copy)
            # print_map(self.component_target_map[self.current_structure_level])
            # aprint(self.id, "ATTACHMENT SITES:")
            # print_map(attachment_sites)

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
            sp = shortest_paths[0]
            # sp = random.sample(shortest_paths, 1)[0]

            self.current_grid_position = np.array([sp[-1][0], sp[-1][1], self.current_structure_level])
            self.current_path = Path()
            for x, y in sp:
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
    """

    def find_attachment_site(self, environment: env.map.Map):
        # hacky solution for the None problem, but here goes:
        if self.current_path is None or self.current_shortest_path is None:
            self.update_local_occupancy_map(environment)

            # TODO: if coming back with same state of map (?),
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites, corner_sites, protruding_sites, most_ccw_sites = \
                legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                       self.local_occupancy_map[self.current_structure_level],
                                       component_marker=self.current_component_marker, local_info=True)

            aprint(self.id, "DETERMINING ATTACHMENT SITES:")
            print_map(attachment_sites)
            aprint(self.id, "CURRENT MAP")
            print_map(self.local_occupancy_map)
            aprint(self.id, "corner_sites = {}, protruding_sites = {}, most_ccw_sites = {}"
                   .format(corner_sites, protruding_sites, most_ccw_sites))

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(self.local_occupancy_map[self.current_structure_level])

            # convert to coordinates
            attachment_sites = np.where(attachment_sites == 1)
            attachment_sites = list(zip(attachment_sites[1], attachment_sites[0]))

            # use the previously defined rules for not closing loops to determine which sites are valid
            # TODO: change the procedure here to something better/more efficient (e.g. "regions of interest")
            # backup = []
            # for site in attachment_sites:
            #     if (site[0], site[1], self.current_structure_level) in self.illegal_sites:
            #         aprint(self.id, "SITE {} REMOVED BECAUSE IN ILLEGAL SITES".format(site))
            #         continue
            #     at_loop_corner, loop_corner_attachable = self.check_loop_corner(
            #         environment, np.array([site[0], site[1], self.current_structure_level]))
            #     allowable_region_attachable = True
            #     if not at_loop_corner:
            #         for x, y, z in self.closing_corners[self.current_structure_level][self.current_component_marker]:
            #             if not environment.check_occupancy_map(np.array([x, y, z])) and \
            #                     x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
            #                 allowable_region_attachable = False
            #                 break
            #     if loop_corner_attachable and allowable_region_attachable:
            #         backup.append(site)
            #     else:
            #         aprint(self.id, "SITE {} REMOVED BECAUSE lca = {}, ara = {}"
            #                .format(site, loop_corner_attachable, allowable_region_attachable))
            # attachment_sites = backup

            if len(attachment_sites) == 0:
                aprint(self.id, "NO LEGAL ATTACHMENT SITES AT LEVEL {} WITH MARKER {}"
                       .format(self.current_structure_level, self.current_component_marker))
                aprint(self.id, "LOCAL MAP:\n{}".format(self.local_occupancy_map))

            backup = []
            for site in corner_sites:
                # if (site[0], site[1], self.current_structure_level) in self.illegal_sites:
                #     aprint(self.id, "SITE {} REMOVED BECAUSE IN ILLEGAL SITES".format(site))
                #     continue
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                if not at_loop_corner:
                    for x, y, z in self.closing_corners[self.current_structure_level][self.current_component_marker]:
                        if not environment.check_occupancy_map(np.array([x, y, z])) and \
                                x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                            allowable_region_attachable = False
                            break
                if loop_corner_attachable and allowable_region_attachable:
                    backup.append(site)
                else:
                    aprint(self.id, "SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                           .format(site, loop_corner_attachable, allowable_region_attachable))
            corner_sites = backup

            backup = []
            for site in protruding_sites:
                # if (site[0], site[1], self.current_structure_level) in self.illegal_sites:
                #     aprint(self.id, "SITE {} REMOVED BECAUSE IN ILLEGAL SITES".format(site))
                #     continue
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                if not at_loop_corner:
                    for x, y, z in self.closing_corners[self.current_structure_level][self.current_component_marker]:
                        if not environment.check_occupancy_map(np.array([x, y, z])) and \
                                x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                            allowable_region_attachable = False
                            break
                if loop_corner_attachable:  # and allowable_region_attachable:
                    backup.append(site)
                else:
                    aprint(self.id, "SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                           .format(site, loop_corner_attachable, allowable_region_attachable))
            protruding_sites = backup

            backup = []
            for site in most_ccw_sites:
                # if (site[0], site[1], self.current_structure_level) in self.illegal_sites:
                #     aprint(self.id, "SITE {} REMOVED BECAUSE IN ILLEGAL SITES".format(site))
                #     continue
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(
                    environment, np.array([site[0], site[1], self.current_structure_level]))
                allowable_region_attachable = True
                if not at_loop_corner:
                    for x, y, z in self.closing_corners[self.current_structure_level][self.current_component_marker]:
                        if not environment.check_occupancy_map(np.array([x, y, z])) and \
                                x <= self.current_grid_position[0] and y <= self.current_grid_position[1]:
                            allowable_region_attachable = False
                            break
                if loop_corner_attachable:  # and allowable_region_attachable:
                    backup.append(site)
                else:
                    aprint(self.id, "SITE {} REMOVED BECAUSE lca = {}, ara = {}"
                           .format(site, loop_corner_attachable, allowable_region_attachable))
            most_ccw_sites = backup

            # need to do the same for row/column sequences in CCW sites as well

            # find the closest corner or protruding site
            # if there are none then take the closest most CCW site
            # all of this while respecting the hole rules, but FOR NOW without
            if len(corner_sites) != 0:
                attachment_sites = corner_sites
            elif len(protruding_sites) != 0:
                attachment_sites = protruding_sites
            else:
                attachment_sites = most_ccw_sites
            # if len(most_ccw_sites) != 0:
            #     attachment_sites = most_ccw_sites
            # elif len(protruding_sites) != 0:
            #     attachment_sites = protruding_sites
            # elif len(corner_sites) != 0:
            #     attachment_sites = corner_sites

            # find the closest one
            shortest_paths = []
            number_adjacent_blocks = []
            for x, y in attachment_sites:
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
            # ordered_by_length = sorted(range(len(shortest_paths)), key=lambda i: len(shortest_paths[i]))
            # shortest_paths = [shortest_paths[i] for i in ordered_by_length]
            # number_adjacent_blocks = [number_adjacent_blocks[i] for i in ordered_by_length]
            # ordered_by_adj = sorted(range(len(shortest_paths)), key=lambda i: number_adjacent_blocks[i])
            # shortest_paths = [shortest_paths[i] for i in ordered_by_adj]
            shortest_paths = sorted(shortest_paths, key=lambda x: len(x))
            sp = shortest_paths[0]

            """
            if len(shortest_paths) == 0:
                self.current_task = Task.FIND_NEXT_COMPONENT
                self.task_history.append(self.current_task)
                self.current_path = None
                for layer in range(self.current_structure_level + 1):
                    self.local_occupancy_map[layer][self.target_map[layer] != 0] = 1
                return
            """
            """
            # find the SOUTH-WESTERN most attachment site
            sorted_by_y = sorted(range(len(attachment_sites)), key=lambda i: attachment_sites[i][1])
            sorted_by_x = sorted(sorted_by_y, key=lambda i: attachment_sites[i][0])
            attachment_sites = [attachment_sites[i] for i in sorted_by_x]

            # use that attachment site
            sp = shortest_path(occupancy_map_copy, (self.current_grid_position[0], self.current_grid_position[1]),
                               (attachment_sites[0][0], attachment_sites[0][1]))

            aprint(self.id, "Attachment sites: {}".format(attachment_sites))
            aprint(self.id, "Chose shortest path: {}".format(sp))
            """

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
                    # TODO: might have to move to the seed first (although localisation should be possible as above)
                    # seed_grid_location = self.component_seed_location(
                    #     self.component_target_map[block_below.grid_position[2], block_below.grid_position[1],
                    #                               block_below.grid_position[0]], self.current_structure_level + 1)
                    # self.current_seed = environment.block_at_position(seed_grid_location)
                    # self.current_task = Task.TRANSPORT_BLOCK

                # have reached next point on shortest path to attachment site
                if self.current_shortest_path is None:
                    aprint(self.id, "shortest path None (previous path None: {})".format(
                        self.path_before_collision_avoidance_none))
                current_spc = self.current_shortest_path[self.current_sp_index]
                aprint(self.id, "REACHED {} ON SHORTEST PATH ({})".format(current_spc, self.current_shortest_path))
                self.current_grid_position = np.array([current_spc[0], current_spc[1], self.current_structure_level])
                aprint(self.id, "GRID POSITION: {}".format(self.current_grid_position))
                self.update_local_occupancy_map(environment)
                if self.current_sp_index == len(self.current_shortest_path) - 1:
                    # have reached intended attachment site and should assess whether block can be placed or not
                    if not environment.check_occupancy_map(self.current_grid_position):
                        # if yes, just place it
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.current_shortest_path = None
                        aprint(self.id, "GOING TO PLACE BLOCK")
                    else:
                        # if no, need to find new attachment site (might just be able to restart this method)
                        self.current_path = None
                        aprint(self.id, "SITE ALREADY OCCUPIED, SEARCH FOR NEW ONE")
                else:
                    # still have to go on, therefore update the current path
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
            self.geometry.position = self.geometry.position + current_direction

    """
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

        # check again whether attachment is allowed
        attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                  environment.occupancy_map[self.current_structure_level],
                                                  component_marker=self.current_component_marker)
        if attachment_sites[self.current_grid_position[1], self.current_grid_position[0]] == 0:
            self.current_path = None
            self.current_task = Task.FIND_ATTACHMENT_SITE
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
    """

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
        if self.current_task == Task.MOVE_TO_PERIMETER and not self.current_block_type_seed:
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.current_path = None  # path is not reset before MOVE_TO_PERIMETER, thus this has to be reset

        if self.check_structure_finished(self.local_occupancy_map):
            self.current_task = Task.LAND
            # aprint(self.id, "LANDING (8)")
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
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    self.previous_task = self.current_task
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    break


class ShortestPathAgent3D(PerimeterFollowingAgentLocal):

    # FOR NOW THIS AGENT WILL BE "OMNISCIENT"/THE BLOCKS WILL BE UNIQUE
    # THEREFORE A NEW transport_block METHOD IS IMPLEMENTED

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0):
        super(ShortestPathAgent3D, self).__init__(position, size, target_map, required_spacing)
        self.highest_observed_level = 0

    # need to change the following:
    # - find_attachment_site
    # - might need change to placement?
    # - doesn't need move_up_layer, move_to_perimeter
    #   - move_up_layer can be found in move_up_layer, find_attachment_site and place_block (more reason to change it)

    def find_attachment_site(self, environment: env.map.Map):
        # component information can be relevant for this algorithm for seeding new layers, but there are other options:
        # - blocks are assumed to be unique, thereby basically making every block a sort of seed
        # - when going to a layer higher up and no block has been placed in that component, make it a seed
        # - covering of previous level seed is not allowed

        # first block placed in component can be seed for that component (if the layer below has been completed?)

        # algorithm outline:
        # - transport_block should know the current seed, which has to be some component seed
        # - once arriving (and updating knowledge about the environment?), where block can be attached (in 3D)
        # - follow shortest path (or potentially other path) to that attachment site
        # - if occupied, do same thing again to next best attachment site
        # - place block

        # find an attachment site by finding the closest allowed/legal attachment site and following the path
        # over the blocks there (this is done for "realism", because the blocks are needed for orientation; it
        # could be worth considering just taking the shortest direct flight path, orientation oneself using a
        # grid on the construction site floor)
        if self.current_path is None:
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites = legal_attachment_sites_3d(self.target_map, environment.occupancy_map, safety_radius=1)

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(environment.occupancy_map)

            # print_map(attachment_sites)

            # convert to coordinates
            attachment_sites = np.where(attachment_sites == 1)
            attachment_sites = list(zip(attachment_sites[2], attachment_sites[1], attachment_sites[0]))

            # aprint(self.id, "ATTACHMENT SITES: {}".format(attachment_sites))

            # find the closest one
            shortest_paths = []
            for x, y, z in attachment_sites:
                sp_and_height = shortest_path_3d_in_2d(occupancy_map_copy, (self.current_grid_position[0],
                                                                            self.current_grid_position[1]), (x, y, z))
                shortest_paths.append(sp_and_height)
            sp_indices = sorted(range(len(shortest_paths)),
                                key=lambda x: len(shortest_paths[x][0]) + shortest_paths[x][1])
            # TODO: need to take additional vertical distance into account for picking best destination
            shortest_paths = [shortest_paths[i] for i in sp_indices]
            attachment_sites = [attachment_sites[i] for i in sp_indices]
            sp = shortest_paths[0][0]
            height = shortest_paths[0][1]
            # index = random.randint(0, len(shortest_paths) - 1)
            # sp = shortest_paths[index][0]
            # height = shortest_paths[index][1]
            self.highest_observed_level = max(height, self.highest_observed_level,
                                              max(np.where(occupancy_map_copy >= 1)[0]))
            self.current_structure_level = self.highest_observed_level

            # aprint(self.id, "SELECTED PATH: {} (AT HEIGHT {})".format(sp, height))

            self.current_grid_position = np.array(
                [attachment_sites[0][0], attachment_sites[0][1], attachment_sites[0][2]])
            self.current_path = Path()
            for x, y in sp:
                current_coordinate = [environment.offset_origin[0] + Block.SIZE * x,
                                      environment.offset_origin[0] + Block.SIZE * y,
                                      self.geometry.size[2] + Block.SIZE * (height + 3) + 2 * self.required_spacing]
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
                self.current_task = Task.PLACE_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None

                # need to check whether the location has already been occupied
                if environment.check_occupancy_map(np.array(self.current_grid_position)):
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

    def transport_block(self, environment: env.map.Map):
        # this new transport_block method approaches the structure, taking the seed position as a reference
        # and once it is over the structure and recognises the block beneath (and has thus achieved localisation),
        # it also goes for finding an attachment site

        # since the highest current structure level is unknown, this method simply flies high enough? this could be
        # based on the highest level this agent has placed a block at -> then one would still have to watch out for
        # the colliding with "unexpected" blocks, and possibly move around or higher

        if self.current_path is None:
            # gain height, fly to seed location and then start search for attachment site
            self.current_path = Path()
            transport_level_z = Block.SIZE * (self.highest_observed_level + 3) + \
                                (self.geometry.size[2] / 2 + self.required_spacing) * 2
            seed_location = self.current_seed.geometry.position
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], transport_level_z])
            self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z])

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
                self.current_grid_position = np.array(self.current_seed.grid_position)
                self.current_path = None
                self.current_task = Task.FIND_ATTACHMENT_SITE
                self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

        # if over structure (i.e. localisation possible), start looking for attachment site
        # block_below = environment.block_below(self.geometry.position)
        # if block_below is not None:
        #     self.current_grid_position = np.array(block_below)
        #     self.current_path = None
        #     self.current_task = Task.FIND_ATTACHMENT_SITE
        #     self.task_history.append(self.current_task)

    def place_block(self, environment: env.map.Map):
        # fly to determined attachment site, lower quadcopter and place block,
        # then switch task back to fetching blocks

        if self.current_path is None:
            # init_z = Block.SIZE * (self.current_structure_level + 2) + self.required_spacing + self.geometry.size[2] / 2
            placement_x = Block.SIZE * self.current_grid_position[0] + environment.offset_origin[0]
            placement_y = Block.SIZE * self.current_grid_position[1] + environment.offset_origin[1]
            placement_z = Block.SIZE * (self.current_grid_position[2] + 1) + self.geometry.size[2] / 2
            self.current_path = Path()
            # self.current_path.add_position([placement_x, placement_y, init_z])
            self.current_path.add_position([placement_x, placement_y, placement_z])

        if environment.check_occupancy_map(self.current_grid_position):
            # a different agent has already placed the block in the meantime
            self.current_path = None
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.task_history.append(self.current_task)
            return

        # check again whether attachment is allowed
        attachment_sites = legal_attachment_sites_3d(self.target_map, environment.occupancy_map, safety_radius=1)
        if attachment_sites[self.current_grid_position[2],
                            self.current_grid_position[1],
                            self.current_grid_position[0]] == 0:
            self.current_path = None
            self.current_task = Task.FIND_ATTACHMENT_SITE
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
                environment.place_block(self.current_grid_position, self.current_block)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
                self.current_block.placed = True
                self.current_block.grid_position = self.current_grid_position
                self.current_block = None
                self.current_path = None

                tm = np.copy(self.target_map)
                np.place(tm, tm >= 1, 1)
                om = np.copy(environment.occupancy_map)
                np.place(om, om >= 1, 1)
                if np.array_equal(tm, om):
                    self.current_task = Task.LAND
                    self.task_history.append(self.current_task)
                    self.current_path = None
                    self.logger.info("Construction finished (YAY).")
                else:
                    self.current_path = None
                    self.current_task = Task.FETCH_BLOCK
                    self.task_history.append(self.current_task)
        else:
            self.geometry.position = self.geometry.position + current_direction

    def advance(self, environment: env.map.Map):
        if self.current_seed is None:
            self.current_seed = environment.blocks[0]
        if self.current_task == Task.MOVE_TO_PERIMETER:
            self.current_task = Task.FIND_ATTACHMENT_SITE
        elif self.current_task == Task.MOVE_UP_LAYER:
            pass

        self.agent_statistics.step(environment)

        if self.current_task == Task.AVOID_COLLISION:
            self.avoid_collision(environment)
        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.PICK_UP_BLOCK:
            self.pick_up_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.collision_possible and self.current_task not in [Task.AVOID_COLLISION, Task.LAND, Task.FINISHED]:
            for a in environment.agents:
                if self is not a and self.collision_potential(a):
                    self.previous_task = self.current_task
                    self.current_task = Task.AVOID_COLLISION
                    self.task_history.append(self.current_task)
                    break
