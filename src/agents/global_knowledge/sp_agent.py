import numpy as np
import random
import env.map
from agents.agent import Agent, Task, check_map
from agents.global_knowledge.gk_agent import GlobalKnowledgeAgent
from env.block import Block
from env.util import shortest_path, shortest_path_3d_in_2d, legal_attachment_sites, legal_attachment_sites_3d
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance


class GlobalShortestPathAgent(GlobalKnowledgeAgent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(GlobalShortestPathAgent, self).__init__(position, size, target_map, required_spacing, printing_enabled)
        self.current_shortest_path = None
        self.current_sp_index = 0
        self.attachment_site_ordering = "shortest_path"  # other is "agent_count"

    def find_attachment_site(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None or self.current_shortest_path is None:
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                      environment.occupancy_map[self.current_structure_level],
                                                      component_marker=self.current_component_marker)

            attachment_map = np.copy(attachment_sites)

            self.aprint("DETERMINING ATTACHMENT SITES ON LEVEL {} WITH MARKER {} AND SEED AT {}:"
                        .format(self.current_structure_level,
                                self.current_component_marker,
                                self.current_seed.grid_position))

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(environment.occupancy_map[self.current_structure_level])

            # convert to coordinates
            attachment_sites = np.where(attachment_sites == 1)
            attachment_sites = list(zip(attachment_sites[1], attachment_sites[0]))

            self.per_search_attachment_site_count["total"].append(len(attachment_sites))

            backup = []
            for site in attachment_sites:
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
            attachment_sites = backup

            if len(attachment_sites) == 0:
                self.aprint("NO LEGAL ATTACHMENT SITES AT LEVEL {} WITH MARKER {}"
                            .format(self.current_structure_level, self.current_component_marker))
                self.aprint("ATTACHMENT MAP:")
                self.aprint(attachment_map, print_as_map=True)

            self.per_search_attachment_site_count["possible"].append(len(attachment_sites))

            # find the closest one
            shortest_paths = []
            for x, y in attachment_sites:
                occupancy_map_copy[y, x] = 1
                sp = shortest_path(occupancy_map_copy, (self.current_grid_position[0],
                                                        self.current_grid_position[1]), (x, y))
                occupancy_map_copy[y, x] = 0
                shortest_paths.append(sp)

            # this should be used later:
            directions = [np.array([x - self.current_grid_position[0], y - self.current_grid_position[1]])
                          for x, y in attachment_sites]
            counts = self.count_in_direction(environment, directions=directions, angle=np.pi / 2)

            # order = sorted(range(len(shortest_paths)), key=lambda i: len(shortest_paths[i]))
            if self.attachment_site_ordering == "shortest_path":
                order = sorted(range(len(shortest_paths)), key=lambda i: (len(shortest_paths[i]), counts[i]))
            elif self.attachment_site_ordering == "agent_count":
                order = sorted(range(len(shortest_paths)), key=lambda i: (counts[i], len(shortest_paths[i])))
            else:
                order = sorted(range(len(shortest_paths)), key=lambda i: random.random())

            shortest_paths = [shortest_paths[i] for i in order]
            attachment_sites = [attachment_sites[i] for i in order]

            # the initial direction of this shortest path would maybe be good to decide based on if this is used...
            new_sp = [(self.current_grid_position[0], attachment_sites[0][1]),
                      (attachment_sites[0][0], attachment_sites[0][1])]

            # find "bends" in the path and only require going there
            sp = shortest_paths[0]
            self.aprint("Original shortest path: {}".format(sp))
            if len(sp) > 1:
                diffs = [(sp[1][0] - sp[0][0], sp[1][1] - sp[0][1])]
                new_sp = [sp[0]]
                for i in range(1, len(sp)):
                    if i < len(sp) - 1:
                        diff = (sp[i + 1][0] - sp[i][0], sp[i + 1][1] - sp[i][1])
                        if diff[0] != diffs[-1][0] or diff[1] != diffs[-1][1]:
                            # a "bend" is happening
                            new_sp.append((sp[i]))
                        diffs.append(diff)
                new_sp.append(sp[-1])
                sp = new_sp

            self.aprint("Shortest path to attachment site: {}".format(sp))

            if not all(sp[-1][i] == attachment_sites[0][i] for i in range(2)):
                self.aprint("SHORTEST PATH DOESN'T LEAD TO INTENDED ATTACHMENT SITE",
                            override_global_printing_enabled=True)
                self.aprint("Current grid position: {}".format(self.current_grid_position),
                            override_global_printing_enabled=True)
                self.aprint("Shortest path: {}".format(sp), override_global_printing_enabled=True)
                self.aprint("Intended attachment site: {}".format(attachment_sites[0]))
                self.aprint("Current component marker: {}".format(self.current_component_marker))
                self.aprint("Current seed at {} and seed's component marker: {}"
                            .format(self.current_seed.grid_position,
                                    self.component_target_map[self.current_seed.grid_position[2],
                                                              self.current_seed.grid_position[1],
                                                              self.current_seed.grid_position[0]]))
                self.aprint("Attachment map:", override_global_printing_enabled=True)
                self.aprint(attachment_map, print_as_map=True, override_global_printing_enabled=True)
                if self.current_component_marker != self.component_target_map[self.current_seed.grid_position[2], self.current_seed.grid_position[1], self.current_seed.grid_position[0]]:
                    self.current_seed = environment.block_at_position(self.component_seed_location(self.current_component_marker))
                    self.recheck_task(environment)
                    return
                else:
                    raise Exception

            # construct the path to that site
            self.current_grid_position = np.array([sp[0][0], sp[0][1], self.current_structure_level])
            self.current_path = Path()
            position = [environment.offset_origin[0] + Block.SIZE * self.current_grid_position[0],
                        environment.offset_origin[0] + Block.SIZE * self.current_grid_position[1],
                        self.geometry.position[2]]
            self.current_path.add_position(position)

            # storing information about the path
            self.current_shortest_path = sp
            self.current_sp_index = 0

        attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                  environment.occupancy_map[self.current_structure_level],
                                                  component_marker=self.current_component_marker)
        if attachment_sites[self.current_shortest_path[-1][1], self.current_shortest_path[-1][0]] == 0:
            self.current_path = None
            self.aprint("Current target for attachment: {}\nAttachment sites:".format(self.current_shortest_path[-1]))
            self.aprint(attachment_sites, print_as_map=True)
            self.find_attachment_site(environment)
            return

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                # have reached next point on shortest path to attachment site
                current_spc = self.current_shortest_path[self.current_sp_index]
                self.current_grid_position = np.array([current_spc[0], current_spc[1], self.current_structure_level])
                self.aprint("REACHED {} ON SHORTEST PATH ({})".format(current_spc, self.current_shortest_path))
                self.aprint("Own position: {}".format(self.geometry.position))
                self.aprint("GRID POSITION: {}".format(self.current_grid_position))
                if self.current_sp_index >= len(self.current_shortest_path) - 1:
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
                    # if the attachment site was determined definitively (corner or protruding), have reached
                    # intended attachment site and should assess whether block can be placed or not
                    if environment.check_occupancy_map(np.array([self.current_shortest_path[-1][0],
                                                                 self.current_shortest_path[-1][1],
                                                                 self.current_grid_position[2]])):
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

                if self.check_component_finished(environment.occupancy_map):
                    self.aprint("FINISHED COMPONENT {} AFTER MOVING TO NEXT BLOCK IN ATTACHMENT SITE"
                                .format(self.current_component_marker))
                    self.recheck_task(environment)
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

        self.per_task_distance_travelled[Task.FIND_ATTACHMENT_SITE] += simple_distance(position_before,
                                                                                       self.geometry.position)

    def place_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            init_z = Block.SIZE * (self.current_structure_level + 2) + self.required_spacing + self.geometry.size[2] / 2
            first_z = Block.SIZE * (self.current_grid_position[2] + 2) + self.geometry.size[2] / 2
            placement_x = Block.SIZE * self.current_grid_position[0] + environment.offset_origin[0]
            placement_y = Block.SIZE * self.current_grid_position[1] + environment.offset_origin[1]
            placement_z = Block.SIZE * (self.current_grid_position[2] + 1) + self.geometry.size[2] / 2
            self.current_path = Path()
            self.current_path.add_position([placement_x, placement_y, init_z])
            self.current_path.add_position([placement_x, placement_y, first_z])
            self.current_path.add_position([placement_x, placement_y, placement_z])
            self.aprint(
                "place_block, height of init, first, placement: {}, {}, {}".format(init_z, first_z, placement_z))
            self.aprint("current grid position: {}, current structure level: {}".format(self.current_grid_position,
                                                                                        self.current_structure_level))

        # check again whether attachment is allowed since other agents placing blocks there may have made it illegal
        if not self.current_block_type_seed:
            if environment.check_occupancy_map(self.current_grid_position):
                self.current_path = None
                self.current_task = Task.FIND_ATTACHMENT_SITE
                self.task_history.append(self.current_task)
                self.find_attachment_site(environment)
                return

            attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                      environment.occupancy_map[self.current_structure_level],
                                                      component_marker=self.current_component_marker)
            if attachment_sites[self.current_grid_position[1], self.current_grid_position[0]] == 0:
                self.current_path = None
                self.current_task = Task.FIND_ATTACHMENT_SITE
                self.task_history.append(self.current_task)
                self.find_attachment_site(environment)
                return

        if self.current_path.current_index != len(self.current_path.positions) - 1:
            next_position, current_direction = self.move(environment)
        else:
            next_position = self.current_path.next()
            current_direction = self.current_path.direction_to_next(self.geometry.position)
            current_direction /= sum(np.sqrt(current_direction ** 2))
            current_direction *= Agent.MOVEMENT_PER_STEP
            self.per_task_step_count[self.current_task] += 1
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # block should now be placed in the environment's occupancy matrix
            if not ret:
                if self.current_block.is_seed:
                    self.current_seed = self.current_block
                    self.components_seeded.append(int(self.current_component_marker))
                elif self.current_component_marker not in self.components_attached:
                    self.components_attached.append(int(self.current_component_marker))

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
                self.attachment_frequency_count.append(self.count_since_last_attachment)
                self.count_since_last_attachment = 0

                environment.place_block(self.current_grid_position, self.current_block)
                self.geometry.attached_geometries.remove(self.current_block.geometry)
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
                    self.current_task = Task.FETCH_BLOCK
                else:
                    self.aprint("AFTER PLACING BLOCK: FETCHING BLOCK (PREVIOUS WAS SEED: {})"
                                .format(self.current_block_type_seed))
                    self.current_task = Task.FETCH_BLOCK
                    if self.current_block_type_seed:
                        self.current_block_type_seed = False
                self.task_history.append(self.current_task)

                self.aprint("RECHECK CALLED FROM AGENT {}".format(self.id))
                for a in environment.agents:
                    a.recheck_task(environment)
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.PLACE_BLOCK] += simple_distance(position_before, self.geometry.position)

    def advance(self, environment: env.map.Map):
        if self.current_task == Task.FINISHED:
            return

        if self.current_task != Task.LAND and self.current_block is None:
            finished = False
            if self.check_structure_finished(environment.occupancy_map):
                finished = True
            elif all([len(environment.seed_stashes[key]) == 0 for key in environment.seed_stashes]) \
                    and all([len(environment.block_stashes[key]) == 0 for key in environment.block_stashes]):
                finished = True
            # elif self.current_block_type_seed \
            #         and all([len(environment.seed_stashes[key]) == 0 for key in environment.seed_stashes]):
            #     finished = True
            # elif not self.current_block_type_seed \
            #         and all([len(environment.block_stashes[key]) == 0 for key in environment.block_stashes]):
            #     finished = True

            if finished:
                self.drop_out_of_swarm = False
                self.wait_for_rejoining = False
                self.rejoining_swarm = False
                self.current_path = None
                self.current_task = Task.LAND
                self.aprint("LANDING (8)")
                self.aprint("")
                self.task_history.append(self.current_task)

        if self.dropping_out_enabled:
            if not self.rejoining_swarm and not self.drop_out_of_swarm and self.step_count > 1000 \
                    and (self.collision_count / self.step_count) > 0.35 \
                    and random.random() < 0.1 \
                    and self.current_task in [Task.RETURN_BLOCK,
                                              Task.FETCH_BLOCK,
                                              Task.TRANSPORT_BLOCK,
                                              Task.WAIT_ON_PERIMETER]:
                if self.current_block is not None:
                    self.current_task = Task.RETURN_BLOCK
                else:
                    self.current_task = Task.LAND
                self.drop_out_of_swarm = True
                self.wait_for_rejoining = False
                self.rejoining_swarm = False
                self.current_path = None

            if self.wait_for_rejoining or self.current_task == Task.REJOIN_SWARM:
                self.rejoin_swarm(environment)

        if self.current_seed is None:
            self.current_seed = environment.blocks[0]
            self.current_component_marker = self.component_target_map[self.current_seed.grid_position[2],
                                                                      self.current_seed.grid_position[1],
                                                                      self.current_seed.grid_position[0]]
            self.next_seed_position = np.copy(self.current_seed.grid_position)

        if self.initial_position is None:
            self.initial_position = np.copy(self.geometry.position)

        self.agent_statistics.step(environment)

        if self.current_task == Task.FETCH_BLOCK:
            self.fetch_block(environment)
        elif self.current_task == Task.PICK_UP_BLOCK:
            self.pick_up_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)
        elif self.current_task == Task.WAIT_ON_PERIMETER:
            self.wait_on_perimeter(environment)
        elif self.current_task == Task.HOVER_OVER_COMPONENT:
            self.hover_over_component(environment)
        elif self.current_task == Task.MOVE_TO_PERIMETER:
            self.move_to_perimeter(environment)
        elif self.current_task == Task.FIND_ATTACHMENT_SITE:
            self.find_attachment_site(environment)
        elif self.current_task == Task.RETURN_BLOCK:
            self.return_block(environment)
        elif self.current_task == Task.PLACE_BLOCK:
            self.place_block(environment)
        elif self.current_task == Task.LAND:
            self.land(environment)

        if self.current_task == Task.MOVE_TO_PERIMETER and not self.current_block_type_seed:
            self.current_task = Task.FIND_ATTACHMENT_SITE
            self.current_path = None
            self.current_shortest_path = None

        if self.current_task not in [Task.FINISHED, Task.LAND, Task.HOVER_OVER_COMPONENT]:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70 \
                    and self.current_path is not None and not self.wait_for_rejoining:
                self.aprint("STUCK")
                self.stuck_count += 1
                self.current_path.add_position([self.geometry.position[0],
                                                self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[
                                                    2] * 2 * random.random()],
                                               self.current_path.current_index)
        elif self.current_task == Task.LAND:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70 \
                    and self.current_path is not None:
                self.stuck_count += 1
                self.current_path.add_position([self.geometry.position[0],
                                                self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] * 2],
                                               self.current_path.current_index)

        # self.collision_queue.append(collision_danger)
        if len(self.collision_queue) == self.collision_queue.maxlen:
            avg = sum(self.collision_queue) / self.collision_queue.maxlen
            self.collision_average_queue.append(avg)

        self.position_queue.append(self.geometry.position.copy())
        self.step_count += 1
        self.count_since_last_attachment += 1
        self.drop_out_statistics["drop_out_of_swarm"].append(self.drop_out_of_swarm)
        self.drop_out_statistics["wait_for_rejoining"].append(self.wait_for_rejoining)
        self.drop_out_statistics["rejoining_swarm"].append(self.rejoining_swarm)

        # updating statistics
