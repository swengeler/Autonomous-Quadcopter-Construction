import random
from typing import List

import env.map
from agents.agent import Agent, Task, check_map
from agents.global_knowledge.gk_agent import GlobalKnowledgeAgent
from env import Block, legal_attachment_sites, shortest_path
from geom.path import *


class GlobalShortestPathAgent(GlobalKnowledgeAgent):
    """
    A class implementing the shortest path algorithm developed for this project using global knowledge.
    """

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
        """
        Move with the goal of finding an attachment site.

        This method is called if the current task is FIND_ATTACHMENT_SITE. If the agent has not planned a path yet,
        it first determines all possible attachment sites in the current component, chooses one according to some
        strategy and then plans a path to move there following the grid structure (not following the grid and still
        counting blocks to maintain information about its position may be faster and may be feasible in a more
        realistic simulation as well). Since this is the global knowledge version of the algorithm, all sites to be
        occupied which do not currently violate the row rule are legal attachment sites. Unless the agent finds the
        planned attachment site to be occupied upon arrival, the task changes to PLACE_BLOCK when that site is reached.

        :param environment: the environment the agent operates in
        """

        position_before = np.copy(self.geometry.position)

        if self.current_path is None or self.current_shortest_path is None:
            # get all legal attachment sites for the current component given the current occupancy matrix
            attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                      environment.occupancy_map[self.current_structure_level],
                                                      component_marker=self.current_component_marker)

            attachment_map = np.copy(attachment_sites)

            # copy occupancy matrix to safely insert attachment sites
            occupancy_map_copy = np.copy(environment.occupancy_map[self.current_structure_level])

            # convert to coordinates
            attachment_sites = np.where(attachment_sites == 1)
            attachment_sites = list(zip(attachment_sites[1], attachment_sites[0]))

            self.per_search_attachment_site_count["total"].append(len(attachment_sites))

            # remove all attachment sites which are not legal because of hole restrictions
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
            attachment_sites = backup

            self.per_search_attachment_site_count["possible"].append(len(attachment_sites))

            # determine shortest paths to all attachment sites
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

            if self.attachment_site_ordering == "shortest_path":
                if self.order_only_one_metric:
                    order = sorted(range(len(shortest_paths)), key=lambda i: (len(shortest_paths[i]), random.random()))
                else:
                    order = sorted(range(len(shortest_paths)), key=lambda i: (len(shortest_paths[i]), counts[i]))
            elif self.attachment_site_ordering == "agent_count":
                if self.order_only_one_metric:
                    order = sorted(range(len(shortest_paths)), key=lambda i: (counts[i], random.random()))
                else:
                    order = sorted(range(len(shortest_paths)), key=lambda i: (counts[i], len(shortest_paths[i])))
            else:
                order = sorted(range(len(shortest_paths)), key=lambda i: random.random())

            shortest_paths = [shortest_paths[i] for i in order]
            attachment_sites = [attachment_sites[i] for i in order]

            # find "bends" in the path and only require going there
            sp = shortest_paths[0]
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

            if not all(sp[-1][i] == attachment_sites[0][i] for i in range(2)):
                # these are problems with a bug that I unfortunately did not have more time to investigate
                # I took the practical approach of "solving" the problem by redoing the search for attachment
                # sites here, which allows construction to continue, if nothing else
                self.aprint("SHORTEST PATH DOESN'T LEAD TO INTENDED ATTACHMENT SITE",
                            override_global_printing_enabled=True)
                self.aprint("Current grid position: {}".format(self.current_grid_position),
                            override_global_printing_enabled=True)
                self.aprint("Shortest path: {}".format(sp), override_global_printing_enabled=True)
                self.aprint("Intended attachment site: {}".format(attachment_sites[0]),
                            override_global_printing_enabled=True)
                self.aprint("Current component marker: {}".format(self.current_component_marker),
                            override_global_printing_enabled=True)
                self.aprint("Current seed at {} and seed's component marker: {}"
                            .format(self.current_seed.grid_position,
                                    self.component_target_map[self.current_seed.grid_position[2],
                                                              self.current_seed.grid_position[1],
                                                              self.current_seed.grid_position[0]]),
                            override_global_printing_enabled=True)
                self.aprint("Attachment map:", override_global_printing_enabled=True)
                self.aprint(attachment_map, print_as_map=True, override_global_printing_enabled=True)
                if check_map(self.component_target_map, self.current_seed.grid_position,
                             lambda x: x != self.current_component_marker) \
                        or check_map(self.component_target_map, self.current_grid_position,
                                     lambda x: x != self.current_component_marker):
                    previous = self.current_seed
                    self.current_seed = environment.block_at_position(self.component_seed_location(
                        self.current_component_marker))
                    if self.current_seed is None:
                        self.current_seed = previous
                        self.current_component_marker = self.component_target_map[self.current_seed.grid_position[2],
                                                                                  self.current_seed.grid_position[1],
                                                                                  self.current_seed.grid_position[0]]
                    self.aprint("Setting new seed to {} with marker {}"
                                .format(self.current_seed.grid_position,
                                        self.component_target_map[self.current_seed.grid_position[2],
                                                                  self.current_seed.grid_position[1],
                                                                  self.current_seed.grid_position[0]]),
                                override_global_printing_enabled=True)
                    self.aprint("Setting new marker to {}".format(self.current_component_marker),
                                override_global_printing_enabled=True)
                    self.current_task = Task.TRANSPORT_BLOCK
                    self.transport_block(environment)
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

            self.current_sp_search_count += 1

        # check again whether attachment site is still available/legal
        attachment_sites = legal_attachment_sites(self.component_target_map[self.current_structure_level],
                                                  environment.occupancy_map[self.current_structure_level],
                                                  component_marker=self.current_component_marker)
        if attachment_sites[self.current_shortest_path[-1][1], self.current_shortest_path[-1][0]] == 0:
            self.current_path = None
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
                if self.current_sp_index >= len(self.current_shortest_path) - 1:
                    # if the attachment site was determined definitively (corner or protruding), have reached
                    # intended attachment site and should assess whether block can be placed or not
                    if not environment.check_occupancy_map(self.current_grid_position):
                        # if yes, just place it
                        self.current_task = Task.PLACE_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        self.current_shortest_path = None
                    else:
                        # if no, need to find new attachment site (might just be able to restart this method)
                        self.current_path = None
                        self.current_shortest_path = None
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
                    self.recheck_task(environment)
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            # update local occupancy map while moving over the structure
            block_below = environment.block_below(self.geometry.position)
            # also need to check whether block is in shortest path
            if block_below is not None and block_below.grid_position[2] == self.current_grid_position[2] \
                    and self.component_target_map[block_below.grid_position[2],
                                                  block_below.grid_position[1],
                                                  block_below.grid_position[0]] == self.current_component_marker:
                # the following might not be enough, might have to check whether block was in the original SP
                if self.current_sp_index < len(self.current_shortest_path) - 1:
                    self.current_grid_position = np.copy(block_below.grid_position)
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FIND_ATTACHMENT_SITE] += simple_distance(position_before,
                                                                                       self.geometry.position)

    def place_block(self, environment: env.map.Map):
        """
        Move with the goal of placing a block.

        This method is called if the current task is PLACE_BLOCK. If the agent has not planned a path yet,
        it determines a path to descend from the current position to a position where it can let go of the block
        to place it. In a more realistic simulation this placement process would likely be much more complex and
        may indeed turn out to be one of the most difficult parts of the low-level quadcopter control necessary
        for the construction task. In this case however, this complexity is not considered. Once the block has been
        placed, if the structure is not finished the task becomes FETCH_BLOCK, otherwise it becomes LAND. Since this
        is the global knowledge version of this algorithm, the agent also notifies all other agents of the change.

        :param environment: the environment the agent operates in
        """

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

        # check again whether attachment is allowed since other agents placing blocks there may have made it illegal
        if not self.current_block_type_seed:
            # check whether site is occupied
            if environment.check_occupancy_map(self.current_grid_position):
                self.current_path = None
                self.current_task = Task.FIND_ATTACHMENT_SITE
                self.task_history.append(self.current_task)
                self.find_attachment_site(environment)
                return

            # check whether site is still legal
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
                if self.current_block.geometry.position[2] > (self.current_grid_position[2] + 1.0) * Block.SIZE:
                    self.aprint("Error: block placed in the air ({})".format(self.current_grid_position[2]))
                    self.current_path.add_position(
                        np.array([self.geometry.position[0], self.geometry.position[1],
                                  (self.current_grid_position[2] + 1) * Block.SIZE + self.geometry.size[2] / 2]))
                    return

                if self.current_block.is_seed:
                    self.current_seed = self.current_block
                    self.components_seeded.append(int(self.current_component_marker))
                    self.seeded_blocks += 1
                else:
                    self.sp_search_count.append(
                        (self.current_sp_search_count, int(self.current_component_marker), self.current_task.name))
                    self.current_sp_search_count = 0
                    if self.current_component_marker not in self.components_attached:
                        self.components_attached.append(int(self.current_component_marker))
                    self.attached_blocks += 1

                if self.rejoining_swarm:
                    self.rejoining_swarm = False

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
                    self.current_task = Task.LAND
                elif self.check_component_finished(self.local_occupancy_map):
                    self.current_task = Task.FETCH_BLOCK
                else:
                    self.current_task = Task.FETCH_BLOCK
                    if self.current_block_type_seed:
                        self.current_block_type_seed = False
                self.task_history.append(self.current_task)

                for a in environment.agents:
                    a.recheck_task(environment)
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.PLACE_BLOCK] += simple_distance(position_before, self.geometry.position)

    def advance(self, environment: env.map.Map):
        """
        Perform the next step of movement according to the current task.

        Aside from calling the respective method for the current task, this method also takes care of some other
        responsibilities, which are not specific to any specific task. In an earlier version, it was decided in
        this method whether to initiate collision avoidance by dodging other agents explicitly and this would be
        the best place to do so should it be reintroduced.

        :param environment: the environment the agent operates in
        """

        if self.current_task == Task.FINISHED:
            return

        if self.current_task != Task.LAND and self.current_block is None:
            finished = False
            if self.check_structure_finished(environment.occupancy_map):
                finished = True
            elif all([len(environment.seed_stashes[key]) == 0 for key in environment.seed_stashes]) \
                    and all([len(environment.block_stashes[key]) == 0 for key in environment.block_stashes]):
                finished = True

            if finished:
                self.drop_out_of_swarm = False
                self.wait_for_rejoining = False
                self.rejoining_swarm = False
                self.current_path = None
                self.current_task = Task.LAND
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

        # since collision avoidance is as basic as it is it could happen that agents are stuck in a position
        # and cannot move past each other, in that case the following makes them move a bit so that their
        # positions change enough for the collision avoidance to take care of the congestion
        if self.current_task not in [Task.FINISHED, Task.LAND, Task.HOVER_OVER_COMPONENT]:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70 \
                    and self.current_path is not None and not self.wait_for_rejoining:
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
                self.current_path.add_position(
                    [self.geometry.position[0] + self.geometry.size[0] * (random.random() - 0.5),
                     self.geometry.position[1] + self.geometry.size[1] * (random.random() - 0.5),
                     self.geometry.position[2] + self.geometry.size[2] * 2], self.current_path.current_index)

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

        # the steps done per layer and component
        if int(self.current_structure_level) not in self.steps_per_layer:
            self.steps_per_layer[int(self.current_structure_level)] = [[0, 0], [0, 0]]
        self.steps_per_layer[int(self.current_structure_level)][
            0 if self.current_block_type_seed else 1][0 if self.current_block is not None else 0] += 1
        if int(self.current_component_marker) not in self.steps_per_component:
            self.steps_per_component[int(self.current_component_marker)] = [[0, 0], [0, 0]]
        self.steps_per_component[int(self.current_component_marker)][
            0 if self.current_block_type_seed else 1][0 if self.current_block is not None else 0] += 1

        # the delay between a component actually being finished and the agent realising that it is
        if self.check_component_finished(environment.occupancy_map):
            if int(self.current_component_marker) not in self.complete_to_switch_delay:
                self.complete_to_switch_delay[int(self.current_component_marker)] = 0
            self.current_component_switch_marker = self.current_component_marker

        if self.current_component_switch_marker != -1:
            if self.check_component_finished(self.local_occupancy_map, self.current_component_switch_marker):
                self.current_component_switch_marker = -1
            else:
                self.complete_to_switch_delay[int(self.current_component_switch_marker)] += 1
