import numpy as np
import random
import env.map
from agents.agent import Task, Agent, check_map
from agents.global_knowledge.gk_agent import GlobalKnowledgeAgent
from env.block import Block
from env.util import legal_attachment_sites
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance, rotation_2d


class GlobalPerimeterFollowingAgent(GlobalKnowledgeAgent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(GlobalPerimeterFollowingAgent, self).__init__(
            position, size, target_map, required_spacing, printing_enabled)
        self.current_stash_position = None

    def find_attachment_site(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            # might consider putting something here
            self.current_path = Path()
            self.current_path.add_position(self.geometry.position)

        # use the following list to keep track of the combination of visited attachment sites and the direction of
        # movement at that point in time; if, during the same "attachment search" one such "site" is revisited, then
        # the agent is stuck in a loop, most likely caused by being trapped in a hole
        if self.current_visited_sites is None:
            self.current_visited_sites = []

        if check_map(self.hole_map, self.current_grid_position, lambda x: x > 1): # need to check if hole is closed
            hole_marker = self.hole_map[self.current_grid_position[2],
                                        self.current_grid_position[1],
                                        self.current_grid_position[0]]
            if np.count_nonzero(environment.occupancy_map[self.hole_boundaries[hole_marker]] != 0) == 0:
                self.current_task = Task.MOVE_TO_PERIMETER
                self.task_history.append(self.current_task)
                self.current_grid_direction = [1, 0, 0]
                return

        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            if not ret:
                # corner of the current block reached, assess next action
                at_loop_corner, loop_corner_attachable = self.check_loop_corner(environment, self.current_grid_position)
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
                    if check_map(self.hole_map, self.current_grid_position, lambda x: x > 1):
                        self.current_task = Task.MOVE_TO_PERIMETER
                        self.task_history.append(self.current_task)
                        self.current_grid_direction = [1, 0, 0]
                    else:
                        self.recheck_task(environment)
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
                        self.aprint("LANDING (2)")
                        self.aprint(True, self.target_map)
                    else:
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

                        sites = legal_attachment_sites(self.target_map[self.current_structure_level],
                                                       environment.occupancy_map[self.current_structure_level],
                                                       component_marker=self.current_component_marker)
                        self.per_search_attachment_site_count["possible"].append(1)
                        self.per_search_attachment_site_count["total"].append(int(np.count_nonzero(sites)))
                else:
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

                if self.check_component_finished(environment.occupancy_map):
                    self.aprint("FINISHED COMPONENT {} AFTER MOVING TO NEXT BLOCK IN ATTACHMENT SITE"
                                .format(self.current_component_marker))
                    self.recheck_task(environment)
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.FIND_ATTACHMENT_SITE] += simple_distance(position_before, self.geometry.position)

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

        if environment.check_occupancy_map(self.current_grid_position):
            self.current_path = None
            self.current_task = Task.TRANSPORT_BLOCK
            self.task_history.append(self.current_task)
            self.transport_block(environment)
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

                if self.current_block.geometry.position[2] > (self.current_grid_position[2] + 0.5) * Block.SIZE:
                    self.logger.error("BLOCK PLACED IN AIR ({})".format(self.current_grid_position[2]))
                    self.current_path.add_position(
                        np.array([self.geometry.position[0], self.geometry.position[1],
                                  (self.current_grid_position[2] + 1) * Block.SIZE + self.geometry.size[2] / 2]))
                    return

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

                if self.rejoining_swarm:
                    self.rejoining_swarm = False

                if self.check_structure_finished(environment.occupancy_map) \
                        or (self.check_layer_finished(environment.occupancy_map)
                            and self.current_structure_level >= self.target_map.shape[0] - 1):
                    self.aprint("AFTER PLACING BLOCK: FINISHED")
                    self.current_task = Task.LAND
                    self.aprint("LANDING (3)")
                elif self.check_component_finished(environment.occupancy_map):
                    self.aprint("AFTER PLACING BLOCK: FINDING NEXT COMPONENT")
                    self.current_task = Task.FETCH_BLOCK
                else:
                    self.aprint("AFTER PLACING BLOCK: FETCHING BLOCK FOR COMPONENT {} (PREVIOUS WAS SEED: {})"
                                .format(self.current_component_marker, self.current_block_type_seed))
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

        if self.current_task != Task.LAND and self.check_structure_finished(environment.occupancy_map):
            self.current_task = Task.LAND
            self.aprint("LANDING (8)")
            self.task_history.append(self.current_task)

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

        if self.current_task not in [Task.FINISHED, Task.LAND, Task.HOVER_OVER_COMPONENT]:
            if len(self.position_queue) == self.position_queue.maxlen \
                    and sum([simple_distance(self.geometry.position, x) for x in self.position_queue]) < 70 \
                    and self.current_path is not None:
                self.aprint("STUCK")
                self.stuck_count += 1
                self.current_path.add_position([self.geometry.position[0],
                                                self.geometry.position[1],
                                                self.geometry.position[2] + self.geometry.size[2] * 2 * random.random()],
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

        self.position_queue.append(self.geometry.position.copy())

        collision_danger = False
        if self.collision_possible and self.current_task:
            for a in environment.agents:
                if self is not a and self.collision_potential(a) and self.collision_potential_visible(a):
                    if self.current_path is None:
                        self.path_before_collision_avoidance_none = True
                    collision_danger = True
                    # self.collision_count += 1
                    break

        self.step_count += 1
        self.collision_queue.append(collision_danger)
        self.count_since_last_attachment += 1
        self.drop_out_statistics["drop_out_of_swarm"].append(self.drop_out_of_swarm)
        self.drop_out_statistics["wait_for_rejoining"].append(self.wait_for_rejoining)
        self.drop_out_statistics["rejoining_swarm"].append(self.rejoining_swarm)
