import numpy as np
import random
import env.map
from abc import abstractmethod
from agents.agent import Task, Agent, check_map
from env.block import Block
from geom.shape import *
from geom.path import Path
from geom.util import simple_distance, rotation_2d


class GlobalKnowledgeAgent(Agent):

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 target_map: np.ndarray,
                 required_spacing: float = 5.0,
                 printing_enabled=True):
        super(GlobalKnowledgeAgent, self).__init__(position, size, target_map, required_spacing, printing_enabled)
        self.wait_for_seed = False
        self.dropping_out_enabled = False
        self.component_ordering = "center"  # others are "percentage", "agents", "distance"
        self.current_dropped_out_count = 0
        self.dropped_out_count = []

    def order_components(self,
                         compared_map: np.ndarray,
                         environment: env.map.Map,
                         candidate_components,
                         order="center"):
        # just using this for the global knowledge version for now

        # first option: ordering by distance measures: either geometric distance or blocks to travel there
        # note that the latter may actually not be applicable in all situations (e.g. when the QC is off-site)
        seed_grid_locations = []
        seed_locations = []
        for m in candidate_components:
            temp = tuple(self.component_seed_location(m))
            seed_grid_locations.append(temp)
            seed_locations.append(np.array([environment.offset_origin[0] + temp[0] * Block.SIZE,
                                            environment.offset_origin[1] + temp[1] * Block.SIZE,
                                            (temp[2] + 0.5) * Block.SIZE]))

        order_by_distance = sorted(range(len(candidate_components)),
                                   key=lambda i: (simple_distance(seed_locations[i], self.geometry.position),
                                                  simple_distance(seed_locations[i], environment.center)))
        # order_by_hor_vert = sorted(range(len(candidate_components)),
        #                            key=lambda i: (abs(self.current_grid_position[0] - seed_grid_locations[i][0]) +
        #                                           abs(self.current_grid_position[1] - seed_grid_locations[i][1]),
        #                                           simple_distance(seed_locations[i], environment.center)))

        # second option: order by the number of agents over that component; realistically this could only ever
        # be an estimate but for the sake of time and implementational difficulty, it is just the count now
        candidate_component_count = [0] * len(candidate_components)
        for a in environment.agents:
            if a is not self:
                for m_idx, m in enumerate(candidate_components):
                    closest_x = int((a.geometry.position[0] - environment.offset_origin[0]) / env.block.Block.SIZE)
                    closest_y = int((a.geometry.position[1] - environment.offset_origin[1]) / env.block.Block.SIZE)
                    if 0 <= closest_x < self.target_map.shape[2] and 0 <= closest_y < self.target_map.shape[1] \
                            and any([self.component_target_map[z, closest_y, closest_x] == m
                                     for z in range(self.target_map.shape[0])]):
                        candidate_component_count[m_idx] += 1
        order_by_agent_count = sorted(range(len(candidate_components)),
                                      key=lambda i: (candidate_component_count[i],
                                                     simple_distance(seed_locations[i], environment.center),
                                                     simple_distance(seed_locations[i], self.geometry.position)))

        # third option: sort by distance to the center of the construction area (to keep outside spaces open)
        # if the distances are the same, then a good option would be to also take one's own distance into account
        order_by_center_distance = sorted(range(len(candidate_components)),
                                          key=lambda i: (simple_distance(seed_locations[i], environment.center),
                                                         simple_distance(seed_locations[i], self.geometry.position)))

        # fourth option: sort by the number of blocks already placed compared to how many should be placed
        # again, it probably makes sense to sort this by some other criteria as well
        percentage_occupied = []
        for cc in candidate_components:
            coords = np.where(self.component_target_map == cc)
            total_count = len(coords[0])
            occupied_count = np.count_nonzero(compared_map[coords] != 0)
            percentage_occupied.append(occupied_count / total_count)
        order_by_percentage = sorted(range(len(candidate_components)),
                                     key=lambda i: (percentage_occupied[i],
                                                    simple_distance(seed_locations[i], environment.center),
                                                    simple_distance(seed_locations[i], self.geometry.position)))

        if order == "center":
            order = order_by_center_distance
        elif order == "percentage":
            order = order_by_percentage
        elif order == "agents":
            order = order_by_agent_count
        elif order == "distance":
            order = order_by_distance
        # elif order == "hor_vert":
        #     order = order_by_hor_vert
        return [candidate_components[i] for i in order]

    def recheck_task(self, environment: env.map.Map):
        changed_task = False
        if self.check_structure_finished(environment.occupancy_map) \
                or (self.current_block is None
                    and all([len(environment.block_stashes[key]) == 0 for key in environment.block_stashes])
                    and all([len(environment.seed_stashes[key]) == 0 for key in environment.seed_stashes])):
            self.current_task = Task.LAND
            self.task_history.append(self.current_task)
            self.aprint("LANDING (11)")
            self.current_path = None
            return True

        # maybe if carrying block for last component and there are no more seeds to go for,
        # just go to position that is not above that seed and hover there ????????

        if self.current_block is not None and self.current_block.geometry in self.geometry.attached_geometries \
                and not self.current_block_type_seed and len(environment.seed_stashes) > 0 \
                and all([len(environment.seed_stashes[key]) == 0 for key in environment.seed_stashes]):
            if self.current_task != Task.HOVER_OVER_COMPONENT \
                    and self.current_component_marker in self.unseeded_component_markers(environment.occupancy_map):
                self.aprint("SWITCHING TO HOVER")
                self.current_task = Task.HOVER_OVER_COMPONENT
                self.task_history.append(self.current_task)
                self.required_distance += 15
                self.wait_for_seed = False
                self.current_path = None
                return
            elif self.current_task == Task.HOVER_OVER_COMPONENT and self.current_component_marker \
                    not in self.unseeded_component_markers(environment.occupancy_map):
                self.aprint("STOPPING HOVER, TASK: {}".format(self.current_task))
                intended_seed_position = self.component_seed_location(self.current_component_marker)
                intended_seed = environment.block_at_position(intended_seed_position)
                self.current_seed = intended_seed
                self.current_task = Task.TRANSPORT_BLOCK
                self.task_history.append(self.current_task)
                self.required_distance -= 15
                self.wait_for_seed = False
                self.current_path = None
            elif self.current_task == Task.TRANSPORT_BLOCK \
                    and not check_map(self.component_target_map,
                                      self.current_seed.grid_position,
                                      lambda x: x == self.current_component_marker):
                intended_seed_position = self.component_seed_location(self.current_component_marker)
                intended_seed = environment.block_at_position(intended_seed_position)
                self.current_seed = intended_seed

        if self.current_task == Task.FETCH_BLOCK or self.current_task == Task.PICK_UP_BLOCK:
            if self.current_block_type_seed:
                if environment.check_occupancy_map(self.next_seed_position):
                    # intended site for seed has been occupied, could now make a choice to attach there instead
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    unfinished = self.unfinished_component_markers(environment.occupancy_map)
                    if len(unseeded) != 0:
                        # can switch to seeding other component
                        unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                        self.current_component_marker = unseeded[0]
                        self.next_seed_position = self.component_seed_location(self.current_component_marker)
                        if self.current_block is not None:
                            self.current_block.color = Block.COLORS_SEEDS[0]
                            self.current_block = None
                        self.current_path = None
                        changed_task = True
                    elif len(unfinished) != 0:
                        # if there are unfinished components left, switch to attach to one of them
                        if self.current_component_marker not in unfinished:
                            unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                            self.current_component_marker = unfinished[0]
                        self.current_seed = environment.block_at_position(
                            self.component_seed_location(self.current_component_marker))
                        self.current_block_type_seed = False
                        if self.current_task != Task.FETCH_BLOCK:
                            self.current_task = Task.FETCH_BLOCK
                            self.task_history.append(self.current_task)
                            if self.current_block is not None:
                                self.current_block.color = Block.COLORS_SEEDS[0]
                                self.current_block = None
                        self.current_path = None
                        changed_task = True
                    else:
                        # there are no unfinished components left on this layer, therefore switch to the next
                        self.current_structure_level += 1
                        return self.recheck_task(environment)
            else:
                unfinished = self.unfinished_component_markers(environment.occupancy_map)
                if self.current_component_marker not in unfinished:
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    if len(unfinished) != 0:
                        # there are unfinished components left, but they may not be seeded yet
                        # if there is the option of seeding or attaching at some component, need to decide which one
                        if len(unfinished) == len(unseeded):
                            # none of the unfinished components are seeded yet, should therefore go for seed
                            unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                            self.current_component_marker = unseeded[0]
                            self.next_seed_position = self.component_seed_location(self.current_component_marker)
                            self.current_block_type_seed = True
                            if self.current_task != Task.FETCH_BLOCK:
                                self.current_task = Task.FETCH_BLOCK
                                self.task_history.append(self.current_task)
                                if self.current_block is not None:
                                    self.current_block.color = "#FFFFFF"
                                    self.current_block = None
                            self.current_path = None
                            changed_task = True
                        elif len(unseeded) == 0:
                            # all of the components are seeded already, should therefore go for block
                            unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                            self.current_component_marker = unfinished[0]
                            self.current_seed = environment.block_at_position(
                                self.component_seed_location(self.current_component_marker))
                            if self.current_block is not None:
                                self.current_block.color = "#FFFFFF"
                                self.current_block = None
                            self.current_path = None
                            changed_task = True
                        else:
                            # some are seeded, some are not, leaving the choice for which to go for
                            # for now, just go for unfinished ones
                            seeded = [c for c in unfinished if c not in unseeded]
                            seeded = self.order_components(environment.occupancy_map, environment, seeded, self.component_ordering)
                            self.current_component_marker = seeded[0]
                            self.current_seed = environment.block_at_position(
                                self.component_seed_location(self.current_component_marker))
                            if self.current_block is not None:
                                self.current_block.color = "#FFFFFF"
                                self.current_block = None
                            self.current_path = None
                            changed_task = True
                    else:
                        # all components are finished, therefore the layer is finished as well
                        self.current_structure_level += 1
                        # I think that this should be enough (to evoke the entire thing above)
                        return self.recheck_task(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK or self.current_task == Task.WAIT_ON_PERIMETER \
                or self.current_task == Task.PLACE_BLOCK:
            # the check for PLACE_BLOCK is there for the (unlikely) case that an agent is pushed away from
            # its intended attachment site so far that some other agents attaches there first
            if self.current_block_type_seed:
                if environment.check_occupancy_map(self.next_seed_position):
                    # intended site for seed has been occupied, could now make a choice to attach there instead
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    unfinished = self.unfinished_component_markers(environment.occupancy_map)
                    if len(unseeded) != 0:
                        # can switch to seeding other component
                        unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                        self.current_component_marker = unseeded[0]
                        self.next_seed_position = self.component_seed_location(self.current_component_marker)
                        self.current_path = None
                        changed_task = True
                        if self.current_task == Task.PLACE_BLOCK:
                            self.current_task = Task.TRANSPORT_BLOCK
                            self.task_history.append(self.current_task)
                    elif len(unfinished) == 0:
                        # there are no more unfinished components on this layer, therefore switch to next
                        self.current_structure_level += 1
                        return self.recheck_task(environment)
                    else:
                        # there are unfinished components left, but no unseeded ones, should return and get normal block
                        unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                        self.current_component_marker = unfinished[0]
                        self.current_seed = environment.block_at_position(
                            self.component_seed_location(self.current_component_marker))
                        self.current_block_type_seed = True
                        self.current_task = Task.RETURN_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        changed_task = True
            else:
                unfinished = self.unfinished_component_markers(environment.occupancy_map)
                if self.current_component_marker not in unfinished:
                    # component is finished
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    if len(unfinished) != 0:
                        # there are still other unfinished components left
                        if len(unfinished) == len(unseeded):
                            # all of these components have not been seeded yet, therefore should fetch a seed for one
                            unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                            self.current_component_marker = unseeded[0]
                            self.next_seed_position = self.component_seed_location(self.current_component_marker)
                            self.current_block_type_seed = False
                            self.current_task = Task.RETURN_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            changed_task = True
                        elif len(unseeded) == 0:
                            # there are no unseeded components left, therefore have free choice
                            unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                            self.current_component_marker = unfinished[0]
                            self.current_seed = environment.block_at_position(
                                self.component_seed_location(self.current_component_marker))
                            self.current_path = None
                            changed_task = True
                        else:
                            # some components are seeded, others not, since already carrying block, go for a seeded one
                            seeded = [c for c in unfinished if c not in unseeded]
                            seeded = self.order_components(environment.occupancy_map, environment, seeded, self.component_ordering)
                            self.current_component_marker = seeded[0]
                            self.current_seed = environment.block_at_position(
                                self.component_seed_location(self.current_component_marker))
                            self.current_path = None
                            changed_task = True
                    else:
                        # all components on this layer are completed
                        self.current_structure_level += 1
                        return self.recheck_task(environment)
        elif self.current_task == Task.MOVE_TO_PERIMETER or self.current_task == Task.FIND_ATTACHMENT_SITE:
            unfinished = self.unfinished_component_markers(environment.occupancy_map)
            if self.current_component_marker not in unfinished:
                # component is finished already, therefore try switching to other seeded component
                unseeded = self.unseeded_component_markers(environment.occupancy_map)
                if len(unfinished) != 0:
                    previous_component_marker = self.current_component_marker
                    # there are still other unfinished components left
                    if len(unfinished) == len(unseeded):
                        # all of these components have not been seeded yet, therefore should fetch a seed for one
                        unseeded = self.order_components(
                            environment.occupancy_map, environment, unseeded, self.component_ordering)
                        self.current_component_marker = unseeded[0]
                        self.next_seed_position = self.component_seed_location(self.current_component_marker)
                        self.current_block_type_seed = False
                        self.current_task = Task.RETURN_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        changed_task = True
                    elif len(unseeded) == 0:
                        # there are no unseeded components left, therefore have free choice, but need to go there first
                        unfinished = self.order_components(
                            environment.occupancy_map, environment, unfinished, self.component_ordering)
                        self.current_component_marker = unfinished[0]
                        self.current_seed = environment.block_at_position(
                            self.component_seed_location(self.current_component_marker))
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        changed_task = True
                    else:
                        # some components are seeded, others not, since already carrying block, go for a seeded one
                        seeded = [c for c in unfinished if c not in unseeded]
                        seeded = self.order_components(
                            environment.occupancy_map, environment, seeded, self.component_ordering)
                        self.current_component_marker = seeded[0]
                        self.current_seed = environment.block_at_position(
                            self.component_seed_location(self.current_component_marker))
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        changed_task = True
                    self.sp_search_count.append(
                        (self.current_sp_search_count, int(previous_component_marker), self.current_task.name))
                    self.current_sp_search_count = 0
                else:
                    # all components on this layer are completed
                    self.current_structure_level += 1
                    return self.recheck_task(environment)
        elif self.current_task == Task.RETURN_BLOCK and not self.drop_out_of_swarm:
            if self.current_block_type_seed:
                # this means that we are currently trying to fetch a normal block
                unfinished = self.unfinished_component_markers(environment.occupancy_map)
                if self.current_component_marker not in unfinished:
                    # the intended component has already been completed
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    if len(unfinished) != 0:
                        if len(unfinished) == len(unseeded):
                            # none of the other possible options have been seeded yet, can therefore choose one to seed
                            unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                            self.current_component_marker = unseeded[0]
                            self.next_seed_position = self.component_seed_location(self.current_component_marker)
                            self.current_task = Task.TRANSPORT_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            changed_task = True
                        elif len(unseeded) == 0:
                            # all the components are seeded already, therefore no choice but to return block
                            unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                            self.current_component_marker = unfinished[0]
                            self.current_seed = environment.block_at_position(
                                self.component_seed_location(self.current_component_marker))
                            self.current_path = None
                            changed_task = True
                        else:
                            # could decide whether to return block or to go and seed an unseeded component
                            unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                            self.current_component_marker = unseeded[0]
                            self.next_seed_position = self.component_seed_location(self.current_component_marker)
                            self.current_task = Task.TRANSPORT_BLOCK
                            self.task_history.append(self.current_task)
                            self.current_path = None
                            changed_task = True
                    else:
                        # the layer is finished, therefore move up
                        self.current_structure_level += 1
                        return self.recheck_task(environment)
            else:
                # this means that we are currently trying to fetch a seed
                if environment.check_occupancy_map(self.next_seed_position):
                    # the originally intended position has already been occupied
                    unseeded = self.unseeded_component_markers(environment.occupancy_map)
                    unfinished = self.unfinished_component_markers(environment.occupancy_map)
                    if len(unseeded) != 0:
                        # can switch to seeding other component
                        unseeded = self.order_components(environment.occupancy_map, environment, unseeded, self.component_ordering)
                        self.current_component_marker = unseeded[0]
                        self.next_seed_position = self.component_seed_location(self.current_component_marker)
                        self.current_path = None
                        changed_task = True
                    elif len(unfinished) != 0:
                        # if there are unfinished components left, switch to attach to one of them
                        if self.current_component_marker not in unfinished:
                            unfinished = self.order_components(environment.occupancy_map, environment, unfinished, self.component_ordering)
                            self.current_component_marker = unfinished[0]
                        self.current_seed = environment.block_at_position(
                            self.component_seed_location(self.current_component_marker))
                        self.current_task = Task.TRANSPORT_BLOCK
                        self.task_history.append(self.current_task)
                        self.current_path = None
                        changed_task = True
                    else:
                        # there are no unfinished components left either, therefore
                        # move up a layer and try the same thing there
                        self.current_structure_level += 1
                        return self.recheck_task(environment)

        if self.current_task == Task.PICK_UP_BLOCK and self.current_path is None and self.current_block is not None:
            self.current_block.color = Block.COLORS_SEEDS[0] if self.current_block_type_seed else "#FFFFFF"
            self.current_block = None

        return changed_task

    def fetch_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            if all([len(val) == 0 for _, val in environment.block_stashes.items()]) \
                    and all([len(val) == 0 for _, val in environment.seed_stashes.items()]):
                self.current_task = Task.LAND
                self.task_history.append(self.current_task)
                self.current_path = None
                self.aprint("LANDING BECAUSE ALL STASHES ARE EMPTY")
                return

            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes

            # given an approximate location for blocks, go there to pick on eup
            min_stash_location = None
            min_distance = float("inf")
            for p in list(stashes.keys()):
                if len(stashes[p]) != 0:
                    distance = simple_distance(self.geometry.position, p)
                    if distance < min_distance:
                        min_distance = distance
                        min_stash_location = p

            if min_stash_location is None:
                if self.current_block_type_seed:
                    self.current_block_type_seed = False
                else:
                    self.current_block_type_seed = True
                self.current_path = None
                self.fetch_block(environment)
                return

            self.current_stash_position = min_stash_location

            # construct path to that location
            # first add a point to get up to the level of movement for fetching blocks
            # which is one above the current construction level
            fetch_level_z = max(self.geometry.position[2], self.geometry.position[2] + Block.SIZE * 2)
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], fetch_level_z])
            self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z],
                                           optional_distance=20)

        # if within a certain distance of the stash, check whether there are many other agents there
        # that should maybe be avoided (issue here might be that other it's is fairly likely due to
        # the low approach to the stashes that other agents push ones already there out of the way; in
        # that case it would this check might still do some good (?))
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
                    if len(stashes[p]) != 0 and any([p[i] != self.current_path.positions[-1][i] for i in range(2)]):
                        distance = simple_distance(self.geometry.position, p)
                        if distance < min_distance:
                            min_distance = distance
                            min_stash_location = p

                if min_stash_location is not None:
                    self.current_stash_position = min_stash_location
                    fetch_level_z = max(self.geometry.position[2], self.geometry.position[2] + Block.SIZE * 2)
                    self.current_path = Path()
                    self.current_path.add_position([min_stash_location[0], min_stash_location[1], fetch_level_z])

        if self.current_block_type_seed:
            if len(environment.seed_stashes[self.current_stash_position]) == 0:
                self.current_path = None
                self.current_stash_position = None
                self.fetch_block(environment)
                return
        else:
            if len(environment.block_stashes[self.current_stash_position]) == 0:
                self.current_path = None
                self.current_stash_position = None
                self.fetch_block(environment)
                return

        # assuming that the if-statement above takes care of setting the path:
        # collision detection should intervene here if necessary
        next_position, current_direction = self.move(environment)
        if simple_distance(self.geometry.position, next_position) <= Agent.MOVEMENT_PER_STEP:
            self.geometry.position = next_position
            ret = self.current_path.advance()

            # if the final point on the path has been reached, determine a block to pick up
            if not ret:
                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                if len(stashes[self.current_stash_position]) == 0:
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
        position_before = np.copy(self.geometry.position)

        # at this point it has been confirmed that there is indeed a block around that location
        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # determine the closest block in that stash (instead of using location, might want to use current_stash?)
            min_block = None
            min_distance = float("inf")
            occupied_blocks = []
            for b in stashes[self.current_stash_position]:
                temp = self.geometry.distance_2d(b.geometry)
                if (not b.is_seed or self.current_block_type_seed) and not b.placed \
                        and not any([b is a.current_block for a in environment.agents]) and temp < min_distance:
                    min_block = b
                    min_distance = temp
                elif any([b is a.current_block for a in environment.agents]):
                    for a in environment.agents:
                        if b is a.current_block:
                            occupied_blocks.append((a.id, b))

            if min_block is None:
                # no more blocks at that location, need to go elsewhere
                self.aprint("EMPTY STASH DURING PICKUP (AT {})".format(self.current_stash_position))
                self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None
                self.fetch_block(environment)
                return

            # otherwise, make the selected block the current block and pick it up
            min_block.color = "red"
            self.current_block = min_block
            self.aprint("Block {} at {} made current_block".format(self.current_block, self.current_block.geometry.position))

            pickup_z = min_block.geometry.position[2] + Block.SIZE / 2 + self.geometry.size[2] / 2
            self.current_path = Path()
            self.current_path.add_position([min_block.geometry.position[0], min_block.geometry.position[1], pickup_z])

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

                stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
                stashes[self.current_stash_position].remove(self.current_block)
                self.geometry.attached_geometries.append(self.current_block.geometry)
                if self.current_block_type_seed:
                    self.current_block.color = "#f44295"
                else:
                    self.current_block.color = "green"

                if self.rejoining_swarm and not self.current_block_type_seed:
                    self.current_task = Task.REJOIN_SWARM
                else:
                    self.aprint("TRANSPORTING BLOCK")
                    self.current_task = Task.TRANSPORT_BLOCK
                self.task_history.append(self.current_task)
                self.current_path = None

                if self.current_block_type_seed and self.next_seed_position is None:
                    self.next_seed_position = self.current_seed.grid_position

                for a in environment.agents:
                    a.recheck_task(environment)
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.PICK_UP_BLOCK] += simple_distance(position_before, self.geometry.position)

    def wait_on_perimeter(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
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

    def hover_over_component(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        # this should only happen when there is one unseeded component left, but there are no more seeds left
        # and therefore some other agent must be transporting that seed to the components
        # in that case it is best to just wait in place until the seed has been attached and hover
        if self.current_path is None:
            coords = np.where(self.component_target_map == self.current_component_marker)
            average_x = np.average(coords[2])
            average_y = np.average(coords[1])

            hover_x = environment.offset_origin[0] + Block.SIZE * average_x
            hover_y = environment.offset_origin[1] + Block.SIZE * average_y
            hover_z = Block.SIZE * (self.current_structure_level + 2) + self.required_distance + \
                      self.geometry.size[2] * 4
            self.current_path = Path()
            self.current_path.add_position([hover_x, hover_y, hover_z], optional_distance=(200, 200, 50))

        reached_end_zone = False
        if self.wait_for_seed:
            next_position, current_direction = self.move(environment, True)
        else:
            next_position, current_direction = self.move(environment)
            if self.current_path.optional_area_reached(self.geometry.position) \
                    and not simple_distance(self.geometry.position + current_direction, next_position) \
                            < simple_distance(self.geometry.position, next_position):
                ret = self.current_path.advance()
                next_position, current_direction = self.move(environment)
                if not ret:
                    reached_end_zone = True

        if not self.wait_for_seed and (simple_distance(self.geometry.position, next_position)
                                       <= Agent.MOVEMENT_PER_STEP or reached_end_zone):
            if not reached_end_zone:
                self.geometry.position = next_position
                ret = self.current_path.advance()

            if reached_end_zone or not ret:
                self.current_static_location = self.geometry.position
                self.wait_for_seed = True
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.HOVER_OVER_COMPONENT] += simple_distance(position_before, self.geometry.position)

    def rejoin_swarm(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if not self.rejoining_swarm and not ((self.collision_count / self.step_count) < 0.34 and random.random() < 0.9):
            return

        self.rejoining_swarm = True
        self.wait_for_rejoining = False
        self.drop_out_of_swarm = False

        if self.current_block is None:
            self.current_task = Task.FETCH_BLOCK
            self.task_history.append(self.current_task)
            self.current_block_type_seed = False
            self.fetch_block(environment)
            return

        # because global information is available, this whole procedure is relatively straigh-forward
        highest_layer = self.current_structure_level
        for z in range(self.current_structure_level, self.target_map.shape[0]):
            if np.count_nonzero(environment.occupancy_map[z]) > 0:
                highest_layer = z

        for z in range(highest_layer + 1):
            for y in range(self.target_map.shape[1]):
                for x in range(self.target_map.shape[2]):
                    self.local_occupancy_map[z, y, x] = environment.occupancy_map[z, y, x]
        self.current_structure_level = highest_layer

        if self.check_structure_finished(environment.occupancy_map):
            self.current_task = Task.LAND
            self.task_history.append(self.current_task)
            self.current_path = None
            self.land(environment)
            return

        # now find some component on that layer
        candidate_components = []
        cc_grid_positions = []
        cc_locations = []
        for m in self.unfinished_component_markers(environment.occupancy_map, highest_layer):
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

        order = sorted(range(len(candidate_components)),
                       key=lambda x: simple_distance(self.geometry.position, cc_locations[x]))

        order = sorted(order, key=lambda x: int(
            candidate_components[x] in self.unseeded_component_markers(environment.occupancy_map, highest_layer)))

        # just pick the closest (for now)
        candidate_components = [candidate_components[i] for i in order]
        cc_grid_positions = [cc_grid_positions[i] for i in order]
        cc_locations = [cc_locations[i] for i in order]

        # this is only possible because of the "cheating" (or if global information is used, although
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
        self.task_history.append(self.current_task)
        self.current_path = None

        if self.current_task == Task.RETURN_BLOCK:
            self.return_block(environment)
        elif self.current_task == Task.TRANSPORT_BLOCK:
            self.transport_block(environment)

        self.per_task_distance_travelled[Task.REJOIN_SWARM] += simple_distance(position_before, self.geometry.position)

    def transport_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        # in this case the seed location is taken as the structure location,
        # since that is where the search for attachment sites would start anyway
        if self.current_path is None:
            self.current_visited_sites = None

            # gain height, fly to seed location and then start search for attachment site
            seed_location = self.current_seed.geometry.position
            transport_level_z = Block.SIZE * self.current_structure_level + Block.SIZE + Block.SIZE + \
                                self.required_distance + self.geometry.size[2] * 1.5
            other_transport_level_z = (self.current_seed.grid_position[2] + 2) * Block.SIZE + self.geometry.size[2] * 2

            # if previously picked up block from stash, consider actually going slightly to the side w/ first point
            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], transport_level_z],
                                           optional_distance=(70, 70, 20))
            self.current_path.add_position([seed_location[0], seed_location[1], transport_level_z],
                                           optional_distance=30)
            self.current_path.add_position([seed_location[0], seed_location[1], other_transport_level_z])

        if self.waiting_on_perimeter_enabled and not environment.check_over_construction_area(self.geometry.position):
            # not over construction area yet
            if environment.distance_to_construction_area(self.geometry.position) <= self.geometry.size[0] * 2:
                # check whether close enough to construction area to warrant looking entering
                if self.area_density_restricted and environment.density_over_construction_area() > 1:
                    # in this case it's too crowded -> don't move in yet
                    self.current_task = Task.WAIT_ON_PERIMETER
                    self.task_history.append(self.current_task)
                    self.previous_path = self.current_path
                    self.current_path = None
                    self.wait_on_perimeter(environment)
                    return

        # since global information is being used, should check the entire time whether seed is being covered
        position_above = [self.current_seed.grid_position[0],
                          self.current_seed.grid_position[1],
                          self.current_seed.grid_position[2] + 1]
        if environment.check_occupancy_map(position_above):
            self.aprint("POSITION ABOVE PREVIOUS SEED ALREADY OCCUPIED")
            # the position is occupied
            block_above_seed = environment.block_at_position(position_above)
            if block_above_seed.is_seed:
                # the "blocking block" is a seed and can therefore also be used for orientation
                self.current_seed = block_above_seed
                self.current_structure_level = self.current_seed.grid_position[2]
            else:
                self.aprint("Current seed {} covered by block, trying to "
                            "find seed of the covering component at {}"
                            .format(self.current_seed.grid_position, block_above_seed.grid_position))

                seed_grid_location = self.component_seed_location(
                    self.component_target_map[block_above_seed.grid_position[2],
                                              block_above_seed.grid_position[1],
                                              block_above_seed.grid_position[0]])
                self.current_seed = environment.block_at_position(seed_grid_location)
                self.current_structure_level = self.current_seed.grid_position[2]
            self.current_path = None
            self.transport_block(environment)
            return

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
                # since this method is also used to move to a seed site with a carried seed after already having
                # found the current seed for localisation, need to check whether we have arrived and should
                # drop off the carried seed
                self.current_path = None
                self.current_grid_position = np.copy(self.current_seed.grid_position)
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

                        # the position is already occupied, need to move to different site
                        # check whether there are even any unseeded sites
                        unseeded = self.unseeded_component_markers(environment.occupancy_map)
                        unfinished = self.unfinished_component_markers(environment.occupancy_map)
                        if len(unseeded) == 0 and len(unfinished) > 0:
                            # this might only be applicable if we know that the block stashes are all exhausted?
                            # should only count block stashes here:
                            counter = 0
                            for _, val in environment.block_stashes.items():
                                if len(val) == 0:
                                    counter += 1
                            if counter >= len(environment.block_stashes):
                                self.current_task = Task.MOVE_TO_PERIMETER
                            else:
                                self.current_task = Task.RETURN_BLOCK
                        else:
                            if len(unseeded) == 0:
                                self.current_structure_level += 1
                            self.recheck_task(environment)
                        self.task_history.append(self.current_task)
                    self.transporting_to_seed_site = False
                    return

                if not self.current_block_type_seed:
                    self.aprint("CARRYING NORMAL BLOCK")
                    # should check whether the component is finished
                    if self.check_component_finished(environment.occupancy_map,
                                                     self.component_target_map[self.current_seed.grid_position[2],
                                                                               self.current_seed.grid_position[1],
                                                                               self.current_seed.grid_position[0]]):
                        self.recheck_task(environment)
                    else:
                        self.current_task = Task.MOVE_TO_PERIMETER
                        self.task_history.append(self.current_task)
                else:
                    self.aprint("REACHED OLD SEED AND NOW TRANSPORTING TO NEW SEED SITE")
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

                if self.check_component_finished(environment.occupancy_map):
                    self.aprint("FINISHED COMPONENT {} AFTER TRANSPORTING".format(self.current_component_marker))
                    self.recheck_task(environment)
                    self.task_history.append(self.current_task)
                    self.current_visited_sites = None
                    self.current_path = None
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.TRANSPORT_BLOCK] += simple_distance(position_before, self.geometry.position)

    def move_to_perimeter(self, environment: env.map.Map):
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

                try:
                    result = all(environment.occupancy_map[self.hole_boundaries[self.hole_map[
                        self.current_grid_position[2], self.current_grid_position[1],
                        self.current_grid_position[0]]]] != 0)
                except (IndexError, KeyError):
                    result = False

                if not self.current_block_type_seed:
                    self.current_blocks_per_attachment += 1

                if environment.block_below(self.geometry.position, self.current_structure_level) is None and \
                        (check_map(self.hole_map, self.current_grid_position, lambda x: x < 2) or not result):
                    # have reached perimeter
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

        self.per_task_distance_travelled[Task.MOVE_TO_PERIMETER] += simple_distance(position_before, self.geometry.position)

    def return_block(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            stashes = environment.seed_stashes if self.current_block_type_seed else environment.block_stashes
            # select the closest block stash
            min_stash_location = None
            min_distance = float("inf")
            for key, value in stashes.items():
                distance = simple_distance(self.geometry.position, key)
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

                if self.current_block_type_seed:
                    self.current_block.color = Block.COLORS_SEEDS[0]
                    environment.seed_stashes[self.current_stash_position].append(self.current_block)
                else:
                    self.current_block.color = "#FFFFFF"
                    environment.block_stashes[self.current_stash_position].append(self.current_block)

                if self.current_block.color == "green":
                    self.current_block.color = "#FFFFFF"

                self.current_block = None
                self.current_path = None
                if self.drop_out_of_swarm:
                    self.current_task = Task.LAND
                else:
                    self.aprint("FETCHING NEW BLOCK AFTER RETURNING ONE")
                    self.current_task = Task.FETCH_BLOCK
                self.task_history.append(self.current_task)

                self.current_block_type_seed = not self.current_block_type_seed
                self.returned_blocks += 1

                for a in environment.agents:
                    a.recheck_task(environment)
        else:
            self.geometry.position = self.geometry.position + current_direction

        self.per_task_distance_travelled[Task.RETURN_BLOCK] += simple_distance(position_before, self.geometry.position)

    def land(self, environment: env.map.Map):
        position_before = np.copy(self.geometry.position)

        if self.current_path is None:
            land_x = self.initial_position[0]
            land_y = self.initial_position[1]
            land_z = Block.SIZE * (self.current_structure_level + 2) + self.geometry.size[2] / 2 + self.required_spacing

            self.current_path = Path()
            self.current_path.add_position([self.geometry.position[0], self.geometry.position[1], land_z])
            self.current_path.add_position([land_x, land_y, land_z])
            self.current_path.add_position([land_x, land_y, self.geometry.size[2] / 2])

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

