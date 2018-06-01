import json
import os
import uuid
import sys
from pprint import pprint
from agents.agent import Task
from agents.global_knowledge.ps_agent import GlobalPerimeterFollowingAgent
from agents.global_knowledge.sp_agent import GlobalShortestPathAgent
from agents.local_knowledge.ps_agent import LocalPerimeterFollowingAgent
from agents.local_knowledge.sp_agent import LocalShortestPathAgent
from env.map import *
from env.util import *
from geom.shape import *
from structures import *

LOAD_DIRECTORY_NAME = "/home/simon/maps/"
SAVE_DIRECTORY_NAME = "/home/simon/results/"

OFFSET_STRUCTURE = 400.0
INTERVAL = 0.0000001
OFFSET_ORIGIN = (OFFSET_STRUCTURE, OFFSET_STRUCTURE)

AGENT_TYPES = {
    "LocalPerimeterFollowingAgent": LocalPerimeterFollowingAgent,
    "LocalShortestPathAgent": LocalShortestPathAgent,
    "GlobalPerimeterFollowingAgent": GlobalPerimeterFollowingAgent,
    "GlobalShortestPathAgent": GlobalShortestPathAgent
}

VALUES = {  # where the first is the default one
    "waiting_on_perimeter_enabled": [False],
    "avoiding_crowded_stashes_enabled": [True, False],
    "transport_avoid_others_enabled": [True, False],
    "seed_if_possible_enabled": [True, False],
    "seeding_strategy": ["distance_center", "distance_self", "agent_count"],
    "component_ordering": ["center", "percentage", "agents", "distance"],
    "attachment_site_order": ["shortest_path", "prioritise", "shortest_travel_path", "agent_count"],
    "attachment_site_ordering": ["shortest_path", "agent_count"]
}


def run_experiment(parameters):
    target_map = np.load(LOAD_DIRECTORY_NAME + parameters["target_map"] + ".npy").astype("int64")
    agent_count = parameters["agent_count"]
    agent_type = AGENT_TYPES[parameters["agent_type"]]  # should have dictionary with names mapping to constructors
    offset_stashes = parameters["offset_stashes"]

    # stuff...
    palette_block = list(sns.color_palette("Blues_d", target_map.shape[0]))
    palette_seed = list(sns.color_palette("Reds_d", target_map.shape[0]))
    hex_palette_block = []
    hex_palette_seed = []
    for i in range(len(palette_block)):
        rgb_block = (int(palette_block[i][0] * 255), int(palette_block[i][1] * 255), int(palette_block[i][2] * 255))
        rgb_seed = (int(palette_seed[i][0] * 255), int(palette_seed[i][1] * 255), int(palette_seed[i][2] * 255))
        hex_palette_block.append("#{:02x}{:02x}{:02x}".format(*rgb_block))
        hex_palette_seed.append("#{:02x}{:02x}{:02x}".format(*rgb_seed))
    Block.COLORS_BLOCKS = hex_palette_block
    Block.COLORS_SEEDS = hex_palette_seed

    # determining the environment extent
    environment_extent = [OFFSET_STRUCTURE * 2 + target_map.shape[2] * Block.SIZE,
                          OFFSET_STRUCTURE * 2 + target_map.shape[1] * Block.SIZE,
                          OFFSET_STRUCTURE * 2 + target_map.shape[1] * Block.SIZE]

    # creating the environment
    environment = Map(target_map, OFFSET_ORIGIN, environment_extent)
    block_count = environment.required_blocks()

    # finding out how many components there are
    dummy_agent = LocalPerimeterFollowingAgent([0, 0, 0], [0, 0, 0], target_map, printing_enabled=False)
    component_target_map = dummy_agent.split_into_components()
    required_seeds = np.max(component_target_map) - 2

    # creating the block_list and a list of initial positions
    block_list = []
    for _ in range(0, block_count):
        block_list.append(create_block(BlockType.INERT))

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    # placing first seed in correct position and the other seeds elsewhere
    block_list[0].is_seed = True
    block_list[0].placed = True
    block_list[0].geometry = GeomBox(list(environment.original_seed_position()), [Block.SIZE] * 3, 0.0)
    block_list[0].grid_position = environment.original_seed_grid_position()
    block_list[0].seed_marked_edge = "down"
    environment.place_block(block_list[0].grid_position, block_list[0])
    processed = [block_list[0]]

    for i in range(1, required_seeds + 1):
        block_list[i].is_seed = True
        block_list[i].color = Block.COLORS_SEEDS[0]
        block_list[i].geometry = GeomBox([OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE / 2,
                                          OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + offset_stashes,
                                          Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        processed.append(block_list[i])

    # placing the simple construction blocks
    chunk_list = split_into_chunks(block_list[(1 + required_seeds):], 4)
    for sl_idx, sl in enumerate(chunk_list):
        for b in sl:
            if sl_idx == 0:
                b.geometry.position = [OFFSET_STRUCTURE - offset_stashes,
                                       OFFSET_STRUCTURE - offset_stashes,
                                       Block.SIZE / 2]
            elif sl_idx == 1:
                b.geometry.position = [OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE + offset_stashes,
                                       OFFSET_STRUCTURE - offset_stashes,
                                       Block.SIZE / 2]
            elif sl_idx == 2:
                b.geometry.position = [OFFSET_STRUCTURE - offset_stashes,
                                       OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + offset_stashes,
                                       Block.SIZE / 2]
            elif sl_idx == 3:
                b.geometry.position = [OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE + offset_stashes,
                                       OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + offset_stashes,
                                       Block.SIZE / 2]
            processed.append(b)

    # creating the agent_list
    agent_list = [agent_type([50, 60, 7.5], [40, 40, 15], target_map, 10.0, False) for _ in range(0, agent_count)]
    for i in range(len(agent_list)):
        agent_list[i].id = i
        agent_list[i].waiting_on_perimeter_enabled = parameters["waiting_on_perimeter_enabled"]
        agent_list[i].avoiding_crowded_stashes_enabled = parameters["avoiding_crowded_stashes_enabled"]
        agent_list[i].transport_avoid_others_enabled = parameters["transport_avoid_others_enabled"]
        if "seed_if_possible_enabled" in parameters:
            agent_list[i].seed_if_possible_enabled = parameters["seed_if_possible_enabled"]
        if "seeding_strategy" in parameters:
            agent_list[i].seeding_strategy = parameters["seeding_strategy"]
        if "component_ordering" in parameters:
            agent_list[i].component_ordering = parameters["component_ordering"]
        if "attachment_site_order" in parameters:
            agent_list[i].attachment_site_order = parameters["attachment_site_order"]
        if "attachment_site_ordering" in parameters:
            agent_list[i].attachment_site_ordering = parameters["attachment_site_ordering"]

    # placing the agents
    processed_counter = 0
    while len(processed) != block_count + agent_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = GeomBox([candidate_x, candidate_y, agent_list[processed_counter].geometry.size[2] / 2],
                                agent_list[processed_counter].geometry.size * 4, 0.0)
        if all([not candidate_box.overlaps(p.geometry) for p in processed]):
            if candidate_box.position[0] - candidate_box.size[0] > \
                    Block.SIZE * target_map.shape[2] + environment.offset_origin[0] \
                    or candidate_box.position[0] + candidate_box.size[0] < environment.offset_origin[0] \
                    or candidate_box.position[1] - candidate_box.size[1] > \
                    Block.SIZE * target_map.shape[1] + environment.offset_origin[1] \
                    or candidate_box.position[1] + candidate_box.size[1] < environment.offset_origin[1]:
                agent_list[processed_counter].geometry.set_to_match(candidate_box)
                agent_list[processed_counter].geometry.size = agent_list[processed_counter].geometry.size / 4
                processed.append(agent_list[processed_counter])
                processed_counter += 1

    # adding the block_list
    environment.add_blocks(block_list)

    # adding the agent list
    environment.add_agents(agent_list)

    # running the main loop of the simulation
    results = {}
    collision_pairs = []

    # miscellaneous setup stuff
    component_markers = [int(cm) for cm in np.unique(dummy_agent.component_target_map) if cm != 0]

    # to terminate the simulation if it gets hung up on something
    no_change_counter = 0
    previous_map = np.copy(environment.occupancy_map)
    np.place(previous_map, previous_map > 1, 1)
    finished_successfully = False
    got_stuck = False

    # performance metrics to keep track of outside the agent
    # most general:
    steps = 0
    collisions = 0
    structure_finished_count = 0
    structure_finished = False

    # agents over construction zone and different components
    agents_over_construction_area = []
    agents_over_components = dict([(cm, []) for cm in component_markers])

    # counts from start to completion of components/layers
    component_completion = dict([(cm, None) for cm in component_markers])
    layer_completion = dict([(l, None) for l in range(target_map.shape[2])])
    current_layer = 0
    started_components = []
    completed_components = []
    started_layers = []
    completed_layers = []
    try:
        while True:
            for a in agent_list:
                a.advance(environment)
            steps += 1

            # agents over construction zone/components
            agents_over_construction_area.append(int(environment.count_over_construction_area()))

            # agents over component and component completion
            for cm in component_markers:
                agents_over_components[cm].append(int(environment.count_over_component(cm)))
                if cm not in started_components and environment.component_started(cm):
                    started_components.append(cm)
                    component_completion[cm] = {"start": steps, "finish": steps}
                if cm not in completed_components and cm in started_components and environment.component_finished(cm):
                    completed_components.append(cm)
                    component_completion[cm]["finish"] = steps

            # layer completion
            if current_layer < target_map.shape[2] and current_layer not in started_layers \
                    and environment.layer_started(current_layer):
                layer_completion[current_layer] = {"start": steps, "finish": steps}
            if current_layer < target_map.shape[2] and current_layer not in completed_layers \
                    and current_layer in started_layers and environment.layer_finished(current_layer):
                layer_completion[current_layer]["finish"] = steps
                current_layer += 1

            if not structure_finished and current_layer >= target_map.shape[2]:
                structure_finished_count = steps
                structure_finished = True
                current_layer -= 1

            if steps % 5000 == 0:
                print("Simulation steps: {}".format(steps))

            if got_stuck or all([a.current_task == Task.FINISHED for a in agent_list]):
                print("Finished construction with {} agents in {} steps ({} colliding)."
                      .format(agent_count, steps, collisions / 2))

                # meta information
                results["parameters"] = parameters
                results["finished_successfully"] = finished_successfully
                results["got_stuck"] = got_stuck

                # outside agents statistics
                results["step_count"] = steps
                results["collision_count"] = collisions
                results["agents_over_construction_area"] = agents_over_construction_area
                results["component_completion"] = component_completion
                results["layer_completion"] = layer_completion
                results["structure_completion"] = structure_finished_count
                results["highest_layer"] = int(environment.highest_block_z)

                # inside agents statistics
                results["total_step_counts"] = [a.step_count for a in agent_list]
                results["returned_block_counts"] = [a.returned_blocks for a in agent_list]
                results["stuck_counts"] = [a.stuck_count for a in agent_list]
                results["attachment_frequency_count"] = [a.attachment_frequency_count for a in agent_list]
                results["components_seeded"] = [a.components_seeded for a in agent_list]
                results["components_attached"] = [a.components_attached for a in agent_list]
                results["per_search_attachment_site_count"] = [a.per_search_attachment_site_count for a in agent_list]
                results["task_stats"] = {}
                for t in list(agent_list[0].per_task_step_count.keys()):
                    step_counts = [a.per_task_step_count[t] for a in agent_list]
                    step_count = {
                        "mean": float(np.mean(step_counts)),
                        "std": float(np.std(step_counts)),
                        "min": int(np.min(step_counts)),
                        "max": int(np.max(step_counts))
                    }
                    collision_avoidance_counts = [a.per_task_collision_avoidance_count[t] for a in agent_list]
                    collision_avoidance_count = {
                        "mean": float(np.mean(collision_avoidance_counts)),
                        "std": float(np.std(collision_avoidance_counts)),
                        "min": int(np.min(collision_avoidance_counts)),
                        "max": int(np.max(collision_avoidance_counts))
                    }
                    distances_travelled = [a.per_task_distance_travelled[t] for a in agent_list]
                    distance_travelled = {
                        "mean": float(np.mean(distances_travelled)),
                        "std": float(np.std(distances_travelled)),
                        "min": int(np.min(distances_travelled)),
                        "max": int(np.max(distances_travelled))
                    }

                    task_results = {
                        "step_count": step_count,
                        "collision_avoidance_count": collision_avoidance_count,
                        "distance_travelled": distance_travelled
                    }

                    results["task_stats"][t.name] = task_results
                break
            if len(agent_list) > 1:
                for a1 in agent_list:
                    for a2 in agent_list:
                        if a1 is not a2 and a1.overlaps(a2) \
                                and not a1.current_task == Task.FINISHED and not a2.current_task == Task.FINISHED:
                            collisions += 1
                            # print("Agent {} ({}) and {} ({}) colliding.".format(a1.id, a1.current_task,
                            #                                                     a2.id, a2.current_task))
                            collision_pairs.append((a1.current_task, a2.current_task))

            # checking whether the occupancy map has been updated/blocks have been placed
            current_map = np.copy(environment.occupancy_map)
            np.place(current_map, current_map > 1, 1)

            if (previous_map == current_map).sum() == 0:
                no_change_counter += 1
            else:
                no_change_counter = 0

            if no_change_counter >= 2000 or steps > 10000000:
                got_stuck = True

            previous_map = current_map

            # time.sleep(INTERVAL)
    except KeyboardInterrupt:
        if not finished_successfully:
            logger.info("Simulation interrupted.")
            print("Interrupted construction with {} agents in {} steps ({} colliding)."
                  .format(agent_count, steps, collisions / 2))
            if got_stuck:
                print("Interrupted because simulation got stuck.")
            if all([a.current_task == Task.FINISHED for a in agent_list]):
                print("Finished construction with {} agents in {} steps ({} colliding)."
                      .format(agent_count, steps, collisions / 2))

                # meta information
                results["parameters"] = parameters
                results["finished_successfully"] = finished_successfully
                results["got_stuck"] = got_stuck

                # outside agents statistics
                results["step_count"] = steps
                results["collision_count"] = collisions
                results["agents_over_construction_area"] = agents_over_construction_area
                results["component_completion"] = component_completion
                results["layer_completion"] = layer_completion
                results["structure_completion"] = structure_finished_count
                results["highest_layer"] = int(environment.highest_block_z)

                # inside agents statistics
                results["total_step_counts"] = [a.step_count for a in agent_list]
                results["returned_block_counts"] = [a.returned_blocks for a in agent_list]
                results["stuck_counts"] = [a.stuck_count for a in agent_list]
                results["attachment_frequency_count"] = [a.attachment_frequency_count for a in agent_list]
                results["components_seeded"] = [a.components_seeded for a in agent_list]
                results["components_attached"] = [a.components_attached for a in agent_list]
                results["per_search_attachment_site_count"] = [a.per_search_attachment_site_count for a in agent_list]
                results["task_stats"] = {}
                for t in list(agent_list[0].per_task_step_count.keys()):
                    step_counts = [a.per_task_step_count[t] for a in agent_list]
                    step_count = {
                        "mean": float(np.mean(step_counts)),
                        "std": float(np.std(step_counts)),
                        "min": int(np.min(step_counts)),
                        "max": int(np.max(step_counts))
                    }
                    collision_avoidance_counts = [a.per_task_collision_avoidance_count[t] for a in agent_list]
                    collision_avoidance_count = {
                        "mean": float(np.mean(collision_avoidance_counts)),
                        "std": float(np.std(collision_avoidance_counts)),
                        "min": int(np.min(collision_avoidance_counts)),
                        "max": int(np.max(collision_avoidance_counts))
                    }
                    distances_travelled = [a.per_task_distance_travelled[t] for a in agent_list]
                    distance_travelled = {
                        "mean": float(np.mean(distances_travelled)),
                        "std": float(np.std(distances_travelled)),
                        "min": int(np.min(distances_travelled)),
                        "max": int(np.max(distances_travelled))
                    }

                    task_results = {
                        "step_count": step_count,
                        "collision_avoidance_count": collision_avoidance_count,
                        "distance_travelled": distance_travelled
                    }

                    results["task_stats"][t.name] = task_results
        else:
            logger.info("Simulation finished successfully.")

    return results


def extra_parameters(agent_type: str):
    parameters = ["waiting_on_perimeter_enabled",
                  "avoiding_crowded_stashes_enabled",
                  "transport_avoid_others_enabled"]
    if agent_type.startswith("Local"):
        parameters.append("seed_if_possible_enabled")
        parameters.append("seeding_strategy")
    if agent_type.startswith("Local"):
        parameters.append("component_ordering")
    if agent_type == "LocalShortestPathAgent":
        parameters.append("attachment_site_order")
    if agent_type == "GlobalShortestPathAgent":
        parameters.append("attachment_site_ordering")
    return parameters


def main(map_name="block_4x4x4"):
    # one experiment is comprised of one map being tested
    # by default:
    # all agent types (saved in different sub-directories)
    # agent counts: 1, 2, 4, 8, 16
    # ten independent runs with all defaults
    # ten independent runs for each value (except dropout for now) with rest defaults
    # per run: save stuff

    # PARAMETERS THAT CAN BE CHANGED:
    # ALL AGENTS: waiting_on_perimeter_enabled, avoiding_crowded_stashes_enabled, transport_avoid_others_enabled
    # LOCAL AGENTS: seed_if_possible_enabled, seeding_strategy
    # GLOBAL AGENTS: component_ordering, (dropping_out_enabled)
    # LocalShortestPathAgent: attachment_site_order
    # GlobalShortestPathAgent: attachment_site_ordering

    # all needed parameters
    # target_map (name), agent_count, agent_type, (offset_stashes)
    # "waiting_on_perimeter_enabled": [False, True],
    # "avoiding_crowded_stashes_enabled": [False, True],
    # "transport_avoid_others_enabled": [False, True],
    # "seed_if_possible_enabled": [False, True],
    # "seeding_strategy": ["distance_self", "distance_center", "agent_count"],
    # "component_ordering": ["center", "percentage", "agents", "distance"],
    # "attachment_site_order": ["shortest_path", "prioritise", "shortest_travel_path", "agent_count"],
    # "attachment_site_ordering": ["shortest_path", "agent_count"]

    # for every map variation:
    # - create 1 folder
    # - for every agent type:
    #   - create 1 folder
    #   - for every tested combination: save that file there

    agent_counts = [1, 2, 4, 8, 16]
    offset = 100

    # create a folder
    if not os.path.exists(SAVE_DIRECTORY_NAME + map_name):
        os.makedirs(SAVE_DIRECTORY_NAME + map_name)

    parameters = []
    for agent_type in AGENT_TYPES:
        for agent_count in agent_counts:
            temp = {
                "target_map": map_name,
                "agent_count": agent_count,
                "agent_type": agent_type,
                "offset_stashes": offset
            }
            temp.update({k: VALUES[k][0] for k in extra_parameters(agent_type)})
            parameters.append(temp)
            for parameter in extra_parameters(agent_type):
                for other_value in VALUES[parameter][1:]:
                    temp = {
                        "target_map": map_name,
                        "agent_count": agent_count,
                        "agent_type": agent_type,
                        "offset_stashes": offset
                    }
                    temp.update({k: VALUES[k][0] for k in extra_parameters(agent_type) if k != parameter})
                    temp[parameter] = other_value
                    parameters.append(temp)

    runs_completed = 0
    for p in parameters:
        results = run_experiment(p)
        try:
            file_name = "{}_{}.json".format(map_name, uuid.uuid4())
            absolute_file_name = SAVE_DIRECTORY_NAME + map_name + "/" + file_name
            with open(absolute_file_name, "w") as file:
                json.dump(results, file)
            runs_completed += 1
            print("Successfully saved results for run with parameters:")
            pprint(p)
            print("RUNS COMPLETED: {}/{}\n\n".format(runs_completed, len(parameters)))
        except KeyboardInterrupt:
            print("Cancelled run with the following parameters:")
            pprint(p)
        except Exception as e:
            print("Error in run with the following parameters:")
            pprint(p)
            raise e

    print(len(parameters))
    pprint(parameters[0])

    # map_name = "block_5x5x10"
    # target_map = block_5x5x10
    # agent_type = LocalShortestPathAgent
    # agent_counts = [4, 6, 8, 12, 16]
    # for ac in agent_counts:
    #     total_results = {"agent_count": ac, "map_name": map_name}
    #     for restricted_count in range(2, ac + 1, 2):
    #         results = run_experiment(target_map, ac, agent_type, restricted_count)
    #         total_results[restricted_count] = {}
    #         total_results[restricted_count]["step_count"] = results["step_count"]
    #         total_results[restricted_count]["collisions"] = results["collisions"]
    #         total_results[restricted_count]["task_counts"] = results["task_counts"]
    #     try:
    #         file_name = "ar_{}_{}_agents.json".format(total_results["map_name"], total_results["agent_count"])
    #         absolute_file_name = SAVE_DIRECTORY_NAME + file_name
    #         with open(absolute_file_name, "w") as file:
    #             json.dump(total_results, file)
    #         print("Successfully saved results for run with {} agent(s).\n".format(ac))
    #     except KeyboardInterrupt:
    #         print("Cancelled run with {} agent(s).".format(ac))
    #     except Exception as e:
    #         print("Error in run with {} agent(s).".format(ac))
    #         raise e


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
