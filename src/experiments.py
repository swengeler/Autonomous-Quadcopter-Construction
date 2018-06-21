import json
import os
import seaborn as sns
import random
from argparse import ArgumentParser
from pprint import pprint

from agents import *
from env.block import *
from env.map import *
from env.util import *
from geom.shape import *

"""
Information about this file
"""

# the default directories to load maps (structures) from and save the results to
LOAD_DIRECTORY_NAME = "/home/simon/maps/"
SAVE_DIRECTORY_NAME = "/home/simon/new_results/"

# alternative directories for loading maps and saving results
LOAD_DIRECTORY_NAME_ALT_1 = "/home/simon/PycharmProjects/LowFidelitySimulation/res/experiment_maps/"
LOAD_DIRECTORY_NAME_ALT_2 = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_experiment_maps/"
SAVE_DIRECTORY_NAME_ALT = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_results/"

# some parameters that were not changed during the experiments (but could be in principle)
OFFSET_STRUCTURE = 400.0
OFFSET_ORIGIN = (OFFSET_STRUCTURE, OFFSET_STRUCTURE)

# a mapping from the string representations of agent types to the actual constructors
AGENT_TYPES = {
    "LocalPerimeterFollowingAgent": LocalPerimeterFollowingAgent,
    "LocalShortestPathAgent": LocalShortestPathAgent,
    "GlobalPerimeterFollowingAgent": GlobalPerimeterFollowingAgent,
    "GlobalShortestPathAgent": GlobalShortestPathAgent
}

# all adjustable parameters and their possible values (the first being the default one)
VALUES = {
    "waiting_on_perimeter_enabled": [False, True],
    "avoiding_crowded_stashes_enabled": [True, False],
    "transport_avoid_others_enabled": [True, False],
    "order_only_one_metric": [False, True],
    "seed_if_possible_enabled": [True, False],
    "seeding_strategy": ["distance_center", "distance_self", "agent_count"],
    "component_ordering": ["center", "distance", "percentage", "agents"],
    "attachment_site_order": ["shortest_path", "prioritise", "shortest_travel_path", "agent_count"],
    "attachment_site_ordering": ["shortest_path", "agent_count"]
}


def run_experiment(parameters):
    """
    Simulate the construction of a structure, using the specified parameter values, and return the results.

    Most importantly the parameters include the name of the structure to build, the agent type to use and
    the number of agents to use. In addition, other parameters relating to the behaviour of that agent type
    are specified (e.g. to choose a strategy to select attachment sites for the shortest path algorithm).
    The function gathers a large number of statistics about the experiment (e.g. the number of steps it took)
    and returns a dictionary containing these.

    :param parameters: a dictionary containing the values for all adjustable parameters
    :return: a dictionary containing statistics of the experiment
    """

    target_map = np.load(LOAD_DIRECTORY_NAME + parameters["target_map"] + ".npy").astype("int64")
    agent_count = parameters["agent_count"]
    agent_type = AGENT_TYPES[parameters["agent_type"]]
    offset_stashes = parameters["offset_stashes"]

    # remnants of the first version of the simulation
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
        block_list.append(Block())

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    # placing first seed in correct position and the other seeds elsewhere
    block_list[0].is_seed = True
    block_list[0].placed = True
    block_list[0].geometry = Geometry(list(environment.original_seed_position()), [Block.SIZE] * 3, 0.0)
    block_list[0].grid_position = environment.original_seed_grid_position()
    block_list[0].seed_marked_edge = "down"
    environment.place_block(block_list[0].grid_position, block_list[0])
    processed = [block_list[0]]

    for i in range(1, required_seeds + 1):
        block_list[i].is_seed = True
        block_list[i].color = Block.COLORS_SEEDS[0]
        block_list[i].geometry = Geometry([OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE / 2,
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
        if "order_only_one_metric" in parameters:
            agent_list[i].order_only_one_metric = parameters["order_only_one_metric"]
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

    # placing the agents using distance instead of collision stuff
    processed_counter = 0
    while len(processed) != block_count + agent_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = Geometry([candidate_x, candidate_y, agent_list[processed_counter].geometry.size[2] / 2],
                                 agent_list[processed_counter].geometry.size, 0.0)
        if all([simple_distance(p.geometry.position, (candidate_x, candidate_y))
                > agent_list[processed_counter].required_distance + 10 for p in processed]):
            if candidate_box.position[0] - candidate_box.size[0] > \
                    Block.SIZE * target_map.shape[2] + environment.offset_origin[0] \
                    or candidate_box.position[0] + candidate_box.size[0] < environment.offset_origin[0] \
                    or candidate_box.position[1] - candidate_box.size[1] > \
                    Block.SIZE * target_map.shape[1] + environment.offset_origin[1] \
                    or candidate_box.position[1] + candidate_box.size[1] < environment.offset_origin[1]:
                agent_list[processed_counter].geometry.set_to_match(candidate_box)
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
    layer_completion = dict([(l, None) for l in range(target_map.shape[0])])
    current_layer = 0
    started_components = []
    completed_components = []
    started_layers = []
    completed_layers = []
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
        if current_layer < target_map.shape[0] and current_layer not in started_layers \
                and environment.layer_started(current_layer):
            layer_completion[current_layer] = {"start": steps, "finish": steps}
        if current_layer < target_map.shape[0] and current_layer not in completed_layers \
                and current_layer in started_layers and environment.layer_finished(current_layer):
            layer_completion[current_layer]["finish"] = steps
            current_layer += 1

        if not structure_finished and current_layer >= target_map.shape[0]:
            structure_finished_count = steps
            structure_finished = True
            current_layer -= 1

        if steps % 5000 == 0:
            print("Simulation steps: {}".format(steps))

        if all([a.current_task == Task.FINISHED for a in agent_list]):
            finished_successfully = True
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

            # new stuff
            results["attached_block_counts"] = [a.attached_blocks for a in agent_list]
            results["seeded_block_counts"] = [a.seeded_blocks for a in agent_list]
            results["sp_number_search_count"] = [a.sp_search_count for a in agent_list]
            results["steps_per_layer"] = [a.steps_per_layer for a in agent_list]
            results["steps_per_component"] = [a.steps_per_component for a in agent_list]
            results["complete_to_switch_delay"] = [a.complete_to_switch_delay for a in agent_list]
            results["blocks_per_attachment"] = [a.blocks_per_attachment for a in agent_list]
            results["steps_per_attachment"] = [a.steps_per_attachment for a in agent_list]
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

        if (previous_map == current_map).all():
            no_change_counter += 1
        else:
            no_change_counter = 0

        if no_change_counter >= 5000 or steps > 1000000:
            got_stuck = True
            break

        previous_map = current_map

    if not finished_successfully:
        print("Interrupted construction with {} agents in {} steps ({} colliding)."
              .format(agent_count, steps, collisions / 2))
        if got_stuck:
            print("Interrupted because simulation got stuck.")
            print("State of the structure:")
            print_map(environment.occupancy_map)

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

        # new stuff
        results["attached_block_counts"] = [a.attached_blocks for a in agent_list]
        results["seeded_block_counts"] = [a.seeded_blocks for a in agent_list]
        results["sp_number_search_count"] = [a.sp_search_count for a in agent_list]
        results["steps_per_layer"] = [a.steps_per_layer for a in agent_list]
        results["steps_per_component"] = [a.steps_per_component for a in agent_list]
        results["complete_to_switch_delay"] = [a.complete_to_switch_delay for a in agent_list]
        results["blocks_per_attachment"] = [a.blocks_per_attachment for a in agent_list]
        results["steps_per_attachment"] = [a.steps_per_attachment for a in agent_list]

    return results


def extra_parameters(agent_type: str):
    """
    Return a list of names of adjustable parameters for the specified agent type.

    :param agent_type: the agent type for which to return parameters
    :return: a list of adjustable parameters names
    """

    parameters = ["waiting_on_perimeter_enabled",
                  "avoiding_crowded_stashes_enabled",
                  "transport_avoid_others_enabled",
                  "order_only_one_metric"]
    if agent_type.startswith("Local"):
        parameters.append("seed_if_possible_enabled")
        parameters.append("seeding_strategy")
    if agent_type.startswith("Global"):
        parameters.append("component_ordering")
    if agent_type == "LocalShortestPathAgent":
        parameters.append("attachment_site_order")
    if agent_type == "GlobalShortestPathAgent":
        parameters.append("attachment_site_ordering")
    return parameters


def short_form(agent_type: str):
    """
    Return the short name of the agent type specified by the argument.

    :param agent_type: long form of the agent type
    :return: corresponding short form of the agent type
    """

    if agent_type == "LocalShortestPathAgent":
        return "LSP"
    if agent_type == "LocalPerimeterFollowingAgent":
        return "LPF"
    if agent_type == "GlobalShortestPathAgent":
        return "GSP"
    if agent_type == "GlobalPerimeterFollowingAgent":
        return "GPF"
    return "NONE"


def long_form(agent_type_abbreviation: str):
    """
    Return the long name of the agent type specified by the argument.

    :param agent_type_abbreviation: short form of the agent type
    :return: corresponding long form of the agent type
    """

    if agent_type_abbreviation == "LSP":
        return "LocalShortestPathAgent"
    if agent_type_abbreviation == "LPF":
        return "LocalPerimeterFollowingAgent"
    if agent_type_abbreviation == "GSP":
        return "GlobalShortestPathAgent"
    if agent_type_abbreviation == "GPF":
        return "GlobalPerimeterFollowingAgent"
    return "NONE"


def parameters_defaults_for_all(map_name, number_runs, agent_counts, offset):
    """
    Return a list of parameter sets for the specified structure, number of runs and agent counts.

    :param map_name: the map/structure name to include in the parameters
    :param number_runs:  the number of runs to include/the number of times to repeat the same set of parameters
    :param agent_counts: a list of agent counts to create sets of parameters for
    :param offset: the offset between the construction area and block stashes (subject to change)
    :return: a list of parameter sets with default values
    """

    parameters = []
    for run in range(number_runs):
        for agent_count in agent_counts:
            for agent_type in AGENT_TYPES:
                temp = {
                    "target_map": map_name,
                    "agent_count": agent_count,
                    "agent_type": agent_type,
                    "offset_stashes": offset,
                    "experiment_name": "defaults",
                    "run": run
                }
                temp.update({k: VALUES[k][0] for k in extra_parameters(agent_type)})
                parameters.append(temp)
    return parameters


def main(map_name="block_4x4x4", skip_existing=False):
    """
    Simulate a number of number of experiments for a specified map/structure and save the results.

    :param map_name: the structure/map to construct
    :param skip_existing: either run experiments with results again and overwrite the files or skip them
    """

    experiment_options = ["all_defaults", "all_adjust_parameters", "one_adjust_parameters",
                          "local_or_global_adjust_parameters"]

    # fixed parameters, which could also become adjustable parameters (especially in the case of offset)
    agent_counts = [1, 2, 4, 8, 12, 16]
    offset = 100

    # create a folder for the given map if it does not exist
    if not os.path.exists(SAVE_DIRECTORY_NAME + map_name):
        os.makedirs(SAVE_DIRECTORY_NAME + map_name)

    # ask the user for the type of experiment they want to run
    print("Please specify the experiment you want to run: ")
    for c_idx, c in enumerate(experiment_options):
        print("[{}]: {}".format(c_idx, c))
    experiment_choice = input("Please specify a number to choose between parameters: ")
    if len(experiment_choice) == 0:
        experiment_choice = experiment_options.index("all_defaults")
    else:
        experiment_choice = int(experiment_choice)

    # ask the user for the number of agents for which to run the experiment for
    agent_count_list = input("Please specify a list of agent counts (or press enter to use the defaults {}): "
                             .format(agent_counts))
    if len(agent_count_list) != 0:
        agent_count_list = [int(x.strip()) for x in agent_count_list.split()]
        agent_counts = agent_count_list
        print("Running experiments for the following agent counts: {}".format(agent_counts))

    # ask the user for the number of independent runs to perform for each configuration
    number_runs = int(input("Please enter the number of repeated runs: "))

    # ask the user for a name for the experiment, or to choose one of the existing options
    if experiment_options[experiment_choice] == "all_defaults":
        experiment_name = "defaults"
    else:
        possible_experiment_names = []
        for map_directory_name in os.listdir(SAVE_DIRECTORY_NAME):
            if os.path.isdir(SAVE_DIRECTORY_NAME):
                for experiment_name in os.listdir(SAVE_DIRECTORY_NAME + map_directory_name):
                    if experiment_name.strip() not in possible_experiment_names:
                        possible_experiment_names.append(experiment_name.strip())
        possible_experiment_names = sorted(possible_experiment_names)

        print("You can choose one of the following experiment names or specify a different one: ")
        for c_idx, c in enumerate(possible_experiment_names):
            print("[{}]: {}".format(c_idx, c))
        experiment_name = input("Number of the experiment name or new name for the experiment: ")
        if len(experiment_name) == 0:
            experiment_name = possible_experiment_names[0]
        else:
            try:
                experiment_number = int(experiment_name)
                experiment_name = possible_experiment_names[experiment_number]
            except ValueError:
                pass
    print()

    # create a directory to save the specific experiment in if it does not exist already
    directory_name = SAVE_DIRECTORY_NAME + map_name + "/" + "{}".format(experiment_name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # compile a list of parameter configurations to use for running the experiments
    if experiment_options[experiment_choice] == "all_defaults":
        parameters = parameters_defaults_for_all(map_name, number_runs, agent_counts, offset)
    elif experiment_options[experiment_choice] == "all_adjust_parameters":
        extra = extra_parameters("all")
        extra_params = {}
        for e in extra:
            print("\nThe choices for parameter {} are:".format(e))
            for c_idx, c in enumerate(VALUES[e]):
                print("[{}]: {}".format(c_idx, c))
            choice = input("Please specify a number to choose between parameters: ")
            if len(choice) == 0:
                extra_params[e] = VALUES[e][0]
            else:
                choice = int(choice)
                extra_params[e] = VALUES[e][choice]

        parameters = []
        for run in range(number_runs):
            for agent_count in agent_counts:
                for agent_type in AGENT_TYPES:
                    temp = {
                        "target_map": map_name,
                        "agent_count": agent_count,
                        "agent_type": agent_type,
                        "offset_stashes": offset,
                        "experiment_name": experiment_name,
                        "run": run
                    }
                    temp.update(extra_params)
                    parameters.append(temp)
    elif experiment_options[experiment_choice] == "one_adjust_parameters":
        agent_type = long_form(input("Please enter the agent type (LSP, LPF, GSP, GPF): "))
        extra = extra_parameters(agent_type)
        extra_params = {}
        for e in extra:
            print("\nThe choices for parameter {} are:".format(e))
            for c_idx, c in enumerate(VALUES[e]):
                print("[{}]: {}".format(c_idx, c))
            choice = input("Please specify a number to choose between parameters: ")
            if len(choice) == 0:
                extra_params[e] = VALUES[e][0]
            else:
                choice = int(choice)
                extra_params[e] = VALUES[e][choice]

        parameters = []
        for run in range(number_runs):
            for agent_count in agent_counts:
                temp = {
                    "target_map": map_name,
                    "agent_count": agent_count,
                    "agent_type": agent_type,
                    "offset_stashes": offset,
                    "experiment_name": experiment_name,
                    "run": run
                }
                temp.update(extra_params)
                parameters.append(temp)
    elif experiment_options[experiment_choice] == "local_or_global_adjust_parameters":
        family = input("Please enter the type (L, G): ")
        agent_types = ["LocalShortestPathAgent", "LocalPerimeterFollowingAgent"] if "L" == family \
            else ["GlobalShortestPathAgent", "GlobalPerimeterFollowingAgent"]
        extra = extra_parameters("Local" if "L" == family else "Global")
        extra_params = {}
        for e in extra:
            print("\nThe choices for parameter {} are:".format(e))
            for c_idx, c in enumerate(VALUES[e]):
                print("[{}]: {}".format(c_idx, c))
            choice = input("Please specify a number to choose between parameters: ")
            if len(choice) == 0:
                extra_params[e] = VALUES[e][0]
            else:
                choice = int(choice)
                extra_params[e] = VALUES[e][choice]

        parameters = []
        for run in range(number_runs):
            for agent_count in agent_counts:
                for agent_type in agent_types:
                    temp = {
                        "target_map": map_name,
                        "agent_count": agent_count,
                        "agent_type": agent_type,
                        "offset_stashes": offset,
                        "experiment_name": experiment_name,
                        "run": run
                    }
                    temp.update(extra_params)
                    parameters.append(temp)
    else:
        parameters = []

    runs_completed = 0
    start_runs_at = 0
    for p in parameters[start_runs_at:]:
        absolute_file_name = "{}_{}_{}.json".format(p["agent_type"], p["agent_count"], p["run"])
        absolute_file_name = directory_name + "/" + absolute_file_name
        if not skip_existing or not os.path.exists(absolute_file_name):
            results = run_experiment(p)
            try:
                absolute_file_name = "{}_{}_{}.json".format(p["agent_type"], p["agent_count"], p["run"])
                absolute_file_name = directory_name + "/" + absolute_file_name
                with open(absolute_file_name, "w") as file:
                    json.dump(results, file)
                runs_completed += 1
                print("Successfully saved results for run {} with {} agents.".format(p["run"], p["agent_count"]))
                print("RUNS COMPLETED: {}/{} (out of total: {}/{})\n\n".format(
                    runs_completed, len(parameters) - start_runs_at, start_runs_at + runs_completed, len(parameters)))
            except KeyboardInterrupt:
                print("Cancelled run with the following parameters:")
                pprint(p)
                break
            except Exception as e:
                print("Error in run with the following parameters:")
                pprint(p)
                raise e


if __name__ == "__main__":
    parser = ArgumentParser(description="Run and save results of simulated construction with quadcopters.")
    parser.add_argument("map_name", type=str, help="Name of the map/structure to build.")
    parser.add_argument("-l", "--load-directory", dest="load_directory", default=0,
                        help="Either a number (0-2) specifying one of three pre-defined "
                             "paths to load maps from or an absolute file path.")
    parser.add_argument("-s", "--save-directory", dest="save_directory", default=0,
                        help="Either a number (0, 1) specifying one of two pre-defined "
                             "paths to save results to or an absolute file path.")
    parser.add_argument("--skip-existing", dest="skip_existing", action='store_true',
                        help="Specifies whether existing files should be overwritten if "
                             "experiments are repeated. The default is to overwrite them.")

    args = parser.parse_args()
    map_name_outer = args.map_name
    try:
        load_directory = int(args.load_directory)
        if load_directory == 1:
            LOAD_DIRECTORY_NAME = LOAD_DIRECTORY_NAME_ALT_1
        elif load_directory == 2:
            LOAD_DIRECTORY_NAME = LOAD_DIRECTORY_NAME_ALT_2
    except ValueError:
        LOAD_DIRECTORY_NAME = args.load_directory
    try:
        save_directory = int(args.save_directory)
        if save_directory == 1:
            SAVE_DIRECTORY_NAME = SAVE_DIRECTORY_NAME_ALT
    except ValueError:
        SAVE_DIRECTORY_NAME = args.save_directory
    skip_existing_outer = args.skip_existing

    main(map_name_outer, skip_existing_outer)

