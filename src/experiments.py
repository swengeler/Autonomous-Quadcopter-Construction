import time
import random
import json
from emergency_structures import emergency_structures
from agents.agent import Task
from agents.local_knowledge.ps_agent import PerimeterFollowingAgentLocal
from agents.local_knowledge.sp_agent import ShortestPathAgentLocal
from env.map import *
from env.util import *
from geom.shape import *
from structures import *


INTERVAL = 0.00001
OFFSET_ORIGIN = (100.0, 100.0)


def scale_map(target_map: np.ndarray, scale_factor=1, axes=(0, 1, 2), scale_factors=None):
    if scale_factors is None:
        z_scale_factor = scale_factor if 0 in axes else 1
        y_scale_factor = scale_factor if 1 in axes else 1
        x_scale_factor = scale_factor if 2 in axes else 1
    else:
        z_scale_factor = scale_factors[0]
        y_scale_factor = scale_factors[1]
        x_scale_factor = scale_factors[2]

    scaled_map = np.zeros((target_map.shape[0] * z_scale_factor,
                           target_map.shape[1] * y_scale_factor,
                           target_map.shape[2] * x_scale_factor), dtype="int64")

    for z in range(target_map.shape[0]):
        for y in range(target_map.shape[1]):
            for x in range(target_map.shape[2]):
                if target_map[z, y, x] != 0:
                    z_start = z * z_scale_factor
                    z_end = (z + 1) * z_scale_factor
                    y_start = y * y_scale_factor
                    y_end = (y + 1) * y_scale_factor
                    x_start = x * x_scale_factor
                    x_end = (x + 1) * x_scale_factor
                    scaled_map[z_start:z_end, y_start:y_end, x_start:x_end] = 1
                    if target_map[z, y, x] == 2:
                        scaled_map[z_start, y_start, x_start] = 2

    return scaled_map


def run_experiment(target_map, agent_count, agent_type):
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
    environment_extent = [2 * OFFSET_ORIGIN[0] + max(target_map.shape[1], target_map.shape[0]) * Block.SIZE] * 3

    # creating the environment
    environment = Map(target_map, OFFSET_ORIGIN, environment_extent)
    block_count = environment.required_blocks()

    # finding out how many components there are
    dummy_agent = PerimeterFollowingAgent([0, 0, 0], [0, 0, 0], target_map)
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
    processed = [block_list[0]]

    for i in range(1, required_seeds + 1):
        block_list[i].is_seed = True
        block_list[i].color = Block.COLORS_SEEDS[0]
        block_list[i].geometry = GeomBox([OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE / 2,
                                          OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + 100,
                                          Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        processed.append(block_list[i])

    # placing the simple construction blocks
    chunk_list = split_into_chunks(block_list[(1 + required_seeds):], 4)
    for sl_idx, sl in enumerate(chunk_list):
        for b in sl:
            if sl_idx == 0:
                b.geometry.position[2] = Block.SIZE / 2
            elif sl_idx == 1:
                b.geometry.position = [OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE + 100, 0.0,
                                       Block.SIZE / 2]
            elif sl_idx == 2:
                b.geometry.position = [0.0, OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + 100,
                                       Block.SIZE / 2]
            elif sl_idx == 3:
                b.geometry.position = [OFFSET_ORIGIN[0] + target_map.shape[2] * Block.SIZE + 100,
                                       OFFSET_ORIGIN[1] + target_map.shape[1] * Block.SIZE + 100, Block.SIZE / 2]
            processed.append(b)

    # creating the agent_list
    agent_list = [agent_type([50, 60, 7.5], [40, 40, 15], target_map, 10.0) for _ in range(0, agent_count)]
    for i in range(len(agent_list)):
        agent_list[i].id = i

    # placing the agents
    processed_counter = 0
    while len(processed) != block_count + agent_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = GeomBox([candidate_x, candidate_y, agent_list[processed_counter].geometry.size[2] / 2],
                                agent_list[processed_counter].geometry.size, 0.0)
        if all([not candidate_box.overlaps(p.geometry) for p in processed]):
            agent_list[processed_counter].geometry.set_to_match(candidate_box)
            processed.append(agent_list[processed_counter])
            processed_counter += 1

    # adding the block_list
    environment.add_blocks(block_list)

    # adding the agent list
    environment.add_agents(agent_list)

    # running the main loop of the simulation
    steps = 0
    collisions = 0
    finished_successfully = False
    results = {}
    try:
        while True:
            for a in agent_list:
                a.advance(environment)
            steps += 1
            if all([a.current_task == Task.FINISHED for a in agent_list]):
                print("Finished construction in {} steps ({} colliding).".format(steps, collisions / 2))
                average_stats = {}
                min_stats = {}
                max_stats = {}
                for k in list(agent_list[0].agent_statistics.task_counter.keys()):
                    task_counters = [a.agent_statistics.task_counter[k] for a in agent_list]
                    average_stats[k.name] = float(np.mean(task_counters))
                    min_stats[k.name] = int(np.min(task_counters))
                    max_stats[k.name] = int(np.max(task_counters))
                # print("Average statistics for agents:")
                # for k in list(average_stats.keys()):
                #     print("{}: {}".format(k, average_stats[k]))
                finished_successfully = True
                results["step_count"] = steps
                results["collisions"] = collisions
                results["average"] = average_stats
                results["min"] = min_stats
                results["max"] = max_stats
                results["finished_successfully"] = finished_successfully
                raise KeyboardInterrupt
            if len(agent_list) > 1:
                for a1 in agent_list:
                    for a2 in agent_list:
                        if a1 is not a2 and a1.overlaps(a2) \
                                and not a1.current_task == Task.FINISHED and not a2.current_task == Task.FINISHED:
                            collisions += 1
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        if not finished_successfully:
            logger.info("Simulation interrupted.")
            print("Interrupted construction in {} steps ({} colliding).".format(steps, collisions / 2))
            average_stats = {}
            min_stats = {}
            max_stats = {}
            for k in list(agent_list[0].agent_statistics.task_counter.keys()):
                task_counters = [a.agent_statistics.task_counter[k] for a in agent_list]
                average_stats[k.name] = float(np.mean(task_counters))
                min_stats[k.name] = int(np.min(task_counters))
                max_stats[k.name] = int(np.max(task_counters))
            # print("Average statistics for agents:")
            # for k in list(average_stats.keys()):
            #     print("{}: {}".format(k, average_stats[k]))
            results["step_count"] = steps
            results["collisions"] = collisions
            results["average"] = average_stats
            results["min"] = min_stats
            results["max"] = max_stats
            results["finished_successfully"] = finished_successfully
        else:
            logger.info("Simulation finished successfully.")

    return results


def main():

    # INFO NEEDED
    # Structures:
    # - occupancy matrices
    # - list of scales for each structure
    #
    # Agents:
    # - list of agent types (thus the following is for each type)
    # - collision avoidance on or off
    # - local information vs global information
    # - seed localisation vs unique block localisation
    # - list of number of agents to run everything with

    target_maps = emergency_structures
    agent_counts = [1, 2, 4, 8]
    scales = [1]
    agent_types = [PerimeterFollowingAgentLocal, ShortestPathAgentLocal]

    for agent_type in agent_types:
        index = 0
        for key in target_maps:
            if key == "tower_more_stilts_10x10x10" or key == "block_10x10x10" or key == "tower_bigger_stilts_10x10x10":
                continue
            for scale in scales:
                scaled_map = scale_map(target_maps[key], scale)
                for agent_count in agent_counts:
                    meta_data = {
                        "agent_type": agent_type.__name__,
                        "target_map": key,
                        "scale": scale,
                        "agent_count": agent_count
                    }
                    try:
                        results = run_experiment(scaled_map, agent_count, agent_type)
                        results["meta_data"] = meta_data
                        file_name = "{}_{}_{}_{}.json".format(meta_data["agent_type"],
                                                              meta_data["target_map"],
                                                              meta_data["scale"],
                                                              meta_data["agent_count"])
                        absolute_file_name = \
                            "/home/simon/PycharmProjects/LowFidelitySimulation/res/experiments/" + file_name
                        with open(absolute_file_name, "w") as file:
                            json.dump(results, file)
                    except KeyboardInterrupt:
                        print("Cancelled experiment run:\n{}".format(meta_data))
                    except Exception as e:
                        print("Experiment run with error:\n{}".format(meta_data))
                        print("Error: {}".format(e))
            index += 1


def emergency():
    pass


if __name__ == "__main__":
    random.seed(101)
    main()
