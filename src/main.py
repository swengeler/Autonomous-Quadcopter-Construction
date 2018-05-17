import time
import random
import queue
from agents.agent import Task
from agents.local_knowledge.ps_agent import PerimeterFollowingAgentLocal
from agents.local_knowledge.sp_agent import ShortestPathAgentLocal
from env.map import *
from env.util import *
from geom.shape import *
from graphics.graphics_2d import Graphics2D
from structures import *
from emergency_structures import emergency_structures

random.seed(202)

request_queue = queue.Queue()
return_queue = queue.Queue()


def main():
    # setting up logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s():%(lineno)d - %(message)s")
    logger = logging.getLogger(__name__)

    # setting global parameters
    interval = 0.1
    paused = False

    # creating the target map
    # could use "requirements" class or dictionary to specify that a certain block should be in a certain place
    # 1: occupied
    # 2: seed

    target_map = emergency_structures["tower_bigger_stilts_10x10x10"]
    target_map = emergency_structures["loop_structure_10x10x10"]

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

    # offset of the described target occupancy map to the origin (only in x/y directions)
    offset_origin = (100.0, 100.0)
    environment_extent = [150.0, 200.0, 200.0]
    environment_extent = [350.0] * 3

    # creating Map object and getting the required number of block_list (of each type)
    environment = Map(target_map, offset_origin, environment_extent)
    block_count = environment.required_blocks()

    # finding out how many components there are
    dummy_agent = PerimeterFollowingAgentLocal([0, 0, 0], [0, 0, 0], target_map)
    component_target_map = dummy_agent.split_into_components()
    required_seeds = np.max(component_target_map) - 2

    # creating the block_list and a list of initial positions
    block_list = []
    for _ in range(0, block_count):
        block_list.append(create_block(BlockType.INERT))

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    # seed(s) and block positions
    block_list[0].is_seed = True
    block_list[0].placed = True
    block_list[0].geometry = GeomBox(list(environment.original_seed_position()), [Block.SIZE] * 3, 0.0)
    block_list[0].grid_position = environment.original_seed_grid_position()
    block_list[0].seed_marked_edge = "down"
    processed = [block_list[0]]

    for i in range(1, required_seeds + 1):
        block_list[i].is_seed = True
        block_list[i].color = Block.COLORS_SEEDS[0]
        block_list[i].geometry = GeomBox([offset_origin[0] + target_map.shape[2] * Block.SIZE / 2,
                                          offset_origin[1] + target_map.shape[1] * Block.SIZE + 100,
                                          Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        processed.append(block_list[i])

    # processed.extend(block_list)
    chunk_list = split_into_chunks(block_list[(1 + required_seeds):], 4)
    for sl_idx, sl in enumerate(chunk_list):
        for b in sl:
            if sl_idx == 0:
                b.geometry.position[2] = Block.SIZE / 2
            elif sl_idx == 1:
                b.geometry.position = [offset_origin[0] + target_map.shape[2] * Block.SIZE + 100, 0.0, Block.SIZE / 2]
            elif sl_idx == 2:
                b.geometry.position = [0.0, offset_origin[1] + target_map.shape[1] * Block.SIZE + 100, Block.SIZE / 2]
            elif sl_idx == 3:
                b.geometry.position = [offset_origin[0] + target_map.shape[2] * Block.SIZE + 100,
                                       offset_origin[1] + target_map.shape[1] * Block.SIZE + 100, Block.SIZE / 2]
            processed.append(b)

    # creating the agent_list
    agent_count = 8
    agent_type = ShortestPathAgentLocal
    agent_list = [agent_type([50, 60, 7.5], [40, 40, 15], target_map, 10.0) for _ in range(0, agent_count)]
    for i in range(len(agent_list)):
        agent_list[i].id = i

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

    # adding the block_list (whose positions are randomly initialised inside the Map object)
    environment.add_blocks(block_list)

    # print("BLOCK POSITIONS: {}".format(environment.block_stashes))
    # print("SEED POSITIONS: {}".format(environment.seed_stashes))

    # adding the agent list
    environment.add_agents(agent_list)

    # starting the tkinter GUI
    # threading.Thread(target=tk_main_loop, args=(environment.environment_extent[0])).start()
    graphics = Graphics2D(environment, request_queue, return_queue, ["top", "front"], interval * 1000, render=True)
    graphics.run()

    # running the main loop of the simulation
    steps = 0
    collisions = 0
    finished_successfully = False
    results = {}
    try:
        while True:
            if not paused:
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
                    print("\nAverage statistics for agents:")
                    for k in list(average_stats.keys()):
                        print("{}: {}".format(k, average_stats[k]))
                    print("\nMin statistics for agents:")
                    for k in list(min_stats.keys()):
                        print("{}: {}".format(k, min_stats[k]))
                    print("\nMax statistics for agents:")
                    for k in list(max_stats.keys()):
                        print("{}: {}".format(k, max_stats[k]))
                    finished_successfully = True
                    results["step_count"] = steps
                    results["collisions"] = collisions
                    results["average"] = average_stats
                    results["min"] = min_stats
                    results["max"] = max_stats
                    results["finished_successfully"] = finished_successfully
                    break
                if len(agent_list) > 1:
                    for a1 in agent_list:
                        for a2 in agent_list:
                            if a1 is not a2 and a1.overlaps(a2) \
                                    and not a1.current_task == Task.FINISHED and not a2.current_task == Task.FINISHED:
                                collisions += 1
                                print("Agent {} and {} colliding.".format(a1.id, a2.id))
                                # paused = True
                # print("CURRENT MAP:")
                # print_map(environment.occupancy_map)
            # submit_to_tkinter(update_window, environment)
            request_queue.put(Graphics2D.UPDATE_REQUEST)
            try:
                val = return_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if val == Graphics2D.SHUTDOWN_RETURN:
                    raise KeyboardInterrupt
                elif val == Graphics2D.PAUSE_PLAY_RETURN:
                    paused = not paused
                elif val == Graphics2D.INTERVAL_RETURN:
                    try:
                        new_interval = return_queue.get_nowait()
                        interval = new_interval
                    except queue.Empty:
                        pass
            time.sleep(interval)
    except KeyboardInterrupt:
        # submit_to_tkinter(stop_tk_thread)
        request_queue.put(Graphics2D.SHUTDOWN_REQUEST)
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
            print("\nAverage statistics for agents:")
            for k in list(average_stats.keys()):
                print("{}: {}".format(k, average_stats[k]))
            print("\nMin statistics for agents:")
            for k in list(min_stats.keys()):
                print("{}: {}".format(k, min_stats[k]))
            print("\nMax statistics for agents:")
            for k in list(max_stats.keys()):
                print("{}: {}".format(k, max_stats[k]))
            results["step_count"] = steps
            results["collisions"] = collisions
            results["average"] = average_stats
            results["min"] = min_stats
            results["max"] = max_stats
            results["finished_successfully"] = finished_successfully
        else:
            logger.info("Simulation finished successfully.")


if __name__ == "__main__":
    main()
