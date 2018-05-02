import numpy as np
import time
import logging
import random
import queue
from agents.util import *
from env.map import *
from env.util import *
from geom.shape import *
from graphics.graphics_2d import Graphics2D
from structures import *

random.seed(202)

request_queue = queue.Queue()
return_queue = queue.Queue()


def main():
    # setting up logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s():%(lineno)d - %(message)s")
    logger = logging.getLogger(__name__)

    # setting global parameters
    interval = 0.3
    paused = False

    # creating the target map
    # could use "requirements" class or dictionary to specify that a certain block should be in a certain place
    # 1: occupied
    # 2: seed

    target_map = tower_solid_5x5

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
    environment_extent = [700.0] * 3

    # creating Map object and getting the required number of block_list (of each type)
    environment = Map(target_map, offset_origin, environment_extent)
    block_count = environment.required_blocks()

    # creating the block_list and a list of initial positions
    block_list = []
    for _ in range(0, block_count):
        block_list.append(create_block(BlockType.INERT))

    # block positions, (for now) only x and y required
    block_list[0].is_seed = True
    block_list[0].placed = True
    # block_list[0].color = Block.COLORS_SEEDS[0]
    block_list[0].geometry = GeomBox(list(environment.seed_position()), [Block.SIZE] * 3, 0.0)
    block_list[0].grid_position = environment.seed_grid_position()
    block_list[0].seed_marked_edge = "down"
    processed = [block_list[0]]
    processed_counter = 1
    while len(processed) != block_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = GeomBox([candidate_x, candidate_y, Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        if not ((environment.offset_origin[0] <= candidate_x <= environment.offset_origin[0]
                 + Block.SIZE * environment.target_map.shape[2]) or
                (environment.offset_origin[1] <= candidate_y <= environment.offset_origin[1]
                 + Block.SIZE * environment.target_map.shape[1])) \
                and all([not candidate_box.overlaps(p.geometry) for p in processed]):
            block_list[processed_counter].geometry.set_to_match(candidate_box)
            processed.append(block_list[processed_counter])
            processed_counter += 1

    # creating the agent_list
    agent_count = 6
    agent_type = AgentType.RANDOM_WALK_AGENT
    agent_list = [create_agent(agent_type, [50, 60, 7.5], [40, 40, 15], target_map, 10.0)
                  for _ in range(0, agent_count)]
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
    # block_1 = create_block(BlockType.INERT, color="green", position=[20, 20, 22.5], rotation=np.pi/4)
    # block_2 = create_block(BlockType.INERT, color="yellow", position=[20, 42, 7.5], rotation=np.pi/4)
    # environment.add_blocks([block_1, block_2])
    #
    # print(block_1.overlaps(block_2))

    # adding the agent list
    environment.add_agents(agent_list)

    # starting the tkinter GUI
    # threading.Thread(target=tk_main_loop, args=(environment.environment_extent[0])).start()
    graphics = Graphics2D(environment, request_queue, return_queue, ["top", "front"], interval * 1000)
    graphics.run()

    # running the main loop of the simulation
    steps = 0
    collisions = 0
    try:
        while True:
            if not paused:
                for a in agent_list:
                    a.advance(environment)
                steps += 1
                if all([a.current_task == Task.FINISHED for a in agent_list]):
                    print("Finished construction in {} steps ({} colliding).".format(steps, collisions / 2))
                    print("Statistics for agent 0:")
                    for k in list(agent_list[0].agent_statistics.task_counter.keys()):
                        print("{}: {}".format(k, agent_list[0].agent_statistics.task_counter[k]))
                    raise KeyboardInterrupt
                if len(agent_list) > 1:
                    for a1 in agent_list:
                        for a2 in agent_list:
                            if a1 is not a2 and a1.overlaps(a2):
                                collisions += 1
                environment.update()
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
        logger.info("Simulation interrupted.")


if __name__ == "__main__":
    main()
