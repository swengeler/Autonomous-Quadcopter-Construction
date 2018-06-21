import queue
import time
import random
import seaborn as sns

from agents import *
from env import *
from geom import *
from graphics.graphics_2d import Graphics2D
from experiments import AGENT_TYPES, long_form


"""
Running this file opens up a simple GUI which can be used to run and monitor single runs of simulated construction.
It should be noted that the GUI is very unwieldy, but it gets done what it needs to.
"""


request_queue = queue.Queue()
return_queue = queue.Queue()
done = False
running = False


def run_simulation(map_name, agent_count, agent_type):
    """
    Simulate and display the construction of a specified structure using
    the specified number of agents of the specified type.

    :param map_name: path to the file containing the occupancy matrix for the target structure
    :param agent_count: the number of agents to use
    :param agent_type: the agent type to use
    """

    global running, done
    running = True

    # setting global parameters
    interval = 0.05
    paused = False

    # creating the target map
    if map_name is None:
        target_map = np.array([[[1, 1, 1], [1, 2, 1], [1, 1, 1]]])
    else:
        target_map = np.load(map_name).astype("int64")

    # changing the block colours for easy differentiation when they are placed at a certain height
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
    offset_structure = 300
    offset_stashes = 100
    offset_origin = (offset_structure, offset_structure)
    environment_extent = [offset_structure * 2 + target_map.shape[2] * Block.SIZE,
                          offset_structure * 2 + target_map.shape[1] * Block.SIZE,
                          offset_structure * 2 + target_map.shape[1] * Block.SIZE]

    # creating Map object and getting the required number of blocks
    environment = Map(target_map, offset_origin, environment_extent)
    block_count = environment.required_blocks()

    # finding out how many components there are
    dummy_agent = LocalPerimeterFollowingAgent([0, 0, 0], [0, 0, 0], target_map)
    component_target_map = dummy_agent.split_into_components()
    required_seeds = np.max(component_target_map) - 2
    Agent.AGENT_ID -= 1

    # creating the blocks and a list of initial positions
    block_list = []
    for _ in range(0, block_count):
        block_list.append(Block())

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    # seed(s) and block positions
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
        block_list[i].geometry = Geometry([offset_origin[0] + target_map.shape[2] * Block.SIZE / 2,
                                           offset_origin[1] + target_map.shape[1] * Block.SIZE + offset_stashes,
                                           Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        processed.append(block_list[i])

    chunk_list = split_into_chunks(block_list[(1 + required_seeds):], 4)
    for sl_idx, sl in enumerate(chunk_list):
        for b in sl:
            if sl_idx == 0:
                b.geometry.position = [offset_structure - offset_stashes, offset_structure - offset_stashes, Block.SIZE / 2]
            elif sl_idx == 1:
                b.geometry.position = [offset_origin[0] + target_map.shape[2] * Block.SIZE + offset_stashes, offset_structure - offset_stashes, Block.SIZE / 2]
            elif sl_idx == 2:
                b.geometry.position = [offset_structure - offset_stashes, offset_origin[1] + target_map.shape[1] * Block.SIZE + offset_stashes, Block.SIZE / 2]
            elif sl_idx == 3:
                b.geometry.position = [offset_origin[0] + target_map.shape[2] * Block.SIZE + offset_stashes,
                                       offset_origin[1] + target_map.shape[1] * Block.SIZE + offset_stashes, Block.SIZE / 2]
            processed.append(b)

    # creating the agents and placing
    agent_type = AGENT_TYPES[agent_type]
    agent_list = [agent_type([50, 60, 7.5], [40, 40, 15], target_map, 10.0) for _ in range(0, agent_count)]
    for a in agent_list:
        a.waiting_on_perimeter_enabled = True

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

    # adding the blocks
    environment.add_blocks(block_list)

    # adding the agents
    environment.add_agents(agent_list)

    request_queue.put((Graphics2D.START_REQUEST, environment))

    # stuck stuff
    no_change_counter = 0
    max_no_change_counter = 0
    previous_map = np.copy(environment.occupancy_map)
    np.place(previous_map, previous_map > 1, 1)

    # running the main loop of the simulation
    steps = 0
    collisions = 0
    finished_successfully = False
    structure_complete = False
    collision_pairs = []

    try:
        while True:
            if not paused:
                for a in agent_list:
                    a.advance(environment)
                steps += 1

                if steps % 1000 == 0:
                    print("Simulation steps: {}".format(steps))

                if all([a.current_task == Task.FINISHED for a in agent_list]):
                    print("Finished construction with {} agents in {} steps ({} colliding)."
                          .format(agent_count, steps, collisions / 2))
                    finished_successfully = True
                    print("\nFinal resulting map:")
                    print_map(environment.occupancy_map)
                    print("\nCollision data:\n{}".format(collision_pairs))
                    break
                elif not structure_complete and agent_list[0].check_structure_finished(environment.occupancy_map):
                    print("Structure complete with {} agent(s) after {} steps.".format(agent_count, steps))
                    structure_complete = True

                if len(agent_list) > 1:
                    for a1 in agent_list:
                        for a2 in agent_list:
                            if a1 is not a2 and a1.overlaps(a2) \
                                    and not a1.current_task == Task.FINISHED and not a2.current_task == Task.FINISHED:
                                collisions += 1
                                print("Agent {} ({}) and {} ({}) colliding.".format(a1.id, a1.current_task,
                                                                                    a2.id, a2.current_task))
                                collision_pairs.append((a1.current_task, a2.current_task))

                current_map = np.copy(environment.occupancy_map)
                np.place(current_map, current_map > 1, 1)
                if (previous_map == current_map).all():
                    no_change_counter += 1
                else:
                    no_change_counter = 0
                if no_change_counter > max_no_change_counter:
                    max_no_change_counter = no_change_counter
                previous_map = current_map

            request_queue.put(Graphics2D.UPDATE_REQUEST)
            try:
                val = return_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if val == Graphics2D.SHUTDOWN_RETURN:
                    raise KeyboardInterrupt
                elif val == Graphics2D.STOP_RETURN:
                    return
                elif val == Graphics2D.PAUSE_PLAY_RETURN:
                    paused = not paused
                elif val == Graphics2D.INTERVAL_RETURN:
                    try:
                        new_interval = return_queue.get_nowait()
                        if new_interval < 0.00001:
                            interval = 0.00001
                        else:
                            interval = new_interval
                    except queue.Empty:
                        pass
                elif val == Graphics2D.START_RETURN:
                    break
            time.sleep(interval)
    except KeyboardInterrupt:
        request_queue.put(Graphics2D.SHUTDOWN_REQUEST)
        done = True
        if not finished_successfully:
            print("Interrupted construction with {} agents in {} steps ({} colliding)."
                  .format(agent_count, steps, collisions / 2))
            print("\nFinal resulting map:")
            print_map(environment.occupancy_map)
            print("\nCollision data:\n{}".format(collision_pairs))


def main():
    """
    Start the simulator GUI and handle input events.
    """

    graphics = Graphics2D(request_queue, return_queue)
    graphics.run()

    map_name = None
    agent_count = 1
    agent_type = "LSP"

    global done
    while not done:
        # listen for events from graphics and if there is one
        try:
            val = return_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            message = None
            if isinstance(val, tuple):
                temp = val
                val = temp[0]
                message = temp[1]
            if val == Graphics2D.AGENT_COUNT_RETURN:
                agent_count = message
            elif val == Graphics2D.AGENT_TYPE_RETURN:
                agent_type = message
            elif val == Graphics2D.MAP_RETURN:
                map_name = message
            elif val == Graphics2D.START_RETURN:
                run_simulation(map_name, agent_count, long_form(agent_type))
            elif val == Graphics2D.SHUTDOWN_RETURN:
                request_queue.put(Graphics2D.SHUTDOWN_REQUEST)
                break


if __name__ == "__main__":
    main()
