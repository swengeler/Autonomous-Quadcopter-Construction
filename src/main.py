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

random.seed(202)

request_queue = queue.Queue()
return_queue = queue.Queue()
'''
master = None
canvas_top = None


def tk_main_loop(env: Map):
    global master, canvas_top

    def timer_tick():
        global master, canvas_top
        try:
            callback, args, kwargs = request_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            return_val = callback(*args, **kwargs)
            result_queue.put(return_val)

        master.after(100, timer_tick)

    def on_closing():
        global done, master
        done = True
        master.destroy()

    master = Tk()
    canvas_top = Canvas(master, width=window_width, height=window_height)
    canvas_top.pack(side="left")
    # canvas_side = Canvas(master, width=window_width, height=window_height)
    # canvas_side.pack(side="right")
    # print(list(master.children.values()))
    # list(master.children.values())[1].create_rectangle(50, 50, 100, 100, fill="blue")
    timer_tick()
    master.protocol("WM_DELETE_WINDOW", on_closing)

    other_top_level = Toplevel(master)
    other_canvas = Canvas(other_top_level, width=window_width, height=window_height)
    other_canvas.pack()

    master.mainloop()


def update_window(env: Map):
    global canvas_top
    canvas_top.delete("all")
    env.draw_grid(canvas_top)
    env.draw_blocks(canvas_top)
    env.draw_agents(canvas_top)


def stop_tk_thread():
    global master
    master.destroy()


def submit_to_tkinter(callback, *args, **kwargs):
    request_queue.put((callback, args, kwargs))
    return result_queue.get()
'''


def main():
    # setting up logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s():%(lineno)d - %(message)s")
    logger = logging.getLogger(__name__)

    # setting global parameters
    interval = 0.2
    paused = False

    # creating the target map
    # could use "requirements" class or dictionary to specify that a certain block should be in a certain place
    # 1: occupied
    # 2: seed
    target_map = np.array([[[0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1],
                            [2, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0]]])

    a_big_h = np.array([[[2, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1]],
                        [[1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1]],
                        [[1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1]],
                        [[1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1, 1]]])

    another_big_h = np.array([[[1, 1, 0, 0, 1, 1],
                               [1, 1, 0, 0, 1, 1],
                               [1, 1, 0, 0, 1, 1],
                               [1, 1, 2, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1],
                               [1, 1, 0, 0, 1, 1],
                               [1, 1, 0, 0, 1, 1],
                               [1, 1, 0, 0, 1, 1]],
                              [[0, 1, 0, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0]]])

    a_big_2 = np.array([[[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])

    other_target_map = np.array([[[2, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]])

    test = np.array([[[1, 1, 1],
                      [1, 2, 1],
                      [1, 1, 1]]])

    test_2 = np.array([[[0, 2, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]])

    test_3 = np.array([[[2, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]]])

    test_3d = np.array([[[1, 1, 1],
                         [1, 2, 1],
                         [1, 1, 1]],
                        [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]])

    test_4 = np.array([[[0, 0, 0, 1, 2, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 0, 0, 0, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0]]])

    test_disjointed = np.array([[[2, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]],
                                [[1, 0, 1],
                                 [0, 0, 0],
                                 [1, 0, 1]]])

    pyramid = np.array([[[2, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1]],
                        [[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]]])

    tower = np.array([[[0, 0, 0, 0],
                       [0, 1, 2, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]],

                      [[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]],

                      [[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]],

                      [[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]],

                      [[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]]])

    other_target_map = tower

    # offset of the described target occupancy map to the origin (only in x/y directions)
    offset_origin = (60.0, 100.0)
    environment_extent = [150.0, 200.0, 200.0]
    environment_extent = [300.0, 300.0, 300.0]

    # creating Map object and getting the required number of block_list (of each type)
    environment = Map(other_target_map, offset_origin, environment_extent)
    block_count = environment.required_blocks()

    # creating the block_list and a list of initial positions
    block_list = []
    for _ in range(0, block_count):
        block_list.append(create_block(BlockType.INERT))

    # block positions, (for now) only x and y required
    block_list[0].is_seed = True
    block_list[0].placed = True
    block_list[0].color = Block.COLORS["seed"]
    block_list[0].geometry = GeomBox(list(environment.seed_position()), [Block.SIZE] * 3, 0.0)
    block_list[0].grid_position = environment.seed_grid_position()
    block_list[0].seed_marked_edge = "down"
    processed = [block_list[0]]
    processed_counter = 1
    while len(processed) != block_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = GeomBox([candidate_x, candidate_y, Block.SIZE / 2], [Block.SIZE] * 3, 0.0)
        if all([not candidate_box.overlaps(p.geometry) for p in processed]):
            block_list[processed_counter].geometry = candidate_box
            processed.append(block_list[processed_counter])
            processed_counter += 1

    # creating the agent_list
    agent_count = 1
    agent_type = AgentType.RANDOM_WALK_AGENT
    agent_list = [create_agent(agent_type, [50, 60, 7.5], [40, 40, 15], other_target_map)
                  for _ in range(0, agent_count)]

    processed_counter = 0
    while len(processed) != block_count + agent_count:
        candidate_x = random.uniform(0.0, environment.environment_extent[0])
        candidate_y = random.uniform(0.0, environment.environment_extent[1])
        candidate_box = GeomBox([candidate_x, candidate_y, agent_list[processed_counter].geometry.size[2] / 2],
                                agent_list[processed_counter].geometry.size, 0.0)
        if all([not candidate_box.overlaps(p.geometry) for p in processed]):
            agent_list[processed_counter].geometry = candidate_box
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
    try:
        while True:
            if not paused:
                for a in agent_list:
                    a.advance(environment)
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
