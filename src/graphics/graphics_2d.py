import numpy as np
import queue
import threading
from typing import List, Tuple
try:
    from Tkinter import *
except ImportError:
    from tkinter import *
from env.map import Map
from env.block import Block


class Graphics2D:
    UPDATE_REQUEST = 0
    SHUTDOWN_REQUEST = 1

    SHUTDOWN_RETURN = 101
    PAUSE_PLAY_RETURN = 102
    INTERVAL_RETURN = 103

    def __init__(self,
                 map: Map,
                 request_queue: queue.Queue,
                 return_queue: queue.Queue,
                 views: List[str] = None,
                 min_update_interval=200,
                 padding: Tuple[float, float] = (20, 20),
                 scale=1,
                 render: bool = True):
        self.map = map
        self.update_queue = request_queue
        self.return_queue = return_queue
        self.views = views if views is not None else ["top", "front"]
        self.update_interval = int(min_update_interval / 2)
        self.padding = padding
        self.scale = scale
        self.render = render
        self.master = None
        self.canvases = dict()

    def run(self):
        threading.Thread(target=self.window_setup).start()

    def draw_grid(self):
        size = Block.SIZE
        if "top" in self.views:
            canvas = self.canvases["top"]
            for i in range(self.map.target_map.shape[2] + 1):
                x_const = self.map.offset_origin[0] + (i - 0.5) * size
                y_start = self.map.environment_extent[1] - (self.map.offset_origin[1] - size / 2)
                y_end = y_start - size * (self.map.target_map.shape[1])
                canvas.create_line(x_const + self.padding[0], y_start + self.padding[1],
                                   x_const + self.padding[0], y_end + self.padding[1])

            for i in range(self.map.target_map.shape[1] + 1):
                x_start = self.map.offset_origin[0] - size / 2
                x_end = x_start + size * (self.map.target_map.shape[2])
                y_const = self.map.environment_extent[1] - (self.map.offset_origin[1] + (i - 0.5) * size)
                canvas.create_line(x_start + self.padding[0], y_const + self.padding[1],
                                   x_end + self.padding[0], y_const + self.padding[1])

        if "front" in self.views:
            canvas = self.canvases["front"]
            for i in range(self.map.target_map.shape[2] + 1):
                x_const = self.map.offset_origin[0] + (i - 0.5) * size
                y_start = self.map.environment_extent[2] + size / 2
                y_end = y_start - size * (self.map.target_map.shape[0])
                canvas.create_line(x_const + self.padding[0], y_start + self.padding[1],
                                   x_const + self.padding[0], y_end + self.padding[1])

            for i in range(self.map.target_map.shape[0] + 1):
                x_start = self.map.offset_origin[0] - size / 2
                x_end = x_start + size * (self.map.target_map.shape[2])
                y_const = self.map.environment_extent[2] - (i - 0.5) * size
                canvas.create_line(x_start + self.padding[0], y_const + self.padding[1],
                                   x_end + self.padding[0], y_const + self.padding[1])

        if "side" in self.views:
            canvas = self.canvases["side"]
            for i in range(self.map.target_map.shape[1] + 1):
                x_const = self.map.offset_origin[1] + (i - 0.5) * size
                y_start = self.map.environment_extent[2] + size / 2
                y_end = y_start - size * (self.map.target_map.shape[0])
                canvas.create_line(x_const + self.padding[0], y_start + self.padding[1],
                                   x_const + self.padding[0], y_end + self.padding[1])

            for i in range(self.map.target_map.shape[0] + 1):
                x_start = self.map.offset_origin[1] - size / 2
                x_end = x_start + size * (self.map.target_map.shape[1])
                y_const = self.map.environment_extent[2] - (i - 0.5) * size
                canvas.create_line(x_start + self.padding[0], y_const + self.padding[1],
                                   x_end + self.padding[0], y_const + self.padding[1])

    def draw_blocks(self):
        size = Block.SIZE
        if "top" in self.views:
            canvas = self.canvases["top"]
            sorted_blocks = sorted(self.map.blocks, key=lambda x: x.geometry.position[2])
            for b in sorted_blocks:
                points = np.concatenate(b.geometry.corner_points_2d()).tolist()
                for p_idx, p in enumerate(points):
                    if p_idx % 2 != 0:
                        points[p_idx] = self.map.environment_extent[1] - p + self.padding[1]
                    else:
                        points[p_idx] = p + self.padding[0]
                canvas.create_polygon(points, fill=b.color, outline="black")

        if "front" in self.views:
            canvas = self.canvases["front"]
            sorted_blocks = sorted(self.map.blocks, key=lambda x: x.geometry.position[1], reverse=True)
            for b in sorted_blocks:
                points = np.array(b.geometry.corner_points_2d())
                min_x = min(points[:, 0]) + self.padding[0]
                max_x = max(points[:, 0]) + self.padding[0]
                z = self.map.environment_extent[2] - (b.geometry.position[2] - size / 2) + self.padding[1]
                canvas.create_polygon([min_x, z - size / 2, max_x, z - size / 2,
                                       max_x, z + size / 2, min_x, z + size / 2], fill=b.color, outline="black")

        if "side" in self.views:
            canvas = self.canvases["side"]
            sorted_blocks = sorted(self.map.blocks, key=lambda x: x.geometry.position[0])
            for b in sorted_blocks:
                points = np.array(b.geometry.corner_points_2d())
                min_y = min(points[:, 1]) + self.padding[0]
                max_y = max(points[:, 1]) + self.padding[0]
                z = self.map.environment_extent[2] - (b.geometry.position[2] - size / 2) + self.padding[1]
                canvas.create_polygon([min_y, z - size / 2, max_y, z - size / 2,
                                       max_y, z + size / 2, min_y, z + size / 2], fill=b.color, outline="black")

    def draw_agents(self):
        if "top" in self.views:
            canvas = self.canvases["top"]
            counter = 0
            for a in self.map.agents:
                if False and a.current_path is not None:
                    canvas.create_line(a.geometry.position[0] + self.padding[0],
                                       self.map.environment_extent[1] - a.geometry.position[1] + self.padding[1],
                                       a.current_path.positions[0][0] + self.padding[0],
                                       self.map.environment_extent[1] - a.current_path.positions[0][1] + self.padding[
                                           1],
                                       fill="blue", width=2)
                    for i in range(0, len(a.current_path.positions) - 1):
                        if a.current_path is not None:
                            canvas.create_line(a.current_path.positions[i][0] + self.padding[0],
                                               self.map.environment_extent[1] - a.current_path.positions[i][1] +
                                               self.padding[1],
                                               a.current_path.positions[i + 1][0] + self.padding[0],
                                               self.map.environment_extent[1] - a.current_path.positions[i + 1][1] +
                                               self.padding[1],
                                               fill="blue", width=2)

                if a.collision_using_geometries:
                    points = np.concatenate(a.collision_avoidance_geometry_with_block.corner_points_2d()).tolist()
                    for p_idx, p in enumerate(points):
                        if p_idx % 2 != 0:
                            points[p_idx] = self.map.environment_extent[1] - p + self.padding[1]
                        else:
                            points[p_idx] = p + self.padding[0]
                    canvas.create_polygon(points, fill="black", outline="black")
                else:
                    canvas.create_oval(a.geometry.position[0] - a.required_distance / 2 + self.padding[0],
                                       self.map.environment_extent[1] - (a.geometry.position[1] -
                                                                         a.required_distance / 2) + self.padding[1],
                                       a.geometry.position[0] + a.required_distance / 2 + self.padding[0],
                                       self.map.environment_extent[1] - (a.geometry.position[1] +
                                                                         a.required_distance / 2) + self.padding[1],
                                       fill="black")

                points = np.concatenate(a.geometry.corner_points_2d()).tolist()
                for p_idx, p in enumerate(points):
                    if p_idx % 2 != 0:
                        points[p_idx] = self.map.environment_extent[1] - p + self.padding[1]
                    else:
                        points[p_idx] = p + self.padding[0]
                canvas.create_polygon(points, fill="blue", outline="black")

                canvas.create_text(a.geometry.position[0] + self.padding[0],
                                   self.map.environment_extent[1] - a.geometry.position[1] + self.padding[1],
                                   fill="white", font="Arial 20 bold", text="{}".format(counter))
                counter += 1

        if "front" in self.views:
            canvas = self.canvases["front"]
            sorted_agents = sorted(self.map.agents, key=lambda x: x.geometry.position[1], reverse=True)
            for a in sorted_agents:
                if a.collision_using_geometries:
                    size = a.collision_avoidance_geometry_with_block.size[2]
                    points = np.array(a.collision_avoidance_geometry_with_block.corner_points_2d())
                    min_x = min(points[:, 0]) + self.padding[0]
                    max_x = max(points[:, 0]) + self.padding[0]
                    z = self.map.environment_extent[2] - (a.collision_avoidance_geometry_with_block.position[2] - Block.SIZE / 2) + self.padding[1]
                    canvas.create_polygon([min_x, z - size / 2, max_x, z - size / 2,
                                           max_x, z + size / 2, min_x, z + size / 2], fill="black", outline="black")
                else:
                    canvas.create_oval(a.geometry.position[0] - a.required_distance / 2 + self.padding[0],
                                       self.map.environment_extent[2] - (a.geometry.position[2] - a.geometry.size[2] -
                                                                         a.required_distance / 2) + self.padding[1],
                                       a.geometry.position[0] + a.required_distance / 2 + self.padding[0],
                                       self.map.environment_extent[2] - (a.geometry.position[2] - a.geometry.size[2] +
                                                                         a.required_distance / 2) + self.padding[1],
                                       fill="black")

            for a in sorted_agents:
                size = a.geometry.size[2]
                points = np.array(a.geometry.corner_points_2d())
                min_x = min(points[:, 0]) + self.padding[0]
                max_x = max(points[:, 0]) + self.padding[0]
                z = self.map.environment_extent[2] - (a.geometry.position[2] - Block.SIZE / 2) + self.padding[1]
                canvas.create_polygon([min_x, z - size / 2, max_x, z - size / 2,
                                       max_x, z + size / 2, min_x, z + size / 2], fill="blue", outline="black")

    def draw_numbers(self):
        # TODO: at least don't show the seed numbers that should be covered if possible
        if "top" in self.views:
            canvas = self.canvases["top"]
            for a in self.map.agents:
                canvas.create_text(a.geometry.position[0] + self.padding[0],
                                   self.map.environment_extent[1] - a.geometry.position[1] + self.padding[1],
                                   fill="white", font="Arial 20 bold", text="{}".format(a.id))

            for b in self.map.placed_blocks:
                if b.is_seed:
                    canvas.create_text(b.geometry.position[0] + self.padding[0],
                                       self.map.environment_extent[1] - b.geometry.position[1] + self.padding[1],
                                       fill="white", font="Arial 10 bold", text="{}".format(b.grid_position[2]))

        if "front" in self.views:
            canvas = self.canvases["front"]
            for a in self.map.agents:
                canvas.create_text(a.geometry.position[0] + self.padding[0],
                                   self.map.environment_extent[2] - a.geometry.position[2] +
                                   self.padding[1] + a.geometry.size[2] / 2,
                                   fill="white", font="Arial 20 italic bold", text="{}".format(a.id))

            for b in self.map.placed_blocks:
                if b.is_seed:
                    canvas.create_text(b.geometry.position[0] + self.padding[0],
                                       self.map.environment_extent[2] - b.geometry.position[2] +
                                       self.padding[1] + b.geometry.size[2] / 2,
                                       fill="white", font="Arial 10 bold", text="{}".format(b.grid_position[2]))

    def update_graphics(self):
        if self.render:
            for c in list(self.canvases.values()):
                c.delete("all")
            self.draw_grid()
            self.draw_agents()
            self.draw_blocks()
            self.draw_numbers()

    def window_setup(self):
        self.master = Tk()

        def on_closing():
            # also need to stop main loop, maybe also using queue?
            self.return_queue.put(Graphics2D.SHUTDOWN_RETURN)
            self.master.destroy()

        def timer_tick():
            try:
                request = self.update_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if request is Graphics2D.UPDATE_REQUEST:
                    self.update_graphics()
                elif request is Graphics2D.SHUTDOWN_REQUEST:
                    self.master.destroy()

            self.master.after(self.update_interval, timer_tick)

        if "top" in self.views:
            cv = Canvas(self.master,
                        width=(self.map.environment_extent[0] + 2 * self.padding[0]),
                        height=(self.map.environment_extent[1] + 2 * self.padding[1]),
                        background="#a7a9bc")
            # cv.pack(side=LEFT)
            cv.grid(row=1, column=0)
            self.canvases["top"] = cv
        if "front" in self.views:
            # m = self.master if len(self.master.children) == 0 else Toplevel(master=self.master)
            # m.protocol("WM_DELETE_WINDOW", on_closing)
            cv = Canvas(self.master,
                        width=(self.map.environment_extent[0] + 2 * self.padding[0]),
                        height=(self.map.environment_extent[2] + 2 * self.padding[1]),
                        background="#a7a9bc")
            # cv.pack(side=LEFT)
            cv.grid(row=1, column=1)
            self.canvases["front"] = cv
        if "side" in self.views:
            # m = self.master if len(self.master.children) == 0 else Toplevel(master=self.master)
            # m.protocol("WM_DELETE_WINDOW", on_closing)
            cv = Canvas(self.master,
                        width=(self.map.environment_extent[1] + 2 * self.padding[0]),
                        height=(self.map.environment_extent[2] + 2 * self.padding[1]),
                        background="#a7a9bc")
            # cv.pack(side=LEFT)
            cv.grid(row=1, column=2)
            self.canvases["side"] = cv

        def callback_pause_play():
            self.return_queue.put(Graphics2D.PAUSE_PLAY_RETURN)

        b = Button(self.master, text="Pause/play", command=callback_pause_play)
        # b.pack(side=BOTTOM)
        b.grid(row=0, column=0)

        f = Frame(self.master)

        f.grid(row=0, column=1)
        l = Label(f, text="Interval (s): ")
        l.pack(side=LEFT)
        e = Entry(f, width=5)
        e.pack(side=LEFT)

        def callback_interval(v):
            value = e.get()
            e.delete(0, "end")
            try:
                value = float(value)
                self.return_queue.put(Graphics2D.INTERVAL_RETURN)
                self.return_queue.put(value)
            except ValueError:
                pass

        e.bind("<Return>", callback_interval)

        # initialise graphics
        for c in list(self.canvases.values()):
            c.delete("all")
        self.draw_grid()
        self.draw_agents()
        self.draw_blocks()

        timer_tick()
        self.master.protocol("WM_DELETE_WINDOW", on_closing)
        self.master.mainloop()
