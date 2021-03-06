import numpy as np
try:
    from Tkinter import *
    from Tkinter.filedialog import asksaveasfilename, askopenfilename
except ImportError:
    from tkinter import *
    from tkinter.filedialog import asksaveasfilename, askopenfilename


class Editor2D(Tk):
    """
    A simple editor for creating structures and saving them for later use.
    """

    def __init__(self):
        Tk.__init__(self)

        # data stuff
        self.number_rows = 4
        self.number_columns = 4
        self.current_layer = 0
        self.number_layers = 1
        self.layers = np.zeros_like(np.ndarray((self.number_layers, self.number_rows, self.number_columns)),
                                    dtype="int64")
        self.shift_activated = False
        self.shift_origin = None
        self.shift_layer = np.zeros_like(self.layers[0])

        # GUI data representation stuff
        self.padding = (20, 20)
        self.cell_size = 30

        # super necessary GUI variables
        self.row_variable = IntVar()
        self.row_variable.set(4)
        self.column_variable = IntVar()
        self.column_variable.set(4)

        # GUI elements
        self.canvas = None
        self.entry_rows = None
        self.entry_columns = None
        self.button_add_layer = None
        self.button_layer_up = None
        self.button_layer_down = None
        self.label_layer = None
        self.checkbox_seed = None
        self.button_show = None
        self.button_save = None
        self.button_load = None
        self.button_clear = None
        self.setup()
        self.redraw()

    def redraw(self):
        """
        Redraw the updated state.
        """

        self.canvas.delete("all")

        # draw grid
        for i in range(self.layers[self.current_layer].shape[1] + 1):
            x_const = self.padding[0] + i * self.cell_size
            y_start = self.padding[1]
            y_end = y_start + self.number_rows * self.cell_size
            self.canvas.create_line(x_const, y_start, x_const, y_end)

        for i in range(self.layers[self.current_layer].shape[0] + 1):
            x_start = self.padding[0]
            x_end = x_start + self.number_columns * self.cell_size
            y_const = self.padding[1] + i * self.cell_size
            self.canvas.create_line(x_start, y_const, x_end, y_const)

        # draw blocks
        for y in range(self.layers[self.current_layer].shape[0]):
            for x in range(self.layers[self.current_layer].shape[1]):
                if self.layers[self.current_layer, y, x] > 0:
                    x_start = self.padding[0] + x * self.cell_size
                    y_start = self.padding[1] + (self.number_rows - 1 - y) * self.cell_size
                    self.canvas.create_rectangle(x_start, y_start, x_start + self.cell_size, y_start + self.cell_size,
                                                 fill=("#329134" if self.layers[self.current_layer, y, x] == 1 else
                                                       "#b53030"))
                if self.shift_activated and self.shift_layer[y, x] > 0:
                    x_start = self.padding[0] + x * self.cell_size
                    y_start = self.padding[1] + (self.number_rows - 1 - y) * self.cell_size
                    self.canvas.create_rectangle(x_start, y_start, x_start + self.cell_size, y_start + self.cell_size,
                                                 fill="#f4e242")

    def setup(self):
        """
        Set up the user interface and its functionality.
        """

        self.wm_title("Map editor")
        self.wm_minsize(800, 400)

        for i in range(50):
            self.rowconfigure(i, weight=1)
            self.columnconfigure(i, weight=1)

        def on_click(e, button):
            if e.x - self.padding[0] < 0 or e.y - self.padding[1] < 0:
                return
            closest_x = int((e.x - self.padding[0]) / self.cell_size)
            closest_y = int((e.y - self.padding[1]) / self.cell_size)
            if closest_y > self.number_rows - 1 or closest_x > self.number_columns - 1:
                return
            print("self.shift_activated: {}".format(self.shift_activated))
            if not self.shift_activated:
                if button == "left":
                    self.layers[self.current_layer, self.number_rows - 1 - closest_y, closest_x] = 1
                elif button == "middle" and self.current_layer == 0:
                    if 2 in self.layers[self.current_layer]:
                        self.layers[self.current_layer][self.layers[self.current_layer] == 2] = 1
                    self.layers[self.current_layer, self.number_rows - 1 - closest_y, closest_x] = 2
                elif button == "right":
                    self.layers[self.current_layer, self.number_rows - 1 - closest_y, closest_x] = 0
            elif button == "left":
                self.shift_activated = not self.shift_activated
                print("SHIFT LAYER:")
                print(self.shift_layer)
                self.layers[self.current_layer][np.where(self.shift_layer == 1)] = 1
            self.redraw()

        def on_shift_click(e, button):
            if e.x - self.padding[0] < 0 or e.y - self.padding[1] < 0:
                return
            closest_x = int((e.x - self.padding[0]) / self.cell_size)
            closest_y = int((e.y - self.padding[1]) / self.cell_size)
            if closest_y > self.number_rows - 1 or closest_x > self.number_columns - 1:
                return

            if not self.shift_activated:
                self.shift_activated = True
            if self.shift_activated:
                self.shift_origin = [closest_x, closest_y]
            print("SHIFT CLICK HAPPENED")

        def on_hover(e):
            if self.shift_activated:
                closest_x = int((e.x - self.padding[0]) / self.cell_size)
                closest_y = int((e.y - self.padding[1]) / self.cell_size)
                if closest_x < 0:
                    closest_x = 0
                if closest_y < 0:
                    closest_y = 0
                if closest_x > self.number_columns - 1:
                    closest_x = self.number_columns - 1
                if closest_y > self.number_rows - 1:
                    closest_y = self.number_rows - 1

                if closest_x < self.shift_origin[0]:
                    x_slice = (closest_x, self.shift_origin[0] + 1)
                else:
                    x_slice = (self.shift_origin[0], closest_x + 1)
                if self.number_rows - 1 - closest_y < self.number_rows - 1 - self.shift_origin[1]:
                    y_slice = (self.number_rows - 1 - closest_y, self.number_rows - self.shift_origin[1])
                else:
                    y_slice = (self.number_rows - 1 - self.shift_origin[1], self.number_rows - closest_y)

                self.shift_layer[self.shift_layer > 0] = 0
                self.shift_layer[slice(*y_slice), slice(*x_slice)] = 1
                self.redraw()

        self.canvas = Canvas(self)
        self.canvas.grid(row=1, column=0, rowspan=49, columnspan=47, sticky="NESW")
        self.canvas.create_rectangle(0, 0, 100000, 100000, fill="green")
        self.canvas.bind("<Button-1>", lambda e: on_click(e, "left"))
        self.canvas.bind("<Button-2>", lambda e: on_click(e, "middle"))
        self.canvas.bind("<Button-3>", lambda e: on_click(e, "right"))
        self.canvas.bind("<Shift-Button-1>", lambda e: on_shift_click(e, "left"))
        self.canvas.bind("<Shift-Button-3>", lambda e: on_shift_click(e, "right"))
        self.canvas.bind("<Motion>", on_hover)

        def on_change(t):
            self.shift_activated = False
            if t == "row":
                self.number_rows = int(self.row_variable.get())
                new_layers = np.zeros_like(np.ndarray((self.number_layers, self.number_rows, self.number_columns)),
                                           dtype="int64")
                diff = self.layers.shape[1] - self.number_rows
                for i in range(self.number_layers):
                    for j in range(self.layers.shape[1] - 1, diff - 1, -1):
                        for k in range(min(self.layers.shape[2], self.number_columns)):
                            new_layers[i, j - diff, k] = self.layers[i, j, k]
                if diff < 0:
                    new_layers[:, :-diff, :] = 0
                self.layers = new_layers
                self.shift_layer = np.zeros_like(self.layers[0])
            elif t == "col":
                self.number_columns = int(self.column_variable.get())
                new_layers = np.zeros_like(np.ndarray((self.number_layers, self.number_rows, self.number_columns)),
                                           dtype="int64")
                for i in range(self.number_layers):
                    for j in range(min(self.layers.shape[1], self.number_rows)):
                        for k in range(min(self.layers.shape[2], self.number_columns)):
                            new_layers[i, j, k] = self.layers[i, j, k]
                self.layers = new_layers
                self.shift_layer = np.zeros_like(self.layers[0])
            self.redraw()

        label_rows = Label(self, text="Rows:")
        label_rows.grid(row=1, column=48, sticky=W)
        self.entry_rows = Spinbox(self, from_=1, to=20, command=lambda: on_change("row"),
                                  textvariable=self.row_variable)
        self.entry_rows.bind("<Return>", lambda e: on_change("row"))
        self.entry_rows.bind("<Tab>", lambda e: on_change("row"))
        self.entry_rows.grid(row=1, column=49)

        label_columns = Label(self, text="Columns:")
        label_columns.grid(row=2, column=48, sticky=W)
        self.entry_columns = Spinbox(self, from_=1, to=20, command=lambda: on_change("col"),
                                     textvariable=self.column_variable)
        self.entry_columns.bind("<Return>", lambda e: on_change("col"))
        self.entry_columns.bind("<Tab>", lambda e: on_change("col"))
        self.entry_columns.grid(row=2, column=49)

        def add_layer():
            self.shift_activated = False
            new_layer = np.zeros_like(np.ndarray((1, self.number_rows, self.number_columns)))
            self.layers = np.vstack((self.layers, new_layer))
            self.number_layers += 1
            self.current_layer = self.number_layers - 1
            self.label_layer["text"] = "{}/{}".format(self.current_layer + 1, self.number_layers)
            self.redraw()

        self.button_add_layer = Button(self, text="Add layer", command=add_layer)
        self.button_add_layer.grid(row=3, column=48, sticky=W)

        def move_layer(t):
            self.shift_activated = False
            if t == "up":
                self.current_layer += 1 if self.current_layer < self.number_layers - 1 else 0
                self.label_layer["text"] = "{}/{}".format(self.current_layer + 1, self.number_layers)
            elif t == "down":
                self.current_layer -= 1 if self.current_layer > 0 else 0
                self.label_layer["text"] = "{}/{}".format(self.current_layer + 1, self.number_layers)
            self.redraw()

        self.button_layer_up = Button(self, text="Layer up", command=lambda: move_layer("up"))
        self.button_layer_up.grid(row=4, column=48, sticky=W)

        self.button_layer_down = Button(self, text="Layer down", command=lambda: move_layer("down"))
        self.button_layer_down.grid(row=5, column=48, sticky=W)

        # label_seed = Label(self, text="Seed (yes/no):")
        # label_seed.grid(row=6, column=48, sticky=W)
        # self.checkbox_seed = Checkbutton(self)
        # self.checkbox_seed.grid(row=6, column=49, sticky=W)

        self.label_layer = Label(self, text="Layer: {}/{}".format(self.current_layer + 1, self.number_layers))
        self.label_layer.grid(row=7, column=48, sticky=W)

        def show():
            np.set_printoptions(threshold=np.nan)
            self.shift_activated = False
            self.layers = self.layers.astype(dtype="int64")
            # string_representation = "np." + np.array2string(self.layers, threshold=np.nan)
            string_representation = "np.{}".format(repr(self.layers))
            np.set_printoptions(threshold=1000)

            new_window = Toplevel(self)
            new_window.resizable(False, False)
            new_window.wm_title("Copyable representation")

            def copy():
                self.clipboard_clear()
                self.clipboard_append(string_representation)
                self.update()

            button_copy = Button(new_window, text="Copy to clipboard", command=copy)
            button_copy.grid(row=0, column=0, sticky="NSEW")

            text_window = Text(new_window, height=20, width=70)
            text_window.config(font=("parts of the structureroboto", 10), undo=True, wrap="word")
            text_window.grid(row=1, column=0, sticky="NSEW")

            scrollbar = Scrollbar(new_window, command=text_window.yview)
            scrollbar.grid(row=1, column=1, sticky="NSEW")

            text_window["yscrollcommand"] = scrollbar.set
            text_window.insert(END, string_representation)
            text_window.focus_set()

        self.button_show = Button(self, text="Show", command=show)
        self.button_show.grid(row=9, column=48, sticky=W)

        def save():
            self.shift_activated = False
            file_name = asksaveasfilename(defaultextension=".npy")
            if file_name is None or len(file_name) == 0:
                return
            np.save(file_name, self.layers, allow_pickle=False)

        self.button_save = Button(self, text="Save", command=save)
        self.button_save.grid(row=10, column=48, sticky=W)

        def load():
            self.shift_activated = False
            file_name = askopenfilename(defaultextension=".npy")
            if file_name is None or len(file_name) == 0:
                return
            self.layers = np.load(file_name).astype(dtype="int64")
            self.number_layers, self.number_rows, self.number_columns = self.layers.shape
            self.entry_rows.delete(0, "end")
            self.entry_rows.insert(0, self.number_rows)
            self.entry_columns.delete(0, "end")
            self.entry_columns.insert(0, self.number_columns)
            self.label_layer["text"] = "{}/{}".format(self.current_layer + 1, self.number_layers)
            self.redraw()

        self.button_load = Button(self, text="Load", command=load)
        self.button_load.grid(row=11, column=48, sticky=W)

        def clear():
            np.place(self.layers[self.current_layer], self.layers[self.current_layer] > 0, 0)
            self.redraw()

        self.button_clear = Button(self, text="Clear", command=clear)
        self.button_clear.grid(row=12, column=48, sticky=W)


if __name__ == "__main__":
    editor = Editor2D()
    editor.mainloop()
