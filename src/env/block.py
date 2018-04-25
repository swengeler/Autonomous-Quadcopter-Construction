import logging
from typing import List
from geom.shape import GeomBox


class Block:

    COLORS = {
        0: "#c70039",
        1: "#57c785",
        2: "#ff5733",
        3: "#00baad",
        4: "#ffc300",
        5: "#3d3d6b"
    }

    SIZE = 15

    def __init__(self,
                 color="white",
                 is_seed: bool = False,
                 seed_marked_edge="down",
                 position: List[float] = None,
                 rotation: float = 0.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geometry = GeomBox(position if position is not None else
                                [0, 0, Block.SIZE / 2], [Block.SIZE] * 3, rotation)
        self.color = color
        self.is_seed = is_seed
        self.seed_marked_edge = seed_marked_edge
        self.placed = False
        self.__grid_position = None

    def overlaps(self, other):
        return self.geometry.overlaps(other.geometry)

    @property
    def grid_position(self):
        return self.__grid_position

    @grid_position.setter
    def grid_position(self, grid_position):
        self.__grid_position = grid_position
        if not self.is_seed:
            self.color = Block.COLORS[self.__grid_position[2]]
