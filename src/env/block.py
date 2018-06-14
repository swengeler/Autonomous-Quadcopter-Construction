import logging
from typing import List

from geom.shape import Geometry


class Block:
    """
    A class representing a block used for construction. It could be extended to e.g. implement communicating blocks.
    """

    COLORS_SEEDS = ["red"]
    COLORS_BLOCKS = ["blue"]

    SIZE = 15

    def __init__(self,
                 color="white",
                 is_seed: bool = False,
                 seed_marked_edge="down",
                 position: List[float] = None,
                 rotation: float = 0.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geometry = Geometry(position if position is not None else
                                [0, 0, Block.SIZE / 2], [Block.SIZE] * 3, rotation)
        self.color = color
        self.is_seed = is_seed
        self.seed_marked_edge = seed_marked_edge
        self.placed = False
        self.__grid_position = None

    def overlaps(self, other):
        """
        Check whether the block's geometry overlaps with some other geometry.

        :param other: the geometry to check against
        :return: True if the geometries overlap, False otherwise
        """

        return self.geometry.overlaps(other.geometry)

    @property
    def grid_position(self):
        return self.__grid_position

    @grid_position.setter
    def grid_position(self, grid_position):
        self.__grid_position = grid_position
        if self.is_seed:
            self.color = Block.COLORS_SEEDS[self.__grid_position[2]]
        else:
            self.color = Block.COLORS_BLOCKS[self.__grid_position[2]]
