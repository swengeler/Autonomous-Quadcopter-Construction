import numpy as np
from geom.util import simple_distance


class Path:

    def __init__(self, positions=None):
        self.positions = []
        if positions is not None:
            for p in positions:
                self.add_position(p)
        self.current_index = 0

    def add_position(self, position, index=None):
        if index is None:
            self.positions.append(np.array(position))
        else:
            self.positions.insert(index, np.array(position))

    def advance(self):
        if len(self.positions) == self.current_index + 1:
            return False
        else:
            self.current_index += 1
            return True

    def retreat(self):
        if self.current_index == 0:
            return False
        else:
            self.current_index -= 1
            return True

    def closest(self, position):
        """Return the closest point on the path."""
        min_position = None
        min_distance = float("inf")
        for p in self.positions:
            temp = simple_distance(p, position)
            if temp < min_distance:
                min_position = p
                min_distance = temp
        return min_position

    def next(self):
        """Return the next (best) point on the path."""
        return self.positions[self.current_index]

    def direction_to_closest(self, position):
        """Return the direction to the closest point on the path."""
        closest = self.closest(position)
        return closest - np.array(position)

    def direction_to_next(self, position):
        """Return the direction to the next (best) point on the path."""
        return self.next() - np.array(position)

