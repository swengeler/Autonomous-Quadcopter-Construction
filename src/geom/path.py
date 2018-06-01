import numpy as np
from geom.util import simple_distance


class Path:

    def __init__(self, positions=None, dodging_path=False):
        self.positions = []
        if positions is not None:
            for p in positions:
                self.add_position(p)
        self.dodging_path = True
        self.current_index = 0
        self.inserted_paths = {}
        self.number_inserted_positions = 0
        self.inserted_indices = []
        self.inserted_sequentially = []
        self.optional_distances = []

    def add_position(self, position, index=None, optional_distance=(0, 0, 0)):
        if index is None:
            self.positions.append(np.array(position))
            self.inserted_sequentially.append(True)
            self.optional_distances.append(optional_distance)
        else:
            self.positions.insert(index, np.array(position))
            self.inserted_sequentially.insert(index, False)
            self.optional_distances.insert(index, optional_distance)

    def advance(self):
        if self.current_index in self.inserted_paths:
            advanced = self.inserted_paths[self.current_index].advance()
            if advanced:
                return True
            del self.inserted_paths[self.current_index]
        if len(self.positions) == self.current_index + 1:
            return False
        else:
            self.current_index += 1
            return True

    def insert_path(self, path, index_key=None):
        if index_key is None:
            self.inserted_paths[self.current_index] = path
        else:
            self.inserted_paths[index_key] = path

    def remove_path(self, index_key):
        del self.inserted_paths[index_key]

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
        if self.current_index in self.inserted_paths:
            # instead take the next from that path
            return self.inserted_paths[self.current_index].next()
        return self.positions[self.current_index]

    def direction_to_closest(self, position):
        """Return the direction to the closest point on the path."""
        closest = self.closest(position)
        return closest - np.array(position)

    def direction_to_next(self, position):
        """Return the direction to the next (best) point on the path."""
        return self.next() - np.array(position)

    def optional_area_reached(self, position):
        """Determine whether the position is close enough to an optional point to count it as having been reached."""
        # e.g. have to reach (x, y), preferably at z, but not necessarily
        optional_position = self.next()
        optional_distance = self.optional_distances[self.current_index]
        # optional_axes = self.optional_distances[self.current_index][1]
        # return simple_distance(position[[a for a in optional_axes]],
        #                        optional_position[[a for a in optional_axes]]) <= optional_distance
        if isinstance(optional_distance, tuple):
            return all([abs(position[i] - optional_position[i]) <= optional_distance[i] for i in range(3)])
        else:
            return simple_distance(position, optional_position) <= optional_distance
