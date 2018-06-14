import numpy as np

from geom.util import simple_distance


class Path:
    """
    A class which encapsulates path planning information. This information mainly consists of ordered way points
    in 3D space. It is the main data structure used for agent movement in the simulation.
    """

    def __init__(self, positions=None):
        self.positions = []
        if positions is not None:
            for p in positions:
                self.add_position(p)
        self.current_index = 0
        self.inserted_paths = {}
        self.number_inserted_positions = 0
        self.inserted_indices = []
        self.inserted_sequentially = []
        self.optional_distances = []

    def add_position(self, position, index=None, optional_distance=(0, 0, 0)):
        """
        Add a position to the path, either at the end or the specified index.

        :param position: the position to add to the path
        :param index: the index at which to insert the new position at
        :param optional_distance: a tuple of distances (representing each axis) used to decide whether the agent
        has reached a point on the path
        """

        if index is None:
            self.positions.append(np.array(position))
            self.inserted_sequentially.append(True)
            self.optional_distances.append(optional_distance)
        else:
            self.positions.insert(index, np.array(position))
            self.inserted_sequentially.insert(index, False)
            self.optional_distances.insert(index, optional_distance)

    def advance(self):
        """
        Advance the path and move on to the next position as the new target position, increasing the index.

        :return: True if the path could be advanced, False if the last position of the path has been reached.
        """

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
        """
        Insert another path to complete first when a certain index on this path has been reached.

        :param path: a path to follow after reaching a certain point on this path
        :param index_key: the index at which to switch to following the other path
        """

        if index_key is None:
            self.inserted_paths[self.current_index] = path
        else:
            self.inserted_paths[index_key] = path

    def remove_path(self, index_key):
        """
        Remove a previously inserted path.

        :param index_key: the index of the path to be removed
        :return:
        """

        del self.inserted_paths[index_key]

    def retreat(self):
        """
        Go back one step on the path, targeting the position before the current one.

        :return: True if retreating is possible, False if already targeting the first position on the path
        """

        if self.current_index == 0:
            return False
        else:
            self.current_index -= 1
            return True

    def closest(self, position):
        """
        Return the position on the path closest to the specified position.

        :return: the closest position on the path
        """

        min_position = None
        min_distance = float("inf")
        for p in self.positions:
            temp = simple_distance(p, position)
            if temp < min_distance:
                min_position = p
                min_distance = temp
        return min_position

    def next(self):
        """
        Return the next (best) point on the path.

        While this method currently returns the next position on the path according to the current index,
        one could implement a more advanced method of choosing the best path (e.g. based on distance and the
        necessity to actually reach certain points or not).

        :return: the next point on the path
        """

        if self.current_index in self.inserted_paths:
            # instead take the next from that path
            return self.inserted_paths[self.current_index].next()
        return self.positions[self.current_index]

    def direction_to_closest(self, position):
        """
        Return the direction to the closest point on the path.

        :param position: the position to calculate the direction from
        :return: a direction vector to the position on the path closest to the specified position
        """

        closest = self.closest(position)
        return closest - np.array(position)

    def direction_to_next(self, position):
        """
        Return the direction to the next (best) point on the path.

        :param position: the position to calculate the direction from
        :return: a direction vector to the next position on the path
        """

        return self.next() - np.array(position)

    def optional_area_reached(self, position):
        """
        Determine whether the position is close enough to an optional point to count it as having been reached.

        :param position: the position to calculate the direction from
        :return: True if the position is close enough to an optional position on the path, False otherwise
        """

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
