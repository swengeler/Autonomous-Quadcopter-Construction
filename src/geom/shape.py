from typing import List

from numpy import cos, sin

from geom.util import *


class Geometry:
    """
    A class used to encapsulate the size, rotation and position of a cubic object
    (either an agent or a block for the purposes of this simulation).
    """

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 rotation: float):
        self.__position = np.array(position)
        self.__size = np.array(size)
        self.__rotation = rotation  # rotation in radians
        self.attached_geometries = []
        self.following_geometries = []

    def set_to_match(self, other_geometry):
        """
        Set all properties of the current geometry to match that of the provided one.

        :param other_geometry: the geometry to match
        """

        self.position = other_geometry.position
        self.size = other_geometry.size
        self.rotation = other_geometry.rotation
        self.attached_geometries = []
        for g in other_geometry.attached_geometries:
            self.attached_geometries.append(g)
        self.following_geometries = []
        for g in other_geometry.following_geometries:
            self.following_geometries.append(g)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        difference = np.array(position) - self.__position
        self.__position += difference
        for g in self.attached_geometries:
            g.position += difference
        for g in self.following_geometries:
            g.position += difference

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        self.__size = np.array(size)

    @property
    def rotation(self):
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation):
        difference = rotation - self.__rotation
        self.__rotation += difference
        for g in self.attached_geometries:
            g.position += difference
        for g in self.following_geometries:
            g.position += difference

    def normals_2d(self):
        """
        Return the normal vectors of the vertical sides ("left"/"right", "front"/"back") of this cubic geometry.

        :return: the normal vectors in 2D
        """

        normal_x = [1, 0]
        normal_y = [0, 1]

        cs = cos(self.__rotation)
        sn = sin(self.__rotation)
        rot_normal_x = np.array([normal_x[0] * cs - normal_x[1] * sn, normal_x[0] * sn + normal_x[1] * cs])
        rot_normal_y = np.array([normal_y[0] * cs - normal_y[1] * sn, normal_y[0] * sn + normal_y[1] * cs])

        return rot_normal_x, rot_normal_y

    def corner_points_2d(self):
        """
        Return the (x, y) corner points of the cube representing this geometry.

        :return: the corner points of the geometry
        """

        x_norm, y_norm = self.normals_2d()

        point_1 = self.position[:2] - x_norm * self.size[0] / 2 - y_norm * self.size[1] / 2
        point_2 = self.position[:2] + x_norm * self.size[0] / 2 - y_norm * self.size[1] / 2
        point_3 = self.position[:2] + x_norm * self.size[0] / 2 + y_norm * self.size[1] / 2
        point_4 = self.position[:2] - x_norm * self.size[0] / 2 + y_norm * self.size[1] / 2

        return point_1, point_2, point_3, point_4

    def overlaps(self, other):
        """
        Check whether this geometry and the other geometry overlap.

        :param other: the other geometry to check against
        :return: True if the geometries overlap, False otherwise
        """

        # check if overlap in height occurs
        if not interval_overlaps(self.position[2] - self.size[2] / 2, self.position[2] + self.size[2] / 2,
                                 other.position[2] - other.size[2] / 2, other.position[2] + other.size[2] / 2):
            return any([g.overlaps(other) for g in self.attached_geometries])

        points_self = list(self.corner_points_2d())
        normals_self = list(self.normals_2d())
        points_other = list(other.corner_points_2d())
        normals_other = list(other.normals_2d())

        for n in normals_self:
            min_self = min_other = float("inf")
            max_self = max_other = -float("inf")
            for p in points_self:
                dot = np.dot(p, n)
                min_self = min(min_self, dot)
                max_self = max(max_self, dot)
            for p in points_other:
                dot = np.dot(p, n)
                min_other = min(min_other, dot)
                max_other = max(max_other, dot)
            if not interval_overlaps(min_self, max_self, min_other, max_other):
                return any([g.overlaps(other) for g in self.attached_geometries])

        for n in normals_other:
            min_self = min_other = float("inf")
            max_self = max_other = -float("inf")
            for p in points_self:
                dot = np.dot(p, n)
                min_self = min(min_self, dot)
                max_self = max(max_self, dot)
            for p in points_other:
                dot = np.dot(p, n)
                min_other = min(min_other, dot)
                max_other = max(max_other, dot)
            if not interval_overlaps(min_self, max_self, min_other, max_other):
                return any([g.overlaps(other) for g in self.attached_geometries])

        return True

    def geometry_is_attached(self, other):
        """
        Check whether the given geometry is attached to this one.

        :param other: the other geometry
        :return: True if it is attached, False otherwise
        """

        for g in self.attached_geometries:
            if g[0] is other:
                return True
        return False

    def distance_2d(self, other):
        """
        Return the distance to the other geometry in the x-y-plane

        :param other: the geometry to calculate the distance to
        :return: the 2D-distance to the specified geometry
        """

        return np.sqrt((self.position[0] - other.position[0]) ** 2 +
                       (self.position[1] - other.position[1]) ** 2)

