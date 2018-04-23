import numpy as np
from typing import List
from numpy import cos, sin
from geom.util import *


class GeomBox:

    def __init__(self,
                 position: List[float],
                 size: List[float],
                 rotation: float):
        self.__position = np.array(position)
        self.__size = np.array(size)
        self.__rotation = rotation  # rotation in radians
        self.attached_geometries = []

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        difference = np.array(position) - self.__position
        self.__position += difference
        for g in self.attached_geometries:
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

    def normals_2d(self):
        normal_x = [1, 0]
        normal_y = [0, 1]

        cs = cos(self.__rotation)
        sn = sin(self.__rotation)
        rot_normal_x = np.array([normal_x[0] * cs - normal_x[1] * sn, normal_x[0] * sn + normal_x[1] * cs])
        rot_normal_y = np.array([normal_y[0] * cs - normal_y[1] * sn, normal_y[0] * sn + normal_y[1] * cs])

        return rot_normal_x, rot_normal_y

    def corner_points_2d(self):
        x_norm, y_norm = self.normals_2d()

        point_1 = self.position[:2] - x_norm * self.size[0] / 2 - y_norm * self.size[1] / 2
        point_2 = self.position[:2] + x_norm * self.size[0] / 2 - y_norm * self.size[1] / 2
        point_3 = self.position[:2] + x_norm * self.size[0] / 2 + y_norm * self.size[1] / 2
        point_4 = self.position[:2] - x_norm * self.size[0] / 2 + y_norm * self.size[1] / 2

        return point_1, point_2, point_3, point_4

    def overlaps(self, other):
        # check if overlap in height occurs
        if not interval_overlaps(self.position[2] - self.size[2] / 2, self.position[2] + self.size[2] / 2,
                                 other.position[2] - other.size[2] / 2, other.position[2] + other.size[2] / 2):
            return False

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
                return False

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
                return False

        # compute all normals for both rectangles (pretty easy since they are rectangles rotated only around z)

        # for each normal, project all points on it, get the min and max of each ON THAT AXIS/NORMAL

        # check if there is overlap between them, if not done, if yes keep trying
        return True

    def distance_3d(self, other):
        return np.sqrt((self.position[0] - other.position[0]) ** 2 +
                       (self.position[1] - other.position[1]) ** 2 +
                       (self.position[2] - other.position[2]) ** 2)

    def distance_2d(self, other):
        return np.sqrt((self.position[0] - other.position[0]) ** 2 +
                       (self.position[1] - other.position[1]) ** 2)


class GridPosition:

    def __init__(self, x: int = 0, y: int = 0, z: int = 0):
        self.x = x
        self.y = y
        self.z = z

