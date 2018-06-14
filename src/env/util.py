import collections
import math
import random
from typing import Tuple

import numpy as np


class WrongInputException(Exception):
    pass


def shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
    """
    Return the shortest path from the start to the goal position using only nonzero positions in the grid.

    :param grid: the 2D occupancy matrix to be used to find the shortest path
    :param start: the starting position for the path
    :param goal: the goal position for the path
    :return: the shortest path as a list of grid positions between the start and goal position
    """

    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0] \
                    and (grid[y2, x2] != 0 or (x2, y2) == goal) and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return path


def shortest_path_3d_in_2d(lattice: np.ndarray, start: Tuple, goal: Tuple):
    """
    Return the shortest path from start to goal position, using only nonzero positions in the lattice.

    This function works similar to the shortest_path function but instead of using only the blocks on one
    layer for determining a possible route, it uses the blocks on all layers in the provided lattice.

    :param lattice: the 3D occupancy matrix to be used to find the shortest path
    :param start: the starting position for the path
    :param goal: the goal position for the path
    :return: the shortest path as a list of grid positions (3D) between the start and goal position
    """

    if len(start) == 3:
        start = start[0:2]
    if len(goal) == 3:
        goal = goal[0:2]

    # determining a grid of locations occupied below that can be used for orientation
    grid = np.zeros_like(lattice[0])
    for z in range(lattice.shape[0]):
        grid[lattice[z] != 0] = 1
    grid[start[1], start[0]] = 1
    grid[goal[1], goal[0]] = 1

    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            # determine the highest layer at which there is a block,
            # i.e. how high the agent has to fly to avoid all collisions
            highest_layer = 0
            for x, y in path:
                for z in range(lattice.shape[0] - 1, -1, -1):
                    if lattice[z, y, x] >= 1:
                        highest_layer = max(z, highest_layer)
                        break
            return path, highest_layer
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0] and grid[y2, x2] != 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    highest_layer = 0
    for x, y in path:
        for z in range(lattice.shape[0] - 1, -1, -1):
            if lattice[z, y, x] >= 1:
                highest_layer = max(z, highest_layer)
                break
    return path, highest_layer


def shortest_path_3d(lattice: np.ndarray, start, goal):
    """
    Return the shortest path from start to goal position, using only nonzero positions in the lattice.

    :param lattice:
    :param start:
    :param goal:
    :return:
    """

    # Note that the way I'm doing this right now basically means the resulting
    # shortest path tries to stick as close to the structure as possible
    processed_lattice = np.zeros_like(lattice)
    processed_lattice = np.concatenate((processed_lattice, np.array([processed_lattice[0]])))
    for z in range(processed_lattice.shape[0]):
        for y in range(processed_lattice.shape[1]):
            for x in range(processed_lattice.shape[2]):
                if z == 0 and lattice[z, y, x] == 0 and any([0 <= x2 < lattice.shape[2] and 0 <= y2 < lattice.shape[1]
                                                             and lattice[z, y2, x2] != 0 for x2, y2
                                                             in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1))]):
                    processed_lattice[z, y, x] = 1
                elif z != 0 and lattice[z - 1, y, x] != 0 and (z == lattice.shape[0] or lattice[z, y, x] == 0):
                    processed_lattice[z, y, x] = 1
    # print("\nORIGINAL MAP:")
    # print_map(lattice)
    for z in range(processed_lattice.shape[0]):
        for y in range(processed_lattice.shape[1]):
            for x in range(processed_lattice.shape[2]):
                if z != 0 and processed_lattice[z - 1, y, x] != 0 \
                        and any([0 <= x2 < processed_lattice.shape[2]
                                 and 0 <= y2 < processed_lattice.shape[1]
                                 and processed_lattice[z, y2, x2] == 1 for x2, y2
                                 in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1))]):
                    processed_lattice[z, y, x] = 2
    # print("\nFINAL LEGAL PATHWAY MAP:")
    # print_map(processed_lattice, custom_symbols={2: "."})
    # TODO: how to solve problem of accounting for multiple layer "climbs"?

    queue = collections.deque([[start]])
    seen = {start}
    path = None
    while queue:
        path = queue.popleft()
        x, y, z = path[-1]
        if x == goal[0] and y == goal[1] and z == goal[2]:
            return path
        for x2, y2, z2 in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
            if 0 <= x2 < processed_lattice.shape[2] and 0 <= y2 < processed_lattice.shape[1] \
                    and 0 <= z2 < processed_lattice.shape[0] and processed_lattice[z2, y2, x2] != 0 \
                    and (x2, y2, z2) not in seen:
                queue.append(path + [(x2, y2, z2)])
                seen.add((x2, y2, z2))
    return path


def print_map_layer(environment: np.ndarray, path=None, custom_symbols=None):
    """
    Print a single layer of a map (2D grid) in nice formatting.

    :param environment: the map/grid to print
    :param path: an optional (shortest) path to highlight
    :param custom_symbols: a dictionary of elements in the given grid mapping to symbols to print them as
    """

    symbols = {
        0: " ",
        1: ".",
        2: "X",
        "corner": "C"
    }

    for x in range(0, (environment.shape[1] + 2) * 2 - 1):
        print("―", end="")
    print("")
    for y in range(environment.shape[0] - 1, -1, -1):
        print("| ", end="")
        for x in range(environment.shape[1]):
            if path is not None and (x, y) in path and environment[y, x] != 0:
                if path.index((x, y)) == 0:
                    print("S", end=" ")
                elif path.index((x, y)) == len(path) - 1:
                    print("E", end=" ")
                else:
                    print("#", end=" ")
            elif custom_symbols is not None and environment[y, x] in custom_symbols.keys():
                print(custom_symbols[environment[y, x]], end=" ")
            elif environment[y, x] in symbols.keys():
                print(symbols[environment[y, x]], end=" ")
            elif environment[y, x] < 0:
                print(symbols["corner"], end=" ")
            else:
                print("?", end=" ")
        print("|")
    for x in range(0, (environment.shape[1] + 2) * 2 - 1):
        print("―", end="")
    print("")


def print_map(environment: np.ndarray, path=None, layer=None, custom_symbols=None):
    """
    Print an entire 3D map in nice formatting.

    :param environment: the map/lattice to print
    :param path: an optional (shortest) path to highlight
    :param layer: an optional layer in which the path should be included
    :param custom_symbols: a dictionary of elements in the given grid mapping to symbols to print them as
    :return:
    """

    if path is not None and layer is None and len(path[0]) != 3:
        raise WrongInputException("Variable 'layer' has to be provided if 2D path is given.")
    elif path is None and layer is not None:
        print("Warning: variable 'path' is None even though 'layer' was provided.")

    if environment.ndim == 2:
        environment = np.array([environment])
    for z in range(environment.shape[0]):
        for x in range(0, (environment.shape[2] + 2) * 2 - 1):
            print("―", end="")
        print("\n| Layer {} |".format(str(z).ljust(environment.shape[2] * 2 - 7)))
        if path is not None and ((len(path[0]) == 2 and z == layer) or len(path[0]) == 3):
            print_map_layer(environment[z], path=path, custom_symbols=custom_symbols)
        else:
            print_map_layer(environment[z], custom_symbols=custom_symbols)


def legal_attachment_sites(target_map: np.ndarray, occupancy_map: np.ndarray, component_marker=None, local_info=False):
    """
    Return information about the legal attachment sites for the given target map, occupancy matrix and component.

    This function is used only for the shortest path algorithm(s), since they have "free" choice of an
    attachment site. The function itself returns only the theoretically possible attachment sites (i.e. those
    not violating the row rule), unless it only attachment sites given local information are allowed. In that case,
    instead of returning only a matrix with marked legal attachment sites, the function also returns three lists
    with different types of attachment sites: corner sites, protruding sites and end-of-row sites.

    :param target_map: an occupancy matrix representing the target structure
    :param occupancy_map: an occupancy matrix representing the current state of the structure
    :param component_marker: the component for which to find attachment sites
    :param local_info: if True take into account that the occupancy matrix only contains local information and return
    additional information, otherwise just return a matrix with all theoretically possible attachment sites
    :return: a matrix with marked attachment sites and possibly lists of different attachment site types
    """

    if local_info:
        return legal_attachment_sites_revisited(target_map, occupancy_map, component_marker, True)

    # input is supposed to be a 2-dimensional layer
    # setting all values larger than 1 (seeds) to 1
    if component_marker is not None:
        occupancy_map = np.copy(occupancy_map)
        np.place(occupancy_map, target_map != component_marker, 0)
        target_map = np.copy(target_map)
        np.place(target_map, target_map != component_marker, 0)

    # identifying all sites that are adjacent to the structure built so far
    legal_sites = np.zeros_like(target_map)
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # can only be legal if a block is actually supposed to be placed there and has not been placed yet
            if target_map[y, x] >= 1 and occupancy_map[y, x] == 0:
                # needs to be adjacent to existing part of structure, but only to 2 or fewer blocks
                counter = 0
                for y2 in (y - 1, y + 1):
                    if 0 <= y2 < legal_sites.shape[0] and occupancy_map[y2, x] >= 1:
                        counter += 1
                for x2 in (x - 1, x + 1):
                    if 0 <= x2 < legal_sites.shape[1] and occupancy_map[y, x2] >= 1:
                        counter += 1
                if 1 <= counter < 3:
                    legal_sites[y, x] = 1

    # given that information is only local, the only legal sites are corners, i.e. adjacent to two other blocks,
    # or positions only adjacent to one block but

    # maybe it's enough to exclude corners around holes until we know that they are surrounded by blocks

    # drastic option: similar to perimeter search, exclude anything NORTH-WEST of corners

    # also identify those sites that still need to be unoccupied for the attachment site to be legal

    # identifying those sites where the row rule would be violated
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # only matters for those sites which have already been identified as potential attachment sites
            if legal_sites[y, x] == 1:
                # check whether the following pattern exists around that block (or in rotated form):
                # [B] [E] ... [A]
                # where (B = block already placed), (E = other potential attachment site), (A = current site)
                # and where at "..." there are only E's
                for diff in (-1, 1):
                    # making it through this loop without a break means that in the x-row, y-column where the block
                    # could be placed, there is either only a block immediately adjacent or any blocks already placed
                    # are separated from the current site by an intended gap

                    counter = 1
                    while 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] == 0 \
                            and target_map[y + counter * diff, x] > 0:
                        counter += 1
                    if counter > 1 and 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] > 0 and target_map[y + counter * diff, x] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        break

                    counter = 1
                    while 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] == 0 \
                            and target_map[y, x + counter * diff] > 0:
                        counter += 1
                    if counter > 1 and 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] > 0 and target_map[y, x + counter * diff] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        break

                # information needed for local information case:
                # - rows/columns where it is important for them to be empty (rather range of coordinates)
                # - if
                # BIG PROBLEM:  if there is nothing in a row/column until the edge of the construction area
                #               then that would mean that the agent would have to explore that entire stretch
                #               of empty space to confirm that placement is OK
                # SOLUTIONS?!:
                # - only consider those rows/columns which are alongside the already existing structure
                #   -> probably not gonna work (think of long-sided loop structure)
                # - could try to identify key sites in the structure (periodically) to make sure that attachments
                #   are legal even if only local information is used
                # - impose some kind of order on occupying attachment sites that means that only legal sites will exist
                #   -> also order sites by number of adjacent blocks -> always go for two

    return legal_sites


def legal_attachment_sites_revisited(target_map: np.ndarray,
                                     occupancy_map: np.ndarray,
                                     component_marker=None,
                                     row_information=False):
    # TODO combine these two into one
    """
    Return information about the legal attachment sites for the given target map, occupancy matrix and component.

    This method is used when the occupancy matrix only contains local information. As described in the function

    :param target_map:
    :param occupancy_map:
    :param component_marker:
    :param row_information: determines the format of the returned end-of-row sites
    :return:
    """

    if component_marker is not None:
        occupancy_map = np.copy(occupancy_map)
        np.place(occupancy_map, target_map != component_marker, 0)
        target_map = np.copy(target_map)
        np.place(target_map, target_map != component_marker, 0)

    # identifying all sites that are adjacent to the structure built so far
    legal_sites = np.zeros_like(target_map)
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # can only be legal if a block is actually supposed to be placed there and has not been placed yet
            if target_map[y, x] >= 1 and occupancy_map[y, x] == 0:
                # needs to be adjacent to existing part of structure, but only to 2 or fewer blocks
                counter = 0
                for y2 in (y - 1, y + 1):
                    if 0 <= y2 < legal_sites.shape[0] and occupancy_map[y2, x] >= 1:
                        counter += 1
                for x2 in (x - 1, x + 1):
                    if 0 <= x2 < legal_sites.shape[1] and occupancy_map[y, x2] >= 1:
                        counter += 1
                if 1 <= counter < 3:
                    legal_sites[y, x] = 1

    # identifying those sites where the row rule would be violated
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            # only matters for those sites which have already been identified as potential attachment sites
            if legal_sites[y, x] == 1:
                # check whether the following pattern exists around that block (or in rotated form):
                # [B] [E] ... [A]
                # where (B = block already placed), (E = other potential attachment site), (A = current site)
                # and where at "..." there are only E's
                for diff in (-1, 1):
                    # making it through this loop without a break means that in the x-row, y-column where the block
                    # could be placed, there is either only a block immediately adjacent or any blocks already placed
                    # are separated from the current site by an intended gap

                    counter = 1
                    while 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] == 0 \
                            and target_map[y + counter * diff, x] > 0:
                        counter += 1
                    if counter > 1 and 0 <= y + counter * diff < legal_sites.shape[0] \
                            and occupancy_map[y + counter * diff, x] > 0 and target_map[y + counter * diff, x] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        break

                    counter = 1
                    while 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] == 0 \
                            and target_map[y, x + counter * diff] > 0:
                        counter += 1
                    if counter > 1 and 0 <= x + counter * diff < legal_sites.shape[1] \
                            and occupancy_map[y, x + counter * diff] > 0 and target_map[y, x + counter * diff] > 0:
                        # have encountered a block already in this row
                        legal_sites[y, x] = 0
                        break

    # now attachment sites need to be sorted into corner/single-block protrusion (trivial attachment) and sites
    # that are in a row/column must be grouped into one such row/column to check
    # an alternative might be that for each row/column only the most CCW site would be a legal attachment site?
    # -> let's try this one first because it's easier
    # -> also choose southwestern-most (?)

    # determine corner sites (two adjacent blocks already placed) and protruding sites (width 1 parts of structure)
    corner_sites = []
    protruding_sites = []
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            if legal_sites[y, x] != 0:
                corner_counter = 0
                protruding_counter = 0
                on_map_counter = 0
                for diff in (-1, 1):
                    y2 = y + diff
                    if 0 <= y2 < legal_sites.shape[0]:
                        on_map_counter += 1
                        if occupancy_map[y2, x] != 0:
                            corner_counter += 1
                        if target_map[y2, x] != 0:
                            protruding_counter += 1
                    x2 = x + diff
                    if 0 <= x2 < legal_sites.shape[1]:
                        on_map_counter += 1
                        if occupancy_map[y, x2] != 0:
                            corner_counter += 1
                        if target_map[y, x2] != 0:
                            protruding_counter += 1

                # should really be checking whether two OPPOSING adjacent blocks are free or not (also diagonally)
                opposite_free_counter = 0
                for c1, c2 in [((-1, -1), (1, 1)), ((-1, 1), (1, -1)), ((0, -1), (0, 1)), ((-1, 0), (1, 0))]:
                    x1 = x + c1[0]
                    y1 = y + c1[1]
                    x2 = x + c2[0]
                    y2 = y + c2[1]
                    counter = 0
                    if x1 < 0 or x1 >= legal_sites.shape[1] \
                            or y1 < 0 or y1 >= legal_sites.shape[0] \
                            or target_map[y1, x1] == 0:
                        counter += 1
                    if counter > 0 and (x2 < 0 or x2 >= legal_sites.shape[1]
                                        or y2 < 0 or y2 >= legal_sites.shape[0]
                                        or target_map[y2, x2] == 0):
                        counter += 1
                    if counter == 2:
                        opposite_free_counter += 1

                if corner_counter == 2:
                    corner_sites.append((x, y))
                if corner_counter <= 2 and opposite_free_counter >= 2:
                    # opposite_free_counter might have to be 2 or more checks might be necessary
                    protruding_sites.append((x, y))

                # diagonal_counter = 0
                # for x2, y2 in ((x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)):
                #     if 0 <= y2 < legal_sites.shape[0] and 0 <= x2 < legal_sites.shape[1] \
                #             and target_map[y2, x2] != 0:
                #         diagonal_counter += 1
                # if protruding_counter <= 2:
                #     if on_map_counter == 3 or on_map_counter == 4:
                #         protruding_sites.append((x, y))
                #     elif on_map_counter == 2:
                #         # also need to check diagonally
                #         if diagonal_counter == 0:
                #             protruding_sites.append((x, y))

    # determine the most CCW site in each row of attachment sites
    most_ccw_row_sites = []
    for y in range(legal_sites.shape[0]):
        for x in range(legal_sites.shape[1]):
            tpl = (x, y)
            if legal_sites[y, x] != 0 and tpl not in corner_sites \
                    and tpl not in protruding_sites and tpl not in most_ccw_row_sites:
                # find all sites in row/column (which might double corner sites at least)
                # there shouldn't be any sites adjacent to both a row and a column because that's what corners are
                # also need to note which orientation the row/column has, which should be easy to determine
                # since all these sites should only be adjacent to one already occupied site
                # note that the directions refer to the position of the adjacent site to the potential attachment site
                adjacent_side = None
                counter = 0
                if 0 <= y - 1 < legal_sites.shape[0] and occupancy_map[y - 1, x] != 0:
                    adjacent_side = "SOUTH"
                    counter += 1
                if 0 <= y + 1 < legal_sites.shape[0] and occupancy_map[y + 1, x] != 0:
                    adjacent_side = "NORTH"
                    counter += 1
                if 0 <= x - 1 < legal_sites.shape[1] and occupancy_map[y, x - 1] != 0:
                    adjacent_side = "WEST"
                    counter += 1
                if 0 <= x + 1 < legal_sites.shape[1] and occupancy_map[y, x + 1] != 0:
                    adjacent_side = "EAST"
                    counter += 1
                if adjacent_side is None:
                    print("Too few adjacent sites for potential row/column attachment site at {}.".format(tpl))
                    continue
                if counter > 1:
                    print("Too many adjacent sites for potential row/column attachment site at {}.".format(tpl))
                    continue

                # this could also just be done in a single direction, but it would be best to set the other
                # positions in the same row/column to 0, so that they don't count in the ongoing loop anymore
                if adjacent_side == "SOUTH" or adjacent_side == "NORTH":
                    # it's a row, therefore look left and right for other blocks
                    row_sites = [tpl]
                    # LEFT
                    x2 = x - 1
                    while x2 >= 0 and legal_sites[y, x2] != 0:
                        row_sites.insert(0, (x2, y))
                        x2 -= 1
                    # RIGHT
                    x2 = x + 1
                    while x2 < legal_sites.shape[1] and legal_sites[y, x2] != 0:
                        row_sites.append((x2, y))
                        x2 += 1

                    # check whether there are any corner sites in this row and if so make all sites illegal
                    reduced_row_sites = [rs for rs in row_sites if rs not in corner_sites]
                    if len(row_sites) != len(reduced_row_sites):
                        # there is some corner site in the list
                        for x2, y2 in reduced_row_sites:
                            legal_sites[y2, x2] = 0
                        print("REMOVED ROW SITES {}".format(reduced_row_sites))
                    else:
                        # find the CCW most site (which is either the first or the last)
                        if not row_information:
                            if adjacent_side == "SOUTH":
                                most_ccw_site = row_sites[0]
                            else:
                                most_ccw_site = row_sites[-1]
                            while most_ccw_site in row_sites:
                                row_sites.remove(most_ccw_site)
                            for x2, y2 in row_sites:
                                legal_sites[y2, x2] = 0
                            most_ccw_row_sites.append(most_ccw_site)
                        else:
                            # if the entire column is supposed to be returned, add all sites in the correct order
                            # to traverse so that correctness of the structure is guaranteed
                            # format: (first site, direction, expected row length)
                            if adjacent_side == "SOUTH":
                                site_info = (row_sites[-1], np.array([-1, 0, 0]), len(row_sites))
                            else:
                                site_info = (row_sites[0], np.array([1, 0, 0]), len(row_sites))
                            # most_ccw_row_sites.append(row_sites if adjacent_side == "NORTH" else row_sites.reverse())
                            if all(site_info[0] != ccw_site[0] for ccw_site in most_ccw_row_sites):
                                most_ccw_row_sites.append(site_info)
                elif adjacent_side == "WEST" or adjacent_side == "EAST":
                    # it's a column, therefore look up and down for other blocks
                    column_sites = [tpl]
                    # DOWN
                    y2 = y - 1
                    while y2 >= 0 and legal_sites[y2, x] != 0:
                        column_sites.insert(0, (x, y2))
                        y2 -= 1
                    # UP
                    y2 = y + 1
                    while y2 < legal_sites.shape[0] and legal_sites[y2, x] != 0:
                        column_sites.append((x, y2))
                        y2 += 1

                    # check whether there are any corner sites in this column and if so make all sites illegal
                    reduced_column_sites = [cs for cs in column_sites if cs not in corner_sites]
                    if len(column_sites) != len(reduced_column_sites):
                        for x2, y2 in reduced_column_sites:
                            legal_sites[y2, x2] = 0
                        print("REMOVED COLUMN SITES {}".format(reduced_column_sites))
                    else:
                        if not row_information:
                            if adjacent_side == "EAST":
                                most_ccw_site = column_sites[0]
                            else:
                                most_ccw_site = column_sites[-1]
                            while most_ccw_site in column_sites:
                                column_sites.remove(most_ccw_site)
                            for x2, y2 in column_sites:
                                legal_sites[y2, x2] = 0
                            most_ccw_row_sites.append(most_ccw_site)
                        else:
                            # if the entire column is supposed to be returned, add all sites in the correct order
                            # to traverse so that correctness of the structure is guaranteed
                            # most_ccw_row_sites.append(column_sites if adjacent_side == "WEST"
                            #                           else column_sites.reverse())
                            if adjacent_side == "EAST":
                                site_info = (column_sites[-1], np.array([0, -1, 0]), len(column_sites))
                            else:
                                site_info = (column_sites[0], np.array([0, 1, 0]), len(column_sites))
                            if all(site_info[0] != ccw_site[0] for ccw_site in most_ccw_row_sites):
                                most_ccw_row_sites.append(site_info)

    # important note on the row/column information: it also is not entirely reliable the way it is handled now
    # since the agent thinks that the row/column ends somewhere it does not end at all
    # one way to amend this would be to use the target map as an indicator of how many sites to count
    # in either case, the agent should follow the row/column direction until there is a site which is in an inner corner
    # (i.e. the location ahead is occupied) or the end of a row/an outer corner (according to the local occupancy map)
    # -> for the latter, it is important that the local occupancy map is updated with the site ahead and to the left of
    #    the agent, so that it is actually guaranteed that this is an outer corner globally as well
    # -> the easiest way to do this thing would therefore probably be to just give the starting point and the direction
    # if there is a block encountered ahead in this row, the entire rest of the row should be considered occupied

    # possible format for information?
    current_attachment_info = {"site": [0, 0, 0], "direction": [0, 0, 0]}

    # the last question then is: how can corner/loop attachment sites be discriminated?
    # -> if there are any sites in the row/column that are in that region?
    # -> maybe those sites should simply be removed
    # -> maybe only if the first site is in that region
    return legal_sites, corner_sites, protruding_sites, most_ccw_row_sites


def legal_attachment_sites_3d(target_map: np.ndarray, occupancy_map: np.ndarray, safety_radius=2):
    # input is supposed to be a 3-dimensional layer

    # for each layer, identify the potential attachment sites in 2D, the 3D version can only get more restrictive
    legal_sites = np.empty_like(target_map, dtype="int64")
    for z in range(target_map.shape[0]):
        legal_sites[z] = legal_attachment_sites(target_map[z], occupancy_map[z])

    # in addition to attaching to adjacent blocks on the next level, attachment where supported from below is also
    # possible; however, it still has to be possible to attach other blocks (row/plane rule?)
    # -> allowed if supported and does not violate row rule on its layer?

    for z in range(1, target_map.shape[0]):
        for y in range(target_map.shape[1]):
            for x in range(target_map.shape[2]):
                # not yet "registered" as legal attachment site and supported from below
                if legal_sites[z, y, x] == 0 and occupancy_map[z, y, x] == 0 \
                        and target_map[z, y, x] >= 1 and occupancy_map[z - 1, y, x] >= 1:
                    legal_sites[z, y, x] = 1
                    # check whether row rule is violated
                    for diff in (-1, 1):
                        counter = 1
                        while 0 <= y + counter * diff < legal_sites.shape[1] \
                                and occupancy_map[z, y + counter * diff, x] == 0 \
                                and target_map[z, y + counter * diff, x] >= 0:
                            counter += 1
                        if counter > 1 and 0 <= y + counter * diff < legal_sites.shape[1] \
                                and occupancy_map[z, y + counter * diff, x] >= 0 \
                                and target_map[z, y + counter * diff, x] >= 0:
                            # have encountered a block already in this row
                            legal_sites[z, y, x] = 0
                            break

                        counter = 1
                        while 0 <= x + counter * diff < legal_sites.shape[2] \
                                and occupancy_map[z, y, x + counter * diff] == 0 \
                                and target_map[z, y, x + counter * diff] >= 0:
                            counter += 1
                        if counter > 1 and 0 <= x + counter * diff < legal_sites.shape[2] \
                                and occupancy_map[z, y, x + counter * diff] >= 0 \
                                and target_map[z, y, x + counter * diff] >= 0:
                            # have encountered a block already in this row
                            legal_sites[z, y, x] = 0
                            break
                if legal_sites[z, y, x] > 0:
                    # check whether quadcopter can still get to other sites, i.e. whether there is any layer below
                    # the current one where a block has to be placed within a 2 block radius of the (x, y) position
                    done = False
                    for z2 in range(z):
                        if not done:
                            for y2 in range(y - safety_radius, y + safety_radius + 1):
                                if not done:
                                    for x2 in range(x - safety_radius, x + safety_radius + 1):
                                        # check if: not the same (x, y) position, within range,
                                        # block supposed to go there and block not yet there
                                        if not (y2 == y and x2 == x) \
                                                and 0 <= y2 < legal_sites.shape[1] \
                                                and 0 <= x2 < legal_sites.shape[2] \
                                                and target_map[z2, y2, x2] >= 1 \
                                                and occupancy_map[z2, y2, x2] == 0:
                                            legal_sites[z, y, x] = 0
                                            done = True
                                            break

    return legal_sites


def neighbourhood(arr: np.ndarray, position, flatten=False):
    # position given as [x, y, z] or [x, y]
    x = position[0]
    y = position[1]
    x_lower = max(x - 1, 0)
    x_upper = min(x + 2, arr.shape[1 if len(position) == 2 else 2])
    y_lower = max(y - 1, 0)
    y_upper = min(y + 2, arr.shape[0 if len(position) == 2 else 1])
    if len(position) == 3:
        z = position[2]
        z_lower = max(z - 1, 0)
        z_upper = min(z + 2, arr.shape[0])
        result = arr[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        return result.flatten() if flatten else result
    else:
        result = arr[y_lower:y_upper, x_lower:x_upper]
        return result.flatten() if flatten else result


def ccw_angle_and_distance(point, origin=(0, 0), ref_point=(0, 1)):
    vector = np.array([point[0] - origin[0], point[1] - origin[1]])
    len_vector = sum(np.sqrt(vector ** 2))
    ref_vector = np.array([ref_point[0] - origin[0], ref_point[1] - origin[1]])
    len_ref_vector = sum(np.sqrt(ref_vector ** 2))

    if len_vector == 0 or len_ref_vector == 0:
        return -np.pi, 0

    signed_angle = np.arctan2(vector[1], vector[0]) - np.arctan2(ref_vector[1], ref_vector[0])
    if signed_angle < 0:
        angle = 2 * np.pi + signed_angle
    else:
        angle = signed_angle

    return angle, len_vector

