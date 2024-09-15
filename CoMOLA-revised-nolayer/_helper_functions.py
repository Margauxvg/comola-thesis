import numpy as np
from collections import deque
from itertools import combinations
import os
import config as cfg


def count_cells(matrix, target_values):
    flattened_matrix = matrix.ravel()
    count = np.sum(np.isin(flattened_matrix, target_values))
    return count


def calculate_area(cell_count, cell_size):
    """Calculate the area of one patch."""
    return cell_count * pow(cell_size, 2)


def calculate_perimeter(cell_combinations, target_index):
    """Calculate the perimeter of one patch."""
    perimeter = sum((x != y) and (x == target_index or y == target_index) for x, y in cell_combinations)
    return perimeter

def get_all_cell_combinations(matrix):
    """Get all combinations of cells within a matrix."""
    rows, cols = len(matrix), len(matrix[0])
    combinations = []

    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                horizontal_comb = (matrix[i][j], matrix[i][j + 1])
                combinations.append(horizontal_comb)
            if i < rows - 1:
                vertical_comb = (matrix[i][j], matrix[i + 1][j])
                combinations.append(vertical_comb)

            if cfg.mapConfig.four_neighbours == False:
                if i < rows - 1 and j < cols - 1:
                    right_diagonal_comb = (matrix[i][j], matrix[i + 1][j + 1])
                    combinations.append(right_diagonal_comb)
                if i != 0 and j != 0:
                    left_diagonal_comb = (matrix[i][j], matrix[i - 1][j - 1])
                    combinations.append(left_diagonal_comb)
    return combinations


def get_file_in_current_directory(file_name, file):
    current_directory = os.path.dirname(os.path.abspath(file))
    return os.path.join(current_directory, file_name)


def open_file(file_path):
    return open(file_path, "w")


def load_file(file, skiprows, dtype):
    if os.path.isfile(file) and os.path.getsize(file) > 0:
        return np.loadtxt(fname=file, skiprows=skiprows, dtype=dtype)
    else:
        print("File does not exist or is empty")
        return None


def update_file(data, file_path):
    """Update a file with some data"""
    f = open_file(file_path)
    f.write(str(data))
    f.close

# def min_cells_between_patches(matrix, patch1_index, patch2_index):
#     """ Calculate the minimum number of cells separating two specific patches in a matrix using BFS."""
#     def is_valid(row, col):
#         return 0 <= row < len(matrix) and 0 <= col < len(matrix[0])

#     def bfs(start, target):
#         visited = set()
#         queue = deque([(start, 0)])  # Queue stores the cell position and distance

#         while queue:
#             (current_row, current_col), distance = queue.popleft()
#             visited.add((current_row, current_col))

#             if matrix[current_row, current_col] == target:
#                 return distance

#             for neighbor in [(current_row - 1, current_col), (current_row + 1, current_col),
#                              (current_row, current_col - 1), (current_row, current_col + 1)]:
#                 neighbor_row, neighbor_col = neighbor
#                 if is_valid(neighbor_row, neighbor_col) and (neighbor_row, neighbor_col) not in visited:
#                     queue.append(((neighbor_row, neighbor_col), distance + 1))

#         return -1  # Patches are not connected

#     # Find a cell in each patch to start the BFS
#     start_cell = np.argwhere(matrix == patch1_index)[0]
#     target_cell = np.argwhere(matrix == patch2_index)[0]

#     distance = bfs(start_cell, target_cell)
#     return distance


# def get_unique_double_combinations(input_list):
#     """ Get all unique double combinations for values in a list."""
#     return list(combinations(input_list, 2))


# def get_landuse_patches(patch_info, landuse_index):
#     """Get the list of patch ids of a specific landuse"""

#     patches = [index for index, patch in enumerate(patch_info) if patch[2] == landuse_index]
#     # The id of every patch corresponds to their index in the info matrix. For instance the patch with id = 1 will be the frist element of the patch_info matrix with index = 0.
#     return [value + 1 for value in patches]


# def calculate_landuse_patches_mean_distance(landuse_index, matrix, patch_info):
#     """Calculate the mean distance of the minimum distance among patches of the same land use"""
#     patches = get_landuse_patches(patch_info, landuse_index)
#     patches_to_compare = get_unique_double_combinations(patches)
#     min_distances = []

#     for patch1_target, patch2_target in patches_to_compare:
#         distance = min_cells_between_patches(matrix, patch1_target, patch2_target)
#         min_distances.append(distance)

#     return sum(min_distances)/len(min_distances)


# def filter_patchinfo(landuse_list, input_matrix):
#     """Return the patch info for all patches whose land use id is in the provided list"""
#     filtered_matrix = [row for row in input_matrix if row[0] in landuse_list]
#     return filtered_matrix


# def sum_column(matrix, column_index):
#     """Calculate the sum of one of the patch info provided (number of cells, perimeter) for a set of patches"""
#     column_sum = sum(row[column_index] for row in matrix)
#     return column_sum




