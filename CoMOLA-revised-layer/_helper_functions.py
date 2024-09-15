import numpy as np
import os
import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.colors import ListedColormap


# Functions to manipulate matrix data

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

def calculate_shannon_diversity(proportion):
    shannon_index = -np.sum(proportion * np.log2(proportion))
    return shannon_index

# Functions for file manipulation

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


# Functions to analysie output data
    
def calculate_mean(set):
    '''Calculate the mean of a set in the form [int1, int2, int3, ..., intn]. Used to calculate the time performance'''
    set_mean = np.mean(set)
    return set_mean

def create_bar_plot(data, title, versions):
    '''Create a custom bar plot. The versions are an array of strings representing the bar labels.'''
    colors = ['skyblue', 'salmon', 'lightgreen']
    
    plt.figure(figsize=(8, 6))
    
    for i, (data, color) in enumerate(zip(data, colors)):
        plt.bar(i, data, color=color, width=0.4, label=versions[i])
        plt.text(i, data + 1, '{:.2f}'.format(data), ha='center', va='bottom', fontsize=10)

    plt.ylabel('Mean Execution Time (seconds)')
    plt.title(title, fontsize=10, style='italic')
    plt.suptitle("Mean Execution Time of Algorithm Versions", fontsize=14, fontweight='bold')
    plt.xticks(range(len(versions)), versions)
    plt.tight_layout()
    plt.show()

def perform_anova_and_tukey(set1, set2, set3, title):
    '''Perform an ANOVA and Turkey test on 3 sets of data '''

    # Combine data and labels for Tukey's test
    data = np.concatenate([set1, set2, set3])
    labels = (['Original Code'] * len(set1)) + (['Post Processing'] * len(set2)) + (['Without Postprocessing'] * len(set3))

    # Statistical comparison (ANOVA)
    f_statistic, p_value = f_oneway(set1, set2, set3)

    if p_value < 0.05:
        print(f"ANOVA: There is a significant difference between at least one pair of means for {title}.")
        # Perform Tukey's HSD post-hoc test
        tukey_result = pairwise_tukeyhsd(data, labels, alpha=0.05)
        print(tukey_result)
    else:
        print(f"ANOVA: No significant difference between the groups for {title}.")

def normalize_fitness_arrays(*fitness_arrays):
    normalized_fitness_arrays = []
    mean_values = []
    std_dev_values = []

    for fitness_arr in fitness_arrays:
        # Calculate mean and standard deviation for each column
        means = np.mean(fitness_arr, axis=0)
        std_devs = np.std(fitness_arr, axis=0)

        # Store mean and standard deviation values
        mean_values.append(means)
        std_dev_values.append(std_devs)

        # Perform Z-score normalization for each column
        normalized_arr = (fitness_arr - means) / std_devs
        normalized_fitness_arrays.append(normalized_arr)

    return mean_values, std_dev_values, normalized_fitness_arrays

def plot_multiple_3d_scatter(*data_sets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors if needed

    descriptions = [
        "Revised algorithm with preprocessing layer",
        "Revised algorithm without preprocessing layer",
        "Original algorithm"
    ]

    for i, data_set in enumerate(data_sets):
        x_vals, y_vals, z_vals = np.array(data_set).T  # Transpose the data set for correct unpacking
        ax.scatter(x_vals, y_vals, z_vals, c=colors[i], label=descriptions[i])

    ax.set_xlabel('Compactness')
    ax.set_ylabel('Compatibility')
    ax.set_zlabel('Economy')
    ax.legend()

    plt.show()

def pareto_dominance_maximize(normalized_fitness_values):
    num_solutions = len(normalized_fitness_values)
    domination_count = np.zeros(num_solutions, dtype=int)

    for i in range(num_solutions):
        for j in range(num_solutions):
            if i != j:
                is_dominated = True
                for k in range(len(normalized_fitness_values[i])):
                    if normalized_fitness_values[i][k] <= normalized_fitness_values[j][k]:
                        is_dominated = False
                        break
                if is_dominated:
                    domination_count[j] += 1

    pareto_front = [index for index, count in enumerate(domination_count) if count == 0]
    return pareto_front

def find_best_fitness_set_single(set1, set2, set3):
    pareto_front_set1 = pareto_dominance_maximize(set1)
    pareto_front_set2 = pareto_dominance_maximize(set2)
    pareto_front_set3 = pareto_dominance_maximize(set3)

    num_pareto_front_set1 = len(pareto_front_set1)
    num_pareto_front_set2 = len(pareto_front_set2)
    num_pareto_front_set3 = len(pareto_front_set3)

    max_pareto_front = max(num_pareto_front_set1, num_pareto_front_set2, num_pareto_front_set3)

    if max_pareto_front == num_pareto_front_set1:
        best_set = set1
        best_set_name = "Revised-NoLayer"
        best_position = pareto_front_set1[0]
    elif max_pareto_front == num_pareto_front_set2:
        best_set = set2
        best_set_name = "Revised-Layer"
        best_position = pareto_front_set2[0]
    else:
        best_set = set3
        best_set_name = "Original"
        best_position = pareto_front_set3[0]

    return best_set[best_position], best_set_name, best_position

def find_best_fitness_sets_with_positions(*sets):
    def find_best_solution(pareto_front):
        if len(pareto_front) > 0:
            return pareto_front[0]
        else:
            return None

    results = []
    for idx, s in enumerate(sets, start=1):
        pareto_front = pareto_dominance_maximize(s)
        best_position = find_best_solution(pareto_front)
        best_solution = None if best_position is None else s[best_position]
        results.append(("Set {}".format(idx), best_solution, best_position))

    return results

def plot_land_use_matrix(matrix, title, subtitle):
    # Define colors for land use types
    colors = ['lightgreen', 'green', 'darkgreen', 'sandybrown', 'sienna', 'darkorange', 'yellow', 'lightblue', 'royalblue', 'darkblue']

    # Create a colormap based on the defined colors
    cmap = ListedColormap(colors)

    # Plot the matrix with cell borders and without axis values
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap=cmap, vmin=0, vmax=9, extent=[0, 10, 0, 10], origin='lower')
    plt.grid(color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])

    # Create a legend outside the plot
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(10)]
    plt.legend(legend_handles, [str(i) for i in range(10)], title='Land Use Index', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.title(subtitle, fontsize=10, style='italic')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
