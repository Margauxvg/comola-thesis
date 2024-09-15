import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

def normalize_fitness_arrays(*fitness_arrays):
    normalized_fitness_arrays = []
    mean_values = []
    std_dev_values = []

    for fitness_arr in fitness_arrays:
        # Extract the fourth column (model type)
        model_type_column = fitness_arr[:, -1]

        # Calculate mean and standard deviation for the first three columns
        means = np.mean(fitness_arr[:, :-1], axis=0)
        std_devs = np.std(fitness_arr[:, :-1], axis=0)

        # Store mean and standard deviation values
        mean_values.append(means)
        std_dev_values.append(std_devs)

        # Perform Z-score normalization for the first three columns
        normalized_columns = (fitness_arr[:, :-1] - means) / std_devs

        # Combine normalized columns with the original fourth column
        normalized_arr = np.column_stack((normalized_columns, model_type_column))
        normalized_fitness_arrays.append(normalized_arr)

    return normalized_fitness_arrays

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
def plot_2d_scatter_with_size(*data_sets):
    fig, ax = plt.subplots()

    descriptions = [
        "Revised algorithm with preprocessing layer",
        "Revised algorithm without preprocessing layer",
        "Original algorithm"
    ]

    for i, data_set in enumerate(data_sets):
        x_vals, y_vals, z_vals = np.array(data_set).T
        sizes = 100 * (z_vals - min(z_vals)) / (max(z_vals) - min(z_vals)) + 10

        ax.scatter(x_vals, y_vals, s=sizes, alpha=0.7, label=descriptions[i])

    ax.set_xlabel('Compactness')
    ax.set_ylabel('Compatibility')
    ax.set_title('Scatter Plot of the Normalized Fitness Scores: Large Experiment', fontweight='bold')  # Bold title
    
    ax.legend()
    
    plt.show()
def pareto_dominance_higher_is_better(normalized_fitness_values):
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
    pareto_front_set1 = pareto_dominance_higher_is_better(set1)
    pareto_front_set2 = pareto_dominance_higher_is_better(set2)
    pareto_front_set3 = pareto_dominance_higher_is_better(set3)

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
        pareto_front = pareto_dominance_higher_is_better(s)
        best_position = find_best_solution(pareto_front)
        best_solution = None if best_position is None else s[best_position]
        results.append(("Set {}".format(idx), best_solution, best_position))

    return results
def plot_land_use_matrix(matrix, title, subtitle):
    # Define colors for land use types
    colors = ['lightgreen', 'green', 'sandybrown', 'sienna', 'yellow', 'lightblue', 'royalblue', 'darkblue']

    # Create a colormap based on the defined colors
    cmap = ListedColormap(colors)

    # Plot the matrix with cell borders and without axis values
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap=cmap, vmin=1, vmax=8, extent=[0, 10, 0, 10], origin='lower')
    plt.grid(color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])

    # Create a legend outside the plot
    legend_uses = ['Arid', 'Recreational', 'Mixed residential - commercial', 'Medical', 'Religious', 'Educational', 'Commercial', 'Public Amenities']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(8)]
    plt.legend(legend_handles, [legend_uses[i] for i in range(8)], title='Land Use Index', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.title(subtitle, fontsize=10, style='italic')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
def plot_2d_scatter_with_size_and_color(data_array):
    # Function to check if a solution is Pareto optimal
    def is_pareto_optimal(current_solution, all_solutions):
        return all(
            any(current_solution[i] >= other_solution[i] for i in range(len(current_solution) - 1))
            for other_solution in all_solutions
        )

    # Filter Pareto front solutions
    pareto_front = [
        solution for solution in data_array if is_pareto_optimal(solution, data_array)
    ]

    # Convert pareto_front to a NumPy array
    pareto_front = np.array(pareto_front)

    # Extract data for plotting
    x_vals = pareto_front[:, 0]  # Assuming the first column corresponds to x values
    y_vals = pareto_front[:, 1]  # Assuming the second column corresponds to y values
    z_vals = pareto_front[:, 2]  # Assuming the third column corresponds to z values (size)
    model_types = pareto_front[:, 3]  # Assuming the fourth column corresponds to model types

    # Assign colors based on model type
    colors = np.where(model_types == 1, 'lightgreen', np.where(model_types == 2, 'salmon', 'skyblue'))

    # Plotting
    plt.scatter(x_vals, y_vals, s=[100 * size for size in z_vals], c=colors, alpha=0.7)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('2D Scatter Plot with Size and Color')
    plt.show()

def plot_3d_scatter_with_size_and_color(data_array):
    # Function to check if a solution is Pareto optimal
    def is_pareto_optimal(current_solution, all_solutions):
        return all(
            any(current_solution[i] >= other_solution[i] for i in range(len(current_solution) - 1))
            for other_solution in all_solutions
        )

    # Filter Pareto front solutions
    pareto_front = [
        solution for solution in data_array if is_pareto_optimal(solution, data_array)
    ]

    # Convert pareto_front to a NumPy array
    pareto_front = np.array(pareto_front)
    # Extract data for plotting
    # Extract data for plotting
    x_vals = pareto_front[:, 0]  # Assuming the first column corresponds to x values
    y_vals = pareto_front[:, 1]  # Assuming the second column corresponds to y values
    z_vals = pareto_front[:, 2]  # Assuming the third column corresponds to z values (size)
    model_types = pareto_front[:, 3]  # Assuming the fourth column corresponds to model types

    # Assign colors based on model type
    colors = np.where(model_types == 1, 'lightgreen', np.where(model_types == 2, 'salmon', 'skyblue'))

    # Ensure non-negative and valid size values
    # valid_sizes = np.clip(z_vals, 0, None)

    # Plotting in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, c=colors, alpha=0.7)

    ax.set_xlabel('Compactness')
    ax.set_ylabel('Compatibility')
    ax.set_zlabel('Economic Benefits')

     # Add legend with labels
    legend_labels = {
        'lightgreen': 'Revised without pre-processing layer',
        'salmon': 'Revised with pre-processing layer',
        'skyblue': 'Original'
    }

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                      for color, label in legend_labels.items()]

    ax.legend(handles=legend_handles, loc='upper right',bbox_to_anchor=(1, 1))

    plt.show()

fitness_revised_nolayer_10_2=np.array([
[82,130,57,1],
[117,151,56,1],
[94,153,41,1],
[90,152,42,1],
[83,129,40,1],
[81,126,46,1],
[88,126,45,1],
[83,130,37,1],
[101,137,58,1],
[78,134,40,1],
[75,112,42,1],
[78,134,40,1],
[82,132,38,1],
[76,112,41,1],
[72,118,42,1],
[87,121,35,1],
[85,121,38,1],
[88,126,37,1],
[81,130,44,1],
[85,127,36,1],
[85,121,38,1],
[85,126,32,1],
[73,115,45,1],
[84,130,34,1],
[84,114,41,1],
[85,121,37,1],
[78,104,42,1],
[91,138,43,1],
[81,126,48,1],
[82,140,33,1],
[82,119,45,1],
[78,133,34,1],
[84,136,32,1],
[86,132,39,1],
[83,115,41,1],
[76,124,47,1],
[74,119,48,1],
[85,125,30,1],
[79,124,34,1],
[82,118,38,1],
[79,115,41,1],
[87,115,29,1],
[82,128,29,1],
[78,128,40,1],
[79,127,46,1],
[81,133,36,1],
[83,133,35,1],
[78,129,43,1],
[79,129,38,1],
[83,131,36,1],
[80,120,41,1],
[83,136,34,1],
[87,130,42,1],
[83,131,37,1],
[75,123,46,1],
[79,126,42,1],
[83,120,45,1],
[84,121,36,1],
[84,124,34,1],
[83,129,41,1],
])

fitness_revised_nolayer_50_10=np.array([
[92,146,43,1],
[102,144,44,1],
[97,135,49,1],
[95,144,46,1],
[96,142,58,1],
[100,134,47,1],
[92,143,49,1],
[103,143,46,1],
[91,125,49,1],
[99,133,35,1],
[91,138,48,1],
[97,135,29,1],
[93,126,41,1],
[97,134,33,1],
[91,145,39,1],
[91,140,42,1],
[93,120,44,1],
[92,118,51,1],
[90,148,55,1],
[92,143,39,1],
[92,127,43,1],
[93,139,35,1],
[82,118,56,1],
[97,145,53,1],
[92,148,53,1],
[91,137,64,1],
[95,145,56,1],
[97,147,50,1],
[91,152,48,1],
[95,142,62,1],
[102,131,40,1],
[97,141,61,1],
[90,151,50,1],
[97,135,40,1],
[95,140,37,1],
[86,140,52,1],
[82,148,45,1],
[91,151,44,1],
[97,139,31,1],
[95,137,45,1],
[90,144,51,1],
[91,142,48,1],
[96,139,34,1],
[86,126,54,1],
[94,139,39,1],
[84,145,48,1],
[93,147,45,1],
[96,135,43,1],
[87,123,52,1],
[81,139,48,1],
[82,126,49,1],
[88,125,47,1],
[92,145,42,1],
[85,135,47,1],
[82,137,48,1],
[78,133,51,1],
[92,145,42,1],
[94,140,41,1],
[88,144,44,1],
[97,139,38,1],
[90,127,46,1],
[98,142,36,1],
[85,136,46,1],
[89,138,44,1],
[100,149,44,1],
[99,145,51,1],
[97,144,52,1],
[95,146,49,1],
[98,151,47,1],
[96,147,48,1],
[81,130,59,1],
[95,142,58,1],
[85,142,48,1],
[90,131,59,1],
[91,130,59,1],
[91,130,59,1],
[89,134,58,1],
[93,142,47,1],
[94,148,44,1],
[90,146,47,1],
[94,141,54,1],
[95,134,47,1],
[101,144,45,1],
[88,134,59,1],
[94,150,43,1],
[94,146,46,1],
[93,135,48,1],
[96,121,34,1],
[95,120,37,1],
[87,148,44,1],
[91,138,55,1],
[85,136,60,1],
[88,142,47,1],
[90,132,58,1],
[88,135,57,1],
[89,134,59,1],
[93,131,58,1],
[92,134,55,1],
[89,139,53,1],
[85,134,62,1],
[90,144,41,1],
[97,142,59,1],
[89,136,60,1],
[89,142,46,1],
[95,141,47,1],
[85,132,54,1],
[84,141,51,1],
[89,129,55,1],
[77,131,58,1],
[96,131,44,1],
[85,142,48,1],
[96,127,52,1],
[90,144,40,1],
[96,130,47,1],
[86,140,50,1],
[89,128,57,1],
[94,138,49,1],
[88,139,51,1],
[89,131,50,1],
[85,142,48,1],
[93,146,38,1],
[79,146,44,1],
[77,144,45,1],
[83,141,56,1],
[95,135,48,1],
[91,130,54,1],
[90,123,57,1],
[88,146,41,1],
[86,147,43,1],
])


#2
fitness_revised_layer_10_2=np.array([
[74,109,36,2],
[78,120,32,2],
[70,131,33,2],
[70,131,33,2],
[70,111,40,2],
[73,118,34,2],
[75,108,34,2],
[73,116,36,2],
[75,108,34,2],
[71,118,39,2],
[79,114,40,2],
[81,120,31,2],
[80,111,38,2],
[77,120,42,2],
[79,122,39,2],
[83,127,53,2],
[80,120,39,2],
[79,120,51,2],
[84,117,30,2],
[74,130,37,2],
[81,121,35,2],
[76,127,35,2],
[80,122,39,2],
[77,115,44,2],
[83,117,39,2],
[85,120,38,2],
[87,119,35,2],
[80,123,49,2],
[79,127,38,2],
[83,112,36,2],
[78,132,40,2],
[85,113,32,2],
[77,120,52,2],
[79,127,38,2],
[77,116,39,2],
[76,118,41,2],
[78,110,39,2],
[73,111,44,2],
[70,121,40,2],
[80,123,37,2],
[71,119,40,2],
[86,130,31,2],
[78,129,38,2],
[81,121,37,2],
[81,137,31,2],
[79,139,33,2],
[81,137,31,2],
[71,111,44,2],
[80,134,34,2],
[84,132,36,2],
[75,112,43,2],
[81,117,42,2],
[78,118,37,2],
[84,125,49,2],
[78,113,39,2],
[78,113,39,2],
[76,107,42,2],
[78,124,38,2],
[73,123,43,2],
[80,109,39,2],
[75,115,42,2],
])

fitness_revised_layer_50_10=np.array([
[100,134,52,2],
[92,139,48,2],
[92,145,39,2],
[101,128,49,2],
[97,137,54,2],
[96,141,38,2],
[92,141,45,2],
[89,138,55,2],
[104,127,46,2],
[92,142,40,2],
[90,140,50,2],
[88,144,56,2],
[91,142,48,2],
[90,145,30,2],
[84,139,57,2],
[86,151,43,2],
[86,137,65,2],
[87,156,38,2],
[85,141,48,2],
[85,142,46,2],
[93,140,31,2],
[90,124,61,2],
[90,135,49,2],
[85,141,48,2],
[88,135,65,2],
[81,145,46,2],
[89,139,53,2],
[84,142,51,2],
[90,141,46,2],
[94,136,33,2],
[97,133,28,2],
[86,141,47,2],
[91,130,53,2],
[98,132,48,2],
[99,126,44,2],
[96,136,48,2],
[81,140,50,2],
[82,136,53,2],
[97,135,44,2],
[94,134,54,2],
[86,130,56,2],
[90,137,48,2],
[91,137,38,2],
[89,149,46,2],
[84,139,47,2],
[78,130,56,2],
[89,145,40,2],
[91,133,38,2],
[89,126,51,2],
[86,138,55,2],
[91,132,47,2],
[84,119,57,2],
[87,136,53,2],
[82,150,32,2],
[91,124,52,2],
[89,129,49,2],
[90,144,36,2],
[88,137,49,2],
[88,141,48,2],
[89,141,45,2],
[88,147,34,2],
[96,131,45,2],
[83,124,60,2],
[90,136,47,2],
[85,140,52,2],
[90,124,56,2],
[82,129,58,2],
[88,144,43,2],
[93,142,41,2],
[92,146,60,2],
[100,141,49,2],
[92,146,48,2],
[86,132,56,2],
[100,145,43,2],
[81,124,60,2],
[81,124,60,2],
[100,139,44,2],
[87,139,49,2],
[102,141,43,2],
[101,144,41,2],
[88,137,54,2],
[90,129,56,2],
[99,138,52,2],
[85,124,58,2],
[93,147,46,2],
[100,147,50,2],
[90,146,51,2],
[97,148,48,2],
[97,148,48,2],
[92,143,56,2],
[99,144,51,2],
[98,145,53,2],
[97,140,57,2],
[95,139,60,2],
[99,137,58,2],
[86,147,52,2],
[99,144,51,2],
[90,146,51,2],
[88,127,59,2],
[96,124,58,2],
[96,126,57,2],
[88,142,54,2],
[91,129,54,2],
[94,134,45,2],
[83,130,57,2],
[92,141,46,2],
[93,152,40,2],
[89,127,55,2],
[94,147,41,2],
[98,124,55,2],
[95,132,48,2],
[89,144,52,2],
[93,138,45,2],
[94,140,43,2],
[89,150,42,2],
[91,142,47,2],
[88,132,55,2],
[89,138,54,2],
[92,137,50,2],
[95,153,38,2],
[87,146,45,2],
[90,132,51,2],
[89,127,55,2],
[93,136,48,2],
[81,146,49,2],
[96,151,40,2],
[85,129,59,2],
[87,150,37,2],
[86,151,38,2],
[97,134,38,2],
[82,149,46,2],
[87,146,42,2],
[87,135,55,2],
[92,145,54,2],
[100,128,43,2],
[84,146,51,2],
[84,148,45,2],
[97,133,46,2],
[86,142,62,2],
[88,134,57,2],
[86,147,43,2],
[91,139,41,2],
[89,130,53,2],
[87,126,57,2],
[89,144,38,2],
[81,133,52,2],
[94,138,51,2],
[87,131,53,2],
[81,131,54,2],
[90,129,56,2],
[84,139,44,2],
[92,143,35,2],
[81,126,59,2],
[89,141,43,2],
[83,139,45,2],
])

#3
fitness_original_10_2=np.array([
[214,594,28750,3],
[290,631,25000,3],
[278,620,26875,3],
[228,593,28125,3],
[290,632,23750,3],
[226,605,28125,3],
[240,606,27500,3],
[282,613,25625,3],
[250,595,28125,3],
[260,616,21250,3],
[246,621,23125,3],
[252,632,20000,3],
[254,628,21875,3],
[238,626,23125,3],
[264,622,19375,3],
[262,642,15000,3],
[228,617,26875,3],
[256,611,24375,3],
[244,613,23750,3],
[254,603,26250,3],
[238,611,31250,3],
[258,619,21250,3],
[276,594,27500,3],
[230,612,27500,3],
[236,607,30625,3],
[296,596,19375,3],
[252,622,24375,3],
[270,593,24375,3],
[216,608,23750,3],
[220,609,23125,3],
[240,605,33125,3],
[268,592,28125,3],
[266,603,25625,3],
[250,607,21875,3],
[242,619,22500,3],
[240,617,24375,3],
[220,627,22500,3],
[244,605,17500,3],
[222,602,24375,3],
[272,600,26875,3],
[234,625,18750,3],
[206,602,26875,3],
[216,613,25625,3],
[210,601,26250,3],
[250,615,24375,3],
[230,586,29375,3],
[226,595,29375,3],
[222,596,29375,3],
[244,611,25625,3],
[292,628,31875,3],
[310,627,31875,3],
[254,613,20625,3],
[272,611,25000,3],
[292,607,28125,3]])

fitness_original_50_10=np.array([
[292,646,28750,3],
[328,632,25625,3],
[292,645,30000,3],
[332,635,18750,3],
[266,600,35000,3],
[268,637,31875,3],
[316,645,19375,3],
[290,635,31250,3],
[302,634,28125,3],
[294,618,33125,3],
[322,625,30625,3],
[324,634,25000,3],
[328,635,21875,3],
[306,613,34375,3],
[308,636,21250,3],
[300,629,31875,3],
[274,614,34375,3],
[276,652,24375,3],
[302,589,26875,3],
[312,652,21875,3],
[306,645,23125,3],
[328,635,19375,3],
[300,637,25625,3],
[302,647,23125,3],
[264,605,31875,3],
[258,632,30000,3],
[270,627,28125,3],
[290,659,23125,3],
[296,623,26875,3],
[270,589,34375,3],
[290,651,26875,3],
[250,616,31250,3],
[308,636,25625,3],
[304,660,20000,3],
[314,644,25000,3],
[290,629,27500,3],
[262,621,29375,3],
[278,609,30625,3],
[284,618,28750,3],
[278,640,27500,3],
[294,601,27500,3],
[334,590,18750,3],
[318,632,18125,3],
[322,614,24375,3],
[326,614,23750,3],
[316,657,21875,3],
[316,611,25625,3],
[254,613,36875,3],
[324,622,21875,3],
[284,638,26875,3],
[252,615,36875,3],
[328,621,12500,3],
[314,618,25625,3],
[246,641,25000,3],
[304,629,32500,3],
[268,640,29375,3],
[244,618,35625,3],
[294,619,33750,3],
[262,632,32500,3],
[294,645,24375,3],
[288,637,31875,3],
[336,611,17500,3],
[260,634,31250,3],
[236,631,31875,3],
[256,640,28750,3],
[248,639,30625,3],
[274,632,26250,3],
[280,618,31250,3],
[290,609,26875,3],
[290,620,24375,3],
[290,608,30625,3],
[272,635,28125,3],
[270,630,33750,3],
[242,643,28750,3],
[288,623,26875,3],
[258,663,21875,3],
[274,666,20000,3],
[304,645,18750,3],
[254,644,26875,3],
[240,646,28750,3],
[294,647,23125,3],
[268,639,29375,3],
[266,637,30000,3],
[254,660,23125,3],
[256,635,30625,3],
[280,631,27500,3],
[240,646,28750,3],
[270,642,26250,3],
[280,619,30000,3],
[250,659,23750,3],
[238,647,27500,3],
[272,642,24375,3],
[274,628,30625,3],
[286,659,21875,3],
[288,632,25625,3],
[310,639,32500,3],
[316,635,35000,3],
[322,644,30625,3],
[306,645,33125,3],
[318,634,33750,3],
[340,646,30000,3],
[316,649,31250,3],
[308,646,31875,3],
[288,613,35625,3],
[302,650,31250,3],
[286,648,33125,3],
[326,635,28750,3],
[288,652,25625,3],
[326,627,37500,3],
[300,638,35625,3],
[332,627,32500,3],
[298,645,34375,3],
[296,634,37500,3],
[292,649,30000,3],
[274,613,41250,3],
[340,622,36875,3],
[322,647,33125,3],
[324,635,30000,3],
[320,628,35625,3],
[302,623,38125,3],
[338,625,35625,3],
[282,650,26250,3],
[274,643,35000,3],
[342,631,30625,3],
[328,636,34375,3],
[308,655,32500,3],
[324,644,38125,3],
[266,648,32500,3],
[298,640,35000,3],
[290,649,31250,3],
[290,641,32500,3],
[306,637,26875,3],
[282,643,35000,3],
[324,634,36875,3],
[288,633,41250,3],
[310,637,40625,3],
[318,649,35000,3],
[282,639,38125,3],
[314,661,28750,3],
[302,645,37500,3],
[296,653,33125,3],
[340,628,34375,3],
[316,657,33750,3],
[356,664,31250,3],
[314,660,36875,3],
[310,648,39375,3],
[326,662,31875,3],
[332,652,38750,3],
[310,662,36875,3],
[320,653,35625,3],
[296,642,42500,3],
[316,666,33125,3]
])

# Call the function with your fitness arrays
normalized_fitness_arrays = normalize_fitness_arrays(
    fitness_revised_nolayer_10_2,
    fitness_revised_nolayer_50_10,
    fitness_revised_layer_10_2,
    fitness_revised_layer_50_10,
    fitness_original_10_2,
    fitness_original_50_10
)

normalized_revised_nolayer_10_2 = np.array(normalized_fitness_arrays[0])
normalized_revised_nolayer_50_10 = np.array(normalized_fitness_arrays[1])
normalized_revised_layer_10_2 = np.array(normalized_fitness_arrays[2])
normalized_revised_layer_50_10 = np.array(normalized_fitness_arrays[3])
normalized_revised_original_10_2 = np.array(normalized_fitness_arrays[4])
normalized_revised_original_50_10 = np.array(normalized_fitness_arrays[5])


# best_set_10_2, algorithm_name_10_2, position_10_2 = find_best_fitness_set_single(normalized_revised_nolayer_10_2,normalized_revised_layer_10_2, normalized_revised_original_10_2)
# print("The best solution for the 10-2 experiment is {} from position {} of the algorithm: {}".format(best_set_10_2, algorithm_name_10_2, position_10_2))


# best_set_50_10, algorithm_name_50_10, position_50_10 = find_best_fitness_set_single(normalized_revised_nolayer_50_10,normalized_revised_layer_50_10, normalized_revised_original_50_10)
# print("The best solution for the 50-10 experiment is {} from position {} of the algorithm: {}".format(best_set_50_10, algorithm_name_50_10, position_50_10))

# best_solutions = find_best_fitness_sets_with_positions(normalized_revised_nolayer_10_2,normalized_revised_nolayer_50_10,normalized_revised_layer_10_2,normalized_revised_layer_50_10,normalized_revised_original_10_2,normalized_revised_original_50_10)

# for result in best_solutions:
#     print("The best solution for {} is {} at position {}".format(result[0], result[1], result[2]))

best_solution_revised_nolayer_10_2 = [
[6, 6, 6, 6, 6, 3, 3, 6, 7, 7], 
[5, 6, 6, 6, 6, 6, 6, 6, 7, 8], 
[5, 6, 6, 6, 6, 6, 6, 6, 3, 3], 
[2, 6, 6, 6, 8, 8, 3, 3, 3, 3], 
[2, 7, 7, 5, 1, 8, 8, 7, 3, 3], 
[7, 7, 7, 7, 7, 8, 8, 7, 3, 3], 
[7, 7, 7, 7, 7, 8, 7, 3, 3, 3], 
[7, 7, 7, 7, 1, 1, 7, 3, 3, 3], 
[3, 3, 7, 7, 1, 1, 7, 3, 3, 3], 
[3, 7, 7, 7, 1, 2, 2, 3, 3, 3]
]

best_solution_revised_nolayer_50_10 = [
[6, 6, 6, 7, 7, 5, 5, 6, 6, 6], 
[1, 6, 6, 7, 6, 6, 6, 6, 7, 8], 
[1, 1, 7, 7, 6, 6, 6, 6, 6, 6], 
[4, 1, 7, 7, 8, 8, 6, 6, 6, 6], 
[4, 7, 7, 2, 1, 8, 8, 6, 3, 3], 
[6, 7, 7, 3, 7, 8, 8, 6, 3, 3], 
[5, 5, 5, 5, 7, 8, 7, 7, 3, 3], 
[5, 5, 7, 7, 7, 2, 7, 7, 3, 3], 
[1, 1, 7, 7, 7, 2, 7, 3, 3, 3], 
[1, 7, 7, 7, 7, 5, 5, 3, 3, 3]
]

best_solution_revised_layer_10_2 = [
[5, 7, 7, 3, 3, 1, 1, 7, 1, 1], 
[3, 7, 7, 3, 5, 5, 7, 7, 7, 8], 
[3, 3, 3, 3, 7, 7, 7, 7, 7, 7], 
[6, 3, 7, 7, 8, 8, 7, 7, 7, 7], 
[6, 7, 7, 7, 2, 8, 8, 5, 6, 6], 
[6, 7, 7, 3, 6, 8, 8, 5, 6, 2], 
[3, 5, 5, 5, 6, 8, 7, 1, 6, 2], 
[3, 3, 6, 6, 3, 3, 7, 1, 2, 2], 
[3, 3, 6, 6, 3, 3, 7, 6, 6, 6], 
[3, 6, 7, 7, 3, 3, 3, 1, 1, 1] 
]

best_solution_revised_layer_50_10 = [
[4, 6, 6, 6, 6, 6, 6, 3, 3, 3], 
[6, 6, 6, 6, 3, 3, 3, 7, 7, 8], 
[6, 6, 6, 6, 3, 3, 3, 3, 4, 4], 
[6, 6, 6, 6, 8, 8, 3, 3, 3, 3], 
[6, 7, 7, 2, 7, 8, 8, 3, 3, 3], 
[5, 7, 7, 2, 7, 8, 8, 3, 3, 3], 
[7, 5, 5, 5, 7, 8, 7, 6, 3, 3], 
[7, 7, 7, 7, 6, 4, 7, 6, 3, 3], 
[7, 7, 7, 7, 6, 4, 7, 4, 4, 4], 
[7, 7, 7, 7, 6, 1, 1, 7, 7, 7]
]

best_solution_original_10_2 = [
[7, 6, 6, 6, 6, 3, 3, 5, 5, 5], 
[6, 6, 6, 6, 6, 6, 5, 6, 7, 8], 
[6, 5, 6, 6, 1, 1, 5, 5, 4, 4], 
[1, 5, 6, 6, 8, 8, 3, 3, 3, 3], 
[1, 7, 7, 6, 4, 8, 8, 5, 6, 6], 
[1, 7, 7, 6, 7, 8, 8, 5, 6, 7], 
[7, 7, 7, 7, 7, 8, 7, 3, 6, 7], 
[7, 7, 7, 7, 6, 7, 7, 3, 7, 7], 
[5, 5, 7, 7, 6, 7, 7, 5, 5, 5], 
[5, 7, 7, 7, 6, 3, 3, 5, 5, 5] 
]

best_solution_original_50_10 = [
[4, 6, 6, 2, 2, 3, 3, 3, 4, 4], 
[6, 6, 6, 2, 3, 3, 3, 6, 7, 8], 
[6, 7, 2, 2, 7, 7, 3, 3, 3, 3], 
[2, 7, 6, 6, 8, 8, 7, 7, 7, 7], 
[2, 7, 7, 6, 3, 8, 8, 5, 6, 6], 
[3, 7, 7, 4, 7, 8, 8, 5, 6, 2], 
[3, 7, 7, 7, 7, 8, 7, 7, 6, 2], 
[3, 3, 7, 7, 1, 2, 7, 7, 2, 2], 
[3, 3, 7, 7, 1, 2, 7, 3, 3, 3], 
[3, 7, 7, 7, 1, 4, 4, 3, 3, 3]
]

input_matrix = [
[2, 6, 6, 1, 1, 2, 2, 3, 4, 4],
[4, 6, 6, 1, 5, 5, 3, 6, 7, 8],
[4, 2, 1, 1, 4, 4, 3, 3, 5, 5],
[5, 2, 6, 6, 8, 8, 1, 1, 1, 1],
[5, 7, 7, 2, 5, 8, 8, 4, 3, 3],
[1, 7, 7, 4, 6, 8, 8, 4, 3, 1],
[5, 2, 2, 2, 6, 8, 7, 5, 3, 1],
[5, 5, 6, 6, 3, 4, 7, 5, 1, 1],
[2, 2, 6, 6, 3, 4, 7, 3, 3, 3],
[2, 6, 7, 7, 3, 2, 2, 1, 1, 1]
]

# Plot all best land use maps
# plot_land_use_matrix(best_solution_revised_nolayer_10_2, "Land Use Map - Best Solution of the Revised algorithm with no layer", "Population size = 10, Max generations = 2")
# plot_land_use_matrix(best_solution_revised_nolayer_50_10, "Land Use Map - Best Solution of the Revised algorithm with no layer", "Population size = 50, Max generations = 10")
# plot_land_use_matrix(best_solution_revised_layer_10_2, "Land Use Map - Best Solution of the Revised algorithm with layer", "Population size = 10, Max generations = 2")
# plot_land_use_matrix(best_solution_revised_layer_50_10, "Land Use Map - Best Solution of the Revised algorithm with layer", "Population size = 50, Max generations = 10")
# plot_land_use_matrix(best_solution_original_10_2, "Land Use Map - Best Solution of the Original algorithm", "Population size = 10, Max generations = 2")
# plot_land_use_matrix(best_solution_original_50_10, "Land Use Map - Best Solution of the Original algorithm", "Population size = 50, Max generations = 10")

# plot_land_use_matrix(input_matrix, "Land Use Map - Input", "")
combined_array = np.concatenate((normalized_revised_layer_10_2, normalized_revised_nolayer_10_2, normalized_revised_original_10_2))

# Plotting
plot_3d_scatter_with_size_and_color(combined_array)

combined_array = np.concatenate((normalized_revised_layer_50_10, normalized_revised_nolayer_50_10, normalized_revised_original_50_10))

# Plotting
plot_3d_scatter_with_size_and_color(combined_array)