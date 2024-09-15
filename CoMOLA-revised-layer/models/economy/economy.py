import sys
import os

main_directory = os.path.dirname(os.path.abspath("_helpers_models.py"))
# Append the current directory to sys.path
sys.path.append(main_directory)

import _helper_models as models
import _helper_functions as hp

economy_landuses = [3, 7]
map_matrix = hp.load_file(hp.get_file_in_current_directory("map.asc", __file__), 6, int)
output_file = hp.get_file_in_current_directory("economy_output.csv", __file__)

models.EconomyOptimizationModel(land_use_classes=economy_landuses, input_matrix=map_matrix, output_file=output_file).perform_calculations()


