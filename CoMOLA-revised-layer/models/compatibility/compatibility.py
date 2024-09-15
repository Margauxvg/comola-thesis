import sys
import os

main_directory = os.path.dirname(os.path.abspath("_helpers_models.py"))
# Append the current directory to sys.path
sys.path.append(main_directory)

import _helper_models as models
import _helper_functions as hp

#Uncomment when using the version without preprocessing layer
#map_matrix = hp.load_file(hp.get_file_in_current_directory("map.asc", __file__), 6, int)
#Uncomment when using the version with preprocessing layer
map_matrix = hp.load_file(hp.get_file_in_current_directory("map_info.asc", __file__), 6, int)

compatibility_matrix = hp.load_file("input\compatibilityMatrix.asc",0,int)
output_file = hp.get_file_in_current_directory("compatibility_output.csv", __file__)

models.CompatibilityOptimizationModel(land_use_classes=None, input_matrix=map_matrix, output_file=output_file,compatibility_matrix=compatibility_matrix).perform_calculations()
