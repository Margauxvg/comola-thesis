from abc import ABCMeta, abstractmethod
import _helper_functions as hp
import numpy as np

class LandUseOptimizationModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, land_use_classes, input_matrix, output_file):
        self.land_use_classes = land_use_classes
        self.input_matrix = input_matrix
        self.output_file = output_file

    # Uncomment when using the version without preprocessing layer
    # @abstractmethod
    # def retrieve_map_info(self):
    #     pass

    @abstractmethod
    def perform_calculations(self):
        pass


class EconomyOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximise the area dedicated for economy land uses.
    1. We define which land uses can benefit the economy
    2. We count the matrix cells which have an economic land use
    '''
    def __init__(self, land_use_classes, input_matrix, output_file):
        super(EconomyOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file)
    
    # Uncomment when using the version without preprocessing layer
    # def retrieve_map_info(self, matrix):
    #     return hp.count_cells(matrix, self.land_use_classes)

    # def perform_calculations(self):
    #     cell_count = self.retrieve_map_info(self.input_matrix)
    #     return hp.update_file(cell_count, self.output_file)
    
    # Uncomment when using the version with preprocessing layer
    def perform_calculations(self):
        cell_count = hp.count_cells(self.input_matrix, self.land_use_classes)
        return hp.update_file(cell_count, self.output_file)


class CompatibilityOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximise the compatibility of cell dispositions.
    1. We derive all cell combinations from the input matrix so that each edge (horizontal, vertical and diagonal in the case of 8 neighbours methods) is accounted for once
    2. We look up the compatibility score of each cell combination and sum those.
    '''
    def __init__(self, land_use_classes, input_matrix, output_file, compatibility_matrix):
        super(CompatibilityOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file)
        self.compatibility_matrix = compatibility_matrix
        
    def calculate_total_compatibility_score(self, cell_combinations, compatibility_matrix):
        results = 0
        for cell_combination in cell_combinations:
            results += compatibility_matrix[cell_combination[0] - 1][cell_combination[1] - 1]
        return results
    
    # Uncomment when using the version without preprocessing layer
    # def retrieve_map_info(self, matrix):
    #     return hp.get_all_cell_combinations(matrix)

    # def perform_calculations(self):
    #     cell_combinations = self.retrieve_map_info(self.input_matrix)
    #     result = self.calculate_total_compatibility_score(cell_combinations, self.compatibility_matrix)
    #     return hp.update_file(result, self.output_file)

    # Uncomment when using the version withpreprocessing layer
    def perform_calculations(self):
        result = self.calculate_total_compatibility_score(self.input_matrix, self.compatibility_matrix)
        return hp.update_file(result, self.output_file)


class CompactOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximize the amount of neighbours of each land use.
    1. We derive all cell combinations from the input matrix so that each edge (horizontal, vertical and diagonal in the case of 8 neighbours methods) is accounted for once
    2. We count all cell combinations of the same land use
    '''
    def __init__(self, land_use_classes, input_matrix, output_file):
        super(CompactOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file)
    
    # Uncomment when using the version without preprocessing layer
    # def retrieve_map_info(self, matrix):
    #     return hp.get_all_cell_combinations(matrix)

    # def perform_calculations(self):
    #     cell_combinations = self.retrieve_map_info(self.input_matrix)
    #     result = sum(1 for cell_combination in cell_combinations if cell_combination[0] == cell_combination[1])
    #     return hp.update_file(result, self.output_file)

    #Uncomment when using the version with preprocessing layer
    def perform_calculations(self):
        result = sum(1 for cell_combination in self.input_matrix if cell_combination[0] == cell_combination[1])
        return hp.update_file(result, self.output_file)
    

class HeterogeneityOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximize the heterogeneity of a set of land uses on the map
    1. We count all the cells of which the land use belongs to the chosen set
    2. We determine the proportions of the count based on the total map
    3. We calculate the shannon diversity index
    '''
    def __init__(self, land_use_classes, input_matrix, output_file):
        super(HeterogeneityOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file)
    
    # Uncomment when using the version without preprocessing layer
    # def retrieve_map_info(self, matrix):
    #     cell_count = hp.count_cells(matrix, self.land_use_classes)
    #     proportion = cell_count /  (len(matrix) * len(matrix[0]))
    #     return proportion

    # def perform_calculations(self):
    #     proportion = self.retrieve_map_info(self.input_matrix)
    #     result = hp.calculate_shannon_diversity(proportion)
    #     return hp.update_file(result, self.output_file)

    #Uncomment when using the version with preprocessing layer
    def perform_calculations(self):
        cell_count = hp.count_cells(self.input_matrix, self.land_use_classes)
        proportion = cell_count /  (len(self.input_matrix) * len(self.input_matrix[0]))
        result = hp.calculate_shannon_diversity(proportion)
        return hp.update_file(result, self.output_file)
    

class BiodiversityOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximise the area dedicated for enatural land uses.
    1. We define which land uses can enhance biodiversity
    2. We count the matrix cells which have an economic land use
    '''
    def __init__(self, land_use_classes, input_matrix, output_file):
        super(BiodiversityOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file)
    
    # Uncomment when using the version without preprocessing layer
    # def retrieve_map_info(self, matrix):
    #     return hp.count_cells(matrix, self.land_use_classes)

    # def perform_calculations(self):
    #     cell_count = self.retrieve_map_info(self.input_matrix)
    #     return hp.update_file(cell_count, self.output_file)
    
    # Uncomment when using the version with preprocessing layer
    def perform_calculations(self):
        cell_count = hp.count_cells(self.input_matrix, self.land_use_classes)
        return hp.update_file(cell_count, self.output_file)
