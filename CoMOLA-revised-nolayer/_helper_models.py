from abc import ABCMeta, abstractmethod
import _helper_functions as hp

class LandUseOptimizationModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, land_use_classes, input_matrix, external_data, output_file):
        self.land_use_classes = land_use_classes
        self.input_matrix = input_matrix
        self.external_data = external_data
        self.output_file = output_file

    @abstractmethod
    def retrieve_map_info(self):
        pass

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
    
    def retrieve_map_info(self, matrix):
        return hp.count_cells(matrix, self.land_use_classes)

    def perform_calculations(self):
        cell_count = self.retrieve_map_info(self.input_matrix)
        return hp.update_file(cell_count, self.output_file)



class CompatibilityOptimizationModel(LandUseOptimizationModel):
    __metaclass__ = ABCMeta
    '''
    We want to maximise the compatibility of cell dispositions.
    1. We derive all cell combinations from the input matrix so that each edge 
       (horizontal, vertical and diagonal in the case of 8 neighbours methods) is accounted for once
    2. We look up the compatibility score of each cell combination and sum those.
    '''
    def __init__(self, land_use_classes, input_matrix, output_file, external_data):
        super(CompatibilityOptimizationModel, self).__init__(land_use_classes, input_matrix, output_file, external_data)
        
    def calculate_total_compatibility_score(self, cell_combinations, compatibility_matrix):
        results = 0
        for cell_combination in cell_combinations:
            results += compatibility_matrix[cell_combination[0] - 1][cell_combination[1] - 1]
        return results
    
    def retrieve_map_info(self, matrix):
        return hp.get_all_cell_combinations(matrix)

    def perform_calculations(self):
        cell_combinations = self.retrieve_map_info(self.input_matrix)
        result = self.calculate_total_compatibility_score(cell_combinations, self.external_data)
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
    
    def retrieve_map_info(self, matrix):
        return hp.get_all_cell_combinations(matrix)

    def perform_calculations(self):
        cell_combinations = self.retrieve_map_info(self.input_matrix)
        result = sum(1 for cell_combination in cell_combinations if cell_combination[0] == cell_combination[1])
        return hp.update_file(result, self.output_file)
    

