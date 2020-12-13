# data structure to store boolean_formula in disjunctive normal form
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from tqdm import tqdm




class Boolean_formula:
    def __init__(self, position_of_relevant_pixel, position_of_not,  number_of_disjunction_term_in_SLS, number_of_relevant_variables = None, total_error_on_validation_set = None ):
        """
              @param position_of_relevant_pixel: indicate if variable has an influence on evaluation of formula
              possible values: 1 variable has influence, 0 variable has not influence
              can be given in number code (uint8) or directly in array code
              @param position_of_not:  indicate if variable negated.
              possible values: 1 not negated, 0 negated
              can be given in number code (uint8) or directly in array code
              @param  number_of_disjunction_term_in_SLS: formula has  number_of_disjunction_term_in_SLS many disjunctions
              @param number_of_relevant_variables: optional because of way the SLS-algorithm is calculated
              the variables are always a multiple of 8
              @param total_error: optional number of errors at train dataset
              """
        self.number_of_disjunction_term_in_SLS = number_of_disjunction_term_in_SLS
        self.variable_pro_term = None # 16

        self.pixel_relevant_in_number_code  = None #  [255, 4, 24, 16 ]
        self.pixel_relevant_in_arrays_code = None # [[1,1,1,1,1,1,1,1, 0,0,0,0,0,1,0,0], [0,0,0,1,1,0,0,0, 0,0,0,1,0,0,0,0]]

        self.pixel_negated_in_number_code = None
        self.pixel_negated_in_arrays_code = None

        self.formula_in_arrays_code = None
        self.number_of_relevant_variables = number_of_relevant_variables
        self.total_error_on_validation_set = total_error_on_validation_set
        self.shape_input_data = None
        self.shape_output_data = None
        self.train_acc = None
        if type(position_of_relevant_pixel) is np.ndarray:
            self.variable_pro_term= self.calc_variable_pro_term(position_of_relevant_pixel)
            self.pixel_relevant_in_number_code, self.pixel_relevant_in_arrays_code = self.fill_pixel_relevant_variabels(position_of_relevant_pixel)
            self.pixel_negated_in_number_code, self.pixel_negated_in_arrays_code = self.fill_negated_variables(position_of_not)
            self.formula_in_arrays_code = self.merge_to_formula(self.pixel_relevant_in_arrays_code
                                                               , self.pixel_negated_in_arrays_code)
        else:
            raise ValueError('type should be np.ndarray not {}'.format(type(position_of_relevant_pixel)))

    # calculate how many variables are in a disjunction
    def calc_variable_pro_term(self, position_of_relevant_pixel):
        if position_of_relevant_pixel.dtype == np.uint8:
            return int(position_of_relevant_pixel.shape[0] * 8 / self.number_of_disjunction_term_in_SLS)

        elif position_of_relevant_pixel.dtype == np.ndarray:
            return position_of_relevant_pixel[0].shape[0]

        else:
            raise ValueError('dtype should be np.uint8 or np.ndarray not {}'.format(position_of_relevant_pixel.dtype))

    # calculate number code (uint8) or array code for self.pixel_relevant_in_...
    # depending if number code (uint8) or array code is given.
    def fill_pixel_relevant_variabels(self, position_of_relevant_pixel):
        if position_of_relevant_pixel.dtype == np.uint8:
                pixel_relevant_in_number_code = position_of_relevant_pixel
                pixel_relevant_in_arrays_code_without_negation = self.transform_number_code_in_arrays_code(position_of_relevant_pixel)
                return pixel_relevant_in_number_code, pixel_relevant_in_arrays_code_without_negation

        elif position_of_relevant_pixel.dtype == np.ndarray:
                pixel_relevant_in_arrays_code_without_negation = position_of_relevant_pixel
                pixel_relevant_in_number_code = self.transform_arrays_code_in_number_code(position_of_relevant_pixel)
                return pixel_relevant_in_number_code, pixel_relevant_in_arrays_code_without_negation
        else:
            raise ValueError('dtype should be np.uint8 or np.ndarray not {}'.format(position_of_relevant_pixel.dtype))

    # calculate number code (uint8) or array code for pixel_negated_in_...
    # depending if number code (uint8) or array code is given.
    def fill_negated_variables(self, position_of_not):
        if position_of_not.dtype == np.uint8:
            pixel_negated_in_number_code = position_of_not
            pixel_negated_in_arrays_code = self.transform_number_code_in_arrays_code(
                position_of_not)
            return pixel_negated_in_number_code, pixel_negated_in_arrays_code

        elif position_of_not.dtype == np.ndarray:
            pixel_negated_in_arrays_code = position_of_not
            pixel_negated_in_number_code = self.transform_arrays_code_in_number_code(position_of_not)
            return pixel_negated_in_number_code, pixel_negated_in_arrays_code
        else:
            raise ValueError('dtype should be np.uint8 or np.ndarray not {}'.format(position_of_not.dtype))

    # [255, 4, 24, 16 ] -> [[1,1,1,1,1,1,1,1, 0,0,0,0,0,1,0,0], [0,0,0,1,1,0,0,0, 0,0,0,1,0,0,0,0]]
    def transform_number_code_in_arrays_code(self, number_code):
        arrays_code = []
        anzahl_number = number_code.shape[0]
        number_per_clause = int(anzahl_number/ self.number_of_disjunction_term_in_SLS)

        for start_number_for_clause in range(0, anzahl_number, number_per_clause):
            array_clause = self.transform_one_number_clause_in_one_array_clause\
                (number_code, start_number_for_clause, number_per_clause)
            arrays_code.append(array_clause)

        return np.array(arrays_code)

    def transform_one_number_clause_in_one_array_clause(self, number_code, start_number_for_clause, number_per_clause ):
        array_clause = [np.unpackbits(number_in_clause) for number_in_clause in number_code[start_number_for_clause:start_number_for_clause + number_per_clause ]]
        return  np.array(array_clause).reshape(-1)

    @staticmethod
    def transform_arrays_code_in_number_code(arrays_code):
        numbercode = [ np.packbits(arrays_code[i-8:i]) for i in range(8,arrays_code.size+1,8)]
        return np.reshape(numbercode, -1)

    # save formula in one array
    # possible values: -1 variable is negated, 0 variable is not important, variable is not negated
    def merge_to_formula(self,pixel_relevant_in_arrays_code, pixel_negated_in_arrays_code):
        formula= []
        for clause_number in range(pixel_relevant_in_arrays_code.shape[0]):
            pixel_negated_clause = pixel_negated_in_arrays_code[clause_number]
            pixel_negated_clause = np.where(pixel_negated_clause == 0, -1, 1)

            pixel_relevant_clause = pixel_relevant_in_arrays_code[clause_number]

            formula_clause = pixel_relevant_clause * pixel_negated_clause
            formula.append(np.array(formula_clause))

        return np.array(formula)

    # cast formula which is saved in one array in two arrays (pixel_relevant and pixel_negated )
    # possible values: -1 variable is negated, 0 variable is not important, variable is not negated
    @staticmethod
    def split_fomula(fomula_in_arrays_code):
        pixel_relevant_in_arrays_code = np.where(fomula_in_arrays_code != 0, 1, 0 )
        pixel_negated_in_arrays_code =  np.where(fomula_in_arrays_code == -1, 0, 1  )
        return pixel_relevant_in_arrays_code, pixel_negated_in_arrays_code

    # print formula at terminal in a nice way
    def pretty_print_formula(self, titel_of_formula = ""):
        print('\n', titel_of_formula)
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        output_string = ""
        for clause in self.formula_in_arrays_code:
            output_string += '('

            for i, variabel in enumerate(clause):
                if variabel == 0:
                    output_string += '  {}'.format(str(i).translate(SUB))
                elif variabel == 1:
                    output_string +=  colored( ' 1{}'.format(str(i).translate(SUB)), 'red')
                elif variabel == -1:
                    output_string +=  colored( '-1{}'.format(str(i).translate(SUB)), 'blue')

                else:
                    raise ValueError('Only 0 and 1 are allowed')
                output_string += ' ∧ '

            output_string = output_string[:-3] + ')'
            output_string += ' ∨  \n'

        output_string= output_string[:- 4]
        print(output_string)

    def built_plot(self, number_of_fomula_to_see, titel_for_picture):
        formula = self.formula_in_arrays_code[number_of_fomula_to_see]
        f = plt.figure()
        ax = f.add_subplot(111)
        pixel_in_pic, height, width = self.calculate_pic_height_width()
        z = np.reshape(formula[: pixel_in_pic], (width,height))
        mesh = ax.pcolormesh(z, cmap='gray', vmin= -1, vmax = 1)

        plt.colorbar(mesh, ax=ax)
        plt.title(titel_for_picture, fontsize=20)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


    def calculate_pic_height_width (self):
        pixel_in_pic = 0
        if self.number_of_relevant_variables:
            pixel_in_pic = self.number_of_relevant_variables
        else:
            pixel_in_pic = self.variable_pro_term
        height = int(np.sqrt(pixel_in_pic))
        width = int(pixel_in_pic/height)
        #width = int(np.sqrt(pixel_in_pic))
        return pixel_in_pic, height, width

    # evaluate data with formula extracted by SLS
    # slow evaluation better use prediction_SLS_fast
    def evaluate_allocation_like_c(self, allocation_arr):
        result = []
        on_off = self.pixel_relevant_in_arrays_code
        pos_neg = self.pixel_negated_in_arrays_code

        print('number allocations: ', allocation_arr.shape[0] )

        for i, allocation in tqdm(enumerate(allocation_arr)):
            # as soon as on clause covers the input of the whole logic formula evaluates to true
            covered_by_any_clause = 0
            for clause_nr in range(self.number_of_disjunction_term_in_SLS):
                # unless we know the opposite, we assume that the clause could cover
                covered_by_clause = 1;
                for position_in_clause in range(allocation_arr.shape[1]):
                    # if difference between formula and allocation than 1
                    result_xor = allocation[position_in_clause] ^ pos_neg[clause_nr][position_in_clause]
                    # if pixel is relevant and there is a difference
                    result_and = result_xor & on_off[clause_nr][position_in_clause]
                    if result_and != 0:
                        # clause does not cover
                        covered_by_clause = 0
                        break
                if covered_by_clause:
                    # there was no pixel which was relevant and was different to the clause
                    covered_by_any_clause = 1
                    break
            # add result for allocation
            result.append(covered_by_any_clause)
        return result






