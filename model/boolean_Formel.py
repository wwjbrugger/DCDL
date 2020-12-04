import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from tqdm import tqdm




class Boolsche_formel:
    def __init__(self, position_of_relevant_pixel, position_of_not,  number_of_product_term, number_of_relevant_variabels = None, total_error = None ):
        self.number_of_product_term = number_of_product_term # exampel 2
        self.variable_pro_term = None # 16

        self.pixel_relevant_in_number_code  = None#  [255, 4, 24, 16 ]
        self.pixel_relevant_in_arrays_code = None# [[1,1,1,1,1,1,1,1, 0,0,0,0,0,1,0,0], [0,0,0,1,1,0,0,0, 0,0,0,1,0,0,0,0]]

        self.pixel_negated_in_number_code = None
        self.pixel_negated_in_arrays_code = None

        self.formel_in_arrays_code = None
        self.number_of_relevant_variabels = number_of_relevant_variabels
        self.total_error = total_error
        self.shape_input_data = None
        self.shape_output_data = None
        if type(position_of_relevant_pixel) is np.ndarray:
            self.variable_pro_term= self.calc_variable_pro_term(position_of_relevant_pixel)
            self.pixel_relevant_in_number_code, self.pixel_relevant_in_arrays_code = self.fill_pixel_relevant_variabels(position_of_relevant_pixel)
            self.pixel_negated_in_number_code, self.pixel_negated_in_arrays_code = self.fill_negated_variabels(position_of_not)
            self.formel_in_arrays_code = self.merge_to_formula(self.pixel_relevant_in_arrays_code
                                                               , self.pixel_negated_in_arrays_code)
        else:
            raise ValueError('type should be np.ndarray not {}'.format(type(position_of_relevant_pixel)))

    def calc_variable_pro_term(self, position_of_relevant_pixel):
        if position_of_relevant_pixel.dtype == np.uint8:
            return int(position_of_relevant_pixel.shape[0] * 8 / self.number_of_product_term)

        elif position_of_relevant_pixel.dtype == np.ndarray:
            return position_of_relevant_pixel[0].shape[0]

        else:
            raise ValueError('dtype should be np.uint8 or np.ndarray not {}'.format(position_of_relevant_pixel.dtype))



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

    def fill_negated_variabels(self, position_of_not):
        if position_of_not.dtype == np.uint8:
            pixel_negated_in_number_code = position_of_not
            pixel_negated_in_arrays_Code = self.transform_number_code_in_arrays_code(
                position_of_not)
            return pixel_negated_in_number_code, pixel_negated_in_arrays_Code

        elif position_of_not.dtype == np.ndarray:
            pixel_negated_in_arrays_Code = position_of_not
            pixel_negated_in_number_code = self.transform_arrays_code_in_number_code(position_of_not)
            return pixel_negated_in_number_code, pixel_negated_in_arrays_Code
        else:
            raise ValueError('dtype should be np.uint8 or np.ndarray not {}'.format(position_of_not.dtype))

    def transform_number_code_in_arrays_code(self, number_code):
        arrays_code = []
        anzahl_number = number_code.shape[0]
        number_per_clause = int(anzahl_number/ self.number_of_product_term)

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

    def merge_to_formula(self,pixel_relevant_in_arrays_code, pixel_negated_in_arrays_Code):
        formula= []
        for clause_number in range(pixel_relevant_in_arrays_code.shape[0]):
            pixel_negated_clause = pixel_negated_in_arrays_Code[clause_number]
            pixel_negated_clause = np.where(pixel_negated_clause == 0, -1, 1)

            pixel_relevant_clause = pixel_relevant_in_arrays_code[clause_number]

            formula_clause = pixel_relevant_clause * pixel_negated_clause
            formula.append(np.array(formula_clause))

        return np.array(formula)

    @staticmethod
    def split_fomula(fomula_in_arrays_code):
        pixel_relevant_in_arrays_code = np.where(fomula_in_arrays_code != 0, 1, 0 )
        pixel_negated_in_arrays_Code =  np.where(fomula_in_arrays_code == -1, 0, 1  )
        return pixel_relevant_in_arrays_code, pixel_negated_in_arrays_Code




    def pretty_print_formula(self, titel_of_formula = ""):
        print('\n', titel_of_formula)
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        output_string = ""
        for clause in self.formel_in_arrays_code:
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
        formula = self.formel_in_arrays_code[number_of_fomula_to_see]
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
        if self.number_of_relevant_variabels:
            pixel_in_pic = self.number_of_relevant_variabels
        else:
            pixel_in_pic = self.variable_pro_term
        height = int(np.sqrt(pixel_in_pic))
        width = int(pixel_in_pic/height)
        #width = int(np.sqrt(pixel_in_pic))
        return pixel_in_pic, height, width

    def evaluate_belegung_like_c(self, belegung_arr):
        result = []
        on_off = self.pixel_relevant_in_arrays_code
        pos_neg = self.pixel_negated_in_arrays_code
        print('Anzahl_Belegungen: ', belegung_arr.shape[0] )
        for i, belegung in tqdm(enumerate(belegung_arr)):
            covered_by_any_clause = 0
            for clause_nr in range(self.number_of_product_term):
                covered_by_clause = 1;
                for position_in_clause in range(belegung_arr.shape[1]):
                    #position = clause_nr * self.variable_pro_term + position_in_clause
                    result_xor = belegung[position_in_clause] ^ pos_neg[clause_nr][position_in_clause]  # wenn unterschei zwischen Formel und belegung 1
                    result_and = result_xor & on_off[clause_nr][position_in_clause]     # wenn pixel relevant ist und ein unterschie 1
                    if result_and != 0:
                        covered_by_clause = 0
                        break
                if covered_by_clause:
                    covered_by_any_clause = 1
                    break
            result.append(covered_by_any_clause)
        return result






