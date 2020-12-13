import visualize_rules_found.helper_methods as help_vis
import numpy as np


def visualize_logic_rule(kernel_approximation, kernel_width, number_of_disjunction_term_in_SLS, set_vmin_vmax):
    # visualize the logic formula found from SLS
    for i, formula in enumerate(kernel_approximation):
        # iterate through formula (as many as channels)
        formula_in_arrays_code = np.reshape(formula.formula_in_arrays_code, (-1, kernel_width, kernel_width))
        # reduce k-many conjunction of formula to 1
        reduced_formula = help_vis.reduce_kernel(formula_in_arrays_code, mode='norm')
        # visualize reduced formula
        help_vis.visualize_single_formula(
            kernel=reduced_formula,
            kernel_width=kernel_width,
            title='k= {}'.format(number_of_disjunction_term_in_SLS),
            set_vmin_vmax=set_vmin_vmax)
