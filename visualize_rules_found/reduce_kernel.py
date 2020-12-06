import helper_methods as helper_methods
import numpy as np

def reduce_SLS_results_of_one_run(one_against_all):

    kernel_approximation = np.load('data/kernel_approximation_label_{}.npy'.format(one_against_all))
    for i, kernel in enumerate(kernel_approximation):
        reduced_kernel = helper_methods.reduce_kernel(kernel, mode='norm')
        #helper_methods.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
        #                             'norm of all found SLS Formel for kernel {}'.format(i))

        """
        reduced_kernel = helper_methods.reduce_kernel(kernel, mode='sum')
        helper_methods.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)),  28,
                                     'Sum of all found SLS Formel for kernel {}'.format(i),
                                     set_vmin_vmax= False)

        reduced_kernel = helper_methods.reduce_kernel(kernel, mode='mean')
        helper_methods.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'mean of all found SLS Formel for kernel {}'.format(i),
                                     set_vmin_vmax= False)

        reduced_kernel = helper_methods.reduce_kernel(kernel, mode='min_max')
        helper_methods.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'min_max of all found SLS Formel for kernel {}'.format(i))
        """
        return reduced_kernel


