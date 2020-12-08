import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import matplotlib.pyplot as plt

def transform_to_boolean(array):
    # cast data from -1 to 0.
    # 0 is interpreted as False
    # 1 as True
    boolean_array = np.maximum(array, 0).astype(np.bool)
    return boolean_array



def visualize_single_formula(kernel, kernel_width, title, set_vmin_vmax):
    f = plt.figure()
    ax = f.add_subplot(111)
    z = np.reshape(kernel, (kernel_width, kernel_width))
    if set_vmin_vmax:
        mesh = ax.pcolormesh(z, cmap='gray', vmin=-1, vmax=1)
    else:
        mesh = ax.pcolormesh(z)
    plt.colorbar(mesh, ax=ax)
    plt.title(title, fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()


def one_class_against_all(array_label, one_class, number_classes_output, kind_of_data):
    """
    converts an array with one_hot_vector for any number of classes into a one_hot_vector,
    whether an example belongs to one class or not
    """
    shape_output = (len(array_label), number_classes_output)
    label_one_class_against_all = np.zeros(shape_output, dtype=int)
    for i, one_hot_vector in enumerate(array_label):
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all[i, 0] = 1
        else:
            label_one_class_against_all[i, -1] = 1
    num_elements_one_class = int(label_one_class_against_all[:, 0].sum())
    num_elements_rest_class = int(label_one_class_against_all[:, 1].sum())
    print('{}   number one label in set: {}     number rest label in set {} '.format(
        kind_of_data, num_elements_one_class, num_elements_rest_class))
    return label_one_class_against_all


def reduce_kernel(input, mode):
    sum = np.sum(input, axis=0)
    if mode in 'sum':
        return sum
    elif mode in 'mean':
        mean = np.mean(input, axis=0)
        return mean
    elif mode in 'min_max':
        min = sum.min()
        max = sum.max()
        min_max = np.where(sum < 0, - sum / min,  sum / max)
        return min_max
    elif mode in 'norm':
        mean = np.mean(sum)
        max_value = sum.max()
        max_centered = max_value-mean
        norm = (sum-mean)/max_centered
        return norm
    else:
        raise ValueError("{} ist not a valid mode.".format(mode))


def convert_to_grey(pic_array):
    """ convert rgb pictures in grey scale pictures """
    pictures_grey = np.empty((pic_array.shape[0], pic_array.shape[1], pic_array.shape[2], 1))
    for i, pic in enumerate(pic_array):
        pictures_grey[i,:,:,0] = np.dot(pic[:,:,:3], [0.2989, 0.5870, 0.1140] )
    return pictures_grey


