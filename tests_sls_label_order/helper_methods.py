import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import matplotlib.pyplot as plt




def calculate_padding_parameter(shape_input_pic, filter_size, stride, ):
    # calculate how many zeros have to be pad on input to perform convolution
    in_height = shape_input_pic[1]
    in_width = shape_input_pic[2]
    out_height = np.ceil(float(in_height) / float(stride))
    out_width = np.ceil(float(in_width) / float(stride))

    pad_along_height = np.max((out_height - 1) * stride +
                              filter_size - in_height, 0)
    pad_along_width = np.max((out_width - 1) * stride +
                             filter_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return ((0, 0), (int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right)), (0, 0))


def data_in_kernel(arr, stepsize=2, width=4):  # kernel views
    # calculates which data are under the kernel/filter of a convolution operation
    npad = calculate_padding_parameter(arr.shape, width, stepsize)
    training_set_padded = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)

    dims = training_set_padded.shape
    temp = [training_set_padded[picture, row:row + width, col:col + width, :]  # .flatten()
            for picture in range(0, dims[0]) for row in range(0, dims[1] - width + 1, stepsize) for col in
            range(0, dims[2] - width + 1, stepsize)]
    out_arr = np.stack(temp, axis=0)

    return out_arr


def permutate_and_flaten(data, label, channel_label):
    # this method has logic to iterate through input data and channel in the method.
    temp = []
    for pic in range(data.shape[0]):
        pic_temp = []
        for channel in range(data.shape[3]):
             pic_temp.append(np.reshape(data[pic,:,:,channel], -1))
        temp.append(np.reshape(pic_temp, -1))
    data_flatten = np.array(temp)
    label_set_flat = label[:, :, :, channel_label].reshape(-1)
    return data_flatten, label_set_flat



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


def one_class_against_all(array_label, one_class, number_classes_output):
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
    print('number one label in set: {}     number rest label in set {} '.format(num_elements_one_class, num_elements_rest_class))
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


