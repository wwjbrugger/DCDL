import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import matplotlib.pyplot as plt
import SLS_Algorithm as SLS
import pickle
from skimage.measure import block_reduce




def calculate_padding_parameter(shape_input_pic, filter_size, stride, ):
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
    # Need padding for convolution
    # npad = ((0, 0), (1, 1), (1, 1), (0, 0))
    # npad = ((0, 0), (0, 0), (0, 0), (0, 0))
    npad = calculate_padding_parameter(arr.shape, width, stepsize)
    training_set_padded = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)

    dims = training_set_padded.shape
    temp = [training_set_padded[picture, row:row + width, col:col + width, :]  # .flatten()
            for picture in range(0, dims[0]) for row in range(0, dims[1] - width + 1, stepsize) for col in
            range(0, dims[2] - width + 1, stepsize)]
    out_arr = np.stack(temp, axis=0)

    # x = [arr[k, i:i+width, j:j+width, :].flatten()
    # [ 5x5 Subbild vom k-ten Bild] .flatten =>
    #   [[0, 1, 2, 3, 4],
    #    [5, 6, 7, 8, 9],
    #    [0, 1, 2, 3, 4],   =>  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
    #    [5, 6, 7, 8, 9],
    #    [0, 1, 2, 3, 4]]
    # for k in range(0, dims[0])  for i in range(0, dims[1] - width + 1, stepsize) for j in range(0, dims[2] - width + 1, stepsize) ], axis=0
    #  für jedes Bild bewege Kernel über das Bild
    # 196 Position ein 4X4 Kernel auf ein 30x30 mit Stride 2 zu plazieren. Pro Position gibt es 256 Channels gibt vor dem stack befehl ein Shape (196,(4,4,256))
    # nach dem Stack  (196,4,4,256)

    return out_arr



def permutate_and_flaten_2(training_set_kernel, label_set, channel_training, channel_label):
    """
    outdated
    @param training_set_kernel:
    @param label_set:
    @param channel_training:
    @param channel_label:
    @return:
    """
    number_kernels = training_set_kernel.shape[0]
    # random_indices = np.random.permutation(number_kernels)
    training_set_flat = training_set_kernel[:, :, :, channel_training].reshape((number_kernels, -1))
    # training_set_flat_permutated = training_set_flat[random_indices]
    label_set_flat = label_set[:, :, :, channel_label].reshape(number_kernels)
    # label_set_flat_permutated = label_set_flat[random_indices]
    return training_set_flat, label_set_flat
    # return training_set_flat_permutated, label_set_flat_permutated


def permutate_and_flaten(data, label, channel_label):
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
    boolean_array = np.maximum(array, 0).astype(np.bool)  # 2,4 for pooled layer
    return boolean_array


def visualize_singel_kernel(kernel, kernel_width, titel, set_vmin_vmax = True):
    f = plt.figure()
    ax = f.add_subplot(111)
    z = np.reshape(kernel, (kernel_width, kernel_width))
    if set_vmin_vmax:
        mesh = ax.pcolormesh(z, cmap='gray', vmin=-1, vmax=1)
    else:
        mesh = ax.pcolormesh(z)
    plt.colorbar(mesh, ax=ax)
    plt.title(titel, fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()


def visualize_multi_kernel(pic_array, label_array, titel):
    """ show 10 first  pictures """
    for i in range(pic_array.shape[3]):
        ax = plt.subplot(4, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        mesh = ax.pcolormesh(pic_array[:, :, 0, i], cmap='gray', vmin=-1, vmax=1)
        plt.colorbar(mesh, ax=ax)
        plt.title(label_array[i])
        plt.gca().set_aspect('equal', adjustable='box')

        # plt.imshow(pic_array[:,:,0,i], cmap=colormap)
        # plt.xlabel(label_array[i])

    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()


def visualize_pic(pic_array, label_array, class_names, titel, colormap, filename = False):
    """ show first 20  pictures in array"""
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if pic_array.shape[3] == 1:
            plt.imshow(pic_array[i,:,:,0], cmap=colormap)
        elif pic_array.shape[3] == 3:
            plt.imshow(pic_array[i], cmap=colormap)
        else:
            raise ValueError('Picture should have 1 or 3 channels not'.format(pic_array.shape[3]))
        plt.xlabel(class_names[np.argmax(label_array[i])])

    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()



def calculate_convolution(data_flat, kernel, true_label):
    label = []
    kernel_flaten = np.reshape(kernel, (-1))
    for row in data_flat:
        label.append(np.dot(row, kernel_flaten))
    return label


def visulize_input_data(pic):
    hight = int(np.sqrt(pic.size))
    pic = np.reshape(pic, (hight, hight))

    plt.imshow(pic, cmap='gray')

    plt.show()


# def one_class_against_all(label_list, one_against_all):
#     """
# converts an array with one_hot_vector for any number of classes into a one_hot_vector,
#  whether an example belongs to 'one' class or all class
#     """
#     shape_output = (len(label_list), 2)
#     label_one_class_against_all = np.zeros(shape_output, dtype=int)
#     for i, one_hot_vector in enumerate(label_list):
#         if one_hot_vector.argmax() == one_against_all:
#             label_one_class_against_all[i, 0] = 1
#         else:
#             label_one_class_against_all[i, -1] = 1
#     return label_one_class_against_all

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


def sls_convolution ( number_of_disjunction_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data_sign, label_sign, used_kernel, result= None, path_to_store = None, SLS_Training = True) :
    kernel_width = used_kernel.shape[0]
    data_flat, label = prepare_data_for_sls(data_sign, label_sign, kernel_width, stride_of_convolution)
    np.save(path_to_store + '_data_flat.npy', data_flat)

    logic_formulas = []
    print('Shape of flatten data: ', data_flat.shape )
    if SLS_Training:
        _, unique_index = np.unique(data_flat, return_index=True, axis=0)
        data_flat = data_flat[unique_index]
        print('Shape of flatten data after making unique', data_flat.shape )
        for channel in range(label.shape[3]):
            print("rule extraction for kernel_conv_1 {} ".format(channel))
            #label_flat = label[:, :, :, channel].reshape(data_flat.shape[0])
            label_flat = label[:, :, :, channel].reshape(-1)[unique_index]

            found_formula = \
                SLS.rule_extraction_with_sls_without_validation(data_flat,label_flat,  number_of_disjunction_term_in_SLS,
                                                               Maximum_Steps_in_SKS)
            found_formula.shape_input_data = data_sign.shape
            found_formula.shape_output_data = label.shape
            logic_formulas.append(found_formula)

            accuracy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
            print("Accuracy of SLS: ", accuracy, "\n")

            if result is not None:
                label_self_calculated = calculate_convolution(data_flat, used_kernel[:, :, :, channel], result)


        if path_to_store is not None:
            pickle.dump(logic_formulas, open(path_to_store, "wb"))
    return logic_formulas
"""
        formel_in_array_code = []
        for formel in logic_formulas:
            formel_in_array_code.append(np.reshape(formel.formula_in_arrays_code, (-1, kernel_width, kernel_width)))
        np.save(path_to_store + '_in_array_code.npy', formel_in_array_code)
"""
def prepare_data_for_sls(data_sign, label_sign, kernel_width, stride_of_convolution):
    data_bool = transform_to_boolean(data_sign)
    label_bool = transform_to_boolean(label_sign)

    data_under_kernel = data_in_kernel(data_bool, stepsize=stride_of_convolution, width=kernel_width)
    number_kernels = data_under_kernel.shape[0]

    data_bool_flat = data_under_kernel.reshape((number_kernels, -1))

    return data_bool_flat, label_bool


def sls_dense_net ( number_of_disjunction_term_in_SLS, Maximum_Steps_in_SKS, data, label, path_to_store = None, SLS_Training = True) :
    data = transform_to_boolean(data)
    data_flat = np.reshape(data, (data.shape[0], -1))
    np.save(path_to_store + '_data_flat.npy', data_flat)
    print('Shape of flatten data: ', data_flat.shape )
    if SLS_Training:
        label_set_one_hot = transform_to_boolean(label)
        label = np.array([label[0] for label in label_set_one_hot])
        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(data_flat,label,  number_of_disjunction_term_in_SLS,
                                                           Maximum_Steps_in_SKS)
        found_formula.shape_input_data = data.shape
        found_formula.shape_output_data = label.shape

        accurancy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
        print("Accurancy of SLS: ", accurancy, '\n')

        if path_to_store is not None:
            pickle.dump(found_formula, open(path_to_store, "wb"))


def prediction_SLS_fast (data_flat, label, found_formula, path_to_store_prediction = None):
    print('Shape of Input Data: ', data_flat.shape)
    if label.ndim == 1: # Output of NN in last layer [1,0,1,0 ...]
        label = np.array([-1 if l == 0 else 1 for l in label])
        print('Calculate prediction')
        prediction = SLS.calc_prediction_in_C(data_flat, label.flatten().shape, found_formula)
        prediction = np.reshape(prediction, label.shape)

    elif label.ndim == 2:  # True Label One-hot-encoded
        label = np.array([-1 if l[1]==0 else 1 for l in label])
        print('Calculate prediction')
        prediction = SLS.calc_prediction_in_C(data_flat, label.flatten().shape , found_formula)
        prediction = np.reshape(prediction, label.shape)

    else: # Output of NN with more than one channel
        prediction = np.empty(label.shape, np.bool)
        for channel in range(label.shape[3]):
            prediction_one_channel = SLS.calc_prediction_in_C(data_flat, label[:, :, :, channel].flatten().shape, found_formula[channel])
            prediction[:, :, :, channel] = np.reshape(prediction_one_channel, label[:, :, :, channel].shape)
    error = np.sum(label != np.where(prediction, 1 ,-1))
    Accuracy = (label.size-error)/label.size
    print('Error of prediction', error)
    print('Acc', Accuracy)
    if path_to_store_prediction is not None:
        np.save(path_to_store_prediction, prediction)
    return Accuracy

def max_pooling (data):
    data_after_max_pooling=block_reduce(data, block_size=(1, 2, 2, 1), func=np.max)
    return data_after_max_pooling

def convert_to_grey(pic_array):
    """ convert rgb pictures in grey scale pictures """
    pictures_grey = np.empty((pic_array.shape[0], pic_array.shape[1], pic_array.shape[2], 1))
    for i, pic in enumerate(pic_array):
        pictures_grey[i,:,:,0] = np.dot(pic[:,:,:3], [0.2989, 0.5870, 0.1140] )
    return pictures_grey



def graph_with_error_bar(x_values, y_values, y_stdr, title = "",x_axis_title="", y_axis_tile='', fix_y_axis= False, ax_out = False, save_path = False, plot_line = False  ):
    if not ax_out:
        fig, ax = plt.subplots()
    else:
        ax = ax_out
    ax.errorbar(x_values, y_values,
                yerr=y_stdr,
                fmt='o')
    if plot_line:
        line = 0 * np.array(y_values) + y_values[0]
        ax.plot(x_values, line, '--r')
    ax.set_xlabel(x_axis_title)
    plt.xticks(rotation=-90)
    ax.set_ylabel(y_axis_tile)
    ax.set_title(title)
    if fix_y_axis:
        min = np.min(y_values)
        max = np.max(y_values)
        #ax.set_ylim((min - 0.05), (max + 0.05))
        ax.set_ylim((0.5), (1))

    if save_path:
        plt.savefig(save_path,
                dpi=300)
    if not ax_out:
        plt.show()


def mark_small_values(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if val == 1:
        color = 'black'
    elif val < 0.05:
        color = 'red'
    else:
        color = 'grey'

    return 'background-color: %s' % color
