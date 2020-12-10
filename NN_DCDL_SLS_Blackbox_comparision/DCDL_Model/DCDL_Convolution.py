import os
import numpy as np
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import SLS_Algorithm as SLS
import NN_DCDL_SLS_Blackbox_comparision.get_data as get_data

class Convolution:
    def __init__(self, properties):
        self.found_formula_list = []
        self.properties = properties
        self.output_shape = None

    def preprocess(self, data, label):
        if label is not None:
            self.output_shape = list(label.shape)
            self.output_shape[0] = None
            label_bool = get_data.transform_to_boolean(label)

        data_bool = get_data.transform_to_boolean(data)


        ## get subsamples of data as they would be under the kernel in a convolution operation
        data_under_kernel = self.data_in_kernel(arr=data_bool,
                                                stepsize=self.properties['stride'],
                                                width=self.properties['kernel'][0])
        # flatten subsamples
        num_data_under_kernel = data_under_kernel.shape[0]
        data_bool_flat = data_under_kernel.reshape((num_data_under_kernel, -1))

        return data_bool_flat, label_bool

    def data_in_kernel(self, arr, stepsize, width):
        # calculates which data are under the kernel/filter of a convolution operation
        npad = self.calculate_padding_parameter(arr.shape, width, stepsize)
        training_set_padded = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)

        dims = training_set_padded.shape
        temp = [training_set_padded[picture, row:row + width, col:col + width, :]  # .flatten()
                for picture in range(0, dims[0]) for row in range(0, dims[1] - width + 1, stepsize) for col in
                range(0, dims[2] - width + 1, stepsize)]
        out_arr = np.stack(temp, axis=0)

        return out_arr

    def calculate_padding_parameter(self, shape_input_pic, filter_size, stride, ):
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

    def train(self, train_data, train_label, validation_data, validation_label):
        train, label_all_channel = self.preprocess(data=train_data,
                                       label=train_label)
        for channel in range(self.properties['num_kernel']):
            label = label_all_channel[:,:,:,channel].flatten()
            if self.properties['SLS_dic']['mode'] in 'rule_extraction_with_sls':
                found_formula = SLS.rule_extraction_with_sls(
                    train=train,
                    train_label=label,
                    number_of_disjunction_term=self.properties['SLS_dic']['number_of_disjunction_term_in_SLS'],
                    maximum_steps_in_SLS=self.properties['SLS_dic']['maximum_steps_in_SLS'],
                    kernel=self.properties['SLS_dic']['init_with_kernel'],
                    p_g1=self.properties['SLS_dic']['p_g1'],
                    p_g2=self.properties['SLS_dic']['p_g2'],
                    p_s=self.properties['SLS_dic']['p_s'],
                    batch=self.properties['SLS_dic']['batch'],
                    cold_restart=self.properties['SLS_dic']['cold_restart'],
                    decay=self.properties['SLS_dic']['decay'],
                    min_prob=self.properties['SLS_dic']['min_prob'],
                    zero_init=self.properties['SLS_dic']['zero_init']
                )
            elif self.properties['SLS_dic']['mode'] in 'rule_extraction_with_sls_val':
                val, val_label_all_channel = self.preprocess(data=validation_data,
                                                 label=validation_label)
                val_label = val_label_all_channel[:, :, :, channel].flatten()
                found_formula = SLS.rule_extraction_with_sls_val(
                    # found_formula_val=SLS.rule_extraction_with_sls(
                    train=train,
                    train_label=label,
                    val=val,
                    val_label=val_label,
                    number_of_disjunction_term=self.properties['SLS_dic']['number_of_disjunction_term_in_SLS'],
                    maximum_steps_in_SLS=self.properties['SLS_dic']['maximum_steps_in_SLS'],
                    kernel=self.properties['SLS_dic']['init_with_kernel'],
                    p_g1=self.properties['SLS_dic']['p_g1'],
                    p_g2=self.properties['SLS_dic']['p_g2'],
                    p_s=self.properties['SLS_dic']['p_s'],
                    batch=self.properties['SLS_dic']['batch'],
                    cold_restart=self.properties['SLS_dic']['cold_restart'],
                    decay=self.properties['SLS_dic']['decay'],
                    min_prob=self.properties['SLS_dic']['min_prob'],
                    zero_init=self.properties['SLS_dic']['zero_init']
                )
            #todo check if found formula is overwritten
            self.found_formula_list.append(found_formula)

    def prediction(self, data, original_label):
        label_shape = self.output_shape
        # number of pictures are constant in convolution
        label_shape[0] = data.shape[0]

        #todo look again
        data_flatten, label = self.preprocess(data=data,
                                       label=original_label)

        prediction = np.empty(label.shape, np.bool)

        for channel in range(self.properties['num_kernel']):
            found_formula = self.found_formula_list[channel]
            flatten_label_shape = [label_shape[0]*label_shape[1]*label_shape[2]]
            prediction_one_channel = SLS.calc_prediction_in_C(data=data_flatten,
                                                  label_shape=flatten_label_shape,
                                                  found_formula=found_formula)
            prediction[:, :, :, channel] = np.reshape(prediction_one_channel, label[:, :, :, channel].shape)
        acc = None
        return prediction, acc