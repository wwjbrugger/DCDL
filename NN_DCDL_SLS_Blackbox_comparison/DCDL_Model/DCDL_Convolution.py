import os
import numpy as np

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import SLS_Algorithm as SLS
import NN_DCDL_SLS_Blackbox_comparison.get_data as get_data


class Convolution:
    # class to approximate a convolutional layer in the neural net
    def __init__(self, properties):
        # place to save found logical formula
        self.found_formula_list = []
        # dic with properties of convolution
        self.properties = properties
        # shape of output
        self.output_shape = None

    def preprocess(self, data, label):
        # preprocess input to perform convolution
        if label is not None :
            # original label is given. This is the case for training
            self.output_shape = list(label.shape)
            self.output_shape[0] =  data.shape[0]
        else:
            # no label is given this is the case for the prediction mode
            self.output_shape[0] = data.shape[0]

        ## get subsamples of data as they would be underthe kernel in a convolution operation
        data_under_kernel = self.data_in_kernel(arr=data,
                                                stepsize=self.properties['stride'],
                                                width=self.properties['kernel'][0])
        # flatten subsamples
        num_data_under_kernel = data_under_kernel.shape[0]
        data_flat = data_under_kernel.reshape((num_data_under_kernel, -1))
        return data_flat, label



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
        # decomposes input image into smaller subimages as they are under the convolution kernel
        train, label_all_channel = self.preprocess(data=train_data,
                                                   label=train_label)
        for channel in range(self.properties['num_kernel']):
            # iterate through channels of the output
            print('channel: {}'.format(channel))
            # get labels of the current channel
            label_one_channel = label_all_channel[:, :, :, channel].flatten()
            if self.properties['SLS_dic']['mode'] in 'rule_extraction_with_sls':
                # use SLS algorithm only with a train set
                found_formula = SLS.rule_extraction_with_sls(
                    train=train,
                    train_label=label_one_channel,
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
                # use SLS algorithm with train and validation set
                val, val_label_all_channel = self.preprocess(data=validation_data,
                                                             label=validation_label)
                val_label_one_chanel = val_label_all_channel[:, :, :, channel].flatten()
                found_formula = SLS.rule_extraction_with_sls_val(
                    train=train,
                    train_label=label_one_channel,
                    val=val,
                    val_label=val_label_one_chanel,
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
            # add formula for current channel to all other formulas
            self.found_formula_list.append(found_formula)

    def prediction(self, data, original_label):
        # decomposes input image into smaller subimages as they are under the convolution kernel
        data_flatten, label = self.preprocess(data=data,
                                              label=original_label)

        output_shape = self.output_shape
        # output will have as many examples as the input.
        output_shape_one_channel = output_shape[:3]


        # create space to write prediction to
        prediction = np.zeros(shape=tuple(output_shape),
                              dtype=np.bool)

        for channel in range(self.properties['num_kernel']):
            # iterate through channels and get formula for current channel
            found_formula = self.found_formula_list[channel]
            flatten_label_shape = [output_shape[0] * output_shape[1] * output_shape[2]]
            prediction_one_channel = SLS.calc_prediction_in_C(data=data_flatten,
                                                              label_shape=flatten_label_shape,
                                                              found_formula=found_formula)
            # cast prediction of one channel to output shape
            prediction[:, :, :, channel] = np.reshape(prediction_one_channel, output_shape_one_channel)

        # acc is calculated by DCDL class
        acc = None
        return prediction, acc
