import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import SLS_Algorithm as SLS
import NN_DCDL_SLS_Blackbox_comparison.get_data as get_data
import numpy as np


class SLS_black_box:
    def __init__(self, mode, arg_min_label, number_of_disjunction_term, maximum_steps_in_SLS,
                 init_with_kernel, p_g1, p_g2, p_s, batch, cold_restart,decay,
                 min_prob, zero_init):
        self.found_formula = None
        self.arg_min_label = arg_min_label
        self.mode = mode
        self.number_of_disjunction_term = number_of_disjunction_term
        self.maximum_steps_in_SLS = maximum_steps_in_SLS
        self.init_with_kernel = init_with_kernel
        self.p_g1 = p_g1
        self.p_g2 = p_g2
        self.p_s = p_s
        self.batch = batch
        self.cold_restart = cold_restart
        self.decay = decay
        self.min_prob = min_prob
        self.zero_init = zero_init
        pass

    def train(self, train_data, train_label, validation_data, validation_label ):
        train, label = self.preprocess(data=train_data,
                                       label=train_label)
        if self.mode in 'rule_extraction_with_sls':
            self.found_formula = SLS.rule_extraction_with_sls(
                    train=train,
                train_label=label,
                number_of_disjunction_term=self.number_of_disjunction_term,
                maximum_steps_in_SLS=self.maximum_steps_in_SLS,
                kernel=self.init_with_kernel,
                p_g1=self.p_g1,
                p_g2=self.p_g2,
                p_s=self.p_s,
                batch=self.batch,
                cold_restart=self.cold_restart,
                decay=self.decay,
                min_prob=self.min_prob,
                zero_init=self.zero_init
                )
        elif self.mode in 'rule_extraction_with_sls_val':
            val, val_label = self.preprocess(data=validation_data,
                                             label=validation_label)
            self.found_formula = SLS.rule_extraction_with_sls_val(
                # found_formula_val=SLS.rule_extraction_with_sls(
                train=train,
                train_label=label,
                val=val,
                val_label=val_label,
                number_of_disjunction_term=self.number_of_disjunction_term,
                maximum_steps_in_SLS=self.maximum_steps_in_SLS,
                kernel=self.init_with_kernel,
                p_g1=self.p_g1,
                p_g2=self.p_g2,
                p_s=self.p_s,
                batch=self.batch,
                cold_restart=self.cold_restart,
                decay=self.decay,
                min_prob=self.min_prob,
                zero_init=self.zero_init
            )
        else:
            raise ValueError('Your mode: {} is not supported, \n'
                             'supported are \'rule_extraction_with_sls\''
                             ' and \'rule_extraction_with_sls_val\''.format(self.mode))

    def prediction(self,data, original_label):
            data_flat, label = self.preprocess(data, original_label)
            prediction = SLS.calc_prediction_in_C(data=data_flat,
                                                  label_shape=label.shape,
                                                  found_formula=self.found_formula)
            num_input = data.shape[0]
            error = np.sum(label != prediction)
            # calculate accuracy
            accuracy = (num_input - error) / num_input
            return accuracy


    def preprocess(self, data, label):

        # flatten data
        shape = data.shape
        data_flat = data.reshape((shape[0], - 1))

        if self.arg_min_label:
            label_flat = np.argmin(label, axis=1)
        else:
            # one hot label should be casted with arg_max to a single number
            label_flat = np.argmax(label, axis=1)

        label_flat = get_data.transform_to_boolean(label_flat)
        if type(data_flat[0][0]) == np.bool_ and type(label_flat[0]) == np.bool_:
            return data_flat, label_flat
        else:
            raise ValueError('Data to process in SLS should be '
                             'from typ bool they are from type: data {}, lable {}'
                             .format(type(data_flat[0][0]), type(label_flat[0])))





