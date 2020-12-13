import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import SLS_Algorithm as SLS
import numpy as np


class Dense:
    def __init__(self, properties):
        self.found_formula = None
        self.properties = properties
        # saves which form the output has to be e.g. [None, 14,14,8}
        # None has to be set to the number of input example
        self.output_shape = None

    def preprocess(self, data, label):
        if label is not None:
            self.output_shape = list(label.shape)
            self.output_shape[0] = data.shape[0]
        else:
            self.output_shape[0] = data.shape[0]

        data_flat = data.reshape((data.shape[0], - 1))
        return data_flat, label



    def train(self, train_data, train_label, validation_data, validation_label):
        train, label = self.preprocess(data=train_data,
                                       label=train_label)

        if self.properties['SLS_dic']['mode'] in 'rule_extraction_with_sls':
            self.found_formula = SLS.rule_extraction_with_sls(
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
            val, val_label = self.preprocess(data=validation_data,
                                             label=validation_label)
            self.found_formula = SLS.rule_extraction_with_sls_val(
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

    def prediction(self, data, original_label):
        # original_label can be None
        data_flat, label = self.preprocess(data, original_label)
        output_shape = self.output_shape

        prediction = SLS.calc_prediction_in_C(data=data_flat,
                                              label_shape=output_shape,
                                              found_formula=self.found_formula)
        acc = None
        return prediction, acc

