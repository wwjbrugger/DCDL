import model.boolean_Formel as bofo
import numpy as np
import parallel_sls.python_wrapper.sls_wrapper as sls_wrapper
import parallel_sls.python_wrapper.data_wrapper as data_wrapper


def rule_extraction_with_sls(data, label, number_of_disjunction_term, maximum_steps_in_SLS, kernel):
    # use SLS with a train, validation and test set
    first_split, second_split = calculate_border_values_train_test_validation(data)
    # number of input variables is rounded up to a multiple of eight
    # C++ implementation stores formula in uint 8 variables
    num_of_features = (8 - data.shape[1] % 8) + data.shape[
        1]
    # how many uint8 variables are needed
    num_of_8_bit_units_to_store_feature =  int(num_of_features / 8)

    #
    training_set_data_packed_continguous, training_set_label_bool_continguous \
        , validation_set_data_packed_continguous, validation_set_label_bool_continguous \
        , test_set_data_packed_continguous, test_set_label_bool_continguous\
        = pack_and_store_contiguous_array_for_sls(data, label,first_split, second_split)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Free space to store formulas found
    pos_neg = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    pos_neg_to_store = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off_to_store = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))


    if not isinstance(kernel, bool):
        # Initialisation with kernel values from neural net not used in experiment
        if kernel.ndim == 1:
            output_relevant, output_negated = bofo.Boolean_formula.split_fomula(kernel)
            output_relevant_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_relevant)
            output_negated_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_negated)
            size_kernel_8bit =  output_relevant_numbers.size
            for i in range(0, number_of_disjunction_term * num_of_8_bit_units_to_store_feature, num_of_8_bit_units_to_store_feature):
                pos_neg[i:i+size_kernel_8bit] = output_negated_numbers
                on_off[i:i+size_kernel_8bit] = output_relevant_numbers
        else:
            raise ValueError("kernel should be one dimensional no {}".format(kernel.ndim))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Start SLS
    sls_obj = sls_wrapper.sls_test(clauses_n=number_of_disjunction_term,
                                   maxSteps=maximum_steps_in_SLS,
                                   p_g1=.5,  # Prob of rand term in H
                                   p_g2=.5,  # Prob of rand literal in H
                                   p_s=.5,  # Prob of rand term in H
                                   data=training_set_data_packed_continguous,
                                   label=training_set_label_bool_continguous,
                                   data_val=validation_set_data_packed_continguous,
                                   label_val=validation_set_label_bool_continguous,
                                   data_test=test_set_data_packed_continguous,  # Data input
                                   label_test=test_set_label_bool_continguous,  # Label input
                                   pos_neg=pos_neg,  # Positive or negative for formula
                                   on_off=on_off,  # Mask for formula
                                   pos_neg_to_store=pos_neg_to_store,  # Positive or negative for formula
                                   on_off_to_store=on_off_to_store,  # Mask for formula
                                   vector_n=first_split,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   vector_n_val=second_split - first_split,
                                   # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   vector_n_test=data.shape[0] - second_split,
                                   # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   features_n=num_of_features,  # of Features
                                   batch=True,
                                   cold_restart=True,
                                   decay=0,
                                   min_prob=0,
                                   zero_init=False
                                   )
    found_formula = bofo.Boolean_formula(on_off_to_store, pos_neg_to_store, number_of_disjunction_term, total_error = sls_obj.total_error)
    # calculate accuracy on train set
    # accuracy = number_of_correct_predictions / total_number_of_prediction
    # The first split is the train set
    found_formula.train_acc = (first_split - found_formula.total_error) / first_split
    return found_formula



def rule_extraction_with_sls_without_test(data, label, number_of_disjunction_term, maximum_steps_in_SLS, kernel  ):
    # run sls wit train and validation data
    # This method returns the indices at which the data set is divided into the training, validation and test set.
    # Since only the training and validation set is used,
    # the training set is assigned the data of the training set + validation set.
    # The validation set is filled with the test set.

    _, train_split = calculate_border_values_train_test_validation(data)

    # number of input variables is rounded up to a multiple of eight
    # C++ implementation stores formula in uint 8 variables
    num_of_features = (8 - data.shape[1] % 8) + data.shape[
        1]
    # how many uint8 variables are needed
    num_of_8_bit_units_to_store_feature = int(num_of_features / 8)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    training_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[:train_split])

    training_set_label_bool_continguous = np.ascontiguousarray(label[:train_split], dtype=np.bool)

    validation_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[train_split:])

    validation_set_label_bool_continguous = np.ascontiguousarray(label[train_split:],
                                                                 dtype=np.bool)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # training_set_data_packed_continguous, training_set_label_bool_continguous \
    #     , validation_set_data_packed_continguous, validation_set_label_bool_continguous \
    #     , test_set_data_packed_continguous, test_set_label_bool_continguous \
    #     = pack_and_store_contiguous_array_for_sls(data, label, first_split, second_split)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Free space to store formulas found
    pos_neg = np.ascontiguousarray(
        np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off = np.ascontiguousarray(
        np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    pos_neg_to_store = np.ascontiguousarray(
        np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off_to_store = np.ascontiguousarray(
        np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))

    if not isinstance(kernel, bool):
        # Initialisation with kernel values from neural net not used in experiment
        if kernel.ndim == 1:
            output_relevant, output_negated = bofo.Boolean_formula.split_fomula(kernel)
            output_relevant_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_relevant)
            output_negated_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_negated)
            size_kernel_8bit = output_relevant_numbers.size
            for i in range(0, number_of_disjunction_term * num_of_8_bit_units_to_store_feature,
                           num_of_8_bit_units_to_store_feature):
                pos_neg[i:i + size_kernel_8bit] = output_negated_numbers
                on_off[i:i + size_kernel_8bit] = output_relevant_numbers
        else:
            raise ValueError("kernel should be one dimensional no {}".format(kernel.ndim))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Start SLS
    sls_obj = sls_wrapper.sls_val(clauses_n=number_of_disjunction_term,
                                   maxSteps=maximum_steps_in_SLS,
                                   p_g1=.5,  # Prob of rand term in H
                                   p_g2=.5,  # Prob of rand literal in H
                                   p_s=.5,  # Prob of rand term in H
                                   data=training_set_data_packed_continguous,
                                   label=training_set_label_bool_continguous,
                                   data_val=validation_set_data_packed_continguous,
                                   label_val=validation_set_label_bool_continguous,
                                   pos_neg=pos_neg,  # Positive or negative for formula
                                   on_off=on_off,  # Mask for formula
                                   pos_neg_to_store=pos_neg_to_store,  # Positive or negative for formula
                                   on_off_to_store=on_off_to_store,  # Mask for formula
                                   vector_n=train_split,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   vector_n_val=data.shape[0] - train_split,
                                   # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   features_n=num_of_features,  # of Features
                                   batch=True,
                                   cold_restart=True,
                                   decay=0,
                                   min_prob=0,
                                   zero_init=False
                                   )
    found_formula = bofo.Boolean_formula(on_off_to_store, pos_neg_to_store, number_of_disjunction_term,
                                         total_error=sls_obj.total_error)
    # calculate accuracy on train set
    # accuracy = number_of_correct_predictions / total_number_of_prediction
    # The first split is the train set
    found_formula.train_acc = (train_split - found_formula.total_error) / train_split
    return found_formula




"""
Input in SLS are values in True/False Form 
"""
def rule_extraction_with_sls_without_validation_and_test(data, label, number_of_disjunction_term, maximum_steps_in_SLS, kernel  ):
    # run SLS with maximal number of training samples

    # number of input variables is rounded up to a multiple of eight
    # C++ implementation stores formula in uint 8 variables
    num_of_features = (8 - data.shape[1]) % 8 + data.shape[1]

    # how many uint8 variables are needed
    num_of_8_bit_units_to_store_feature = int(num_of_features / 8)

    # pack data in C## compatible arrays

    training_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(data)

    training_set_label_bool_continguous = np.ascontiguousarray(label, dtype=np.bool)



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Free space to store formulas found
    pos_neg = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    pos_neg_to_store = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off_to_store = np.ascontiguousarray(np.empty((number_of_disjunction_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))

    if not isinstance(kernel, bool):
        # Initialisation with kernel values from neural net not used in experiment
        if kernel.ndim == 1:
            output_relevant, output_negated = bofo.Boolean_formula.split_fomula(kernel)
            output_relevant_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_relevant)
            output_negated_numbers = bofo.Boolean_formula.transform_arrays_code_in_number_code(output_negated)
            size_kernel_8bit =  output_relevant_numbers.size
            for i in range(0, number_of_disjunction_term * num_of_8_bit_units_to_store_feature, num_of_8_bit_units_to_store_feature):
                pos_neg[i:i+size_kernel_8bit] = output_negated_numbers
                on_off[i:i+size_kernel_8bit] = output_relevant_numbers
        else:
            raise ValueError("kernel should be one dimensional no {}".format(kernel.ndim))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Start SLS
    sls_obj = sls_wrapper.sls(clauses_n=number_of_disjunction_term,
                                   maxSteps=maximum_steps_in_SLS,
                                   p_g1=.5,  # Prob of rand term in H
                                   p_g2=.5,  # Prob of rand literal in H
                                   p_s=.5,  # Prob of rand term in H
                                   data=training_set_data_packed_continguous,
                                   label=training_set_label_bool_continguous,
                                   pos_neg=pos_neg,  # Positive or negative for formula
                                   on_off=on_off,  # Mask for formula
                                   pos_neg_to_store=pos_neg_to_store,  # Positive or negative for formula
                                   on_off_to_store=on_off_to_store,  # Mask for formula
                                   vector_n=int(data.shape[0]),  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   features_n=num_of_features,  # of Features
                                   batch=True,
                                   cold_restart=True,
                                   decay=0,
                                   min_prob=0,
                                   zero_init=False
                                   )

    found_formula = bofo.Boolean_formula(on_off_to_store, pos_neg_to_store, number_of_disjunction_term, total_error = sls_obj.total_error)
    # calculate accuracy on train set
    # accuracy = number_of_correct_predictions / total_number_of_prediction
    # The first split is the train set
    found_formula.train_acc = (data.shape[0] - found_formula.total_error) / data.shape[0]
    return found_formula
    #return bofo.Boolean_formula(on_off_to_store, pos_neg_to_store, number_of_disjunction_term, total_error = sls_obj.total_error)



def calc_prediction_in_C(data, label_shape, found_formula ):
    # use C++ code to calculate prediction for given data with found formula
    num_anzahl_input_data = int(data.shape[0])
    num_of_features = found_formula.variable_pro_term
    number_of_disjunction_term = found_formula.number_of_disjunction_term_in_SLS
    data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(data)
    space_label_bool_continguous = np.ascontiguousarray(np.empty(label_shape,np.bool), dtype=np.bool)
    pos_neg_to_store = np.ascontiguousarray(found_formula.pixel_negated_in_number_code.copy(), dtype=np.uint8)
    on_off_to_store = np.ascontiguousarray( found_formula.pixel_relevant_in_number_code, dtype=np.uint8)
    prediction_obj = sls_wrapper.calc_prediction(data_packed_continguous,
                                                 space_label_bool_continguous,
                                                 pos_neg_to_store,
                                                 on_off_to_store,
                                                 num_anzahl_input_data,
                                                 number_of_disjunction_term,
                                                 num_of_features)
    return space_label_bool_continguous


def pack_and_store_contiguous_array_for_sls(data, label,first_split, second_split):
    # pack data in C## compatible arrays

    training_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[:first_split])


    training_set_label_bool_continguous = np.ascontiguousarray(label[:first_split], dtype=np.bool)

    validation_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[first_split:second_split])

    validation_set_label_bool_continguous = np.ascontiguousarray(label[first_split:second_split],
                                                                 dtype=np.bool)

    test_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[second_split:])

    test_set_label_bool_continguous = np.ascontiguousarray(label[second_split:], dtype=np.bool)

    return training_set_data_packed_continguous, training_set_label_bool_continguous \
        , validation_set_data_packed_continguous, validation_set_label_bool_continguous \
        , test_set_data_packed_continguous, test_set_label_bool_continguous


def calculate_border_values_train_test_validation(data):
    # calculate border to split data in  in 2/3 train data , 1/6 validation data und 1/6 test data
    first_split = int(data.shape[0] * 2 / 3)
    second_split = int(data.shape[0] * 2 / 3) + int(
        (data.shape[0] - int(data.shape[0] * 2 / 3)) * 1 / 2)
    return first_split, second_split


