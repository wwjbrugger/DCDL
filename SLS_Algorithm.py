import model.boolean_Formel as bofo
import numpy as np
import parallel_sls.python_wrapper.sls_wrapper as sls_wrapper
import parallel_sls.python_wrapper.data_wrapper as data_wrapper


def rule_extraction_with_sls(data, label, Number_of_Product_term, Maximum_Steps_in_SKS):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Dataset stats  flat_training_set.shape[1] = (196, 4, 4, 256) sollte sein (196,4,4,1)
    # data.shape[1] Anzahl
    first_split, second_split = calculate_border_values_train_test_validation(data)
    num_of_features = (8 - data.shape[1] % 8) + data.shape[
        1]  # anzahl an eingabewerte gerunfet auf das nächstgrößere Vielfaches von 8  # data.shape[1] % 8 + data.shape[1] # nötigerspeicherplatz in 8 bit stücken

    num_of_8_bit_units_to_store_feature =  int(num_of_features / 8)

    training_set_data_packed_continguous, training_set_label_bool_continguous \
        , validation_set_data_packed_continguous, validation_set_label_bool_continguous \
        , test_set_data_packed_continguous, test_set_label_bool_continguous\
        = pack_and_store_contiguous_array_for_sls(data, label,first_split, second_split)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Free space to store formulas found
    pos_neg = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    pos_neg_to_store = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off_to_store = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Start SLS
    sls_obj = sls_wrapper.sls_test(clauses_n=Number_of_Product_term,
                                   maxSteps=Maximum_Steps_in_SKS,
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


   # boolsche_formel  = get_boolsche_formel(on_off_to_store, pos_neg_to_store)
  #  variable_on_off= [np.unpackbits(x)] for x in on_off_to_store]

    return bofo.Boolsche_formel(on_off_to_store, pos_neg_to_store, Number_of_Product_term)

"""
Input in SLS are values in True/False Form 
"""
def rule_extraction_with_sls_without_validation(data, label, Number_of_Product_term, Maximum_Steps_in_SKS, kernel = False ):
    first_split, second_split = int(data.shape[0]), int(data.shape[0])
    num_of_features = (8 - data.shape[1]) % 8 + data.shape[1] # anzahl an eingabewerte gerundet auf das nächstgrößere Vielfaches von 8  # data.shape[1] % 8 + data.shape[1] # nötigerspeicherplatz in 8 bit stücken
    num_of_8_bit_units_to_store_feature = int(num_of_features / 8) 

    training_set_data_packed_continguous, training_set_label_bool_continguous \
        ,_, _ \
        , _, _\
        = pack_and_store_contiguous_array_for_sls(data, label,first_split, second_split)

    validation_set_data_packed_continguous, validation_set_label_bool_continguous \
    , test_set_data_packed_continguous, test_set_label_bool_continguous =\
        training_set_data_packed_continguous, training_set_label_bool_continguous\
            , training_set_data_packed_continguous, training_set_label_bool_continguous


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Free space to store formulas found
    pos_neg = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    pos_neg_to_store = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))
    on_off_to_store = np.ascontiguousarray(np.empty((Number_of_Product_term * num_of_8_bit_units_to_store_feature,), dtype=np.uint8))

    if not isinstance(kernel, bool): # initalsation with kernel values
        if kernel.ndim == 1:
            output_relevant, output_negated = bofo.Boolsche_formel.split_fomula(kernel)
            output_relevant_numbers = bofo.Boolsche_formel.transform_arrays_code_in_number_code(output_relevant)
            output_negated_numbers = bofo.Boolsche_formel.transform_arrays_code_in_number_code(output_negated)
            size_kernel_8bit =  output_relevant_numbers.size
            for i in range(0, Number_of_Product_term * num_of_8_bit_units_to_store_feature, num_of_8_bit_units_to_store_feature):
                pos_neg[i:i+size_kernel_8bit] = output_negated_numbers
                on_off[i:i+size_kernel_8bit] = output_relevant_numbers
        else:
            raise ValueError("kernel should be one dimensional no {}".format(kernel.ndim))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Start SLS
    sls_obj = sls_wrapper.sls(clauses_n=Number_of_Product_term,
                                   maxSteps=Maximum_Steps_in_SKS,
                                   p_g1=.5,  # Prob of rand term in H
                                   p_g2=.5,  # Prob of rand literal in H
                                   p_s=.5,  # Prob of rand term in H
                                   data=training_set_data_packed_continguous,
                                   label=training_set_label_bool_continguous,
                                   pos_neg=pos_neg,  # Positive or negative for formula
                                   on_off=on_off,  # Mask for formula
                                   pos_neg_to_store=pos_neg_to_store,  # Positive or negative for formula
                                   on_off_to_store=on_off_to_store,  # Mask for formula
                                   vector_n=first_split,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                                   features_n=num_of_features,  # of Features
                                   batch=True,
                                   cold_restart=True,
                                   decay=0,
                                   min_prob=0,
                                   zero_init=False
                                   )
   # boolsche_formel  = get_boolsche_formel(on_off_to_store, pos_neg_to_store)
  #  variable_on_off= [np.unpackbits(x)] for x in on_off_to_store]

    return bofo.Boolsche_formel(on_off_to_store, pos_neg_to_store, Number_of_Product_term, total_error = sls_obj.total_error)

def calc_prediction_in_C(data, label_shape, found_formula ):
    num_anzahl_input_data = int(data.shape[0])
    num_of_features = found_formula.variable_pro_term
    Number_of_Product_term = found_formula.number_of_product_term
    data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(data)
    space_label_bool_continguous = np.ascontiguousarray(np.empty(label_shape,np.bool), dtype=np.bool)
    pos_neg_to_store = np.ascontiguousarray(found_formula.pixel_negated_in_number_code.copy(), dtype=np.uint8)
    on_off_to_store = np.ascontiguousarray( found_formula.pixel_relevant_in_number_code, dtype=np.uint8)
    prediction_obj = sls_wrapper.calc_prediction(data_packed_continguous,
                                                 space_label_bool_continguous,
                                                 pos_neg_to_store,
                                                 on_off_to_store,
                                                 num_anzahl_input_data,
                                                 Number_of_Product_term,
                                                 num_of_features)
    return space_label_bool_continguous


def pack_and_store_contiguous_array_for_sls(data, label,first_split, second_split):
   # first_split, second_split = calculate_border_values_train_test_validation(data)

    training_set_data_packed_continguous = data_wrapper.binary_to_packed_uint8_continguous(
        data[:first_split])
    # input_data_in_SKS = data[:first_split]  # shape(130,4,4,256)

    training_set_label_bool_continguous = np.ascontiguousarray(label[:first_split], dtype=np.bool)  # (1,14,14,256)
    input_label_in_SKS = label[:first_split]  # shape(196,)

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
    # berechnet grenzen um pixel in bild (#196) in 2/3 trainingset, 1/6 validationset und 1/6 testset
    first_split = int(data.shape[0] * 2 / 3)
    second_split = int(data.shape[0] * 2 / 3) + int(
        (data.shape[0] - int(data.shape[0] * 2 / 3)) * 1 / 2)
    return first_split, second_split


