import copy
import numpy as np

import NN_DCDL_SLS_Blackbox_comparison.DCDL_Model.DCDL_Dense as DCDL_Dense
import NN_DCDL_SLS_Blackbox_comparison.DCDL_Model.DCDL_Convolution as DCDL_Convolution
import NN_DCDL_SLS_Blackbox_comparison.DCDL_Model.DCDL_Max_Pooling as DCDL_Max_Pooling
import NN_DCDL_SLS_Blackbox_comparison.get_data as get_data
class DCDL:
    def __init__(self, operations, arg_min_label ):
        if not isinstance(operations, dict):
            # check if operations is a dict
            raise ValueError ('operations object has to be a dictionary but it is a {}'.format(type(operations)))

        # saves operation. to perform operations they should have the form {int(position) : object_description_dic}
        self.operations = copy.deepcopy(operations)
        self.arg_min_label = arg_min_label

        # create DCDL subobjects
        for key, object_description_dic in self.operations.items():
            DCDL_operation = self.get_operation_object(object_description_dic=object_description_dic)
            self.operations[key]['operation'] = DCDL_operation

    def get_operation_object(self, object_description_dic):
        # get a string of which kind the operation is and returns an object of this kind
        kind_of_operation = object_description_dic['kind']
        if kind_of_operation in 'convolution':
            return DCDL_Convolution.Convolution(object_description_dic['properties'])
        elif kind_of_operation in 'max_pool':
            return DCDL_Max_Pooling.Max_Pooling(object_description_dic['properties'])
        elif kind_of_operation in 'dense':
            return DCDL_Dense.Dense(object_description_dic['properties'])
        else:
            raise ValueError('kind of operation is not supported. \n'
                             'your operation is {}'
                             'possible operations are: convolution, max_pool, dense')

    def train(self, train_data, validation_data,
              use_prediction_operation_before, DCDL_data_dic, DCDL_val_dic):
        #train DCDL
        # get operation in the right order
        order_index = list(self.operations.keys())
        order_index.sort()

        current_train_data = train_data
        current_validation_data = validation_data
        # iterate through dic
        for key in order_index:
            # get operation to train
            operation_dic = self.operations[key]
            operation_name = operation_dic['name']

            print('------ DCDL train to approximate operation {} -------'.format(operation_name))
            operation = operation_dic['operation']
            # get label
            current_train_label = DCDL_data_dic[operation_name]
            current_validation_label = DCDL_val_dic[operation_name]
            # train operation
            operation.train(train_data=current_train_data,
                            train_label=current_train_label,
                            validation_data=current_validation_data,
                            validation_label = current_validation_label)
            #update current_train_data and current_validation_data
            if use_prediction_operation_before:
                #update training and validation data with
                # prediction of the operation just learned
                current_train_data, acc_train=operation.prediction(data=current_train_data,
                                                                   original_label= current_train_label)
                current_validation_data, acc_val=operation.prediction(data=current_validation_data,
                                                                      original_label=current_validation_label)
            else:
                # update training and validation data with
                # prediction of the operation just learned
                current_train_data = current_train_label
                current_validation_data = current_validation_label


    def prediction(self, data, original_label):

        data, label = self.preprocess(data, original_label)
        num_input = data.shape[0]


        order_index = list(self.operations.keys())
        order_index.sort()

        current_data = data

        # iterate through dic
        for key in order_index:
            # get operation to train
            operation_dic = self.operations[key]
            operation = operation_dic['operation']
            current_data, _  = operation.prediction(data=current_data,
                                                    original_label = None)
        # last prediction is prediction of DCDL
        prediction = current_data

        # calculate accuracy
        error = np.sum(label != prediction)
        accuracy = (num_input - error) / num_input
        return accuracy


    def preprocess(self, data, label):
        if self.arg_min_label:
            label_flat = np.argmin(label, axis=1)
        else:
            # one hot label should be casted with arg_max to a single number
            label_flat = np.argmax(label, axis=1)

        label_flat = get_data.transform_to_boolean(label_flat)

        # DCDL has as input pictures in the shape [num_pic, width, height, channels]
        if type(data[0][0][0][0]) == np.bool_ and type(label_flat[0]) == np.bool_:
            return data, label_flat
        else:
            raise ValueError('Data to process in SLS should be '
                             'from typ bool they are from type: data {}, lable {}'
                             .format(type(data[0][0][0][0]), type(label_flat[0])))



