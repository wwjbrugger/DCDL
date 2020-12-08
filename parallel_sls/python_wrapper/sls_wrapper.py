import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import os

# load the library by creating an instance of CDLL by calling the constructor
libc = ct.CDLL(os.environ.get('BLD_PATH', "parallel_sls/bld/Parallel_SLS_shared"))
sls_func_obj = libc.sls
#sls_func_obj.restype = None
sls_func_obj.restype = ct.c_uint32
sls_func_obj.argtypes = [ct.c_uint32,  # clauses_n
                         ct.c_uint32,  # maxSteps
                         ct.c_float,  # p_g1
                         ct.c_float,  # p_g2
                         ct.c_float,  # p_gs
                         ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data
                         ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label
                         ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg
                         ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off
                         ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg to store
                         ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off to store
                         ct.c_uint32,  # vector_n
                         ct.c_uint32,  # features_n
                         ct.c_bool,   # batch
                         ct.c_bool,   # cold_restart
                         ct.c_float,  # decay
                         ct.c_float,  # min_prob
                         ct.c_bool    # zero_init]
                         ]

sls_val_func_obj = libc.sls_val
sls_val_func_obj.restype = ct.c_uint32
sls_val_func_obj.argtypes = [ct.c_uint32,  # clauses_n
                             ct.c_uint32,  # maxSteps
                             ct.c_float,  # p_g1
                             ct.c_float,  # p_g2
                             ct.c_float,  # p_gs
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data
                             ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data val
                             ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label val
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg to store
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off to store
                             ct.c_uint32,  # vector_n
                             ct.c_uint32,  # vector_n val
                             ct.c_uint32,  # features_n
                             ct.c_bool,   # batch
                             ct.c_bool,   # cold_restart
                             ct.c_float,  # decay
                             ct.c_float,  # min_prob
                             ct.c_bool    # zero_init]
                             ]

sls_test_func_obj = libc.sls_test
sls_test_func_obj.restype = ct.c_uint32
sls_test_func_obj.argtypes = [ct.c_uint32,  # clauses_n
                             ct.c_uint32,  # maxSteps
                             ct.c_float,  # p_g1
                             ct.c_float,  # p_g2
                             ct.c_float,  # p_gs
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data
                             ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data val
                             ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label val
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data test
                             ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label test
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg to store
                             ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off to store
                             ct.c_uint32,  # vector_n
                             ct.c_uint32,  # vector_n val
                             ct.c_uint32,  # vector_n test
                             ct.c_uint32,  # features_n
                             ct.c_bool,   # batch
                             ct.c_bool,   # cold_restart
                             ct.c_float,  # decay
                             ct.c_float,  # min_prob
                             ct.c_bool    # zero_init]
                             ]

prediction_obj = libc.calc_prediction
prediction_obj.restype = None
prediction_obj.argtypes = [ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # data
                           ndpointer(ct.c_bool, flags="C_CONTIGUOUS"),   # label
                           ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # pos_neg to store
                           ndpointer(ct.c_uint8, flags="C_CONTIGUOUS"),  # on_off to store
                           ct.c_uint32,  # vector_n
                           ct.c_uint32,  # clauses_n
                           ct.c_uint32  # features_n
                          ]


class sls(object):
    def __init__(self,
                 clauses_n,  # of DNFs
                 maxSteps,  # of Updates
                 p_g1,  # Prob of rand term in H
                 p_g2,  # Prob of rand literal in H
                 p_s,  # Prob of rand term in H
                 data,  # Data input
                 label,  # Label input
                 pos_neg,  # Positive or negative for formula
                 on_off,  # Mask for formula
                 pos_neg_to_store,  # Positive or negative for formula
                 on_off_to_store,  # Mask for formula
                 vector_n,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 features_n,  # of Features
                 batch,  # If score calculation should be done batchwise for given clause
                 cold_restart,  # Restar if stuck in bad local minimum
                 decay,  # Decay factor, could be zero. Up to min_prob
                 min_prob,  # Not decay below this threshold
                 zero_init  # Wether to go bigger steps in case of no sucess
                 ):
        total_error = sls_func_obj(clauses_n,
                     maxSteps,
                     p_g1,
                     p_g2,
                     p_s,
                     data,
                     label,
                     pos_neg,
                     on_off,
                     pos_neg_to_store,
                     on_off_to_store,
                     vector_n,
                     features_n,
                     batch,
                     cold_restart,
                     decay,
                     min_prob,
                     zero_init)
        self.total_error = total_error


class sls_val(object):
    def __init__(self,
                 clauses_n,  # of DNFs
                 maxSteps,  # of Updates
                 p_g1,  # Prob of rand term in H
                 p_g2,  # Prob of rand literal in H
                 p_s,  # Prob of rand term in H
                 data,  # Data input
                 label,  # Label input
                 data_val,  # Data input
                 label_val,  # Label input
                 pos_neg,  # Positive or engative for formula
                 on_off,  # Mask for formula
                 pos_neg_to_store,  # Positive or engative for formula
                 on_off_to_store,  # Mask for formula
                 vector_n,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 vector_n_val,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 features_n,  # of Features
                 batch,  # If score calculation should be done batchwise for given clause
                 cold_restart,  # Restar if stuck in bad local minimum
                 decay,  # Decay factor, could be zero. Up to min_prob
                 min_prob,  # Not decay below this threshold
                 zero_init  # Wether to go bigger steps in case of no sucess
                 ):
        total_error = sls_val_func_obj(clauses_n,
                         maxSteps,
                         p_g1,
                         p_g2,
                         p_s,
                         data,
                         label,
                         data_val,
                         label_val,
                         pos_neg,
                         on_off,
                         pos_neg_to_store,
                         on_off_to_store,
                         vector_n,
                         vector_n_val,
                         features_n,
                         batch,
                         cold_restart,
                         decay,
                         min_prob,
                         zero_init)
        self.total_error = total_error

class sls_test(object):
    def __init__(self,
                 clauses_n,  # of DNFs
                 maxSteps,  # of Updates
                 p_g1,  # Prob of rand term in H
                 p_g2,  # Prob of rand literal in H
                 p_s,  # Prob of rand term in H
                 data,  # Data input
                 label,  # Label input
                 data_val,  # Data input
                 label_val,  # Label input
                 data_test,  # Data input
                 label_test,  # Label input
                 pos_neg,  # Positive or engative for formula
                 on_off,  # Mask for formula
                 pos_neg_to_store,  # Positive or negative for formula
                 on_off_to_store,  # Mask for formula
                 vector_n,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 vector_n_val,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 vector_n_test,  # of data vectors !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
                 features_n,  # of Features
                 batch,  # If score calculation should be done batchwise for given clause
                 cold_restart = True,  # Restar if stuck in bad local minimum
                 decay = 0,  # Decay factor, could be zero. Up to min_prob
                 min_prob = 0,  # Not decay below this threshold
                 zero_init = False  # Wether to go bigger steps in case of no sucess
                 ):
        total_error = sls_test_func_obj(clauses_n,
                         maxSteps,
                         p_g1,
                         p_g2,
                         p_s,
                         data,
                         label,
                         data_val,
                         label_val,
                         data_test,
                         label_test,
                         pos_neg,
                         on_off,
                         pos_neg_to_store,
                         on_off_to_store,
                         vector_n,
                         vector_n_val,
                         vector_n_test,
                         features_n,
                         batch,
                         cold_restart,
                         decay,
                         min_prob,
                         zero_init)
        self.total_error = total_error

class calc_prediction(object):
    def __init__(self, data, prediction_label, pos_neg_to_store,  on_off_to_store, vector_n,  clauses_n, features_n):
        prediction_obj(data, prediction_label, pos_neg_to_store,  on_off_to_store, vector_n,  clauses_n, features_n)
