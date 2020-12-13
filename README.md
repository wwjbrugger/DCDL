## Structure of the repository

#####  NN_DCDL_SLS_Blackbox_comparison
compares the DCDL approach with the SLS black box approach and the neural network.

 #####  visualize_rules_found Module
Visualizes the rules found with the SLS algorithm. 

#### compare_dither
Implementation of various dither algorithms and test them on the neural net

#### tests_sls_label_order
Experiment the influence of the performance of the SLS algorithm,
if the one-hot-label are inverted from [1,0] to [0,1]. 
Good starting point, if SLS Algorithm should be used alone. 

#####  SLS_Algorithm.py
Interface to the SLS implementation on Python side. 
Converts lists to binary arays 
calls C++ code of the SLS implementation. 

#####  parallel_sls module 
C++ implementation of the SLS algorithm 

#####  model module 
Structure of the neural networks for the experiments 
Structure for storing the extracted logical rules 

#### install necessary packages 
`sudo pip install -r requirements.txt`
