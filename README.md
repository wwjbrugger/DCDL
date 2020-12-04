## Structure of the repository

#####  comparision_DCDL_vs_SLS module
compares the DCDL approach with the SLS black box approach and the neural network. 
 #####  visualize_rules_found Module
Visualizes the rules found with the SLS algorithm. 

#### compare_dither
Implementation of various dither algorithms and test them on the neural net

#####  SLS_Algorithm.py
Interface to the SLS implementation on Python side. 
Converts lists to binary arays 
calls C++ code of the SLS implementation. 
#####  parallel_sls module 
C++ implementation of the SLS algorithm 
#####  model module 
Structure of the neural networks for the experiments 
Structure for storing the extracted logical rules 
