# Comparision DCDL vs SLS blackbox

Scripts for the experiment to compare the approach DCDL with plain (blackbox) SLS.
The basic idea of the DCDL approach is to approximate the individual layers of the neural network with
logical rules which are found with the SLS algorithm. 
The (blackbox) SLS approach is trained with the input data and label. 

The experiment is run for MNIST, Fashion, Cifar dataset.
For each of the 10 labels the experiments where repeated 3 times. 

Insert point is the **acc_main.py** script, in which the meta parameters
like data set and the label for the one against all tests can be set. 

In the script **acc_data_generation.py** the intermediate results of the neural network
are extracted which are necessary to approximate the network with logical rules. 

The script **sls_one_against_all.py** contains the methods to investigate how the
SLS algorithm performs directly on the data set.

The script **acc_extracting_pictures.py** the DCDL approach is trained. 
The methods for making predictions with DCDL are used there. 
##### Update possibility (was not changed to be consistent with existing experiment results):
#####    rename acc_extracting_pictures.py in extract_DCDL_Rules

The script **dithering.py** uses the floyd-steinberg algorithm to dither pictures (set pixel to 0 or 1).

The script **acc_train** preprocess the pictures and train and evaluate the neural net. 

The script **evaluate_result_frame.py** analyse the results of several runs of the experiment.

The script **helper_methods.py** contain methods which are needed at multiple scripts .
See comments in methods for more details. 

The script **acc_reduce_kernel.py** was used for visualization of the kernels 
##### Update possibility (was not changed to be consistent with existing experiment results):
#####   delete this script it is not used in the experiment  