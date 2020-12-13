# Comparision Neural Net, DCDL and SLS blackbox

Scripts for the experiment to compare the approach neural net, DCDL with plain (blackbox) SLS.
The basic idea of the DCDL approach is to approximate the individual layers of the neural network with
logical rules which are found with the SLS algorithm. 

The (blackbox) SLS label approach is trained with the input data and label.

The (blackbox) SLS prediction approach is trained with the input data and the predicted label of the net .

###Experimental setup
The experiment is run for MNIST, Fashion, Cifar dataset.
For each of the 10 labels the experiments where repeated 3 times. 

### How to run one experiment 

Use set_settings_cifar.py and set_settings_numbers_fashion.py to 
set the settings with which you would like to run the experiment. 

You can start an experiment with `python [dataset] [label]`
- The first parameter set is the dataset to use. The keyword numbers will use the MNIST dataset.
  The other options are fashion for FASHION-MNIST and cifar for CIFAR
- The second parameter is the label you want to use as the 'one' in one against all testings. 

### How to run a set of experiments
If you want to run a set of experiments use the bash script **run_set_of_experiments.sh**
in the terminal with  
`bash run_set_of_experiments.sh -d [dataset] -n [numbers of repetions] -t [threats to use]`.
A repetition includes all 10 labels of a data set 

### Structure of the folder 

- DCDL_Model folder to contains all classes needed for the DCDL approach 
- Neural_net_model folder contains the model of th neural net
- SLS_black_box_model folder contains class for the SLS black box approaches 


-  The folder terminal_out contains the output of the start script, 
   if it is started with run_set_of_experiments.sh .
   
- The folder results contains pandas frames with the results of the experiment 
e.g. for the cifar dataset with the settings Arg_max_0 and label 0 against all:
  
|                           |   Neural network |       DCDL |   SLS BB prediction |   SLS BB label | setup             |
|:-------------------------:|:----------------:|:----------:|:-------------------:|:--------------:|:------------------|
| setup                     |       nan        | nan        |          nan        |     nan        | Arg_min_0_cifar_0 |
| Training time             |        74.8313   | 223.991    |          290.049    |     238.864    | nan               |
| Training set              |         0.740169 |   0.71573  |            0.664963 |       0.679516 | nan               |
| Validation set            |         0.753273 |   0.712991 |            0.688822 |       0.695871 | nan               |
| Test set                  |         0.736368 |   0.709355 |            0.687844 |       0.692346 | nan               |
| Test set Similarity to NN |         1        |   0.847924 |            0.805403 |       0.78089  | nan               |

- The folder settings contains settings with which the experiments where run.


