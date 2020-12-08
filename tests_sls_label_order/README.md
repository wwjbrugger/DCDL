## tests_sls_label_order


In this folder, the SLS algorithm is tested in mode one against all for the MNIST data set
#### SLS-Algorithm
The SLS Algorithm can run in three modes: (train, train_val, train_val_test)
- **train_val_test** The algorithm is trained on a train set, validated on a validation set and tested on a test set (not used)
- **train_val** The algorithm is trained on a train set, validated on a validation set
- **train** The algorithm is trained on a train set, this train set is used as validation set in the algorithm   

#### one_hot_label
Because the SLS-Algorithm works with binary values we have to cast
the 10 label in the MNIST dataset to 2. 
This is done by using one against all tests.

#### balacing 
Without balancing the rest class would have many more instances.
after balancing the statistics are as follows

|       | One class | Rest class |
|:-----:|:---------:|:----------:|
| Train |    5431   |    5427    |
|  Val  |    492    |     495    |
|  Test |    980    |     981    |





### Example results 
The values in the tables are the accuracy scores on a test dataset

one-hot-label can have the following form:
- positive_label [1,0] instance is part of the **one** against all class
- inverse_label [1,0] instance is part of the **rest** class

- val means SLS is running with validation set
- no_val means SLS is running without validation set

|  one class  |   val_positive_label |   val_inverse_label |   no_val_positive_label |   no_val_inverse_label |
|:-----------:|:--------------------:|:-------------------:|:-----------------------:|:---------------------:|
|  0 |                 0.95 |                0.91 |                    0.96 |                   0.89 |
|  1 |                 0.98 |                0.92 |                    0.98 |                   0.91 |
|  2 |                 0.85 |                0.81 |                    0.86 |                   0.8  |
|  3 |                 0.83 |                0.81 |                    0.86 |                   0.81 |
|  4 |                 0.86 |                0.75 |                    0.86 |                   0.77 |
|  5 |                 0.85 |                0.77 |                    0.86 |                   0.76 |
|  6 |                 0.91 |                0.9  |                    0.93 |                   0.89 |
|  7 |                 0.91 |                0.88 |                    0.92 |                   0.88 |
|  8 |                 0.8  |                0.8  |                    0.79 |                   0.77 |
|  9 |                 0.85 |                0.79 |                    0.86 |                   0.84 |


|           approaches            |   mean   |
|-----------------------|:--------:|
| val_positive_label    | 0.878 |
| val_inverse_label     | 0.833 |
| no_val_positive_label | **0.888** |
| no_val_inverse_label  | 0.831 |
