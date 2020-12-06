from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import statistics
import sys
import numpy as np
from scipy.stats import stats

import comparison_DCDL_vs_SLS.helper_methods as helper_methods
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def return_label(path):
    "path has the form e.g. label_0__20200509-170342 so this method returns e.g. 0"
    return int(path[6])

def load_tables(path_to_results):
    # get path to a folder and read in all files in this folder
    label = []
    files = [f for f in listdir(path_to_results) if isfile(join(path_to_results, f))]
    # sort files after label
    files = sorted(files, key=return_label)
    tables = []
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    for file_name in files:
        # "path has the form e.g. label_0__20200509-170342 so this method returns e.g. label_0"
        label.append(file_name[:7])
        path = join(path_to_results, file_name)
        # load pandas-frames with results from experiment
        pd_file = pickle.load(open(path, "rb"))
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        pd_file['label'] = file_name[:7] +'_'+ path_to_results.split('/')[1]
        #print(pd_file)
        tables.append(pd_file)
    return tables, label


def sim_DCDL_NN(suffix):
    # shows similarity between DCDL approach and the prediction of the neural net on training data for every label
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        path_to_results = join('data', dataset, suffix)
        titel = 'DCDL similarity \n   label predicted from NN for train data \n'+ dataset
        table, label = load_tables(path_to_results)
        d = defaultdict(list)
        x_values=[]
        y_values=[]
        y_stdr=[]
        for i in range (len(table)):
            # get similarity values of DCDL label are used as key in dic

            d[label[i]].append(table[i].at[0, 'DCDL'])
        for value in d:
            # iterate through similarity values of DCDL for one label
            m, s = calculate_mean_std(d[value])
            y_values.append(m)
            y_stdr.append(s)
            x_values.append(value[-1])
        # plot similarity between DCDL approach and the prediction of the neural net on training data for every label with std
        helper_methods.graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="label", y_axis_tile='similarity [%]',
                                  save_path='evaluation/sim_DCDL_NN_{}.png'.format(dataset))

def average_accurancy_on_test_data(path_to_results, titel, ax):
    # compare accuracy of DCDL, BB Prediction, BB train and Neural Network
    tables, label = load_tables(path_to_results)
    if titel:
        titel = 'average accurancy on test data \n {}'.format(titel)
    deep_rule_set = []
    sls_black_box_prediction = []
    sls_black_box_label = []
    neural_net = []
    for table in tables:
        # get accuracy of all approaches
        deep_rule_set.append(table.at[3,'DCDL'])
        sls_black_box_prediction.append(table.at[3,'BB Prediction'])
        sls_black_box_label.append(table.at[3, 'BB train'])
        neural_net.append(np.float64(table.at[3,'Neural network']))
    # calculate statistics for approaches
    mean_deep_rule_set, stdr_deep_rule_set = calculate_mean_std(deep_rule_set)
    mean_sls_black_box_prediction, stdr_sls_black_box_prediction = calculate_mean_std(sls_black_box_prediction)
    mean_neural_net, stdr_neural_net = calculate_mean_std(neural_net)
    mean_sls_black_box_label, stdr_sls_black_box_label = calculate_mean_std(sls_black_box_label)
    # plot results for accuracy of all approaches
    x_values = ['DCDL', 'BB\nPrediction', 'BB\nLabel', 'NN']
    y_values = [mean_deep_rule_set,  mean_sls_black_box_prediction, mean_sls_black_box_label, mean_neural_net]
    y_stdr=[stdr_deep_rule_set, stdr_sls_black_box_prediction, stdr_sls_black_box_label, stdr_neural_net]
    helper_methods.graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="",
                         y_axis_tile='accuracy [%]', ax_out=ax)

def DCDL_minus_SLS_prediction(suffix):
    # for similarity prediction of the neural net for training data are the label
    # shows difference between similarity of DCDL approach and similarity of SLS backbox prediction

    x_values = []
    y_values = []
    y_stdr = []

    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        # iterate through datasets
        # get pandas frames with results for dataset
        path_to_result = join('data', dataset, suffix)
        table, label = load_tables(path_to_result)
        Concat_minus_SLS_prediction = []
        for pd_file in table:
            #for every pandas frame calculate   similarity DCDL -  similarity SLS Prediction
            Concat_minus_SLS_prediction.append(pd_file.at[0, 'DCDL'] - pd_file.at[0, 'BB Prediction'])
        mean, stdr = calculate_mean_std(Concat_minus_SLS_prediction)

        x_values.append(dataset)
        y_values.append(mean)
        y_stdr.append(stdr)
    helper_methods.graph_with_error_bar(x_values, y_values, y_stdr, title='', x_axis_title=" ", y_axis_tile='sim. diff. DCDL - SLS [%]',
                         save_path= 'evaluation/similarity_diff_SLS_DCDL.png' )


def calculate_mean_std(array):
    mean = statistics.mean(array)
    standard_derivation = statistics.stdev(array)
    return mean, standard_derivation

def students_t_test(suffix):
    # uses t-test from scipy
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        print ('\033[94m', '\n', dataset, ' students-t-test', '\033[0m')
        path_to_result = join('data', dataset, suffix)
        # get all result frames of one dataset
        tables, label = load_tables(path_to_result)
        methods = ['DCDL', 'SLS BB prediction', 'SLS BB train', 'Neural network']
        # uses t-test from scipy
        scipy_student_t_test_p_value = pd.DataFrame(0,index= methods, columns= methods, dtype=float)
        for i in range(len(methods)):
            # fill diagonals with ones
            scipy_student_t_test_p_value.at[methods[i], methods[i]] = 1

            for j in range(i+1, len(methods), 1):
                # iterate through approaches
                col_1 = []
                col_2 = []
                for table in tables:
                    # get all accuracy values
                    col_1.append(table.at[3,methods[i]])
                    col_2.append(table.at[3,methods[j]])
                # calculate t-statistic between accuracy values of approches
                t_statistic, two_tailed_p_test = stats.ttest_ind(col_1, col_2)
                # add p values
                scipy_student_t_test_p_value.at[methods[i], methods[j]] = two_tailed_p_test
                scipy_student_t_test_p_value.at[methods[j], methods[i]] = two_tailed_p_test
                if two_tailed_p_test > 0.05:
                    # accept H_0 hypothesis that both approaches have same mean
                    print('{} and {} can have th same mean p_value = {}'.format(methods[i], methods[j],
                                                                                two_tailed_p_test))
                else:
                    # reject H_0 hypothesis that both approaches have same mean
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(methods[i], methods[j],
                                                                                          two_tailed_p_test))
        # save result of student-t-test for dataset as html file
        with pd.option_context('display.precision', 2):
            html = scipy_student_t_test_p_value.style.applymap(helper_methods.mark_small_values).render()
        with open('evaluation/students-test_scipy{}.html'.format(dataset), "w") as f:
            f.write(html)

# from https://gist.github.com/jensdebruijn/13e8eeda85eb8644ac2a4ac4c3b8e732
# Python implementation of the Nadeau and Bengio correction of dependent Student's t-test
# using the equation stated in https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf

from scipy.stats import t
from math import sqrt
from statistics import stdev

def corrected_dependent_ttest(data1, data2, n_training_folds, n_test_folds, alpha):
    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)]
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = n_test_folds / n_training_folds
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p

def corrected_dependent_t_test( n_training_folds, n_test_folds, alpha, suffix):
    # uses t-test from Nadeau and Bengio correction of dependent Student's t-test
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        print ('\033[94m', '\n', dataset, ' students-t-test', '\033[0m')
        path_to_result = join('data', dataset, suffix)
        # get all result frames of one dataset
        tables, label = load_tables(path_to_result)

        methods = [ 'DCDL', 'SLS BB prediction', 'SLS BB train', 'Neural network']

        corrected_dependent_t_test = pd.DataFrame(0,index= methods, columns= methods, dtype=float)
        for i in range(len(methods)):
            # fill diagonals with ones
            corrected_dependent_t_test.at[methods[i], methods[i]] = 1

            for j in range(i+1, len(methods), 1):
                # iterate through approaches
                col_1 = []
                col_2 = []
                for table in tables:
                    # get all accuracy values of approaches
                    col_1.append(table.at[3,methods[i]])
                    col_2.append(table.at[3,methods[j]])
                # calculate t-statistic between accuracy values of approches
                t_stat, df, cv, p = corrected_dependent_ttest(data1 = col_1,
                                                                           data2 = col_2,
                                                                           n_training_folds = n_training_folds,
                                                                           n_test_folds = n_test_folds,
                                                                           alpha = alpha)
                # add p values
                corrected_dependent_t_test.at[methods[i], methods[j]] = p
                corrected_dependent_t_test.at[methods[j], methods[i]] = p
                if p > 0.05:
                    # accept H_0 hypothesis that both approaches have same mean
                    print('{} and {} can have th same mean p_value = {}'.format(methods[i], methods[j],
                                                                                p))
                else:
                    # reject H_0 hypothesis that both approaches have same mean
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(methods[i], methods[j],
                                                                                          p))
        # save result of student-t-test for dataset as html file
        with pd.option_context('display.precision', 2):
            html = corrected_dependent_t_test.style.applymap(helper_methods.mark_small_values).render()
        with open('evaluation/students-test_corrected_{}.html'.format(dataset), "w") as f:
            f.write(html)

def average_accurancy(suffix):
    # show comparison accuracy of DCDL, BB Prediction, BB train and Neural Network for all three dataset in one plot
    gs = gridspec.GridSpec(4, 4)
    position = [plt.subplot(gs[:2, :2]), plt.subplot(gs[:2, 2:]), plt.subplot(gs[2:4, 1:3])]
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        path_to_results = join('data', dataset, suffix)
        average_accurancy_on_test_data(path_to_results, '', position[i])

    plt.tight_layout()
    plt.savefig('evaluation/accuracy_different_approaches.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # hyperparameter for corrected_dependent_t_test
    n_training_folds = 1
    n_test_folds = 1
    alpha = 0.5
    suffix = 'old_results'
    corrected_dependent_t_test(n_training_folds=n_training_folds,
                               n_test_folds = n_test_folds,
                               alpha = alpha,
                               suffix = suffix)
    students_t_test(suffix= suffix)
    sim_DCDL_NN(suffix = suffix)

    DCDL_minus_SLS_prediction(suffix = suffix)

    average_accurancy(suffix = suffix)


