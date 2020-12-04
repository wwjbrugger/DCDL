from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import statistics
import sys
import numpy as np
from scipy.stats import stats

import comparison_DCDL_vs_SLS.helper_methods as help
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def return_label(path):
    return int(path[6])

def load_tables(path_to_results):
    label = []
    files = [f for f in listdir(path_to_results) if isfile(join(path_to_results, f))]
    files = sorted(files, key=return_label)
    tables = []
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    for file_name in files:
        label.append(file_name[:7])
        path = join(path_to_results, file_name)
        pd_file = pickle.load(open(path, "rb"))
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        pd_file['label'] = file_name[:7] +'_'+ path_to_results.split('/')[1]
        #print(pd_file)
        tables.append(pd_file)
    return tables, label


def sim_DCDL_NN():
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        path_to_results = join('data', dataset, 'results')
        titel = 'Deep_Rule_Set similarity \n   label predicted from NN for train data \n'+ dataset
        table, label = load_tables(path_to_results)
        d = defaultdict(list)
        x_values=[]
        y_values=[]
        y_stdr=[]
        for i in range (len(table)):
            d[label[i]].append(table[i].at[0, 'Concat'])
        for value in d:
            m, s = calculate_mean_std(d[value])
            y_values.append(m)
            y_stdr.append(s)
            x_values.append(value[-1])
        help.graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="label", y_axis_tile='similarity [%]',
                                  save_path='evaluation/sim_DCDL_NN_{}.png'.format(dataset))

def average_accurancy_on_test_data(path_to_results, titel, ax):
    tables, label = load_tables(path_to_results)
    if titel:
        titel = 'average accurancy on test data \n {}'.format(titel)
    deep_rule_set = []
    sls_black_box_prediction = []
    sls_black_box_label = []
    neural_net = []
    for table in tables:
        deep_rule_set.append(table.at[3,'Concat'])
        sls_black_box_prediction.append(table.at[3,'SLS prediction'])
        sls_black_box_label.append(table.at[3, 'SLS train'])
        neural_net.append(np.float64(table.at[3,'Neural network']))
    mean_deep_rule_set, stdr_deep_rule_set = calculate_mean_std(deep_rule_set)
    mean_sls_black_box_prediction, stdr_sls_black_box_prediction = calculate_mean_std(sls_black_box_prediction)
    mean_neural_net, stdr_neural_net = calculate_mean_std(neural_net)
    mean_sls_black_box_label, stdr_sls_black_box_label = calculate_mean_std(sls_black_box_label)

    x_values = ['DCDL', 'BB\nPrediction', 'BB\nLabel', 'NN']
    y_values = [mean_deep_rule_set,  mean_sls_black_box_prediction, mean_sls_black_box_label, mean_neural_net]
    y_stdr=[stdr_deep_rule_set, stdr_sls_black_box_prediction, stdr_sls_black_box_label, stdr_neural_net]
    help.graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="",
                         y_axis_tile='accuracy [%]', ax_out=ax)

def DCDL_minus_SLS_prediction():
    x_values = []
    y_values = []
    y_stdr = []

    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        path_to_result = join('data', dataset, 'results')
        table, label = load_tables(path_to_result)
        Concat_minus_SLS_prediction = []
        for pd_file in table:
            Concat_minus_SLS_prediction.append(pd_file.at[0, 'Concat'] - pd_file.at[0, 'SLS prediction'])
        mean, stdr = calculate_mean_std(Concat_minus_SLS_prediction)

        x_values.append(dataset)
        y_values.append(mean)
        y_stdr.append(stdr)
    help.graph_with_error_bar(x_values, y_values, y_stdr, title='', x_axis_title=" ", y_axis_tile='sim. diff. DCDL - SLS [%]',
                         save_path= 'evaluation/similarity_diff_SLS_DCDL.png' )


def calculate_mean_std(array):
    mean = statistics.mean(array)
    standard_derivation = statistics.stdev(array)
    return mean, standard_derivation

def students_t_test():
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        print ('\033[94m', '\n', dataset, ' students-t-test', '\033[0m')
        path_to_result = join('data', dataset, 'results')
        tables, label = load_tables(path_to_result)
        methods = ['Concat', 'SLS prediction', 'SLS train', 'Neural network']
        df2 = pd.DataFrame(0,index= methods, columns= methods, dtype=float)
        for i in range(len(methods)):
            df2.at[methods[i], methods[i]] = 1
            for j in range(i+1, len(methods), 1):
                col_1 = []
                col_2 = []
                for table in tables:
                    col_1.append(table.at[3,methods[i]])
                    col_2.append(table.at[3,methods[j]])

                t_statistic, two_tailed_p_test = stats.ttest_ind(col_1, col_2)
                df2.at[methods[i], methods[j]] = two_tailed_p_test
                df2.at[methods[j], methods[i]] = two_tailed_p_test
                if two_tailed_p_test > 0.05:
                    print('{} and {} can have th same mean p_value = {}'.format(methods[i], methods[j],
                                                                                two_tailed_p_test))
                else:
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(methods[i], methods[j],
                                                                                          two_tailed_p_test))
        with pd.option_context('display.precision', 2):
            html = df2.style.applymap(help.mark_small_values).render()
        with open('evaluation/students-test_{}.html'.format(dataset), "w") as f:
            f.write(html)


def average_accurancy():
    gs = gridspec.GridSpec(4, 4)
    position = [plt.subplot(gs[:2, :2]), plt.subplot(gs[:2, 2:]), plt.subplot(gs[2:4, 1:3])]
    for i, dataset in enumerate(['numbers', 'fashion', 'cifar']):
        path_to_results = join('data', dataset, 'results')
        average_accurancy_on_test_data(path_to_results, '', position[i])

    plt.tight_layout()
    plt.savefig('evaluation/accuracy_different_approaches.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    students_t_test()
    sim_DCDL_NN()

    DCDL_minus_SLS_prediction()

    average_accurancy()


