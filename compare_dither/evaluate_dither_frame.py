import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import pandas as pd
import helper_methods as helper_methods


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
        # path has the form e.g. label_0__20200509-170342 so this method returns e.g. label_0
        label.append(file_name[:7])
        path = join(path_to_results, file_name)
        pd_file = pickle.load(open(path, "rb"))
        pd_file['label'] = file_name[:7]
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        #print(pd_file)
        tables.append(pd_file)
    return tables, label



def result_single_label(dataset):
    # plot statistics for tried dither approaches for one dataset and one label
    path = 'dither_frames/' + dataset
    #  Update possibility (was not changed to be consistent with existing experiment results)
    # delete for loop there is only one path and i is used two times
    for i, path_to_results in enumerate([path]):
        # get all result frames of one dataset
        table, label = load_tables(path_to_results)
        # concatenate all frames to one so for 30 tables with shape [1,5]
        # make one table with shape [30,5]
        table = pd.concat(table)
        #  Update possibility (was not changed to be consistent with existing experiment results)
        # the hardcoded 10 should be replaced
        for i in range(10):
            titel = dataset + ' Label ' + str(i)
            # get all tables which where measured for on label value
            sub_table = table[table['label'].str.contains(str(i))]
            # get for tried dither approaches the column name for the test dataset
            cols_test = [c for c in sub_table.columns if 'test' in c.lower()]
            # get test result
            #  Update possibility (was not changed to be consistent with existing experiment results)
            # replace df with test_df
            df = sub_table[cols_test]
            # get dither method names
            x_values = [text.split('_', 1)[0] for text in df.columns]
            # calculate statistics
            y_values=df.mean(axis = 0).tolist()
            y_stdr = df.std(axis = 0).tolist()
            # plot statistics for tried dither approaches for one dataset and one label
            fig, ax = plt.subplots()
            save_path = 'results/single_label/' + dataset + '/label_' + str(i)
            helper_methods.graph_with_error_bar(x_values, y_values, y_stdr, title=titel,fix_y_axis=True , y_axis_tile='accuracy [%]', ax_out=ax,
                                      save_path=save_path)

def result_dataset(dataset, ax):
    # calculate average accuracy of dither methods for one dataset
    path ='dither_frames/' + dataset
    #  Update possibility (was not changed to be consistent with existing experiment results)
    # delete for loop there is only one path
    for i, path_to_results in enumerate([path]):
        # get all result frames of one dataset
        table, label = load_tables(path_to_results)
        # concatenate all frames to one so for 30 tables with shape [1,5]
        # make one table with shape [30,5]
        table = pd.concat(table)
        titel = 'average perfomance on {}'.format(dataset)

        # get for tried dither approaches the column name for the test dataset
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        # get test result
        df = table[cols_test]
        # get method names
        x_values = [text.split('_', 1)[0] for text in df.columns]
        # calculate statistics
        y_values = df.mean(axis=0).tolist()
        y_stdr = df.std(axis=0).tolist()

        helper_methods.graph_with_error_bar(x_values, y_values, y_stdr, title=titel, fix_y_axis=True,
                                  y_axis_tile='accuracy [%]', ax_out=ax, plot_line=True
                                  )

def t_statistik(dataset):
    # calculate student-t-test if dither approaches have the same mean
    # uses t-test from scipy
    print('\033[94m', '\n', dataset, ' students-t-test', '\033[0m')
    #  Update possibility (was not changed to be consistent with existing experiment results)
    # delete for loop there is only one path
    # rename path_cifar for e.g. path to results
    path_cifar = 'dither_frames/' + dataset
    for i, path_to_results in enumerate([path_cifar]):
        # get all result frames of one dataset
        table, label = load_tables(path_to_results)
        # concatenate all frames to one so for 30 tables with shape [1,5]
        # make one table with shape [30,5]
        table = pd.concat(table)
        # get for tried dither approaches the column name for the test dataset
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        # get test result
        #  Update possibility (was not changed to be consistent with existing experiment results)
        # replace df table with  test_df
        table = table[cols_test]
        # get dither method names
        short_col = [c.split('_')[0] for c in cols_test]
        # pandas frame to save t statistic
        #  Update possibility (was not changed to be consistent with existing experiment results)
        #  change name of df2 to results_t_statistic
        df2 = pd.DataFrame(0, index=short_col, columns=short_col, dtype=float)
        for i in range(len(table.columns)):
            # fill diagonals with ones
            df2.at[short_col[i], short_col[i]] = 1
            for j in range(i + 1, len(table.columns), 1):
                # iterate through approaches
                # get test accuracy values
                col_1 = table.iloc[:, i]
                col_1_name = table.columns[i]
                col_2 = table.iloc[:, j]
                col_2_name = table.columns[j]

                # calculate t-statistic between accuracy values of approches
                t_statistic, two_tailed_p_test = stats.ttest_ind(col_1, col_2)
                # add p values
                df2.at[short_col[i], short_col[j]] = two_tailed_p_test
                df2.at[short_col[j], short_col[i]] = two_tailed_p_test
                if two_tailed_p_test > 0.05:
                    # accept H_0 hypothesis that both approaches have same mean
                    print('{} and {} can have th same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                          two_tailed_p_test))
                else:
                    # reject H_0 hypothesis that both approaches have same mean
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                                      two_tailed_p_test))
        # save result of student-t-test for dataset as html file
        with pd.option_context('display.precision', 2):
            html = df2.style.applymap(helper_methods.mark_small_values).render()
        with open('results/students-test_{}.html'.format(dataset), "w") as f:
            f.write(html)


def run():
    datasets = ['mnist', 'cifar', 'fashion']
    for dataset in datasets:
        # plot statistics for tried dither approaches for one dataset and one label
        result_single_label(dataset)
        # calculate student-t-test if dither approaches have the same mean.
        t_statistik(dataset)

    gs = gridspec.GridSpec(4, 4)
    position = [plt.subplot(gs[:2, :2]), plt.subplot(gs[:2, 2:]), plt.subplot(gs[2:4, 1:3])]
    for i, dataset in enumerate(['mnist', 'fashion', 'cifar']):
        # calculate average accuracy of dither methods for one dataset
        result_dataset(dataset, position[i])
    plt.tight_layout()
    plt.savefig('results/average_performance.png', dpi=300)
    plt.show()
if __name__ == '__main__':
    run()


