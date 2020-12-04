import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import pandas as pd
import helper_methods as help


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
        pd_file['label'] = file_name[:7]
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        #print(pd_file)
        tables.append(pd_file)
    return tables, label



def result_single_label(dataset):
    path = 'dither_frames/' + dataset
    for i, path_to_results in enumerate([path]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        for i in range(10):
            titel = dataset + ' Label ' + str(i)
            sub_table = table[table['label'].str.contains(str(i))]
            cols_test = [c for c in sub_table.columns if 'test' in c.lower()]
            df = sub_table[cols_test]
            x_values = [text.split('_', 1)[0] for text in df.columns]
            y_values=df.mean(axis = 0).tolist()
            y_stdr = df.std(axis = 0).tolist()
            fig, ax = plt.subplots()
            save_path = 'results/single_label/' + dataset + '/label_' + str(i)
            help.graph_with_error_bar(x_values, y_values, y_stdr, title=titel,fix_y_axis=True , y_axis_tile='accuracy [%]', ax_out=ax,
                                      save_path=save_path)

def result_dataset(dataset, ax):

    path ='dither_frames/' + dataset
    for i, path_to_results in enumerate([path]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        titel = 'average perfomance on {}'.format(dataset)
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        df = table[cols_test]
        x_values = [text.split('_', 1)[0] for text in df.columns]
        y_values = df.mean(axis=0).tolist()
        y_stdr = df.std(axis=0).tolist()

        help.graph_with_error_bar(x_values, y_values, y_stdr, title=titel, fix_y_axis=True,
                                  y_axis_tile='accuracy [%]', ax_out=ax, plot_line=True
                                  )

def t_statistik(dataset):
    print('\033[94m', '\n', dataset, ' students-t-test', '\033[0m')
    path_cifar = 'dither_frames/' + dataset
    for i, path_to_results in enumerate([path_cifar]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        table = table[cols_test]
        short_col = [c.split('_')[0] for c in cols_test]
        df2 = pd.DataFrame(0, index=short_col, columns=short_col, dtype=float)
        for i in range(len(table.columns)):
            df2.at[short_col[i], short_col[i]] = 1
            for j in range(i + 1, len(table.columns), 1):
                col_1 = table.iloc[:, i]
                col_1_name = table.columns[i]
                col_2 = table.iloc[:, j]
                col_2_name = table.columns[j]

                t_statistic, two_tailed_p_test = stats.ttest_ind(col_1, col_2)
                df2.at[short_col[i], short_col[j]] = two_tailed_p_test
                df2.at[short_col[j], short_col[i]] = two_tailed_p_test
                if two_tailed_p_test > 0.05:
                    print('{} and {} can have th same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                          two_tailed_p_test))
                else:
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                                      two_tailed_p_test))
        with pd.option_context('display.precision', 2):
            html = df2.style.applymap(help.mark_small_values).render()
        with open('results/students-test_{}.html'.format(dataset), "w") as f:
            f.write(html)


def run():
    datasets = ['mnist', 'cifar', 'fashion']
    for dataset in datasets:
        result_single_label(dataset)
        t_statistik(dataset)

    gs = gridspec.GridSpec(4, 4)
    position = [plt.subplot(gs[:2, :2]), plt.subplot(gs[:2, 2:]), plt.subplot(gs[2:4, 1:3])]
    for i, dataset in enumerate(['mnist', 'fashion', 'cifar']):
        result_dataset(dataset, position[i])
    plt.tight_layout()
    plt.savefig('results/average_performance.png', dpi=300)
    plt.show()
if __name__ == '__main__':
    run()


