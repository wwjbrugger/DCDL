from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pickle
from pathlib import Path

import pandas as pd
import statistics
import sys
import numpy as np
from scipy.stats import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pk
import seaborn as sns; sns.set_theme()

import NN_DCDL_SLS_Blackbox_comparison.visualize as visualize


# ---------------------------------define settings for analysis--------------------------
def get_analyze_settings():
    analyze_settings_dic = {
        # name of the experiment you want to analyze
        'setup_names': ['Arg_max_0', 'Arg_min_0'],
        # path where the results are saved 
        'path_result_dic': Path.cwd(),
        # where visualization should be stored
        'save_path_visualization': Path.cwd() / 'visualization_results',
        # datasets which should be analyzed
        'datasets': ['numbers', 'fashion', 'cifar'],
        # all column names which are in result frmes
        'column_names': ['Neural network', 'DCDL', 'SLS BB prediction', 'SLS BB label', 'setup'],
        # which approaches should be analyzed
        'approaches': ['Neural network', 'DCDL', 'SLS BB prediction', 'SLS BB label', ],
        # name of rows in data_set
        'row_names': ['Training set', 'Validation set', 'Test set', 'setup'],
        # which statistics to use
        'statistics_to_use': ['students_t_test_ind','students_t_test_rel', 'kruskal_wallis', 'friedmanchisquare'],
        # datasets used mnist and numbers are the same dataset
        'datasets': ['numbers', 'fashion', 'cifar']
    }
    return analyze_settings_dic


# ------------------------------------------ load results ---------------------------------

def get_result_frames(analyze_settings_dic):
    result_frames_dic = {}

    # get all experiments in result folder
    experiment_folder = [f.stem for f in
                         analyze_settings_dic['path_result_dic'].iterdir()
                         if f.is_dir()]
    for setup in analyze_settings_dic['setup_names']:
        # iterate through experiments, which should be analyzed
        if setup in experiment_folder:
            result_frames_dic[setup] = {}
            path_to_experiment = analyze_settings_dic['path_result_dic'] / setup
            # get results of an experiment
            data_set_folders = [f for f in path_to_experiment.iterdir()
                                if f.is_dir()]

            for data_set_folder in data_set_folders:
                # iterate though data sets
                data_set = data_set_folder.stem
                result_frames_dic[setup][data_set] = {}
                label_folders = [f for f in data_set_folder.iterdir()
                                 if f.is_dir()]

                for label_folder in label_folders:
                    # iterate through  labels
                    label = label_folder.stem
                    result_frames_dic[setup][data_set][label] = {}
                    for i, result_file in enumerate(
                            label_folder.iterdir()):
                        # iterate through results
                        with open(result_file, 'rb') as f:
                            result_frame = pk.load(f)  # .astype(float)
                        result_frames_dic[setup][data_set][label][i] = result_frame
        else:
            # requested setup is not in experiment_folder
            raise ValueError(
                'requested setup is not in experiment_folder\n'
                'you requested: {}\n '
                'possible values are {}'.format(
                    setup,
                    experiment_folder
                )
            )
    return result_frames_dic


# ------------------------------------------ statistical tests ---------------------------------


def accuracy_significance_methods(result_frames_dic, statistics_to_use, approaches, save_path):
    # calculate accuracy significance between approaches
    for experiment, experiment_dics in result_frames_dic.items():
        # iterate through experiments
        for dataset, dataset_dic in experiment_dics.items():
            # iterate through datasets
            accuracy_values = get_accuracy_values(
                dataset=dataset,
                dataset_dic=dataset_dic,
                approaches=approaches)
            if 'students_t_test_ind' in statistics_to_use:
                student_t_test_ind(approaches=approaches,
                               accuracy_values=accuracy_values,
                               save_path=save_path / experiment / dataset
                               )
            if 'students_t_test_rel' in statistics_to_use:
                student_t_test_rel(approaches=approaches,
                                   accuracy_values=accuracy_values,
                                   save_path=save_path / experiment / dataset
                                   )
            if 'kruskal_wallis' in statistics_to_use:
                kruskal_wallis_test(
                    approaches=approaches,
                    accuracy_values=accuracy_values,
                    save_path=save_path / experiment / dataset)
            if 'friedmanchisquare' in statistics_to_use:
                friedmanchisquare_test(
                    approaches=approaches,
                    accuracy_values=accuracy_values,
                    save_path=save_path / experiment / dataset
                )


def get_accuracy_values(dataset, dataset_dic, approaches):
    # returns all accuracy for a dataset for the approaches which are requested
    accuracy_frame = pd.DataFrame(columns=approaches)
    for label, label_dic in dataset_dic.items():
        # iterate through labels
        for result_key, result_frame in label_dic.items():
            # get accuracy of all approaches
            for approach in approaches:
                row_name = '{}_{}_measurment_{}'.format(dataset,
                                                        label,
                                                        result_key)
                accuracy_frame.at[row_name, approach] = result_frame.at['Test set', approach]
    return accuracy_frame


def student_t_test_ind(approaches, accuracy_values, save_path):
    # calculate the two sided unpaired students t-test from scipy
    # it compare all approaches with each other
    #Calculate the T-test for the means of two independent samples of scores
    student_t_test_ind_frame = pd.DataFrame()
    for i in range(len(approaches)):
        for j in range(i, len(approaches), 1):
            # iterate through approaches
            approach_i = approaches[i]
            approach_j = approaches[j]
            values_i = accuracy_values.loc[:, approach_i]
            values_j = accuracy_values.loc[:, approach_j]
            t_statistic, two_tailed_p_test = stats.ttest_ind(values_i, values_j)
            student_t_test_ind_frame.at[approach_i, approach_j] = two_tailed_p_test

    save_path.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.subplots()
    ax = sns.heatmap(student_t_test_ind_frame, ax=ax, annot=True, fmt="0.3f", cmap="autumn", vmin=0, vmax=0.05)
    plt.xticks(rotation=45)
    fig.canvas.start_event_loop(sys.float_info.min)
    path = save_path / 'students-test_scipy_ind.png'
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def student_t_test_rel(approaches, accuracy_values, save_path):
    # calculate the two sided paired students t-test from scipy
    # it compare all approaches with each other
    student_t_test_rel_frame = pd.DataFrame()
    for i in range(len(approaches)):
        for j in range(i, len(approaches), 1):
            # iterate through approaches
            approach_i = approaches[i]
            approach_j = approaches[j]
            values_i = accuracy_values.loc[:, approach_i]
            values_j = accuracy_values.loc[:, approach_j]
            t_statistic, two_tailed_p_test = stats.ttest_rel(values_i, values_j)
            student_t_test_rel_frame.at[approach_i, approach_j] = two_tailed_p_test


        save_path.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(4, 2))
        ax = fig.subplots()
        ax = sns.heatmap(student_t_test_rel_frame, ax=ax, annot=True, fmt="0.3f", cmap="autumn", vmin=0, vmax=0.05)
        plt.xticks(rotation=45)
        fig.canvas.start_event_loop(sys.float_info.min)
        path = save_path / 'students-test_scipy_rel.png'
        fig.savefig(path, bbox_inches='tight', dpi=100)
        plt.close(fig)

def kruskal_wallis_test(approaches, accuracy_values, save_path):
    # The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal.
    # It is a non-parametric version of ANOVA.
    # The test works on 2 or more independent samples, which may have different sizes.
    # Note that rejecting the null hypothesis does not indicate which of the groups differs.
    # Post hoc comparisons between groups are required to determine which groups are different.

    kruskal_wallis_test_frame = pd.DataFrame()
    # perform kruskal wallis test
    # returns (The Kruskal-Wallis H statistic corrected for ties,
    # The p-value for the test using the assumption that H has a chi square distribution)
    # depack values from accuracy frame
    input_kruskal = [accuracy_values.loc[:,approach].to_numpy() for approach in approaches]
    #  the syntax *expression appears in the function call, expression must evaluate to an iterable.
    #  Elements from this iterable are treated as if they were additional positional arguments
    statistic, p_value = stats.kruskal(*input_kruskal)
    kruskal_wallis_test_frame.at['result', 'statistic'] = statistic
    kruskal_wallis_test_frame.at['result', 'p-value'] = p_value


    save_path.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.subplots()
    ax = sns.heatmap(kruskal_wallis_test_frame, ax=ax, annot=True, fmt="0.3f", cmap="autumn", vmin=0, vmax=0.05)
    plt.xticks(rotation=45)
    fig.canvas.start_event_loop(sys.float_info.min)
    path = save_path / 'kruskal_wallis_test.png'
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def friedmanchisquare_test(approaches, accuracy_values, save_path):
    # The Friedman test tests the null hypothesis that repeated measurements
    # of the same individuals have the same distribution.
    # It is often used to test for consistency among measurements obtained in different ways.
    # For example, if two measurement techniques are used on the same set of individuals,
    # the Friedman test can be used to determine if the two measurement techniques are consistent

    friedmanchisquare_test_frame = pd.DataFrame()
    # returns
    # The associated p-value assuming that the test statistic has a chi squared distribution.
    # depack values from accuracy frame
    input_friedmanchisquare = [accuracy_values.loc[:, approach].to_numpy() for approach in approaches]
    #  the syntax *expression appears in the function call, expression must evaluate to an iterable.
    #  Elements from this iterable are treated as if they were additional positional arguments
    statistic, p_value = stats.friedmanchisquare(*input_friedmanchisquare)
    friedmanchisquare_test_frame.at['result', 'statistic'] = statistic
    friedmanchisquare_test_frame.at['result', 'p-value'] = p_value


    save_path.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.subplots()
    ax = sns.heatmap(friedmanchisquare_test_frame, ax=ax, annot=True, fmt="0.3f", cmap="autumn", vmin=0, vmax=0.05)
    plt.xticks(rotation=45)
    fig.canvas.start_event_loop(sys.float_info.min)
    path = save_path / 'friedmanchisquare_test.png'
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)

        # ----------------------------------------- visualize results -----------------------------------


def average_accuracy_on_test_data_all_datasets(result_frames_dic, save_path):
    # show comparison accuracy of DCDL, BB Prediction, BB train and Neural Network
    # for all three dataset in one plot
    for experiment, experiment_dics in result_frames_dic.items():
        # iterate through experiments
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)
        gs = fig.add_gridspec(4, 4)
        position = {'numbers': fig.add_subplot(gs[:2, :2]),
                    'fashion': fig.add_subplot(gs[:2, 2:]),
                    'cifar': fig.add_subplot(gs[2:4, 1:3])
                    }

        for dataset, dataset_dic in experiment_dics.items():
            # iterate through datasets
            average_accuracy_on_test_data_single_dataset(
                dataset_dic=dataset_dic,
                title='',
                ax=position[dataset])

        # visualize accuracy results for all datasets
        plt.tight_layout()
        path = save_path / experiment / 'test_accuracy'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def average_accuracy_on_test_data_single_dataset(dataset_dic, title, ax):
    # compare accuracy of DCDL, BB Prediction, BB train and Neural Network
    # for a single dataset
    DCDL_results = []
    BB_Prediction_results = []
    BB_Label_results = []
    Neural_net_results = []

    for label, label_dic in dataset_dic.items():
        # iterate through labels
        for result_key, result_frame in label_dic.items():
            # get accuracy of all approaches
            DCDL_results.append(result_frame.at['Test set', 'DCDL'])
            BB_Prediction_results.append(result_frame.at['Test set', 'SLS BB prediction'])
            BB_Label_results.append(result_frame.at['Test set', 'SLS BB label'])
            Neural_net_results.append(np.float64(result_frame.at['Test set', 'Neural network']))

    # calculate statistics for approaches
    mean_DCDL_results, stdr_DCDL_results = calculate_mean_std(DCDL_results)
    mean_BB_Prediction_results, stdr_BB_Prediction_results = calculate_mean_std(BB_Prediction_results)
    mean_Neural_net_results, stdr_Neural_net_results = calculate_mean_std(Neural_net_results)
    mean_BB_Label_results, stdr_BB_Label_results = calculate_mean_std(BB_Label_results)

    # plot results for accuracy of all approaches
    x_values = ['DCDL', 'BB\nPrediction', 'BB\nLabel', 'NN']
    y_values = [mean_DCDL_results, mean_BB_Prediction_results, mean_BB_Label_results, mean_Neural_net_results]
    y_stdr = [stdr_DCDL_results, stdr_BB_Prediction_results, stdr_BB_Label_results, stdr_Neural_net_results]

    # visualize results for one dataset
    visualize.graph_with_error_bar(
        x_values=x_values,
        y_values=y_values,
        y_stdr=y_stdr,
        title=title,
        x_axis_title="",
        y_axis_tile='accuracy [%]',
        fix_y_axis=False,
        ax_out=ax,
        save_path=False,
        plot_line=False,
        xticks_rotation=0  # -90

    )


def similarity_difference_DCDL_SLS_prediction(analyze_settings_dic, title, save_path):
    # when we look at similarity to the neural network
    # we want to know how good other approaches
    # can predict the decisions of the neural net
    # this methods shows the difference between the DCDL approach
    # and SLS backbox approach. which was trained with the prediction of the net

    for experiment, experiment_dics in result_frames_dic.items():
        # iterate through experiments
        x_values = []
        y_values = []
        y_stdr = []
        datasets = analyze_settings_dic['datasets'].copy()
        for dataset in datasets:
            dataset_dic = experiment_dics[dataset]
            # iterate through datasets
            # get pandas frames with results for dataset
            sim_difference = []
            for label, label_dic in dataset_dic.items():
                # iterate through labels
                for result_key, result_frame in label_dic.items():
                    # get accuracy of DCDL and BB_Prediction approach
                    DCDL_results = result_frame.at['Test set Similarity to NN ', 'DCDL']
                    BB_Prediction_results = result_frame.at['Test set Similarity to NN ', 'SLS BB prediction']
                    # append diference
                    sim_difference.append(DCDL_results - BB_Prediction_results)
            # calculate statistics for one dataset
            mean, stdr = calculate_mean_std(sim_difference)
            y_values.append(mean)
            y_stdr.append(stdr)
        # in graph we want to use 'mnist' instead of 'numbers
        datasets = [dataset.replace('numbers', 'mnist') for dataset in datasets]
        # show similarity difference DCDL SLS_prediction for one experiment
        path = save_path / experiment / 'similarity_difference_DCDL_SLS_prediction.png'
        visualize.graph_with_error_bar(
            x_values=datasets,
            y_values=y_values,
            y_stdr=y_stdr,
            title=title,
            x_axis_title="",
            y_axis_tile='sim. diff. DCDL - SLS [%]',
            fix_y_axis=False,
            ax_out=False,
            save_path=path,
            plot_line=False,
            xticks_rotation=0
        )

    # ----------------------------------------- helper methods -----------------------------------


def calculate_mean_std(array):
    mean = statistics.mean(array)
    standard_derivation = statistics.stdev(array)
    return mean, standard_derivation


# ----------------------------------------- main method ---------------------------------------


if __name__ == '__main__':
    analyze_settings_dic = get_analyze_settings()

    result_frames_dic = get_result_frames(
        analyze_settings_dic=analyze_settings_dic
    )

    # average_accuracy_on_test_data_all_datasets(
    #     result_frames_dic=result_frames_dic,
    #     save_path=analyze_settings_dic['save_path_visualization']
    # )
    #
    # similarity_difference_DCDL_SLS_prediction(
    #     analyze_settings_dic=analyze_settings_dic,
    #     title='',
    #     save_path=analyze_settings_dic['save_path_visualization'],
    # )

    accuracy_significance_methods(
        result_frames_dic=result_frames_dic,
        statistics_to_use=analyze_settings_dic['statistics_to_use'],
        approaches=analyze_settings_dic['approaches'],
        save_path=analyze_settings_dic['save_path_visualization']
    )
