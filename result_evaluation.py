import os
import utils
import pandas as pd
import re
import numpy as np


def append_result_summary_val_copy(baselines):
    """
    unite all results to one csv file, just copying all values
    :param baselines: list of strings defining the baselines to be used, congruent with the files' prefixes
    """
    # prepare output file
    output_file_path = 'results/all-values.csv'
    if not os.path.isfile(output_file_path):
        utils.write_info_to_csv(output_file_path, ['baseline', 'corpus', 'weighted', 'method', 'pos', 'metaphor',
                                                   'similarity_value', 'baseline_value', 'test_stat', 'p_value'])
    directory = 'results'
    for prefix in baselines:
        # iterate all relevant files (in results, starting with prefix)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a relevant file
            if os.path.isfile(f) and re.match(prefix, filename):
                # extract important info from filename (corpus, distance-measure, weighted/unweighted)
                splitted_infos = filename.split('_')
                corpus = splitted_infos[1]
                distance_measure = splitted_infos[4]
                weighted = splitted_infos[6]
                # read csv with pd:
                df = pd.read_csv(f)
                # write to new csv file:
                for row in df.itertuples():
                    utils.write_info_to_csv(output_file_path, [prefix, corpus, weighted, distance_measure, row.pos,
                                                               f'{row.metaphor_id} {row.metaphor_name}',
                                                               row.mean_similarity, row.baseline_performance,
                                                               row.test_statistic, row.p_value], 'a')


def confront_results_for_one_param(parameter, baselines):
    """
    collect relevant info from all results (results/all-values.csv) that allow closer interpretation of one parameter
    shaping the values, differentiating values from two different baselines
    save these infos to csv
    :param parameter: one of "metaphor", "pos", "corpus", "weighted", "method". Defines parameter from which perspective
    the results can be analyzed closer
    :param baselines: list of strings defining the baselines to be used
    """
    # TODO: maybe generate file with IDs that contain all possibilities for the parameters and make stuff less random
    #  and complex to use
    # prepare output file
    output_file_path = f'results/confront_files/{parameter}_confront.csv'
    to_be_calculated = ['mean_baseline_', 'mean_test_stat_', 'mean_p_value_', 'amount_pos_sign_',
                        'amount_pos_insign_', 'amount_neg_sign_', 'amount_neg_insign_']
    csv_headings = []
    for calculation in to_be_calculated:
        for baseline in baselines:
            csv_headings.append(calculation + baseline)
    utils.write_info_to_csv(output_file_path, [parameter, 'mean_similarity'] + csv_headings)
    # read all_values.csv
    df = pd.read_csv('results/all-values.csv')
    # iterate values for parameters
    for param_value in set(df[parameter].tolist()):
        print(param_value)
        # prepare dataframes
        filtered_df = df[df[parameter] == param_value]
        bl_dfs = []
        for baseline in baselines:
            bl_dfs.append(filtered_df[filtered_df['baseline'] == baseline])
        # calculate needed values. mean similarity should be the same, as is independent from baseline
        mean_similarity = np.mean(filtered_df['similarity_value'].tolist())
        mean_vals = calculate_mean_values_from_dfs(bl_dfs, ['baseline_value', 'test_stat', 'p_value'])
        amount_vals = calculate_amounts_from_dfs(bl_dfs, ['pos_sign', 'pos_insign', 'neg_sign', 'neg_insign'])
        utils.write_info_to_csv(output_file_path, [param_value, mean_similarity] + mean_vals + amount_vals, 'a')


# TODO: maybe remove this and the corresponding files. Is it even mentioned in text?
def metaphor_confront_for_one_param(parameter):
    """
    analyze different metaphors from the perspective of a parameter, trying to provide the data for an answer on the
    question: does the parameter have an impact on the results for the different metaphors?
    save info to csv file
    :param parameter: one of "corpus", "weighted", "method", "pos"
    """
    # prepare output file
    output_file_path = f'results/confront_files/{parameter}_metaphor_confront.csv'
    # read all_values.csv
    df = pd.read_csv('results/all-values.csv')
    param_values = set(df[parameter].tolist())
    to_be_calculated = ['mean_similarity_', 'mean_baseline_', 'mean_test_stat_', 'mean_p_value_', 'amount_pos_sign_',
                        'amount_pos_insign_', 'amount_neg_sign_', 'amount_neg_insign_']
    csv_headings = []
    for calculation in to_be_calculated:
        for param_value in param_values:
            csv_headings.append(calculation + param_value)

    utils.write_info_to_csv(output_file_path, ['metaphor'] + csv_headings)
    # iterate metaphors
    for metaphor in set(df['metaphor'].tolist()):
        # prepare dataframes
        filtered_df = df[df['metaphor'] == metaphor]
        param_dfs = [filtered_df[filtered_df[parameter] == param_value] for param_value in param_values]
        # calculate needed values
        mean_vals = calculate_mean_values_from_dfs(param_dfs, ['similarity_value', 'baseline_value', 'test_stat','p_value'])
        amount_vals = calculate_amounts_from_dfs(param_dfs, ['pos_sign', 'pos_insign', 'neg_sign', 'neg_insign'])
        utils.write_info_to_csv(output_file_path, [metaphor] + mean_vals + amount_vals, 'a')


def calculate_mean_values_from_dfs(dfs, value_names):
    """
    calculate mean values in given pandas dataframes
    :param dfs: list of dataframes to be analyzed
    :param value_names: values to be analyzed, should match column name in dataframe
    :return: list of amounts, sorted by amount_name first, then dataframe
    """
    value_list = []
    for value_name in value_names:
        for df in dfs:
            try:
                value_list.append(np.mean(df[value_name].tolist()))
            except KeyError:
                print(f'Value name {value_name} is not a column in the dataframe')
    return value_list


def calculate_amounts_from_dfs(dfs, amount_names):
    """
    calculate amount of defined significant/insignificant positive/negative values in given pandas dataframes
    :param dfs: list of dataframes to be analyzed
    :param amount_names: should contain "pos" for positive values (test statistic > 0) and "insign" for insignificant
    values (p-value > 0,05). Else calculate negative and significant value.
    :return: list of amounts, sorted by amount_name first, then dataframe
    """
    amounts = []
    for amount_name in amount_names:
        for df in dfs:
            # filter for positive or negative test stats
            if 'pos' in amount_name:
                filtered_df = df[df['test_stat'] > 0]
            else:
                filtered_df = df[df['test_stat'] < 0]
            # differentiate significant and insignificant values
            if 'insign' in amount_name:
                insign_df = filtered_df[filtered_df['p_value'] > 0.05]
                amounts.append(len(insign_df.index))
            else:
                sign_df = filtered_df[filtered_df['p_value'] < 0.05]
                amounts.append(len(sign_df.index))
    return amounts

