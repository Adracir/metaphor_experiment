import csv

import calc
import plot
import pandas as pd
import random
import numpy as np
import warnings
from gensim.models import KeyedVectors
import os
import re

warnings.filterwarnings(action='ignore')

WORD_DOMAIN_SETS_FILE = "data/word_sets.csv"


# TODO: unify meta info for all functions,
#  https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings
# TODO: generally break functions in different parts. one function should only do one thing.


def load_word_set_from_df_by_metaphor_id(df, metaphor_id, pos='all', weights=False):
    """
    load set of words from a given pandas dataframe generated from the file data/word_sets.csv
    :param df: dataframe
    :param metaphor_id: for each set of words, 'metaphor-nr'_'domain-nr', for example 1_1 for the first domain of the
    first metaphor
    :param pos: pos-tag, 'NOUN', 'VERB', 'ADJ' or 'all'
    :param weights: should weighted words be double in the set?
    :return: a list of words to which all of the above applies
    """
    df = df[((df['metaphor_id'].str.contains(metaphor_id)) & (df['metaphor_id'].str.contains('\.'))) | (
            df['metaphor_id'] == metaphor_id)]
    word_list = df['word_pos'].tolist()
    if weights:
        double_words_df = df[df['weight'] == 2]
        for dw in double_words_df['word_pos'].tolist():
            word_list.append(dw)
    if not pos == 'all':
        filtered = filter(lambda word: pos in word, word_list)
        word_list = list(filtered)
    return word_list


def vectorize_word_list(word_list, keyed_vectors):
    """
    retrieve the corresponding word vectors to a list of words
    :param word_list: list of words
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :return: a list of word vectors, a set of words that were not in the vocabulary of the model
    """
    word_vecs = []
    unknown_words = []
    for word in word_list:
        try:
            word_vecs.append(keyed_vectors[word])
        except KeyError:
            unknown_words.append(word)
    # TODO: maybe find better way to analyze unknown words
    return word_vecs, set(unknown_words)


def create_random_word_vector_sets(num, keyed_vectors, length):
    """
    create random vector sets
    :param num: number of vector sets wanted
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param length: number of vectors per set
    :return: list with (num) lists, each containing (length) random vectors from the given model
    """
    vector_sets = []
    for n in range(num):
        temp = []
        for l in range(length):
            temp.append(keyed_vectors[random.choice(keyed_vectors.index_to_key)])
        vector_sets.append(temp)
    return vector_sets


def get_nr_of_metaphors_from_dataframe(df):
    """
    count the number of distinct metaphors using the metaphor_id from the pandas dataframe
    :param df: given pandas dataframe, retrieved from data/word_sets.csv
    :return: total number of metaphors given in the dataframe
    """
    metaphor_ids = []
    str_list = set(df['metaphor_id'].tolist())
    for s in str_list:
        if '.' in s:
            s_parts = s.split('.')
            for s_part in s_parts:
                i = int(s_part.split('_')[0])
                metaphor_ids.append(i)
        else:
            metaphor_ids.append(int(s.split('_')[0]))
    return max(metaphor_ids)


def execute_experiment(keyed_vectors, model_name, similarity_measure, random_vector_sets, pos_tags=['all', 'ADJ', 'VERB', 'NOUN'], weights=False):
    """
    execute the main experiment of this work, calculating the similarity between pairs of word domains.
    Save results to a csv file in results folder
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param model_name: name of the Word2Vec model, used to name the csv file
    :param similarity_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance measure
    :param random_vector_sets: list of lists, containing random metaphors from the model, used as a baseline
    :param pos_tags: POS tags to be analyzed
    :param weights: whether some words should get a double weight
    :return: a list of words that were not in the vocabulary of the model during the execution of the experiment
    """
    # read data from word sets input csv
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)

    # prepare output csv file
    output_file_path = f'results/{model_name}_{similarity_measure}_{"-".join(pos_tags)}' \
                       f'{"_weighted_" if weights else "_"}results.csv'
    write_info_to_csv(output_file_path, ['metaphor_id', 'metaphor_name', 'pos', 'mean_similarity',
                                         'baseline_performance', 'test_statistic', 'p_value'])

    unknown_words = []
    for pos in pos_tags:
        # iterate all metaphors
        for i in range(1, get_nr_of_metaphors_from_dataframe(df) + 1):
            # load 2 word sets for metaphor
            metaphor_ids = [f'{i}_1', f'{i}_2']
            word_set1 = load_word_set_from_df_by_metaphor_id(df, metaphor_ids[0], pos, weights)
            word_set2 = load_word_set_from_df_by_metaphor_id(df, metaphor_ids[1], pos, weights)
            df1 = df[(df['metaphor_id'].str.contains(metaphor_ids[0])) | (df['metaphor_id'] == metaphor_ids[0])]
            df2 = df[(df['metaphor_id'].str.contains(metaphor_ids[1])) | (df['metaphor_id'] == metaphor_ids[1])]
            metaphor_name = f'{df1["domain"].tolist()[0]} is {df2["domain"].tolist()[0]}'
            # initiate 100 random word vector sets
            # random_vector_sets = create_random_word_vector_sets(100, model, len(word_set1))
            random_similarities = []
            # vectorize first word list and keep unknown words in list
            vectorized1 = vectorize_word_list(word_set1, keyed_vectors)
            word_vecs1 = vectorized1[0]
            unknown_words.extend(vectorized1[1])
            # calculate random baseline distance, depending on chosen method
            for random_vecs in random_vector_sets:
                random_similarities.append(np.mean(calc.generate_similarities(word_vecs1, random_vecs, similarity_measure)))
            random_similarity = np.mean(random_similarities)
            # vectorize first word list and keep unknown words in list
            vectorized2 = vectorize_word_list(word_set2, keyed_vectors)
            word_vecs2 = vectorized2[0]
            unknown_words.extend(vectorized2[1])
            # calculate distance between the two domain word sets
            similarities = calc.generate_similarities(word_vecs1, word_vecs2, similarity_measure)
            mean_similarity = np.mean(similarities)
            ttest = calc.t_test(similarities, random_similarity)
            test_statistic = ttest[0]
            p_value = ttest[1]
            # write results to csv file
            write_info_to_csv(output_file_path, [i, metaphor_name, pos, mean_similarity, random_similarity, test_statistic, p_value], 'a')
    return set(unknown_words)


def create_result_summary_val_copy(baselines):
    """
    unite all results to one csv file, just copying all values
    :param baselines: list of baselines to be used, e.g. "saved" (saved random vectors in /data) or "mixed" (created on the
    fly during the execution of the experiment)
    """
    # prepare output file
    output_file_path = 'results/all-values.csv'
    if not os.path.isfile(output_file_path):
        write_info_to_csv(output_file_path, ['baseline', 'corpus', 'weighted', 'method', 'pos', 'metaphor',
                                             'similarity_value', 'baseline_value', 'test_stat', 'p_value'])
    directory = 'results'
    for baseline in baselines:
        prefix = 'BL-.*' if baseline == "saved" else 'word2vec.*'
        # iterate all relevant files (in results, starting with prefix)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a relevant file
            if os.path.isfile(f) and re.match(prefix, filename):
                # extract important info from filename (corpus, distance-measure, weighted/unweighted)
                splitted_infos = filename.split('_')
                corpus = splitted_infos[1]
                distance_measure = splitted_infos[4]
                # TODO: maybe simplify by renaming files
                weighted = "weighted" if splitted_infos[6] == "weighted" else "unweighted"
                # read csv with pd:
                df = pd.read_csv(f)
                # write to new csv file:
                for row in df.itertuples():
                    write_info_to_csv(output_file_path, [baseline, corpus, weighted, distance_measure, row.pos,
                                                         f'{row.metaphor_id} {row.metaphor_name}',
                                                         row.mean_similarity, row.baseline_performance,
                                                         row.test_statistic, row.p_value], 'a')


def confront_results_for_one_param(parameter):
    """
    collect relevant info from all results (results/all-values.csv) that allow closer interpretation of one parameter
    shaping the values, differentiating values from two different baselines
    save these infos to csv
    :param parameter: one of "metaphor", "pos", "corpus", "weighted", "method". Defines parameter from which perspective
    the results can be analyzed closer
    """
    # prepare output file
    output_file_path = f'results/{parameter}_confront.csv'
    to_be_calculated = ['mean_baseline_', 'mean_test_stat_', 'mean_p_value_', 'amount_pos_sign_',
                        'amount_pos_insign_', 'amount_neg_sign_', 'amount_neg_insign_']
    csv_headings = []
    baselines = ["saved", "mixed"]
    for calculation in to_be_calculated:
        for baseline in baselines:
            csv_headings.append(calculation + baseline)
    write_info_to_csv(output_file_path, [parameter, 'mean_similarity'] + csv_headings)
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
        write_info_to_csv(output_file_path, [param_value, mean_similarity] + mean_vals + amount_vals, 'a')


def metaphor_confront_for_one_param(parameter):
    """
    analyze different metaphors from the perspective of a parameter, trying to provide the data for an answer on the
    question: does the parameter have an impact on the results for the different metaphors?
    save info to csv file
    :param parameter: one of "corpus", "weighted", "method", "pos"
    """
    # prepare output file
    output_file_path = f'results/{parameter}_metaphor_confront.csv'
    # read all_values.csv
    df = pd.read_csv('results/all-values.csv')
    param_values = set(df[parameter].tolist())
    to_be_calculated = ['mean_similarity_', 'mean_baseline_', 'mean_test_stat_', 'mean_p_value_', 'amount_pos_sign_',
                        'amount_pos_insign_', 'amount_neg_sign_', 'amount_neg_insign_']
    csv_headings = []
    for calculation in to_be_calculated:
        for param_value in param_values:
            csv_headings.append(calculation + param_value)

    write_info_to_csv(output_file_path, ['metaphor'] + csv_headings)
    # iterate metaphors
    for metaphor in set(df['metaphor'].tolist()):
        # prepare dataframes
        filtered_df = df[df['metaphor'] == metaphor]
        param_dfs = [filtered_df[filtered_df[parameter] == param_value] for param_value in param_values]
        # calculate needed values
        mean_vals = calculate_mean_values_from_dfs(param_dfs, ['similarity_value', 'baseline_value', 'test_stat','p_value'])
        amount_vals = calculate_amounts_from_dfs(param_dfs, ['pos_sign', 'pos_insign', 'neg_sign', 'neg_insign'])
        write_info_to_csv(output_file_path, [metaphor] + mean_vals + amount_vals, 'a')


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


def write_info_to_csv(output_file_path, arr, mode='w'):
    """
    write array to csv
    :param output_file_path: path to which the file should be saved
    :param arr: containing all row values that should be written
    :param mode: csv writer mode: 'w' for writing to a new file, 'a' for appending an existing one
    """
    with open(output_file_path, mode=mode, newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(arr)


if __name__ == '__main__':
    # load word vectors
    # keyed_vectors1 = KeyedVectors.load("models/word2vec_gutenberg_1-8000u16001-26000_skipgram.wordvectors", mmap='r')
    # keyed_vectors2 = KeyedVectors.load("models/word2vec_wiki_1-200000_skipgram.wordvectors", mmap='r')
    # generate one large set of random vectors per model for all calculations
    # random_vector_sets = create_random_word_vector_sets(250, keyed_vectors, 24)
    # rvs = np.asarray(random_vector_sets)
    # np.save('data/gutenberg_random_vector_sets.npy', rvs)
    # load saved random vectors
    # random_vector_sets1 = np.load('data/gutenberg_random_vector_sets.npy')
    # random_vector_sets2 = np.load('data/wiki_random_vector_sets.npy')
    # create random vectors
    # random_vector_sets1 = create_random_word_vector_sets(100, keyed_vectors1, 24)
    # random_vector_sets2 = create_random_word_vector_sets(100, keyed_vectors2, 24)
    # calculate all possible results
    '''for measure in ["cosine", "canberra", "euclidian", "manhattan"]:
        for weight in [True, False]:
            execute_experiment(keyed_vectors1, 'word2vec_gutenberg_1-8000u16001-26000_skipgram',
                               random_vector_sets=random_vector_sets1, similarity_measure=measure, weights=weight)
            execute_experiment(keyed_vectors2, 'word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2,
                               similarity_measure=measure, weights=weight)
            print(f'measure {measure}, weight {weight} done')'''
    # visualize all results in plots
    '''for measure in ["cosine", "canberra", "euclidian", "manhattan"]:
        for pos in ["all", "ADJ", "VERB", "NOUN"]:
            for pref in ["BL-", ""]:
                plot.output_to_plot(f'results/{pref}word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_all-ADJ-VERB-NOUN_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_wiki_1-200000_skipgram_{measure}_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_wiki_1-200000_skipgram_{measure}_all-ADJ-VERB-NOUN_results.csv', pos=pos)'''
    # add results to summary all-values.csv file
    # create_result_summary_val_copy(["two"])
    # generate confront files
    # for param in ["metaphor", "pos", "corpus", "weighted", "method"]:
       # confront_results_for_one_param(param)
