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
    :param metaphor_id: for each set of words, 'metaphor-nr'_'domain-nr', for example 1_1 for the first domain of the first metaphor
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


def create_random_word_vector_sets(num, keyed_vectors, len):
    """
    create random vector sets
    :param num: number of vector sets wanted
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param len: number of vectors per set
    :return: list with len num of lists with len len, containing random vectors from the given model
    """
    vector_sets = []
    for n in range(num):
        temp = []
        for l in range(len):
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
    output_file_path = f'results/{model_name}_{similarity_measure}_{"-".join(pos_tags)}{"_weighted_" if weights else "_"}results.csv'
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['metaphor_id', 'metaphor_name', 'pos', 'mean_similarity', 'baseline_performance', 'test_statistic', 'p_value'])

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
            with open(output_file_path, mode='a', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=',')
                writer.writerow([i, metaphor_name, pos, mean_similarity, random_similarity, test_statistic, p_value])
    return set(unknown_words)


# TODO: decide whether and to which extent to include these files and this code
def create_result_summary():
    """
    create a summary csv file from all existing results, containing maximum and minimum as well as mean values for
    the different modalities.
    """
    # iterate all relevant files (in results, starting with "BL")
    output_file_path = 'results/summary_BL_results.csv'
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['Baseline', 'Korpus', 'Gewichtet', 'Methode', 'POS', 'Art des Werts', 'Metapher', 'Wert', 'Baseline-Wert', 'Test-Stat', 'P-Value'])
    directory = 'results'
    pos_arr = ['all', 'ADJ', 'VERB', 'NOUN']
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a relevant file
        baseline = 'new' if re.match('BL-.*', filename) else ('old' if re.match('word2vec.*?', filename) else '')
        if os.path.isfile(f) and baseline:
            # extract important info from filename (baseline, corpus, distance-measure, weighted/unweighted)
            splitted_infos = filename.split('_')
            corpus = splitted_infos[1]
            distance_measure = splitted_infos[4]
            weighted = "Ja" if splitted_infos[6] == "weighted" else "Nein"
            # read csv with pd:
            df = pd.read_csv(f)
            # write to new csv file:
            # iterating POS
            for pos in pos_arr:
                pos_df = df[df['pos'] == pos]
                metaphor_names = pos_df['metaphor_name'].tolist()
                metaphor_numbers = pos_df['metaphor_id'].tolist()
                similarities = pos_df['mean_similarity'].tolist()
                baseline_performance = pos_df['baseline_performance'].tolist()
                test_statistic = pos_df['test_statistic'].tolist()
                p_value = pos_df['p_value'].tolist()
                # all info in line for max & min test stat
                i_max_test_stat = test_statistic.index(max(test_statistic))
                i_min_test_stat = test_statistic.index(min(test_statistic))
                # all info in line for max & min mean_similarity
                i_max_sim = similarities.index(max(similarities))
                i_min_sim = similarities.index(min(similarities))
                # write info in output file
                with open(output_file_path, mode='a', newline='') as output_file:
                    writer = csv.writer(output_file, delimiter=',')
                    writer.writerow([baseline, corpus, weighted, distance_measure, pos, 'max (Test Stat)',
                                     f'{metaphor_numbers[i_max_test_stat]} {metaphor_names[i_max_test_stat]}',
                                     similarities[i_max_test_stat], baseline_performance[i_max_test_stat],
                                     test_statistic[i_max_test_stat], p_value[i_max_test_stat]])
                    writer.writerow([baseline, corpus, weighted, distance_measure, pos, 'min (Test Stat)',
                                     f'{metaphor_numbers[i_min_test_stat]} {metaphor_names[i_min_test_stat]}',
                                     similarities[i_min_test_stat], baseline_performance[i_min_test_stat],
                                     test_statistic[i_min_test_stat], p_value[i_min_test_stat]])
                    writer.writerow([baseline, corpus, weighted, distance_measure, pos, 'max (similarity)',
                                     f'{metaphor_numbers[i_max_sim]} {metaphor_names[i_max_sim]}',
                                     similarities[i_max_sim], baseline_performance[i_max_sim],
                                     test_statistic[i_max_sim], p_value[i_max_sim]])
                    writer.writerow([baseline, corpus, weighted, distance_measure, pos, 'min (similarity)',
                                     f'{metaphor_numbers[i_min_sim]} {metaphor_names[i_min_sim]}',
                                     similarities[i_min_sim], baseline_performance[i_min_sim],
                                     test_statistic[i_min_sim], p_value[i_min_sim]])
                    # mean values
                    writer.writerow([baseline, corpus, weighted, distance_measure, pos, 'mean',
                                     '',
                                     np.mean(similarities), np.mean(baseline_performance),
                                     np.mean(test_statistic), np.mean(p_value)])


def create_result_summary_val_copy(baseline):
    """
    unite all results to one csv file, just copying all values
    """
    # iterate all relevant files (in results, starting with "BL")
    output_file_path = 'results/all-values.csv'
    if not os.path.isfile(output_file_path):
        with open(output_file_path, mode='w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            writer.writerow(['Baseline', 'Korpus', 'Gewichtet', 'Methode', 'POS', 'Art des Werts', 'Metapher', 'Wert', 'Baseline-Wert', 'Test-Stat', 'P-Value'])
    directory = 'results'
    prefix = 'BL-.*' if baseline=="saved" else 'word2vec.*'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a relevant file
        if os.path.isfile(f) and re.match(prefix, filename) :
            # extract important info from filename (corpus, distance-measure, weighted/unweighted)
            splitted_infos = filename.split('_')
            corpus = splitted_infos[1]
            distance_measure = splitted_infos[4]
            weighted = "Ja" if splitted_infos[6] == "weighted" else "Nein"
            # read csv with pd:
            df = pd.read_csv(f)
            # write to new csv file:
            with open(output_file_path, mode='a', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=',')
                for row in df.itertuples():
                    writer.writerow([baseline, corpus, weighted, distance_measure, row.pos, 'direct',
                                     f'{row.metaphor_id} {row.metaphor_name}',
                                     row.mean_similarity, row.baseline_performance,
                                     row.test_statistic, row.p_value])


if __name__ == '__main__':
    create_result_summary_val_copy("saved")
    # TODO: push random vector sets? or are they to big?
    '''keyed_vectors1 = KeyedVectors.load("models/word2vec_gutenberg_1-8000u16001-26000_skipgram.wordvectors", mmap='r')
    keyed_vectors2 = KeyedVectors.load("models/word2vec_wiki_1-200000_skipgram.wordvectors", mmap='r')
    # generate one large set of random vectors per model for all calculations
    # random_vector_sets = create_random_word_vector_sets(250, keyed_vectors, 24)
    # rvs = np.asarray(random_vector_sets)
    # np.save('data/gutenberg_random_vector_sets.npy', rvs)
    # random_vector_sets1 = np.load('data/gutenberg_random_vector_sets.npy')
    # random_vector_sets2 = np.load('data/wiki_random_vector_sets.npy')
    random_vector_sets1 = create_random_word_vector_sets(100, keyed_vectors1, 24)
    random_vector_sets2 = create_random_word_vector_sets(100, keyed_vectors2, 24)
    execute_experiment(keyed_vectors1, 'word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='cosine',
                           weights=True)
    execute_experiment(keyed_vectors2, 'word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2,
                       similarity_measure='cosine',
                       weights=True)'''
    '''for measure in ["cosine", "canberra", "euclidian", "manhattan"]:
        for pos in ["all", "ADJ", "VERB", "NOUN"]:
            for pref in ["BL-", ""]:
                plot.output_to_plot(f'results/{pref}word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_all-ADJ-VERB-NOUN_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_wiki_1-200000_skipgram_{measure}_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}word2vec_wiki_1-200000_skipgram_{measure}_all-ADJ-VERB-NOUN_results.csv', pos=pos)'''
