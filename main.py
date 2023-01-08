import csv

import calc
import plot
# import embeddings
import pandas as pd
import random
import numpy as np
import warnings
from gensim.models import KeyedVectors
import os

warnings.filterwarnings(action='ignore')

# ignores tensorflow warnings and infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WORD_DOMAIN_SETS_FILE = "data/word_sets.csv"


# TODO: make requirements.txt,
#   also find out how to deal with resources that had to be downloaded from nltk in the code
#   (e.g. nltk.download('universal_tagset'))
# TODO: unify meta info for all functions,
#  https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings
# TODO: only allow either KeyedVectors or Word2Vec or Glove model, remove all if-else options regarding this problem


def load_word_set_from_csv_by_metaphor_id(df, metaphor_id, pos='all', weights=False):
    """
    loads set of words from a given pandas dataframe generated from the file data/word_sets.csv
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


def vectorize_word_list(word_list, keyedvectors):
    """
    a helper method to retrieve the corresponding word vectors to a list of words
    :param word_list: list of words
    :param model: Word2Vec model to retrieve the word vectors from
    :return: a list of word vectors, a set of words that were not in the vocabulary of the model
    """
    word_vecs = []
    unknown_words = []
    for word in word_list:
        try:
            word_vecs.append(keyedvectors[word])
        except KeyError:
            unknown_words.append(word)
    # TODO: maybe find better way to analyze unknown words
    return word_vecs, set(unknown_words)


def create_random_word_vector_sets(num, keyedvectors, len):
    """
    helps create random vector sets
    :param num: number of vector sets wanted
    :param model: Word2Vec model to retrieve the vectors from
    :param len: number of vectors per set
    :return: list with len num of lists with len len, containing random vectors from the given model
    """
    vector_sets = []
    for n in range(num):
        temp = []
        for l in range(len):
            temp.append(keyedvectors[random.choice(keyedvectors.index_to_key)])
        vector_sets.append(temp)
    return vector_sets


def get_nr_of_metaphors_from_dataframe(df):
    """
    counts the number of distinct metaphors using the metaphor_id from the pandas dataframe
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
    executes the main experiment of this work, calculating the similarity between pairs of word domains,
    thought to represent metaphorical connections. Saves results to a csv file in results folder
    :param model: the Word2Vec model to be used
    :param model_name: name of the model, used to name the csv file
    :param similarity_measure: kind of distance measure to be used: cosine, manhattan, canberra or euclidian
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
        writer.writerow(['metaphor-id', 'metaphor-name', 'pos', 'mean_similarity', 'baseline_performance', 'test_statistic', 'p_value'])

    unknown_words = []
    for pos in pos_tags:
        # iterate all metaphors
        for i in range(1, get_nr_of_metaphors_from_dataframe(df) + 1):
            # load 2 word sets for metaphor
            metaphor_ids = [f'{i}_1', f'{i}_2']
            word_set1 = load_word_set_from_csv_by_metaphor_id(df, metaphor_ids[0], pos, weights)
            word_set2 = load_word_set_from_csv_by_metaphor_id(df, metaphor_ids[1], pos, weights)
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


if __name__ == '__main__':
    keyed_vectors1 = KeyedVectors.load("models/word2vec_gutenberg_1-8000u16001-26000_skipgram.wordvectors", mmap='r')
    keyed_vectors2 = KeyedVectors.load("models/word2vec_wiki_1-200000_skipgram.wordvectors", mmap='r')
    # generate one large set of random vectors per model for all calculations
    # random_vector_sets = create_random_word_vector_sets(250, keyed_vectors, 24)
    # rvs = np.asarray(random_vector_sets)
    # np.save('data/gutenberg_random_vector_sets.npy', rvs)
    random_vector_sets1 = np.load('data/gutenberg_random_vector_sets.npy')
    random_vector_sets2 = np.load('data/wiki_random_vector_sets.npy')
    # TODO: regenerate results
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='cosine',
                           weights=False)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2,
                       similarity_measure='cosine',
                       weights=False)
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='cosine',
                           weights=True)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='cosine',
                           weights=True)
    print("Cosine done")
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='euclidian',
                           weights=False)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='euclidian',
                           weights=False)
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='euclidian',
                           weights=True)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='euclidian',
                           weights=True)
    print("Euclidian done")
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='manhattan',
                           weights=False)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='manhattan',
                           weights=False)
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='manhattan',
                           weights=True)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='manhattan',
                           weights=True)
    print("Manhattan done")
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='canberra',
                           weights=False)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='canberra',
                           weights=False)
    execute_experiment(keyed_vectors1, 'BL-word2vec_gutenberg_1-8000u16001-26000_skipgram', random_vector_sets=random_vector_sets1, similarity_measure='canberra',
                           weights=True)
    execute_experiment(keyed_vectors2, 'BL-word2vec_wiki_1-200000_skipgram', random_vector_sets=random_vector_sets2, similarity_measure='canberra',
                           weights=True)
    print("Canberra done")
    '''for pos in ["all", "ADJ", "VERB", "NOUN"]:
        plot.output_to_plot('results/word2vec_gutenberg_1-8000u16001-26000_skipgram_euclidian_all-ADJ-VERB-NOUN_results.csv', pos=pos)
        plot.output_to_plot('results/word2vec_gutenberg_1-8000u16001-26000_skipgram_euclidian_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
        plot.output_to_plot('results/word2vec_wiki_1-200000_skipgram_euclidian_all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
        plot.output_to_plot('results/word2vec_wiki_1-200000_skipgram_euclidian_all-ADJ-VERB-NOUN_results.csv', pos=pos)'''

