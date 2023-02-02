import calc
import plot
import utils
import result_evaluation
import pandas as pd
import random
import numpy as np
import warnings
from gensim.models import KeyedVectors

warnings.filterwarnings(action='ignore')

WORD_DOMAIN_SETS_FILE = "data/word_sets.csv"


def execute_experiment(keyed_vectors, model_name, similarity_measure, random_vector_sets, prefix,
                       pos_tags=['all', 'ADJ', 'VERB', 'NOUN'], weights=False):
    """
    execute the main experiment of this work, calculating the similarity between pairs of word domains.
    Save results to a csv file in results folder
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param model_name: name of the Word2Vec model, used to name the csv file
    :param similarity_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used
    similarity/distance measure
    :param random_vector_sets: list of lists, containing random metaphors from the model, used as a baseline
    :param prefix: string that signifies what baseline was used, appended to beginning of output filename
    :param pos_tags: POS tags to be analyzed
    :param weights: whether some words should get a double weight
    :return: a list of words that were not in the vocabulary of the model during the execution of the experiment
    """
    # read data from word sets input csv
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)

    # prepare output csv file
    output_file_path = f'results/{prefix}-{model_name}_{similarity_measure}_{"-".join(pos_tags)}' \
                       f'{"_weighted_" if weights else "_unweighted_"}results.csv'
    utils.write_info_to_csv(output_file_path, ['metaphor_id', 'metaphor_name', 'pos', 'mean_similarity',
                                               'baseline_performance', 'test_statistic', 'p_value'])

    unknown_words = []
    for pos in pos_tags:
        # iterate all metaphors
        for i in range(1, get_nr_of_metaphors_from_dataframe(df) + 1):
            # load 2 word sets for metaphor
            metaphor_ids = [f'{i}_1', f'{i}_2']
            word_set1 = load_word_set_from_df_by_metaphor_id(df, metaphor_ids[0], pos, weights)
            word_set2 = load_word_set_from_df_by_metaphor_id(df, metaphor_ids[1], pos, weights)
            # generate metaphor name from dataframe
            df1 = df[(df['metaphor_id'].str.contains(metaphor_ids[0])) | (df['metaphor_id'] == metaphor_ids[0])]
            df2 = df[(df['metaphor_id'].str.contains(metaphor_ids[1])) | (df['metaphor_id'] == metaphor_ids[1])]
            metaphor_name = f'{df1["domain"].tolist()[0]} is {df2["domain"].tolist()[0]}'
            # create random baseline with random word vectors from parameter and the first word set
            random_similarities = []
            # vectorize first word list and keep unknown words in list
            vectorized1 = vectorize_word_list(word_set1, keyed_vectors)
            word_vecs1 = vectorized1[0]
            unknown_words.extend(vectorized1[1])
            # calculate random baseline distance, depending on chosen method
            for random_vecs in random_vector_sets:
                random_similarities.append(np.mean(calc.generate_similarities(word_vecs1, random_vecs,
                                                                              similarity_measure)))
            random_similarity = np.mean(random_similarities)
            # vectorize first word list and keep unknown words in list
            vectorized2 = vectorize_word_list(word_set2, keyed_vectors)
            word_vecs2 = vectorized2[0]
            unknown_words.extend(vectorized2[1])
            # calculate distance/similarity between the two domain word sets
            similarities = calc.generate_similarities(word_vecs1, word_vecs2, similarity_measure)
            mean_similarity = np.mean(similarities)
            # carry out t test with baseline
            ttest = calc.t_test(similarities, random_similarity)
            test_statistic = ttest[0]
            p_value = ttest[1]
            # write results to csv file
            utils.write_info_to_csv(output_file_path, [i, metaphor_name, pos, mean_similarity, random_similarity,
                                                       test_statistic, p_value], 'a')
    return set(unknown_words)


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
    df = df[((df['metaphor_id'].str.contains(metaphor_id)) & (df['metaphor_id'].str.contains(r'\.'))) | (
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
    :return: a list of word vectors and a set of words that were not in the vocabulary of the model
    """
    word_vecs = []
    unknown_words = []
    for word in word_list:
        try:
            word_vecs.append(keyed_vectors[word])
        except KeyError:
            unknown_words.append(word)
    return word_vecs, set(unknown_words)


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


if __name__ == '__main__':
    # load word vectors
    '''keyed_vectors1 = KeyedVectors.load("models/word2vec_gutenberg_1-8000u16001-26000_skipgram.wordvectors", mmap='r')
    keyed_vectors2 = KeyedVectors.load("models/word2vec_wiki_1-200000_skipgram.wordvectors", mmap='r')'''
    # generate one large set of random vectors per model for all calculations
    '''random_vector_sets1 = create_random_word_vector_sets(250, keyed_vectors1, 24)
    random_vector_sets2 = create_random_word_vector_sets(250, keyed_vectors2, 24)
    rvs1 = np.asarray(random_vector_sets1)
    np.save('data/gutenberg_random_vector_sets.npy', rvs1)
    rvs2 = np.asarray(random_vector_sets2)
    np.save('data/wiki_random_vector_sets.npy', rvs2)'''
    # load saved random vectors
    '''random_vector_sets1 = np.load('data/gutenberg_random_vector_sets.npy')
    random_vector_sets2 = np.load('data/wiki_random_vector_sets.npy')'''
    # calculate all possible results
    '''for measure in ["cosine", "canberra", "euclidian", "manhattan"]:
        for weight in [True, False]:
            execute_experiment(keyed_vectors1, 'word2vec_gutenberg_1-8000u16001-26000_skipgram',
                               random_vector_sets=random_vector_sets1, prefix='savedBL', similarity_measure=measure, weights=weight)
            execute_experiment(keyed_vectors2, 'word2vec_wiki_1-200000_skipgram', 
                               random_vector_sets=random_vector_sets2, prefix='savedBL', similarity_measure=measure, weights=weight)
            execute_experiment(keyed_vectors1, 'word2vec_gutenberg_1-8000u16001-26000_skipgram',
                               random_vector_sets=create_random_word_vector_sets(250, keyed_vectors1, 24),
                               prefix='mixedBL', similarity_measure=measure, weights=weight)
            execute_experiment(keyed_vectors2, 'word2vec_wiki_1-200000_skipgram',
                               random_vector_sets=create_random_word_vector_sets(250, keyed_vectors2, 24),
                               prefix='mixedBL', similarity_measure=measure, weights=weight)
            print(f'measure {measure}, weight {weight} done')'''
    # visualize all results in plots
    '''for measure in ["cosine", "canberra", "euclidian", "manhattan"]:
        for pos in ["all", "ADJ", "VERB", "NOUN"]:
            for pref in ["savedBL", "mixedBL"]:
                plot.output_to_plot(f'results/{pref}-word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_'
                                    f'all-ADJ-VERB-NOUN_unweighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}-word2vec_gutenberg_1-8000u16001-26000_skipgram_{measure}_'
                                    f'all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}-word2vec_wiki_1-200000_skipgram_{measure}_'
                                    f'all-ADJ-VERB-NOUN_weighted_results.csv', pos=pos)
                plot.output_to_plot(f'results/{pref}-word2vec_wiki_1-200000_skipgram_{measure}_'
                                    f'all-ADJ-VERB-NOUN_unweighted_results.csv', pos=pos)'''
    # add results to summary all-values.csv file
    '''result_evaluation.append_result_summary_val_copy(['mixedBL'])'''
    # generate confront files
    for param in ["metaphor", "pos", "corpus", "weighted", "method"]:
        result_evaluation.confront_results_for_one_param(param, ['savedBL', 'mixedBL'])
