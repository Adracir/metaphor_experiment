import csv

import weat
# import embeddings
import pandas as pd
import random
import numpy as np
import warnings
from gensim.models import Word2Vec, KeyedVectors
import os

warnings.filterwarnings(action='ignore')

# ignores tensorflow warnings and infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WORD_DOMAIN_SETS_FILE = "word_sets.csv"


# TODO: move code to different files, e.g. "corpora.py", "embeddings.py" etc.
# TODO: train own embeddings with "real" database
#   wikipedia
#   gutenberg?
# TODO: make requirements.txt,
#   also find out how to deal with resources that had to be downloaded from nltk in the code
#   (e.g. nltk.download('universal_tagset'))
# TODO: unify meta info for all functions,
#  https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings
# TODO: only allow either KeyedVectors or Word2Vec model, remove all if-else options regarding this problem


def load_word_set_from_csv_by_metaphor_id(df, metaphor_id, pos='', weights=False):
    df = df[((df['metaphor_id'].str.contains(metaphor_id)) & (df['metaphor_id'].str.contains('\.'))) | (
            df['metaphor_id'] == metaphor_id)]
    word_set = df['word_pos'].tolist()
    if weights:
        double_words_df = df[df['weight'] == 2]
        for dw in double_words_df['word_pos'].tolist():
            word_set.append(dw)
    if not pos == '':
        filtered = filter(lambda word: pos in word, word_set)
        word_set = list(filtered)
    # print(f'word_set for metaphor_id {metaphor_id} and pos {pos}: {word_set}')
    return word_set


# TODO: decide: keep or throw away?
# TODO: maybe include weights predefined in word_sets.csv to stress especially useful words
def create_mean_vector_from_multiple(word_list, model=None):
    """
    calculates the mean vector multiple words
    :param word_list: list of words, in the preprocessed form (pos-tagged with "word"_"tag", can be obtained using
    preprocess_text_for_word_embedding_creation
    :param model: Word2Vec-model
    :return: one vector, same length as all vectors in model, that combines all word vectors from the word list
    according to the used model by giving the average values
    """
    # TODO: check again if any pre-defined method from word2vec might be used
    # get vectors for words
    word_vecs = vectorize_word_list(word_list, model)
    # combine word_vecs1,2,3... to np.array a [[num1, num2, num3...],[num1, num2, num3], ...]
    a = np.column_stack([word_vec for word_vec in word_vecs])
    # calculate mean vector
    vec = np.mean(a, axis=1)
    return vec


def vectorize_word_list(word_list, model):
    word_vecs = []
    unknown_words = []
    for word in word_list:
        try:
            if type(model) == KeyedVectors:
                word_vecs.append(model[word.split('_')[0]])
            elif model:  # model is Word2vec
                word_vecs.append(model.wv[word])
            else:
                word_vecs.append(word)
        except KeyError:
            unknown_words.append(word)
    # TODO: maybe find better way to analyze unknown words
    return (word_vecs, set(unknown_words))


def create_random_word_vector_sets(num, model, len):
    vector_sets = []
    for n in range(num):
        temp = []
        for l in range(len):
            if type(model) == KeyedVectors:
                temp.append(model[random.choice(model.index_to_key)])
            else:
                temp.append(model.wv[random.choice(model.wv.index_to_key)])
        vector_sets.append(temp)
    return vector_sets


def get_nr_of_metaphors_from_dataframe(df):
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


# TODO: for interpretation, make also p-values https://www.scribbr.com/statistics/p-value/
# method compare_each gives extremely low values, but still better than baseline
def execute_experiment(model, model_name, method, similarity_measure, pos_tags=[''], weights=False):
    # read data from word sets input csv
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)

    # prepare output csv file
    output_file_path = f'results/{model_name}_{method}_{similarity_measure}_{"_".join(pos_tags)}{"_weighted" if weights else ""}results.csv'
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['metaphor-id', 'metaphor-name', 'pos', 'similarity', 'baseline_performance'])

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
            similarity = 0
            # initiate 100 random word vector sets
            random_vector_sets = create_random_word_vector_sets(100, model, len(word_set1))
            # calculate random baseline similarity, depending on chosen method
            random_similarity_sum = 0
            # TODO: improve embeddings, many words give keyerror at the moment
            # TODO: also improve error handling
            # calculate similarity between both sets, depending on chosen method
            if method == "mean_vector":
                mean_vec1 = create_mean_vector_from_multiple(word_set1, model)
                mean_vec2 = create_mean_vector_from_multiple(word_set2, model)
                # TODO: maybe use model.wv.n_similarity?
                similarity = weat.similarity(mean_vec1, mean_vec2, similarity_measure)
                # TODO: maybe improve way of getting mean random similarity
                for random_vecs in random_vector_sets:
                    random_mean_vec = create_mean_vector_from_multiple(random_vecs)
                    random_similarity_sum += weat.similarity(mean_vec1, random_mean_vec, similarity_measure)
            elif method == "compare_each":
                vectorized1 = vectorize_word_list(word_set1, model)
                word_vecs1 = vectorized1[0]
                unknown_words.extend(vectorized1[1])
                vectorized2 = vectorize_word_list(word_set2, model)
                word_vecs2 = vectorized2[0]
                unknown_words.extend(vectorized2[1])
                similarity = weat.set_s(word_vecs1, word_vecs2, similarity_measure)
                for random_vecs in random_vector_sets:
                    random_similarity_sum += weat.set_s(word_vecs1, random_vecs, similarity_measure)

            random_similarity = random_similarity_sum / len(random_vector_sets)

            # write results to csv file
            with open(output_file_path, mode='a', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=',')
                pos_string = "all" if pos == '' else pos
                writer.writerow([i, metaphor_name, pos_string, similarity, random_similarity])
    return set(unknown_words)


if __name__ == '__main__':
    # model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    model = Word2Vec.load("models/word2vec_wiki_1-200000_skipgram_more_vocab2.model")
    # print('lala')
    # embeddings.evaluate_embeddings(model)
    # uw = execute_experiment(model, 'word2vec_wiki_1-200000_skipgram', 'compare_each', similarity_measure='cosine',
    #               pos_tags=['', 'ADJ', 'VERB', 'NOUN'], weights=True)
    # for w in uw:
    #    print(f'Wort nicht enthalten: {w}')
