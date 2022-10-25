import csv

import nltk

import weat
import pandas as pd
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents
import numpy as np
# import corpora

import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec, KeyedVectors

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


def load_pretrained_embeddings(glove_file):
    """ function to read in pretrained embedding vectors from a 6b glove file
        :param glove_file:      the path to the glove file (vocabulary of 400k words)
    """
    embedding_dict = dict()
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split(' ')
            word = row[0]  # get word
            vector = [float(i) for i in row[1:]]  # get vector
            embedding_dict[word] = vector
    return embedding_dict


# TODO: only for testing purposes, remove!
def read_text_data():
    # Reads ‘alice.txt’ file
    sample = open(file='data/alice_in_wonderland.txt', encoding="utf8")
    return sample.read()


def preprocess_text_for_word_embedding_creation(s):
    # TODO: remove this code snippet, should be done in preprocessing
    # Replaces escape character with space
    # f = s.replace("\n", " ")

    text_data = []

    tokenized_sents = []
    # iterate through each sentence in the file
    for i in sent_tokenize(s):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        tokenized_sents.append(temp)
    # TODO: improve code here! (but it seems, second for-loop cannot be avoided)
    #   maybe just make one-liners from for-loops
    #   maybe compare performance of two versions
    # tag the sentences with the universal tagset. pos_tag_sents has improved performance in confront to pos_tag
    pos_tagged = pos_tag_sents(sentences=tokenized_sents, tagset="universal", lang="eng")

    # connect tag and word to string to simplify usage in later trained model
    for sent in pos_tagged:
        tag_connected_sent = []
        for t in sent:
            tag_connected_sent.append(t[0] + "_" + t[1])
            # TODO: is this method sensible? It for sure solves the KeyError-problem!
        text_data.append(tag_connected_sent)
    return text_data


def make_word_emb_model(data):
    # TODO: try if skipgram or CBOW (here, default) work better!
    return gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)


# TODO: include weights?
def load_word_set_from_csv_by_metaphor_id(df, metaphor_id, pos):
    df = df[(df['metaphor_id'].str.contains(metaphor_id)) | (df['metaphor_id'] == metaphor_id)]
    word_set = df['word_pos'].tolist()
    if not pos == '':
        filtered = filter(lambda word: pos in word, word_set)
        word_set = list(filtered)
    print(f'word_set for metaphor_id {metaphor_id} and pos {pos}: {word_set}')
    return word_set


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
    if type(model) == KeyedVectors:
        word_vecs = [model[word.split('_')[0]] for word in word_list]
    elif model:  # model is Word2vec
        word_vecs = [model.wv[word] for word in word_list]
    else:
        word_vecs = word_list
    return word_vecs


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


# method compare_each gives extremely low values, but still better than baseline
def execute_experiment(model, method, similarity_measure, pos_tags=['']):
    # read data from word sets input csv
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)

    # prepare output csv file
    # TODO: improve naming, depending on used model
    output_file_path = f'results/googlenews_{method}_{similarity_measure}_{"_".join(pos_tags)}results.csv'
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        # TODO: include metaphor-name
        writer.writerow(['metaphor-id', 'metaphor-name', 'pos', 'similarity', 'baseline_performance'])

    # TODO: more general, takes all info from csv file, maybe implement
    # get all different metaphor ids
    # metaphor_ids = set(df['metaphor_id'].values)
    # clean up metaphor ids that contain multiple metaphors
    # sort list
    # iterate all pos-tags given for experiment
    for pos in pos_tags:
        # TODO: implicates knowledge about content of csv, maybe improve (see above)
        # iterate all metaphors
        for i in range(1, 12):
            # TODO: include other iterations with different word forms (pos)
            # load 2 word sets for metaphor
            metaphor_ids = [f'{i}_1', f'{i}_2']
            word_set1 = load_word_set_from_csv_by_metaphor_id(df, metaphor_ids[0], pos)
            word_set2 = load_word_set_from_csv_by_metaphor_id(df, metaphor_ids[1], pos)
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
                similarity = weat.similarity(mean_vec1, mean_vec2, similarity_measure)
                # TODO: maybe improve way of getting mean random similarity
                for random_vecs in random_vector_sets:
                    random_mean_vec = create_mean_vector_from_multiple(random_vecs)
                    random_similarity_sum += weat.similarity(mean_vec1, random_mean_vec, similarity_measure)
            elif method == "compare_each":
                word_vecs1 = vectorize_word_list(word_set1, model)
                word_vecs2 = vectorize_word_list(word_set2, model)
                similarity = weat.set_s(word_vecs1, word_vecs2, similarity_measure)
                for random_vecs in random_vector_sets:
                    random_similarity_sum += weat.set_s(word_vecs1, random_vecs, similarity_measure)

            random_similarity = random_similarity_sum / len(random_vector_sets)

            # write results to csv file
            with open(output_file_path, mode='a', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=',')
                pos_string = "all" if pos == '' else pos
                writer.writerow([i, metaphor_name, pos_string, similarity, random_similarity])


if __name__ == '__main__':
    # s = corpora.preprocess_wiki_dump()
    # s = read_text_data()
    # data = preprocess_text_for_word_embedding_creation(s)
    # model = make_word_emb_model(data)
    model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)    # TODO: training needed for better results?
    # print("Cosine similarity between 'be' " +
          # "and 'is' - CBOW : ",
          # model.wv.similarity("be_VERB", "is_VERB"))
    execute_experiment(model, 'compare_each', similarity_measure='manhattan')

    '''
    # prints first 10 entries from vocab
    word_key = ''
    for index, word in enumerate(model.wv.index_to_key):
        if index == 10:
            word_key = word
            break
        print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")
    print(f"word_key: {word_key}")
    vector = model.wv[word_key]
    print(f"vector by word_key {word_key}: {vector}")'''
    # model.save("wiki_word2vec.model")

    # TODO: visualize results!
    #   maybe similarity inside of sets (similarity matrix?)
    #   similarity numbers of sets and baseline with matplotlib.pyplot Streu- oder Liniendiagramm
