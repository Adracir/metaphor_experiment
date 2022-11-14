import csv

import weat
import pandas as pd
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents, pos_tag
import numpy as np
from scipy.stats import pearsonr, spearmanr
# import corpora
import time
import warnings
import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.utils as gu
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


# TODO: still needed?
def read_text_data(file):
    sample = open(file, encoding="utf8")
    return sample.read()


def preprocess_text_for_word_embedding_creation(filename):
    with open(filename, encoding='utf8') as file:
        s = file.read()
        text_data = []
        tokenized = []
        print('starting to sent_tokenize')
        # takes a long time in nltk 3.7, that's why I downgraded to nltk 3.6.5 until upgrade is coming
        # https://github.com/nltk/nltk/issues/3013
        sents = sent_tokenize(s)
        print('starting to simple_preprocess')
        for sent in sents:
            tokenized.append(gu.simple_preprocess(sent))
        print('starting to pos-tag')
        tagged = pos_tag_sents(tokenized, tagset='universal', lang='eng')
        # TODO: is this method sensible? It for sure solves the KeyError-problem!
        print('formatting pos-tags')
        [text_data.append([t[0] + "_" + t[1] for t in sent]) for sent in tagged]
        return text_data


def make_word_emb_model(data, sg=0):
    # TODO: play around with different settings!
    return gensim.models.Word2Vec(data, min_count=1, sg=sg, vector_size=100, window=5)


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


def evaluate_embeddings(model, similarity_measure='cosine'):
    """
    method to print out the evaluation of a given model in correlation (Pearson and Spearman) to a human-based list of
    words (based on Rubenstein, H., & Goodenough, J. (1965). Contextual correlates of synonymy. Commun. ACM, 8, 627–633.
    https://doi.org/10.1145/365628.365657)
    For a good-functioning model, the first value is expected to be as high as possible, the pvalue is expected to be
    less than 0.05.
    :param model: either a Word2Vec model containing word vectors with keys formed as "Word_POS" or KeyedVectors
    """
    df = pd.read_csv('data/human_relatedness.csv')
    gold_standard_relatedness = [float(x) for x in df['synonymy'].tolist()]
    words1 = df['word1'].tolist()
    words2 = df['word2'].tolist()
    embedding_relatedness = []
    for i in range(0, len(words1)):
        if type(model) == Word2Vec:
            vec_word1 = model.wv[words1[i] + '_NOUN']
            vec_word2 = model.wv[words2[i] + '_NOUN']
        elif type(model) == KeyedVectors:
            vec_word1 = model[words1[i]]
            vec_word2 = model[words2[i]]
        else:
            break
        embedding_relatedness.append(weat.similarity(vec_word1, vec_word2, similarity_measure))
    if similarity_measure != 'cosine':
        for i in range(len(embedding_relatedness)):
            d = embedding_relatedness[i]
            d_normalized = d / max(embedding_relatedness)
            embedding_relatedness[i] = 1 - d_normalized
    print(pearsonr(gold_standard_relatedness, embedding_relatedness))
    print(spearmanr(gold_standard_relatedness, embedding_relatedness))


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


# method compare_each gives extremely low values, but still better than baseline
def execute_experiment(model, model_name, method, similarity_measure, pos_tags=[''], weights=False):
    # read data from word sets input csv
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)

    # prepare output csv file
    output_file_path = f'results/{model_name}_{method}_{similarity_measure}_{"_".join(pos_tags)}{"_weighted" if weights else ""}results.csv'
    with open(output_file_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['metaphor-id', 'metaphor-name', 'pos', 'similarity', 'baseline_performance'])

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
    # start = time.time()
    # print(f'text read in {time.time() - start} seconds')
    # data = preprocess_text_for_word_embedding_creation('data/wiki/cleaned_texts_from_1_to_10000.txt')
    # end = time.time()
    # print(f'preprocessing finished, time taken in secs: {end - start}')
    # model = make_word_emb_model(data, sg=1)
    # model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    model = Word2Vec.load("models/word2vec_wiki_1-100000_skipgram.model")
    # execute_experiment(model, 'word2vec_wiki_1-100000_skipgram', 'compare_each', similarity_measure='cosine',
    #               pos_tags=['', 'ADJ', 'VERB', 'NOUN'], weights=True)
    # execute_experiment(model, 'word2vec_wiki_1-100000_skipgram', 'compare_each', similarity_measure='manhattan',
    #               pos_tags=['', 'ADJ', 'VERB', 'NOUN'], weights=True)
    # vec1 = model.wv['love_NOUN']
    # vec2 = model.wv['warmth_NOUN']
    # print(weat.similarity(vec1, vec2, 'manhattan'))
    print('model loaded')
    sents = preprocess_text_for_word_embedding_creation('data/wiki/cleaned_texts_from_100001_to_200000.txt')
    print('sents preprocessed')
    # model.build_vocab(sents, update=True)
    model.train(sents, total_examples=model.corpus_count + len(sents), epochs=10)
    print('model trained')
    evaluate_embeddings(model)
    model.save("models/word2vec_wiki_1-200000_skipgram.model")
    print('model saved')
    # execute_experiment(model, 'word2vec_wiki_1-10000_skipgram' ,'compare_each', similarity_measure='cosine',
    #               weights=True)

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

    # TODO: visualize results!
    #   maybe similarity inside of sets (similarity matrix?)
    #   similarity numbers of sets and baseline with matplotlib.pyplot Streu- oder Liniendiagramm
