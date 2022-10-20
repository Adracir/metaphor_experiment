import nltk

import weat
import pandas as pd
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents
import numpy as np
import corpora

import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

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
    # TODO: do I also have to remove stopwords and special characters like here?
    #   https://medium.com/analytics-vidhya/word-similarity-word2vec-natural-language-processing-fe085f9f03e7?

    # Replaces escape character with space
    f = s.replace("\n", " ")

    text_data = []

    tokenized_sents = []
    # iterate through each sentence in the file
    for i in sent_tokenize(f):
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

# TODO: maybe load two word sets by metaphor_id


# TODO: structure of csv has changed, update
def load_word_set_from_csv(word_set):
    """ function to read prepared word set from a csv of the following structure
        word, pos (part of speech), set (belonging to word set), metaphor_id (distinguishing the different metaphors
        that consist of a target and source domain word set each)
            :param word_set:     the set to which the words belong
    """
    df = pd.read_csv(WORD_DOMAIN_SETS_FILE)
    df = df[df['set'] == word_set]
    return df['word'].tolist()


# TODO: maybe include weights predefined in word_sets.csv to stress especially useful words
def create_mean_vector_from_multiple(word_list, model):
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
    word_vecs = [model.wv[word] for word in word_list]
    # combine word_vecs1,2,3... to np.array a [[num1, num2, num3...],[num1, num2, num3], ...]
    a = np.column_stack([word_vec for word_vec in word_vecs])
    # calculate mean vector
    vec = np.mean(a, axis=1)
    return vec


if __name__ == '__main__':
    s = corpora.preprocess_wiki_dump()
    # s = read_text_data()
    data = preprocess_text_for_word_embedding_creation(s)
    model = make_word_emb_model(data)
    # model = Word2Vec.load("word2vec.model")
    # TODO: training needed for better results?
    print("Cosine similarity between 'be' " +
          "and 'is' - CBOW : ",
          model.wv.similarity("be_VERB", "is_VERB"))
    # test create_mean_vector_from_multiple
    # word_list = ("hatter_NOUN", "alice_NOUN")
    # print(create_mean_vector_from_multiple(word_list, model))
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
    model.save("wiki_word2vec.model")

    '''
    # uses pre-trained glove embeddings to compare sets from csv list by cosine_similarity
    # importance_set = load_word_set_from_csv("importance")
    # size_set = load_word_set_from_csv("size")
    active_set = load_word_set_from_csv("active")
    life_set = load_word_set_from_csv("life")
    # get dictionary of word --> embedding from glove file
    emb_dict = load_pretrained_embeddings('glove.6B.50d.txt')
    A = [emb_dict[x] for x in active_set]
    B = [emb_dict[x] for x in life_set]
    baseline_set = [random.choice(list(emb_dict.values())) for x in life_set]
    print('similarity importance and size: {}'.format(weat.set_s(A, B)))
    print('baseline similarity, importance and random: {}'.format(weat.set_s(A, baseline_set)))
    '''
    # TODO: improve baseline, use sets that are similarly homogenous to the source sets?
    """
    # up is good, bad is down
    X_list = ['raise', 'rise', 'lift', 'climb', 'mount', 'reach', 'surge', 'elevate', 'height', 'top', 'mountain',
              'elevation', 'raise', 'ascent', 'peak', 'summit', 'tip', 'high', 'tall', 'top', 'upper', 'large',
              'rising',
              'elevated', 'raised', 'upward', 'ascending', 'up', 'above']
    Y_list = ['drop', 'fall', 'sink', 'decline', 'decrease', 'descend', 'slip', 'lower', 'floor', 'decline', 'descent',
              'bottom', 'ground', 'sinking', 'slope', 'underside', 'low', 'down', 'downward', 'descending', 'sliding',
              'below', 'bottom', 'declining', 'down', 'under', 'underneath']
    # attribute words
    A_list = ['care', 'help', 'donate', 'cheer', 'improve', 'heal', 'better', 'purify', 'right', 'reason',
              'charity', 'humanity', 'welfare', 'kindness', 'virtue', 'decency', 'quality', 'improvement', 'good',
              'better', 'able', 'whole', 'fine', 'healthy', 'moral', 'whole', 'fair', 'excellent', 'great',
              'favorable', 'wonderful', 'nice']
    B_list = ['lose', 'fail', 'suffer', 'impair', 'worsen', 'aggravate', 'deteriorate', 'sicken', 'fault', 'error',
              'evil', 'damage', 'harm', 'badness', 'bad', 'deterioration', 'bad', 'sad', 'dangerous', 'terrible',
              'sick', 'difficult', 'serious', 'unfortunate', 'painful', 'cruel', 'evil']
    """

    # TODO: visualize results!
    #   maybe similarity inside of sets (similarity matrix?)
    #   similarity numbers of sets and baseline with matplotlib.pyplot Streu- oder Liniendiagramm
