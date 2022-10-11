import nltk

import weat
import pandas as pd
import random
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

FILE_NAME = "word_sets.csv"

#TODO: train own embeddings with "real" database
#   wikipedia
#   gutenberg?
#TODO: make requirements.txt

def load_pretrained_embeddings(glove_file):
    """ function to read in pretrained embedding vectors from a 6b glove file
        :param glove_file:      the path to the glove file (vocabulary of 400k words)
    """
    embedding_dict = dict()
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split(' ')
            word = row[0] # get word
            vector = [float(i) for i in row[1:]] # get vector
            embedding_dict[word] = vector
    return embedding_dict

def read_text_data():
    #  Reads ‘alice.txt’ file
    sample = open(file='data/alice_in_wonderland.txt', encoding="utf8")
    return sample.read()

def preprocess_text(s):
    # Replaces escape character with space
    f = s.replace("\n", " ")

    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
        # TODO: improve code here! (but it seems, second for loop cannot be avoided)
        tagged = nltk.pos_tag(temp)
        # TODO: wie mit Pluralen etc umgehen? POS-Tags zusammenfassen?
        #   Liste hier: https://www.guru99.com/pos-tagging-chunking-nltk.html
        temp2 = []
        for t in tagged:
            temp2.append(t[0] + "_" + t[1])
        data.append(temp2)
    return data

def make_word_emb_model(data):
    # TODO: try if skipgram or CBOW (here, default) work better!
    return gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 100, window = 5)

#TODO: make word sets real word sets or rename to list? connected with PoS problem
#TODO: maybe load two word sets by metaphor_id
#TODO: maybe make metaphor_id array, so that one word set can belong to multiple metaphors (e.g. life)
def load_word_set_from_csv(set):
    """ function to read prepared word set from a csv of the following structure
        word, pos (part of speech), set (belonging to word set), metaphor_id (distinguishing the different metaphors
            that consist of a target and source domain word set each)
            :param set:     the set to which the words belong
    """
    df = pd.read_csv(FILE_NAME)
    df = df[df['set'] == set]
    return df['word'].tolist()


if __name__ == '__main__':
    #s = read_text_data()
    #data = preprocess_text(s)
    #model = make_word_emb_model(data)
    model = Word2Vec.load("word2vec.model")
    #TODO: training needed for better results?
    print("Cosine similarity between 'machines' " +
          "and 'cat' - CBOW : ",
          model.wv.similarity("machines_NNS", "cat_NN"))
    '''word_key = ''
    for index, word in enumerate(model.wv.index_to_key):
        if index == 10:
            word_key = word
            break
        print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")
    print(f"word_key: {word_key}")
    vector = model.wv[word_key]
    print(f"vector by word_key {word_key}: {vector}")'''
    # TODO: KeyError: "Key 'said' not present" bei model.wv[word_key]
    #   erstmal nur model.wv.get_vector(word_key) nutzen
    #   ansonsten nochmal nach anderen Lösungen suchen
    #   wie z.B. similarity nutzen?
    model.save("word2vec.model")


    '''print("Cosine similarity between 'bottle' " +
          "and 'cake' - CBOW : ",
          model.wv.similarity('bottle', 'cake'))'''

    '''# importance_set = load_word_set_from_csv("importance")
    #size_set = load_word_set_from_csv("size")
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
    # TODO: improve baseline, use sets that are similarly homogenous to the source sets
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
    #   similarity inside of sets (similarity matrix?)
    #   similarity numbers of sets and baseline with matplotlib.pyplot Streu- oder Liniendiagramm
