import weat
import pandas as pd
import random

FILE_NAME = "word_sets.csv"

#TODO: train own embeddings
#TODO: include PoS-Tagging
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
   # importance_set = load_word_set_from_csv("importance")
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
    # TODO: include PoS
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
    # TODO: rewrite for only 2 sets of associations instead of 4

    # TODO: visualize results!
    #   similarity inside of sets (similarity matrix?)
    #   similarity numbers of sets and baseline with matplotlib.pyplot Streu- oder Liniendiagramm
