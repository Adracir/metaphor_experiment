import weat
import pandas as pd

FILE_NAME = "word_sets.csv"

#TODO: github!
#TODO: train own embeddings
#TODO: include PoS-Tagging
#TODO: make requirements.txt
#TODO: implement baseline (random word set?)

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

#TODO: make word sets real word sets or rename to list?
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

    importance_set = load_word_set_from_csv("importance")
    size_set = load_word_set_from_csv("size")
    # get dictionary of word --> embedding from glove file
    emb_dict = load_pretrained_embeddings('glove.6B.50d.txt')
    A = [emb_dict[x] for x in importance_set]
    B = [emb_dict[x] for x in size_set]
    print(weat.set_s(A, A))
    # TODO: include POS
    """X_list = ['raise', 'rise', 'lift', 'climb', 'mount', 'reach', 'surge', 'elevate', 'height', 'top', 'mountain',
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
"""
    # vectorize
    X = [emb_dict[x] for x in X_list]
    Y = [emb_dict[x] for x in Y_list]
    A = [emb_dict[x] for x in A_list]
    B = [emb_dict[x] for x in B_list]
    print('The test statistic for this category is: {}'.format(weat.test_statistic(X, Y, A, B)))
    print('The test effect size is: {}'.format(weat.effect_size(X, Y, A, B)))
"""

    # TODO: visualize results!
