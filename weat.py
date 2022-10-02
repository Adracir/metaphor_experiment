# based on
# https://github.com/e-mckinnie/WEAT
# https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
from scipy import spatial
import numpy as np

#TODO: adapt WEAT to fit my problems
def cosine_similarity(vec1, vec2):
    """calculate cosine similarity of two word vectors
    :param vec1: vector of first word
    :param vec2: vector of second word"""
    return 1-spatial.distance.cosine(vec1, vec2)

def set_s(A, B):
    """calculate cosine similarity between two sets of words
    :param A: source set
    :param B: target set"""
    all_similarities = []
    for a in A:
        for b in B:
            all_similarities.append(cosine_similarity(a, b))
    return np.mean(all_similarities)

def s(w, A, B):
    """calculate bias score of attribute word w and two target sets A and B
    :param w: attribute word
    :param A: target set
    :param B: other target set"""
    cos_wa = np.array([cosine_similarity(w, a) for a in A])
    cos_wb = np.array([cosine_similarity(w, b) for b in B])
    return np.mean(cos_wa) - np.mean(cos_wb)


def test_statistic(X, Y, A, B):
    """calculate bias score of attribute sets and target sets
    :param X: attribute set
    :param Y: other attribute set
    :param A: target set
    :param B: other target set"""
    s_x = np.array([s(x, A, B) for x in X])
    s_y = np.array([s(y, A, B) for y in Y])
    return np.mean(s_x) - np.mean(s_y)


def effect_size(X, Y, A, B):
    """calculate effect size of bias for attribute sets and target sets
    :param X: attribute set
    :param Y: other attribute set
    :param A: target set
    :param B: other target set"""
    s_x = np.array([s(x, A, B) for x in X])
    s_y = np.array([s(y, A, B) for y in Y])
    return (np.mean(s_x) - np.mean(s_y))/ np.std(np.concatenate((s_x, s_y)))
