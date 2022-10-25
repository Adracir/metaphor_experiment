# based on
# https://github.com/e-mckinnie/WEAT
# https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
from scipy import spatial
import numpy as np


# TODO: maybe rename class

# TODO: maybe include more measures
# TODO: for euclidian, manhattan and canberra, include similarity, not distance!
def similarity(vec1, vec2, similarity_measure):
    """calculate similarity of two word vectors
    :param vec1: vector of first word
    :param vec2: vector of second word
    :param similarity_measure: kind of similarity measure to be used: cosine, manhattan, canberra or euclidian"""
    if similarity_measure == 'cosine':
        return 1-spatial.distance.cosine(vec1, vec2)
    elif similarity_measure == 'manhattan':
        return spatial.distance.cityblock(vec1, vec2)
    elif similarity_measure == 'canberra':
        return spatial.distance.canberra(vec1, vec2)
    elif similarity_measure == 'euclidian':
        return np.linalg.norm(vec1 - vec2)


def cosine_similarity(vec1, vec2):
    """calculate cosine similarity of two word vectors
    :param vec1: vector of first word
    :param vec2: vector of second word"""
    return 1-spatial.distance.cosine(vec1, vec2)


def set_s(A, B, similarity_measure):
    """calculate similarity between two sets of words
    :param A: source set
    :param B: target set
    :param similarity_measure: kind of similarity measure to be used: cosine, manhattan, canberra or euclidian"""
    all_similarities = []
    for a in A:
        for b in B:
            all_similarities.append(similarity(a, b, similarity_measure))
    return np.mean(all_similarities)


'''def s(w, A, B):
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
    return (np.mean(s_x) - np.mean(s_y))/ np.std(np.concatenate((s_x, s_y)))'''
