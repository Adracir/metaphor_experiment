from scipy import spatial
from scipy.stats import ttest_1samp
import numpy as np


def distance(vec1, vec2, distance_measure):
    """
    calculate distance of two word vectors
    :param vec1: vector of first word
    :param vec2: vector of second word
    :param distance_measure: kind of distance measure to be used: cosine, manhattan, canberra or euclidian
    :return: result of the distance calculation between the two vectors
    """
    if distance_measure == 'cosine':
        return spatial.distance.cosine(vec1, vec2)
    elif distance_measure == 'manhattan':
        return spatial.distance.cityblock(vec1, vec2)
    elif distance_measure == 'canberra':
        return spatial.distance.canberra(vec1, vec2)
    elif distance_measure == 'euclidian':
        return np.linalg.norm(vec1 - vec2)


def generate_similarities(A, B, distance_measure):
    """
    calculate similarities between two sets of words
    :param A: source set
    :param B: target set
    :param distance_measure: kind of distance measure to be used: cosine, manhattan, canberra or euclidian
    :return: array containing all calculated similarities between all of the values of source and target set
    """
    all_similarities = []

    for a in A:
        for b in B:
            all_similarities.append(distance(a, b, distance_measure))
    # normalize distance and subtract from 1 to generate distance
    for i in range(len(all_similarities)):
        d = all_similarities[i]
        if distance_measure != 'cosine':
            d_normalized = d / max(all_similarities)
        else:
            d_normalized = d
        all_similarities[i] = 1 - d_normalized
    return all_similarities


def t_test(similarities, baseline):
    """
    execute two-sided t-test for set of values and a baseline
    :param similarities: sample observation
    :param baseline: value standing for the baseline, null hypothesis
    :return: test statistic (positive or negative), p-value
    """
    return ttest_1samp(similarities, baseline)
