# based on
# https://github.com/e-mckinnie/WEAT
# https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
from scipy import spatial
import numpy as np


# TODO: maybe rename class
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


# TODO: maybe include normalization also for cosine, test if it works generally
def set_s(A, B, similarity_measure):
    """calculate similarity between two sets of words
    :param A: source set
    :param B: target set
    :param similarity_measure: kind of similarity measure to be used: cosine, manhattan, canberra or euclidian"""
    all_similarities = []
    for a in A:
        for b in B:
            all_similarities.append(similarity(a, b, similarity_measure))
    if similarity_measure != 'cosine':
        # normalize distance and subtract from 1 to generate similarity
        for i in range(len(all_similarities)):
            d = all_similarities[i]
            d_normalized = d / max(all_similarities)
            all_similarities[i] = 1 - d_normalized
    return np.mean(all_similarities)
