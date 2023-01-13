
import calc
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag_sents
import gensim
import gensim.utils as gu
from gensim.models import Word2Vec, KeyedVectors
from scipy.stats import pearsonr, spearmanr


def preprocess_text_for_word_embedding_creation(filename):
    """
    preprocess raw text from a file into a format that can be transformed in a Word2Vec model:
    1. sentence tokenization
    2. gensim simple preprocessing (word tokenization and some cleaning)
    3. pos-tagging and appending tags to the corresponding words with a "_"
    :param filename: relative path to the file containing the text
    :return: list containing a list of pos-tagged words for each sentence
    """
    with open(filename, encoding='utf8') as file:
        s = file.read()
        text_data = []
        tokenized = []
        print('starting to sent_tokenize')
        # TODO: test with nltk 3.8
        # takes a long time in nltk 3.7, that's why I downgraded to nltk 3.6.5 until upgrade is coming
        # https://github.com/nltk/nltk/issues/3013
        sents = sent_tokenize(s)
        print('starting to simple_preprocess')
        # simple-preprocess sentences using gu (word tokenization, etc.)
        for sent in sents:
            tokenized.append(gu.simple_preprocess(sent, min_len=1, max_len=30))
        print('starting to pos-tag')
        # pos-tag words using nltk
        tagged = pos_tag_sents(tokenized, tagset='universal', lang='eng')
        print('formatting pos-tags')
        # connect pos-tag to word
        [text_data.append([t[0] + "_" + t[1] for t in sent]) for sent in tagged]
        return text_data


def make_word_emb_model(data, sg=1, vec_dim=100):
    """
    initialize and train a Word2Vec model with gensim from the given data
    :param data: list of lists, containing tokenized words in tokenized sentences
    (as can be generated from raw text with preprocess_text_for_word_embedding_creation(filename))
    :param sg: if 0, method CBOW is used, if 1, Skipgram
    :param vec_dim: defines the dimensions of the resulting vectors
    """
    return gensim.models.Word2Vec(data, min_count=1, sg=sg, vector_size=vec_dim, window=5)


def evaluate_embeddings(keyed_vectors, distance_measure='cosine'):
    """
    print the evaluation of a given model in correlation (Pearson and Spearman) to a human-based list of
    words (based on Rubenstein, H., & Goodenough, J. (1965). Contextual correlates of synonymy. Commun. ACM, 8, 627–633.
    https://doi.org/10.1145/365628.365657)
    For a well-functioning model, the first value is expected to be as high as possible, the pvalue is expected to be
    less than 0.05
    :param keyed_vectors: keyed vectors from a Word2Vec model
    :param distance_measure: one of 'cosine', 'manhattan', 'canberra', 'euclidian'. Determines used similarity/distance measure
    """
    df = pd.read_csv('data/human_relatedness.csv')
    gold_standard_relatedness = [float(x) for x in df['synonymy'].tolist()]
    words1 = df['word1'].tolist()
    words2 = df['word2'].tolist()
    embedding_relatedness = []
    for i in range(0, len(words1)):
        vec_word1 = keyed_vectors[words1[i] + '_NOUN']
        vec_word2 = keyed_vectors[words2[i] + '_NOUN']
        embedding_relatedness.append(calc.distance(vec_word1, vec_word2, distance_measure))
    embedding_relatedness = calc.normalize_and_reverse_distances(embedding_relatedness, distance_measure)
    print(pearsonr(gold_standard_relatedness, embedding_relatedness))
    print(spearmanr(gold_standard_relatedness, embedding_relatedness))


# data = preprocess_text_for_word_embedding_creation('data/wiki/cleaned_texts_from_1_to_3000.txt')
# print(data)
# print('sents preprocessed')
# model = make_word_emb_model(data, sg=1)
keyed_vectors = KeyedVectors.load('models/word2vec_gutenberg_1-8000u16001-26000_skipgram.wordvectors', mmap='r')
'''model = Word2Vec.load("models/word2vec_wiki_1-3000_skipgram_better-preprocessing.model")
sents = preprocess_text_for_word_embedding_creation('data/wiki/cleaned_texts_from_1_to_10000.txt')
print('sents preprocessed')
model.build_vocab(sents, update=True)
model.train(sents, total_examples=model.corpus_count, epochs=10)
word_vectors = model.wv
word_vectors.save('models/word2vec_wiki_1-10000_skipgram.wordvectors')
'''
evaluate_embeddings(keyed_vectors)
