
import weat
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents, pos_tag
import gensim
import gensim.utils as gu
from gensim.models import Word2Vec, KeyedVectors
from scipy.stats import pearsonr, spearmanr


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
            tokenized.append(gu.simple_preprocess(sent, min_len=1, max_len=30))
        print('starting to pos-tag')
        tagged = pos_tag_sents(tokenized, tagset='universal', lang='eng')
        # TODO: is this method sensible? It for sure solves the KeyError-problem!
        print('formatting pos-tags')
        [text_data.append([t[0] + "_" + t[1] for t in sent]) for sent in tagged]
        return text_data


def make_word_emb_model(data, sg=0, vec_dim=100):
    # TODO: play around with different settings!
    return gensim.models.Word2Vec(data, min_count=1, sg=sg, vector_size=vec_dim, window=5)


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


# data = preprocess_text_for_word_embedding_creation('data/gutenberg/cleaned_texts_from_1_to_4000.txt')
# print('sents preprocessed')
# model = make_word_emb_model(data, sg=1)
# model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
# model1 = Word2Vec.load("models/word2vec_wiki_1-200000_skipgram_more_vocab2.model")
# model = Word2Vec.load("models/word2vec_wiki_1-300000_skipgram.model")

# sents = preprocess_text_for_word_embedding_creation('data/wiki/cleaned_texts_from_300001_to_400000.txt')
# print('sents preprocessed')
# model.build_vocab(sents, update=True)
# model.train(sents, total_examples=model.corpus_count, epochs=10)
'''print('model trained')
evaluate_embeddings(model)
model.save("models/word2vec_gutenberg_1-4000_skipgram.model")
print('model saved')'''

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