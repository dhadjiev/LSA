import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('english')

def tokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters or years between 1000 and 3000 (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]|[1-3][0-9]{3}', token):
            filtered_tokens.append(token)
    return filtered_tokens

class corpusDecompositor(object):
    
    self.token_vocabulary_ = []
    self.stopwords_ = nltk.corpus.stopwords.words('english')
    
    def _tokenize
    
    def __init__(self, text, vectorizer)
        token_vocabulary = []
    
    def fit(self, x, y=None)
for i in corpus:
    allwords_tokenized = tokenize(i)
    token_vocabulary.extend(allwords_tokenized)



#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=2, stop_words=stopwords,
                                 use_idf=True, tokenizer=tokenize, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus) #fit the vectorizer to corpus

#print(tfidf_matrix.shape)

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=70, algorithm='randomized')

from sklearn.preprocessing import Normalizer

lsa = make_pipeline(svd, Normalizer(copy=False))
dtm = lsa.fit_transform(tfidf_matrix)

#terms = tfidf_vectorizer.get_feature_names()
#from scipy.sparse import spmatrix
#xdtm = spmatrix.todense(dtm)
from sklearn.metrics.pairwise import cosine_distances
dist = cosine_distances(dtm)

from scipy.cluster.hierarchy import ward
linkage_matrix = ward(dist)


#linkage_matrix = scipy.sparse.csr_matrix.todense(linkage_matrix)
#from sklearn.cluster import ward_tree
#linkage_matrix = ward_tree(dtm, pdist, return_distance = 'true')
#linkage_matrix = linkage(pdist, 'ward', 'cosine')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 20)) # set size
from scipy.cluster import hierarchy
ax = hierarchy.dendrogram(linkage_matrix, orientation="right", labels=titles);
plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    leaf_font_size = 24)
plt.tight_layout() #show plot with tight layout

assignments = hierarchy.fcluster(linkage_matrix,0.5,'distance')

cluster_output = pd.DataFrame({'title':titles, 'cluster':assignments})

#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/