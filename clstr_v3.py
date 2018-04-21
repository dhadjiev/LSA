#import numpy as np
from __future__ import print_function
import nltk
import re
#import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

#define stopwords
try:
    stopwords = nltk.corpus.stopwords.words('english')
except:
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

stopwords.extend(['\'s', '\'d', '\'nt', '\'ll', '\'re', 'a.m.', 'p.m.', 'u.s.'])

#define custom tokenizer
def tokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    try:
        nltk.sent_tokenize(text)
    except:
        nltk.download('punkt')
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters or years between 1000 and 3000 (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]|[1-3][0-9]{3}', token):
            filtered_tokens.append(token)
    return filtered_tokens


def buildModel(text):
    text_model = []
    for i in text:
        allwords_tokenized = tokenize(i)
        text_model.extend(allwords_tokenized)
    return text_model

#build vector model of the text

class corpusVectorizer(Object):
    
    def __init__(self, minf=2, maxngram = 3):    
        #self.text = text
        self.minf = minf
        self.maxngram = maxngram        
        self._vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                                min_df=minf, stop_words=stopwords,
                                                use_idf=True, tokenizer=tokenize, 
                                                ngram_range=(1,maxngram))
    def _summarize(self, X, y = None):
        
    def fit(self, raw_docs, y = None):
        self.fit_transform(raw_documents)
        return self
    
    def fit_transform(self, raw_docs, y = None):
        
        
        
    
def buildVectorModel(text, minf=2, maxngram=3):
    """
    Build a Term frequency * Inverse document frequency vector model
    of the source corpus.
    
    Parameters
    ----------
    text: list, a corpus of articles
    
    minf: int, minimal number of documents, that a term should be present in,
    to be considered in the vector model
    defalt value 2
    
    maxngram: int, maximum range of the n_gram 
    ex: 1 - only single words, 2 - single words and bigrams,
    3 - single words, bigrams and trigrams(where considerable)
    default value 3 (up to trigrams)
    
    Returns
    -------
    DTM : Sparse matrix 
    
    """
    
    #define vectorizer parameters
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(text) #fit the vectorizer to model
    #print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()
    return tfidf_matrix, terms #save to file

def saveMatrix(matrix, index, columns, path='matrix.csv'):
    try:
        dmtx = matrix.todense()
    except:
        dmtx = matrix
    mtx_frm = pd.DataFrame(data=dmtx, index = index, columns = columns)
    mtx_frm.to_csv(path, encoding='utf8')

def saveList(list, path='list.csv'):
    lst_frm = pd.DataFrame(data = list)
    lst_frm.to_csv(path, index=False, index_label=False , encoding='utf8')

def lsaModel(matrix, components=70):
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=components, algorithm='randomized')
    from sklearn.preprocessing import Normalizer
    lsa = make_pipeline(svd, Normalizer(copy=False))
    dtm = lsa.fit_transform(matrix)
    svd.components_ #to file
    #svd.singular_values_
    return dtm

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import sparsefuncs, check_array, as_float_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot

class lsaTransform(BaseEstimator, TransformerMixin):
    """ Transformer class that derives from the scikit-learn TruncatedSVD.
        This transformer performs linear dimensionality reduction by means of
        truncated singular value decomposition (SVD). Contrary to PCA, this
        estimator does not center the data before computing the singular value
        decomposition. This means it can work with scipy.sparse matrices
        efficiently.
        For our implementation the ARPACK algorithm has been omitted. """ 
    
    def __init__(self, n_iter=5, random_state=None):
        """ Ininitialisation of the base algorithm variables.
        
        n_iter : int, optional (default 5)
        Number of iterations for randomized SVD solver.
        The default is larger than the default in `randomized_svd` to handle 
        sparse matrices that may have large slowly decaying spectrum. 
        
        random_state : int or RandomState, optional
        (Seed for) pseudo-random number generator. If not given, the
        numpy.random singleton is used. """
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit LSI model on matrix X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        Returns
        -------
        self : object
        
        Returns the transformer object.
        """
        self.fit_transform(X)
        return self
    
    def fit_transforn(self, X, n_components=70, y=None):
        """Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        n_components : int, default = 70
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for sentence clustering taks.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        
        
        X = as_float_array(X, copy=False)
        #random_state = check_random_state(self.random_state)
        
        # If sparse and not csr or csc, convert to csr
        import scipy.sparse as sp
        
        if sp.issparse(X) and X.getformat() not in ["csr", "csc"]:
            X = X.tocsr()
            
        
        #fitting the decomposition
        k = n_components
        n_features = X.shape[1]
        if k >= n_features:
            raise ValueError("n_components must be < n_features;"
                             " got %d >= %d" % (k, n_features))
        
        #definition of the randomized SVD parameters
        n_oversamples=10
        
        """
        n_oversamples: int (default is 10)
        
        Additional number of random vectors to sample the range of X so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of X is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
        
        """
        
        n_iter='auto'
        
        """
        n_iter: int or 'auto' (default is 'auto')
        
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
        
        """
        power_iteration_normalizer='auto'
        
        """
        power_iteration_normalizer: 'auto' (default), 'QR', 'LU', 'none'
        
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.
        
        """
    
        flip_sign=True, random_state=0
        
        """
        flip_sign: boolean, (True by default)
        
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
        
        """
        
        transpose='auto'
        
        """
        transpose: True, False or 'auto' (default)
        
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
        
        """
        
        flip_sign=True
        
        """
        flip_sign: boolean, (True by default)
        
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
        
        """
        
        random_state=0
        
        """
        random_state: RandomState or an int seed (0 by default)
        
        A random number generator instance to make behavior
        
        """
        
        U, Sigma, VT = randomized_svd(X, n_components,
                                      n_iter = self.n_iter,
                                      random_state = self.random_state)
        
        self.doc_dist_ = U
        self.components_ = VT
        self.singular_values_ = Sigma
        
        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        
        #self.term_space_ = np.dot(Sigma, VT)
        
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        
        if sp.issparse(X):
            _, full_var = sparsefuncs.mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        
        return X_transformed        

        def transform(self, X):
            """Perform dimensionality reduction on X.
            
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                New data.
                
            Returns
            -------
            X_new : array, shape (n_samples, n_components)
                Reduced version of X. This will always be a dense array.
            """
            X = check_array(X, accept_sparse='csr')
            return safe_sparse_dot(X, self.components_.T)
        
        def save_doc_dist_(self, path='doc_dist_.csv'):
            saveMatrix(self.doc_dist_, titles, titles, path)
            
        def save_term_dist_(self, path='term_dist_.csv'):
            saveMatrix(self.term_space_, terms, terms, path)
            
def distanceModel(mtx):
    from sklearn.metrics.pairwise import cosine_distances
    dist = cosine_distances(mtx)
    return dist


def kMeansCluster(lsa, titles, clusters=5):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=clusters)
    km.fit(lsa)
    clusts=km.labels_.tolist()
    articles = {'title': titles, 'cluster':clusts}
    frame = pd.DataFrame(articles, index=[clusts], columns=['title'])
    #frame['cluster'].value_counts()
    for i in range(clusters): 
        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.manifold import MDS
    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(distanceModel(lsa))  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusts, title=titles))
    groups = df.groupby('label')
    # set up plot
    fig, ax = plt.subplots(figsize=(22, 14)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #iterate through groups to layer the plot
    #note that I use the cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                color=cluster_colors[name], 
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
                       axis= 'x',          # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       bottom='off',      # ticks along the bottom edge are off
                       top='off',         # ticks along the top edge are off
                       labelbottom='off')
        ax.tick_params(\
                       axis= 'y',         # changes apply to the y-axis
                       which='both',      # both major and minor ticks are affected
                       left='off',      # ticks along the bottom edge are off
                       top='off',         # ticks along the top edge are off
                       labelleft='off')

    #add label in x,y position with the label as the article title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)   
    plt.show() #show the plot

    #uncomment the below to save the plot if need be
    #plt.savefig('clusters_small_noaxes.png', dpi=200)
 
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    from scipy.cluster import hierarchy
    ddata = hierarchy.dendrogram(*args, **kwargs)

    import matplotlib.pyplot as plt
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def hierarchyCluster(lsa, titles, cutoff=1.5):
    from scipy.cluster.hierarchy import ward
    #scipy.cluster.hierarchy
    import matplotlib.pyplot as plt
    linkage_matrix = ward(distanceModel(lsa))
    from scipy.cluster import hierarchy
    assignments = hierarchy.fcluster(linkage_matrix, cutoff, criterion='distance')
    print(assignments)
    cluster_output = pd.DataFrame({'title':titles, 'cluster':assignments})
    print(cluster_output)
    fig, ax = plt.subplots(figsize=(22, 14)) # set size
    ax = fancy_dendrogram(linkage_matrix, max_d=cutoff, labels=titles);
    plt.tick_params(\
                    axis= 'x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off',
                    leaf_font_size = 24.)
    plt.tight_layout() #show plot with tight layout
    plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters


#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/