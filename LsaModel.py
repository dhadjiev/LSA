# -*- coding: utf-8 -*-
"""Used to preform vectorization and LSA on a list of documents.
Documents beging a list of str. Where a document can be a single sentence, sentences, paragraphs or 
any amount of text.
"""
from __future__ import print_function

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class LsaModel(object):
    """Wrapper around sklearn's CountVectorizer, TfidfVectorizer and TruncatedSVD.
     Used to vectorize a list of documents with CountVectorizer/TfidfVectorizer
    and perform LSA (latent semantic analysis/indexing) with TruncatedSVD.

    Parameters
    ----------
    documents: list of str, list of documents to vectorize.
        Documents can be a single sentence, sentences, paragraphs, etc. 

    Attributes
    ----------
    vectorizer: default=None. The vectorizer used to perform the vectorization on the ``documents``.
        Use `build_vectorizer` to initialize. See `build_vectorizer` for more info.
        
    doc_term_matrix: default=None. The resulting document-term matrix from running `build_vectorizer`.
        Use `build_vectorizer` to initialize. See `build_vectorizer` for more info.
        
    svd: default=None. The TruncatedSVD instance used to perform LSA.
        Use `perform_svd` to initialize. See `perform_svd` for more info.
    
    doc_coords: default=None. The normalized document coordinates from performing LSA.
        Use `perform_svd` to initialize. See `perform_svd` for more info.
    """
    
    def __init__(self, documents):
        self.documents = documents
        # use `build_vectorizer` method to init
        self.vectorizer = None
        self.doc_term_matrix = None
        
        # use `perform_svd method` to init 
        self.svd = None
        self.doc_coords = None

    def build_vectorizer(self,
                    vectorizer_type='tfidf',
                    ngram_range=(1, 1),
                    stop_words=None,
                    max_df=1.0,
                    min_df=1,
                    max_features=None,
                    use_idf=True):
        """Wrapper around sklearn's CountVectorizer and TfidfVectorizer.
        creates a vectorizer and transforms the input documents with it.
        The paramaters' descriptions is copied from sklearn's vectorizers.
        Parameters
        ----------
        vectorizer_type: str, what vectorizer to use, choices are: "binary", "count" and "tfidf"

        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.

        stop_words : string {'english'}, list, or None (default)
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.

            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.

            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.

        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.

        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.

        max_features : int or None, default=None
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.

            This parameter is ignored if vocabulary is not None.

        use_idf : boolean, default=True, used when ``vectorizer_type`` == 'tfidf'
            Enable inverse-document-frequency reweighting.

        Returns
        -------
        tuple, (vectorizer, doc_term_matrix).
            The selected vectorizer and document-term matrix of the vectorized ``documents``
        """

        vectorizer_type = vectorizer_type.strip().lower()

        if vectorizer_type == 'binary':
            self.vectorizer = CountVectorizer(binary=True,
                                        stop_words=stop_words,
                                        ngram_range=ngram_range,
                                        max_df=max_df,
                                        min_df=min_df,
                                        max_features=max_features,)
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(stop_words=stop_words,
                                        ngram_range=ngram_range,
                                        max_df=max_df,
                                        min_df=min_df,
                                        max_features=max_features)
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=stop_words,
                                        ngram_range=ngram_range,
                                        max_df=max_df,
                                        min_df=min_df,
                                        max_features=max_features,
                                        use_idf=use_idf,
                                        norm='l2')
        else:
            raise Exception(
                "Wrong vectorizer type! Possible values: 'binary', 'count' or 'tfidf'")

        # build document term matrix and convert it's values to float
        self.doc_term_matrix = self.vectorizer.fit_transform(self.documents).astype(float)

        return self.vectorizer, self.doc_term_matrix
    
    def perform_svd(self, n_components=2, algorithm="randomized"):
        """Perform SVD to reduce dimensions.
            The paramaters' descriptions is copied from sklearn's TruncatedSVD.
        
        Parameters
        ----------
        n_components : int, default = 2
            Desired dimensionality of output data.
            Must be strictly less than the number of features.
            The default value is useful for visualisation. For LSA, a value of
            100 is recommended.
        algorithm : string, default = "randomized"
            SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
            (scipy.sparse.linalg.svds), or "randomized" for the randomized
            algorithm due to Halko (2009).

        Returns
        -------
        normalized document coordinates : array, shape (n_documents, n_components)
        """

        # build svd and normalization pipeline 
        self.svd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
        lsa = make_pipeline(self.svd, Normalizer(copy=False))

        # Perform Dimensionality reduction with LSA
        # Fit LSA. Use algorithm = “randomized” for large datasets
        self.doc_coords = lsa.fit_transform(self.doc_term_matrix)

        return self.doc_coords
    
    def docs_to_csv(self, path='lsa_documents.csv'):
        """Save the document coordinates resulting from `perform_svd`
        """
        if self.svd is None:
            print('"svd" not initialized!')
            print('Use the "perform_svd" method on the model to initialize "svd".')
            return
        # get rank of the truncatedSVD
        num_components = self.doc_coords.shape[1]
        # list of components for pd.DataFrame
        components = ['Component ' + str(c) for c in range(num_components)]
        # build dataframe
        doc_df = pd.DataFrame(data=self.doc_coords, index=self.documents, columns=components)
        # save lsa documents as CSV
        doc_df.to_csv(path, encoding='utf8')
    
    def terms_to_csv(self, path='lsa_components.csv'):
        """Save the term vectors resulting from `perform_svd`
        """
        if self.svd is None:
            print('"svd" not initialized!')
            print('Use the "perform_svd" method on the model to initialize "svd".')
            return
        # get rank of the truncatedSVD
        num_components = self.svd.components_.shape[0]

        term_vectors = self.svd.components_
        # list of components for pd.DataFrame
        components = ['Component ' + str(c) for c in range(num_components)]
        # list of all term names
        feat_names = self.vectorizer.get_feature_names()
        # build dataframe
        terms_df = pd.DataFrame(data=term_vectors, index=components, columns=feat_names)
        # save lsa components as CSV
        terms_df.to_csv(TERMS_CSV, encoding='utf8')

    def feature_names(self):
        """Returns a list of term names built by `build_vectorizer`
        """
        
        return self.vectorizer.get_feature_names()
