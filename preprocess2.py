import nltk
import re
import pandas as pd

stop_list = nltk.corpus.stopwords.words('english')

#extend the stop list
def extend_stop_list_from_df(latest_review, secondary_category):
    if secondary_category not in latest_review.columns:
        print(f"Column '{secondary_category}' not found in the DataFrame.")
        return

    # Extract the specified column and convert it to a list of words
    column_values = latest_review[secondary_category].str.split().sum()

    # Extend the stop_list with the words from column_values
    stop_list.extend(column_values)

import gensim
stemmer = nltk.stem.porter.PorterStemmer()

def load_corpus(dir):
    # dir is a directory with plain text files to load.
    corpus = nltk.corpus.PlaintextCorpusReader(dir, '.+\.txt')
    return corpus

def corpus2docs(corpus, stem = False):
    # corpus is a object returned by load_corpus that represents a corpus.
    fids = corpus.fileids()
    docs1 = []
    for fid in fids:
        doc_raw = corpus.raw(fid)
        doc = nltk.word_tokenize(doc_raw)
        docs1.append(doc)
    docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs3 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs2]
    docs4 = [[w for w in doc if w not in stop_list] for doc in docs3]
    docs5 = [[word if word != "NaN" else "" for word in doc] for doc in docs4]

    return docs5

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

    