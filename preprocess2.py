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

def corpus2docs(corpus, stem=False):
    # corpus is an object returned by load_corpus that represents a corpus.
    fids = corpus.fileids()
    docs1 = []
    for fid in fids:
        doc_raw = corpus.raw(fid)
        doc = nltk.word_tokenize(doc_raw)
        docs1.append(doc)
    docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs3 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs2]

    # Initialize Laplace smoothing parameters
    smoothing_parameter = 1  # Adjust as needed

    # Extend the stop_list with the words from column_values
    docs4 = []
    for doc in docs3:
        smoothed_doc = []
        for word in doc:
            if word not in stop_list:
                smoothed_doc.append(word)  # Exclude stopwords
                # Apply Laplace smoothing here:
                word_count = doc.count(word) + smoothing_parameter
                total_word_count = len(doc) + (smoothing_parameter * len(set(doc)))
                word_probability = word_count / total_word_count
                # Replace the word with its Laplace-smoothed form
                smoothed_doc.extend([word] * int(round(word_probability * 1000)))  # You can adjust the multiplier
        docs4.append(smoothed_doc)
    docs5 = [[word if word != "nan" else "" for word in doc] for doc in docs4]
    return docs5

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

    