import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn import model_selection
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# function to plot the wordcloud
def plot_wordcloud(text,title):
    # initiate wordcloud
    # wordcloud is not case sensitive, stopwords default on, include_number off, no_normalize_plurals on to take out 's'
    wcd = WordCloud(background_color='white', random_state = 2023)
    wordcloud_1 = wcd.generate(text)
    
    # plot wordplot
    plt.figure(figsize=(16,13))
    plt.imshow(wordcloud_1, interpolation='bilinear')
    plt.axis("off")    
    plt.title(title, fontsize=20)    
    return plt



# function to remove plural
def remove_plural_preprocessor(text):
    # Apply the default stop words removal in CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    words = vectorizer.build_analyzer()(text)
    
    # Remove only 1 "s" ending from each remaining word
    words = [re.sub(r's(?<!ss)$', '', word) for word in words]
    
    # Join the remaining words and return the preprocessed text
    return ' '.join(words)



# naive bayes classifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_naive_bayes_classifier(data, text_col):
    # Apply the text preprocessing step to the 'reviews' column
    data['reviews'] = data[text_col].apply(remove_plural_preprocessor)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['reviews'], data['true_sentiment'], test_size=0.4, random_state=2023)

    # Create a Naive Bayes Classifier pipeline
    naive_bayes_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),  # Convert reviews into numerical features
        ('classifier', MultinomialNB())      # Naive Bayes Classifier
    ])

    # Train the pipeline on the training data
    naive_bayes_pipeline.fit(X_train, y_train)
    
    # Get predictions for both train and test sets
    train_predictions = naive_bayes_pipeline.predict(X_train)
    test_predictions = naive_bayes_pipeline.predict(X_test)
    
    # Calculate accuracy for train and test sets
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Generate confusion matrix for test set
    confusion_mat = confusion_matrix(y_test, test_predictions, labels=['positive', 'neutral', 'negative'])
    
    return naive_bayes_pipeline, train_accuracy, test_accuracy, confusion_mat

# Function to get accuracy for train and test sets
def get_accuracy(classifier, data, text_col):
    preprocessed_text = data[text_col].apply(remove_plural_preprocessor)
    predictions = classifier.predict(preprocessed_text)
    true_sentiment = data['true_sentiment']
    accuracy = accuracy_score(true_sentiment, predictions)
    return accuracy

# Function to get confusion matrix
def get_confusion_matrix(classifier, data, text_col):
    preprocessed_text = data[text_col].apply(remove_plural_preprocessor)
    predictions = classifier.predict(preprocessed_text)
    true_sentiment = data['true_sentiment']
    confusion_mat = confusion_matrix(true_sentiment, predictions, labels=['positive', 'neutral', 'negative'])
    return confusion_mat

