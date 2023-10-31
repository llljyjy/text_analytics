# naive_bayes_utils.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import inflect
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS


# different types of preprocessor to understand the impact of different text preprocessing techniques

# most basic preprocessor, only converting all to lower cases
def lowercase_preprocessor(text):
    if isinstance(text, str):
        return text.lower()
    return text

# try out with only common english stopwords
def stopword_preprocessor(text):
    stop_words = CountVectorizer(stop_words='english').get_stop_words()
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# try out with both custom and common english stopwords
def customized_stopword_preprocessor(text):
    custom_stopwords = {'skin','product','face','cream','serum','moisturizer','makeup','eye','sunscreen','eyes','products','s'}
    all_stopwords = STOPWORDS.union(custom_stopwords)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return ' '.join(filtered_words)

# experiment on converting text back to its original format 
def lemmatizer_preprocessor(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# only to convert plural words back to its single form 
p = inflect.engine()
def remove_plural_preprocessor(text):
    words = text.split()
    singular_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
    return ' '.join(singular_words)


# naive bayes classifer pipeline to finetune preprocessors, define target variables, and input features 
# preprocessors -> vectorizer -> train -> evaluate
class NaiveBayesClassifier:

    def __init__(self, data, text_col, preprocessors=None):
        self.data = data
        self.text_col = text_col
        self.classifier = None
        self.preprocessors_map = {
            'lowercase': lowercase_preprocessor,
            'stopword': stopword_preprocessor,
            'lemmatizer': lemmatizer_preprocessor,
            'remove_plural': remove_plural_preprocessor,
            'custom_stopword': customized_stopword_preprocessor
        }
        
        if preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    def apply_preprocessors(self, text):
        for preprocessor_name in self.preprocessors:
            func = self.preprocessors_map.get(preprocessor_name)
            if func:
                text = func(text)
        return text

    def build_pipeline(self, vectorizer_type):
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer()
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

        classifier = MultinomialNB()

        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

    def train(self, vectorizer_type='count', use_additional_features=False):
        # Preprocessing
        self.data['reviews_processed'] = self.data[self.text_col].apply(self.apply_preprocessors)

        if use_additional_features:
            X = self.data[['reviews_processed', 'helpfulness', 'price_usd', 'length']]
        else:
            X = self.data[['reviews_processed']]

        y = self.data['true_sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2023)
        
        pipeline = self.build_pipeline(vectorizer_type)
        pipeline.fit(X_train['reviews_processed'], y_train) # We only vectorize the text data
        self.classifier = pipeline

        train_metrics = self.evaluate_classifier(self.classifier, X_train['reviews_processed'], y_train)
        test_metrics = self.evaluate_classifier(self.classifier, X_test['reviews_processed'], y_test)
        
        return train_metrics, test_metrics


    def evaluate_classifier(self, classifier, X, y):
        predictions = classifier.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        unique_labels = np.unique(np.concatenate((y, predictions)))
        confusion_mat = confusion_matrix(y, predictions, labels=unique_labels)

        precision = precision_score(y, predictions, average='weighted', labels=unique_labels, zero_division=0)
        recall = recall_score(y, predictions, average='weighted', labels=unique_labels)
        f1 = f1_score(y, predictions, average='weighted', labels=unique_labels)

        # Calculate F2 score using its formula: F2 = (1 + 2^2) * (precision * recall) / ((2^2 * precision) + recall)
        beta = 2
        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

        metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'confusion_mat': confusion_mat
        }
        
        return metrics

    def display_results(self, metrics):
        for metric, value in metrics.items():
            if metric != 'confusion_mat':
                if isinstance(value, np.ndarray):
                    print(f"{metric}:\n{value}")
                else:
                    print(f"{metric}: {value:.5f}")

            
    

    
class NaiveBayesVisualization:
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.data = classifier.data
        self.validator = Validator(classifier) 

    def plot_individual_confusion_matrices(self, X, y, classes=['positive', 'negative', 'neutral']):
        # Check for the unique classes in y and only keep those present in the classes list
        unique_classes = np.unique(y)
        classes = [cls for cls in classes if cls in unique_classes]

        # Get the confusion matrix using evaluate_classifier
        _, full_confusion = self.classifier.evaluate_classifier(self.classifier.classifier, X, y)

        for class_label in classes:
            # Extract binary confusion matrix for the specific class
            true_pos = full_confusion[classes.index(class_label), classes.index(class_label)]
            false_pos = sum(full_confusion[:, classes.index(class_label)]) - true_pos
            false_neg = sum(full_confusion[classes.index(class_label), :]) - true_pos
            true_neg = np.sum(full_confusion) - true_pos - false_pos - false_neg

            # Construct the binary confusion matrix for this class
            binary_confusion = np.array([
                [true_pos, false_pos],
                [false_neg, true_neg]
            ])

            # Plot
            self.plot_confusion_matrix(binary_confusion, classes=[class_label, 'not_' + class_label])

    def plot_confusion_matrix(self, matrix, classes, cmap=plt.cm.Blues):
        plt.figure(figsize=(5, 3))
        sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap,
                    xticklabels=classes,
                    yticklabels=classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()


    def display_incorrect_samples(self, n_samples_per_sentiment=3):
        self.data['prediction'] = self.validator.predict_batch(self.data['reviews_processed'].tolist())
        incorrect_samples = self.data[self.data['true_sentiment'] != self.data['prediction']]

        for sentiment in ['positive', 'negative', 'neutral']:
            # Check if the sentiment exists in the true sentiments
            if sentiment in self.data['true_sentiment'].unique():
                incorrect_of_sentiment = incorrect_samples[incorrect_samples['true_sentiment'] == sentiment].head(n_samples_per_sentiment)
                for _, row in incorrect_of_sentiment.iterrows():
                    print(f"Review: {row['review_text']}")
                    print(f"Review Procrssed: {row['reviews_processed']}")
                    print(f"Actual Sentiment: {row['true_sentiment']}")
                    print(f"Predicted Sentiment: {row['prediction']}\n")

    def show_top_features(self, n_features=10):
        output_str = "x"

        if 'count' in self.classifier.classifier.named_steps or 'tfidf' in self.classifier.classifier.named_steps:
            if 'count' in self.classifier.classifier.named_steps:
                vectorizer = self.classifier.classifier.named_steps['count']
            else:
                vectorizer = self.classifier.classifier.named_steps['tfidf']

            feature_names = vectorizer.get_feature_names_out()
            classifier_coef = self.classifier.classifier.named_steps['classifier'].coef_

            for class_index, class_label in enumerate(self.classifier.classifier.named_steps['classifier'].classes_):
                top_indices = classifier_coef[class_index].argsort()[-n_features:][::-1]
                top_features = [feature_names[i] for i in top_indices]
                output_str += f"Top features for class {class_label}:\n"
                output_str += ", ".join(top_features)
                output_str += "\n\n"

        return output_str





class Validator:
    
    def __init__(self, classifier):
        self.classifier = classifier

    def preprocess_text(self, text):
        # Apply all preprocessors in sequence to the provided text
        return self.classifier.apply_preprocessors(text)

    def predict_single(self, text):
        processed_text = self.preprocess_text(text)
        return self._predict(processed_text)

    def predict_batch(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self._predict(processed_texts)

    def _predict(self, processed_texts):
        return self.classifier.classifier.predict(processed_texts)

    def validate(self):
        text = input("Enter the text for validation: ")
        prediction = self.predict_sentiment(text)
        print(f"The predicted sentiment for the given text is: {prediction}")

