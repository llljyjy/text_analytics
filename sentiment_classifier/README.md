# Sentiment Analysis

## Overview

This repository encompasses various methods for sentiment analysis. The definitions and functions related to the models are stored separately in `naive_bayes_utils.py` and are imported for easier code maintenance and readability.

You can play around with the code in `sentiment_training.ipynb`

---

## 🚀 Model Training Baseline

We start with a baseline model trained on a Naive Bayes classifier.

```python
# Code snippet related to the baseline model can be added here, if needed
```

## 🔍 Advanced EDA and Model Improvement

Several text treatment techniques have been explored to gauge their effectiveness.

- **Advanced EDA**: For exploratory data analysis.
  
- **Model Improvement**: For enhancing the model training process.

## 🛠️ Validation

Before validating the models, ensure the necessary model is initiated:

```python
# Using our custom stopword model
cus_visualization = NaiveBayesVisualization(cus_sw_nb_classifier)
cus_validator = Validator(cus_sw_nb_classifier)

# Example using a longer negative review text
text = 'input your text'
prediction = cus_validator.predict_single(text)
print(f"The predicted sentiment for the given text is: {prediction}")
```

For a direct jump to validation:

```python
# Import the filtered DataFrame
df_cleaned = pd.read_csv('data/df_cleaned.csv')

# Train the model
preprocessors = ['lowercase','stopword']
sw_nb_classifier = NaiveBayesClassifier(data=df_cleaned, text_col='review_text', preprocessors=preprocessors)
train_metrics, test_metrics = sw_nb_classifier.train(vectorizer_type='count', use_additional_features=False)
```

## 🤖 Combined Model

Use the combined model by following the code below:

```python
data = ['put your text here']
preds_weighted = weighted_predictions(cus_validator, us_validator, data, 0.6, 0.4)
print(f"Prediction with preds_weighted is {convert_to_sentiment(preds_weighted[0])}")
```

## ⚙️ Tuneable Parameters

### Preprocessors:
- `lowercase_preprocessor`
- `stopword_preprocessor`
- `lemmatizer_preprocessor`
- `remove_plural_preprocessor`
- `customized_stopword_preprocessor`

### Vectorizer:
- `count`
- `tfidf`

### Others:
- `bi_gram`: binary
- `use_additional_features`: binary
