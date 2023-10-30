import streamlit as st
import gensim
from gensim import corpora, models
import re
import os
import shutil
import heapq
import pandas as pd
#### topic labeling ##########

product_keywords = ['cream', 'foundation', 'lipstick', 'perfume', 'makeup', 'acne', 'irritated skin', 'worst product', 'bad product', 'poor quality', 'harsh', ' harsh chemicals', 'toxic chemicals', 'greasey cream', 'hand cream', 'oily', 'break out', 'cured acne', 'smooth', 'soft']

marketing_keywords = ['good deal', 'gift set', 'promotions', 'festive offer', 'end of season sale', 'bundle offer', 'dicounts', 'cupon code']

last_mile = ['Late delivery', 'service', 'poor service', 'delivery agent rude', 'horrible delivery', 'quick delivery', 'ontime', 'very late', 'poor delivery','experience', 'not happy', 'satistifed']


industries = {
    'Product Department': product_keywords,
    'Marketing Team': marketing_keywords,
    'Last Mile': last_mile
}


# def label_topic(text):
#     """
#     Given a piece of text, this function returns the industry label that best matches the topics discussed in the text.
#     """
#     # Count the number of occurrences of each keyword in the text for each industry
#     counts = {}
#     for industry, keywords in industries.items():
#         count = sum([1 for keyword in keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)])
#         counts[industry] = count
    
#     # Return the industry with the highest count
#     return max(counts, key=counts.get)

def label_topic(text):
    """
    Given a piece of text, this function returns the top two departments labels that best match the topics discussed in the text.
    """
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum([1 for keyword in keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)])
        counts[industry] = count
    
    # Get the top two industries based on their counts
    top_industries = heapq.nlargest(2, counts, key=counts.get)
    
    # If only one industry was found, return it
    if len(top_industries) == 1:
        return top_industries[0]
    # If two industries were found, return them both
    else:
        return top_industries

def preprocess_text(text):
    # Replace this with your own preprocessing code
    # This example simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]

    return preprocessed_text

def perform_topic_modeling(transcript_text,num_topics=1,num_words= 5):
    """
    this function performs topic modelling on a given text.
    """
    preprocessed_text = preprocess_text(transcript_text)
    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
     # Extract the most probable words for each topic
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics

st.set_page_config(layout="wide")

choice = st.sidebar.selectbox("Select your choice", ["On Text","On CSV"])

if choice == "On Text":
    st.subheader("Topic Modeling and Labeling App")
    
    text_input=st.text_area("Paste your input text",height = 400)
    
    if text_input is not None:
        if st.button("analyze text"):
            col1, col2, col3 =st.columns([1,1,1])
            with col1:
                st.info("Text is below")
                st.success(text_input)
            with col2:
                topics= perform_topic_modeling(text_input)
                st.info("Topics in the text")
                for topic in topics:
                    st.success(f"{topic[0]}: {', '.join(topic[1])}")
            with col3:
                st.info("Topic Labeling")
                labeling_text = text_input
                industry = label_topic(labeling_text)
                st.markdown("**Topic Labeling Industry Wise**")
                st.write(industry)
                 
elif choice == "On CSV":
    st.subheader("Topic Modeling and Labeling on CSV File")
    upload_csv = st.file_uploader("Upload your CSV file", type=['csv'])
    if upload_csv is not None:
        if st.button("Analyze CSV File"):
            col1, col2 = st.columns([1,2])
            with col1:
                st.info("CSV File uploaded")
                csv_file = upload_csv.name
                with open(os.path.join(csv_file),"wb") as f: 
                    f.write(upload_csv.getbuffer()) 
                print(csv_file)
                df = pd.read_csv(csv_file, encoding= 'unicode_escape')
                st.dataframe(df)
            with col2:
                data_list = df['Data'].tolist()
                industry_list = []
                for i in data_list:
                    industry = label_topic(i)
                    industry_list.append(industry)
                df['Industry'] = industry_list
                st.info("Topic Modeling and Labeling")
                st.dataframe(df)               
                
