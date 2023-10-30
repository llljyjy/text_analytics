import streamlit as st
import pandas as pd
st.set_page_config(page_title="About", page_icon='📊')

st.header("📊 About The Project")

st.markdown("---")

st.markdown("### 🎯 Goal :")

st.write("""Sephora stands as the preeminent luxury goods conglomerate globally, boasting best-in-class e-commerce experiences. Beside reaching customers through channels such as Facebook Live Shopping, and Instagram Checkout, Sephora has also prioritised digital means such as integrating in-store technologies to engage clients, and personalizing product or service recommendations based on customer data. Furthermore, in today’s social media fueled world, global brands tag an additional premium to monitor reviews and feedback and quickly act on them. This ability to turnaround the feedback gives brands an edge over their competitors. However, given the large number of products offered by Sephora, and its global presence, this means that Sephora must sieve through large swaths of data and get them to the relevant departments.  

As such, in alignment its core values and business trajectory, Sephora should leverage a data-driven approach, applying text analytics to sort through large amounts of reviews and feedback, to enhance customer satisfaction and foster greater customer loyalty. The intent would be to channel these reviews to the relevant departments for them to act upon them, improving the turnaround time experienced by the customer.""")

    
st.markdown("### 🔬 Project Overview :")

st.write("""The project begins with web scraping weekly job postings posted last week of data engineering roles from Glassdoor in the US. The collected data includes job titles, 
                company names, job locations, job descriptions, salaries, education requirements, and required skills. The data is named like "glassdoor-data-engineer-15-2023.csv"
                where 15 is the week number the data was scraped in and 2023 is the year, then it's stored locally on data/raw/ folder then it's uploaded to an AWS S3 Bucket containing 
                only the raw uncleaned data. The data is then cleaned and preprocessed to remove irrelevant information and ensure consistency, the duplicates are dropped
                then it's joined with the initial cleaned data in another S3 Bucket containing only one csv file that contains all the job postings. All of this is automated in a data pipeline
                using MageAI.""")

st.write("""Exploratory Data Analysis (EDA) is performed on the cleaned data to gain insights into trends and patterns. This includes identifying 
                the most common job titles, the industries and locations with the highest demand, and the most commonly required skills and education 
                degrees. EDA also involves creating visualizations to aid in understanding the data.""")

st.write("""After EDA, feature engineering is performed to create new features that may improve the accuracy of the salary prediction model. 
                This includes creating dummy variables for categorical features such as location, education level, and seniority.""")

st.write("""The salary prediction model is built using a random forest regressor. Finally, the model is deployed in a web application using Streamlit, 
                allowing users to input their own data and receive a salary prediction based on the model.""")

st.markdown("---")