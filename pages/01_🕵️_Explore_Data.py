import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt



st.set_page_config(page_title="Review Tracker", page_icon='👨‍💻')
df = pd.read_csv("C:\\Users\\User\\Desktop\\Textanalytics\\project\\data\\product_info.csv")
reviwes_df = pd.read_csv("C:\\Users\\User\\Desktop\\Textanalytics\\project\\data\\reviews_df.csv")

def show_explore_page():

    st.header("🕵️ Exploring Sephora Product Dashboard")
    st.write(f":blue[{len(reviwes_df)} Customer Reviews Analysed]")

    st.info("##### Gain insights into what your customer says about you")


    # Most in Demand Degrees
    st.markdown("#### 🎓 the distribution of sentiment")

    degree_counts = reviwes_df['true_sentiment'].value_counts().reset_index()
    degree_counts.columns = ['Sentiment', 'Count']

    sentiment_chart = alt.Chart(degree_counts).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
        x=alt.Y('Sentiment:N', title=None),
        y=alt.X('Count:Q', title=None),
        color="Sentiment"
    ).properties(
        height=500
    ).configure_axis(
        labelFontSize=18,
        labelAngle=0,
        grid=False
    ).configure_text(
        fontSize=16,
        fontWeight='bold'
    ).configure_legend(
        disable=True
    )

    st.altair_chart(sentiment_chart, use_container_width=True)

show_explore_page()