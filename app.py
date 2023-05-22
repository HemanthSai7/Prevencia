import streamlit as st

import pandas as pd

st.title("Prevencia")

#generate a beautiful UI
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">A Machine Learning App for Predicting Hate Speech on Twitter</p>', unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="big-font">By: <a href="https://www.github.com/HemanthSai7/">Hemanth Sai Garladinne</a></p>', unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">This app predicts the **Hate Speech** on Twitter!</p>', unsafe_allow_html=True)

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        tweet = st.sidebar.text_input("Tweet")
        features = {'tweet': tweet}
        data = pd.DataFrame(features, index=[0])
        return data
    input_df = user_input_features()


