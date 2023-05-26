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



