import pandas as pd
import streamlit as st

@st.cache_data
def load_rainfall_data(path="data/rainfall.csv"):
    df = pd.read_csv(path)
    return df
