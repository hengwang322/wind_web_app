import os

import streamlit as st
import pandas as pd
import src.pages as pages
from pymongo import MongoClient


@st.cache(hash_funcs={MongoClient: id})
def connect():
    MONGO_URI = os.environ['MONGO_URI']
    client = MongoClient(MONGO_URI)
    return client


def main():
    st.set_page_config(page_title='Wind Power Prediction', page_icon='🎐')

    st.markdown(
        """
    <style>
    .css-1aumxhk {
    padding: 0
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Navigation")
    PAGES = {
        'Welcome': pages.welcome,
        'Real-time Forecast': pages.forecast,
        'Historical Data': pages.historical,
        'Model Performance': pages.performance,
        'Model Explainability': pages.explain,
        'About': pages.about,
    }
    page_select = st.sidebar.radio("Go to...", list(PAGES.keys()))

    client = connect()

    PAGES[page_select](client)


if __name__ == '__main__':
    main()
