import streamlit as st
import pandas as pd
import src.pages as pages


def main():
    st.sidebar.subheader("Navigation")
    PAGES = {
        'Welcome': pages.welcome,
        'Real-time Forecast': pages.forecast,
        'Historical Data': pages.historical,
        'Model Performance': pages.performance,
        'Model Explainability': pages.explain,
        'About': pages.about
    }
    page_select = st.sidebar.radio("Go to...", list(PAGES.keys()))

    PAGES[page_select]()


if __name__ == '__main__':
    main()
