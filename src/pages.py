import os
import pickle

import pandas as pd
import numpy as np
import arrow
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from xgboost import DMatrix, XGBRegressor
from shap import summary_plot, waterfall_plot
import streamlit as st

from .plot import plot_map, plot_forecast, plot_weather, plot_historical, plot_error, format_title
from .data import FARM_LIST, FARM_NAME_LIST, connect_db, fetch_data
from .models import MODEL_FILE, TRAIN_LOG_FILE, transform_data

tz = 'Australia/Sydney'


def show_gif(icon='default'):
    if pd.isnull(icon):
        icon = 'default'

    url_prefix = 'https://raw.githubusercontent.com/hengwang322/wind_web_app/master/resources/'
    url_suffix = '.gif'
    st.sidebar.markdown(f'''<img src="{url_prefix}{icon}{url_suffix}" alt="404 Wind farm not found :(" 
    width="270" height="210" style="padding-left: 20px">''', unsafe_allow_html=True)


def load_overview_df():
    df = pd.read_csv('https://services.aremi.data61.io/aemo/v6/csv/wind')
    df.set_index('DUID', inplace=True)
    return df


@st.cache(persist=True, suppress_st_warning=True, ttl=600)
def load_data(farm, limit, dropna=False):
    df = fetch_data(farm, limit=limit)
    # all time displayed in the web app is converted to AEST
    df.time = df.time.apply(lambda t: arrow.get(
        t).to(tz).format('YYYY-MM-DD HH:mm:SS'))
    if dropna:
        df.dropna(inplace=True)
    return df


def welcome():
    st.header('Welcome to Wind Power Predictions')
    st.markdown('''<p style="text-align:justify;">
                This a demo for medium-range wind power predictions for major wind farms in South Australia.</p>
                <p style="text-align:justify;">
                Please choose an option in the sidebar. "<b>Real-time Forecast</b>" will show you today and tomorrow's hourly power predictions. You can view historical data since Jun 2018 in "<b>Historical Data</b>". To evaluate how well the models perform, choose "<b>Model Performance</b>". The effects of predictors on model outcomes are explained in  "<b>Model Explainability</b>". If you want to wish more about the web app and the author, please go to the "<b>About</b>" section.</p>
                ''', unsafe_allow_html=True)

    show_gif(icon='default')

    farms = load_overview_df()
    st.plotly_chart(plot_map(farms), use_container_width=True)
    if st.checkbox('Show raw farm data'):
        format_dict = dict()
        for item in list(farms.dtypes[farms.dtypes == 'float'].index):
            format_dict[item] = "{:.1f}"
        st.write(farms[farms.Region == 'SA1'].style.format(format_dict))
        st.markdown('Data is provided by [The Australian Renewable\
                    Energy Mapping Infrastructure Project (AREMI)](https://nationalmap.gov.au/renewables/)')


def forecast():
    farm_select = st.sidebar.selectbox('Select a farm', FARM_NAME_LIST)
    farm = FARM_LIST[FARM_NAME_LIST.index(farm_select)]
    df = load_data(farm, limit=24*4, dropna=False)

    latest = df[df.isna().any(1)].index[0]-1

    show_gif(icon=df.loc[latest, 'icon'])

    temp = f'''Current temperature: {round(df.loc[latest,"temperature"],1)} °C'''
    wind = f'''Wind Speed: {round(df.loc[latest,"wind_speed"],1)} m/s; 
                Wind Gust: {round(df.loc[latest,"wind_gust"],1)} m/s'''
    misc = f'''Humidity: {round(df.loc[latest,"humidity"]*100)} %; 
                 Precipitation: {round(df.loc[latest,"precipitation"],2)} mm'''
    st.sidebar.markdown(
        f'<div style="color: grey; font-size: large">{temp}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="color: grey; font-size: small">{wind}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="color: grey; font-size: small">{misc}</div>', unsafe_allow_html=True)

    st.plotly_chart(plot_forecast(df, farm), use_container_width=True)

    if st.checkbox('Plot weather data'):
        st.plotly_chart(plot_weather(df, farm),
                        use_container_width=True)
        st.markdown(
            'Weather data is provided by [**Dark Sky API**](https://darksky.net/poweredby/)')
    if st.checkbox('Show raw data'):
        st.dataframe(df.drop('icon', axis=1).set_index(
            'time').style.format("{:.1f}"))
        st.markdown(
            'Weather data is provided by [**Dark Sky API**](https://darksky.net/poweredby/)')

    if st.button('Tooltip'):
        st.markdown('''<p style="text-align:justify;">
                    There are various buttons on the top-left corner of the graph. You can use them to pan, zoom or download the graph, or toggle what to show when hovering over the traces. You can also click on the legend of a trace to hide it.</p>
                    ''', unsafe_allow_html=True)

    if st.button('About the Data'):
        st.markdown('''<p style="text-align:justify;">
                    All data is collected in its own timezone, which is updated hourly. Power data is the average output of the wind farm within the hour.</p>
                    ''', unsafe_allow_html=True)


def historical():
    range_dict = {'1 Week': 7*24, '1 Month': 30*24, '3 Months': 90 *
                  24, '6 Months': 180*24, '1 Year': 365*24, 'All time': None}
    range_select = st.selectbox(
        'Select how much historic data you would like to view', list(range_dict.keys()), index=1)
    farm_select = st.sidebar.selectbox('Select a farm', FARM_NAME_LIST)
    farm = FARM_LIST[FARM_NAME_LIST.index(farm_select)]
    show_gif(icon='default')
    df = load_data(farm, limit=range_dict[range_select], dropna=True)

    st.plotly_chart(plot_historical(df, farm), use_container_width=True)

    if st.checkbox('Plot historical weather data'):
        st.plotly_chart(plot_weather(df, farm), use_container_width=True)
        st.markdown(
            'Weather data is provided by [**Dark Sky API**](https://darksky.net/poweredby/)')
    if st.checkbox('Show raw historical data'):
        st.write(df.drop(['icon'], axis=1).set_index(
            'time').style.format("{:.1f}"))
        st.markdown(
            'Weather data is provided by [**Dark Sky API**](https://darksky.net/poweredby/)')

    if st.button('Tooltip'):
        st.markdown('''<p style="text-align:justify;">
                     There are various buttons on the top-left corner of the graph.You can use them to pan, zoom
                     or download the graph, or toggle what to show when hovering over the traces. You can also click on the
                     legend of a trace to hide it.</p>
                     ''', unsafe_allow_html=True)


def performance():
    farm_select = st.sidebar.selectbox('Select a farm', FARM_NAME_LIST)
    farm = FARM_LIST[FARM_NAME_LIST.index(farm_select)]
    show_gif(icon='default')
    df = load_data(farm, limit=30*24, dropna=True)
    error = df.copy(deep=True)
    # todo: calculate from database instead
    error['date'] = pd.to_datetime(error.time).dt.date
    error = error[['prediction', 'actual', 'date']].groupby(
        by='date').mean().reset_index()

    error['error'] = error['actual'] - error['prediction']
    st.plotly_chart(plot_error(error, farm), use_container_width=True)

    train_log = pd.read_csv(TRAIN_LOG_FILE, sep='\t')
    train_farm = train_log[train_log['model_name'] == farm]
    train_farm.reset_index(drop=True, inplace=True)
    latest_version = arrow.get(
        train_farm.timestamp.iloc[-1]).format('YYYY-MM-DD')

    actual = np.array(error.actual)
    prediction = np.array(error.prediction)

    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - prediction[j]) / actual[j]
        else:
            res[j] = prediction[j] / np.mean(actual)

    mape = np.mean(np.abs(res))*100

    st.markdown(f'''<p style="text-align:justify;">
    Model Version: {farm}-{latest_version}.<br>
    The prediction model on {farm_select} shows a day average root mean square error (RMSE) of  
    <b>{mse(actual,prediction,squared=False):.2f}</b> and a mean absolute percentage 
    error (MAPE) of <b>{mape:.2f}%</b>.</p>
    <p style="text-align:justify;">An accurate daily prediction can help market operators plan 
    ahead and farm owners profit from energy bidding for the next day</p>
    ''', unsafe_allow_html=True)

    if st.checkbox("Show log for past trainings"):
        st.write(train_farm)


@st.cache(persist=True, suppress_st_warning=True, hash_funcs={XGBRegressor: id}, ttl=600)
def load_models():
    return pickle.load(open(MODEL_FILE, "rb"))


def explain():
    st.header('Model Explainability')
    st.subheader(
        'Use SHAP to explain the output of the machine learning model')
    st.markdown('''<p style="text-align:justify;">
    Model explainability is the ability to explain the internal mechanics of a model in human terms. It is an important tool to make reasoning behind each decision in machine learning transparent and repeatable.</p>
    <p style="text-align:justify;">
    <a href="https://github.com/slundberg/shap" target="_blank">SHAP</a> (SHapley Additive exPlanations) is a python module that uses game theoretic approach to explain machine learning models. It will be used to explore the explainability of all XGBoost models for different wind farms.</p>
    ''', unsafe_allow_html=True)

    farm_select = st.sidebar.selectbox('Select a farm', FARM_NAME_LIST)
    show_gif(icon='default')
    farm = FARM_LIST[FARM_NAME_LIST.index(farm_select)]
    df = load_data(farm, limit=200)
    models = load_models()
    model = models[farm]
    with st.spinner("Running Calculations..."):
        X, _ = transform_data(df)
        shap_val = model.get_booster().predict(DMatrix(X), pred_contribs=True)
        expected_val = shap_val[0][-1]
        shap_val = np.delete(shap_val, obj=-1, axis=1)

    col_name = [format_title(col) for col in list(X.columns)]
    st.subheader("Feature importance based on SHAP value")
    if st.checkbox('Show more...', key='importance', value=True):
        st.markdown('''<p style="text-align:justify;">
        The following plot summarizes feature importance based on SHAP values (i.e. how much each feature changes the model outcome when conditioning on that feature). The features are sorted by the sum of the magnitudes of SHAP values. The colour represents feature value, while red is high and blue is low.</p>
        <p style="text-align:justify;">
        For example, if a red (high feature value) data point shows a positive SHAP value, it increases the predicted value; if the SHAP value is negative, it lowers the predicted value. A point far away from zero point also has a higher impact (either negative or positive) than that is near zero point.</p>
        ''', unsafe_allow_html=True)

        with st.spinner("Plotting..."):
            summary_plot(shap_val, X, show=False, feature_names=col_name)
            st.pyplot(bbox_inches='tight', dpi=150)
            plt.clf()

    st.subheader("Feature contribution for individual prediction")
    if st.checkbox('Show more...', key='contribution'):
        st.markdown('''<p style="text-align:justify;">
        The waterfall plot below demonstrates how much each feature contributes to pushing the model from the baseline value (indicated by <i>E[f(X)]</i>) to model output (indicated by <i>f(X)</i>) in an intuitive manner.</p>
        <p style="text-align:justify;">
        The plot can show all the individual predictions for today & tomorrow (48 h in total). Use the slider below to choose which hour's prediction you'd like to view.</p>
        ''', unsafe_allow_html=True)

        with st.spinner("Plotting..."):
            pred = df[-48:].reset_index(drop=True).prediction
            i = st.slider('Select the hour', min_value=0,
                          max_value=47, value=0)
            plt.figure(figsize=(18, 2))
            plt.rcParams.update({'font.size': 20})
            plt.plot(range(0, 48), pred, c='#1e88e5')
            plt.scatter(i, pred[i], c='#ff0d57', s=300)
            plt.xlim(0, 47)
            plt.xticks(np.arange(0, 48, step=6))
            plt.xlabel('Hour')
            st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0.01)
            plt.clf()

            waterfall_plot(expected_val, shap_val[-48:][i],
                           feature_names=col_name, max_display=10, show=False)
            st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0)
            plt.clf()


def about():
    st.markdown('''
    <h1><a href="https://drive.google.com/file/d/14tvyZ9Lt3peM-9B2ZbAmw6HeRacYUzgW/view" target="_blank">🎐</a>Wind Power Prediction: About</h1>
    <h3>The Project</h3>
    <p style="text-align:justify;">
    This is an interactive web app for a <b>medium-range wind power prediction model</b>. Wind power prediction is critical in <b>integrating</b> the highly intermittent wind power into our existing power grid. An accurate medium-range (usually up to 48 h) prediction can help wind farms <b>maximize the profit</b> of energy trading and help energy market operators <b>minimize the cost</b> of grid stabilization and <b>reduce carbon emission</b> for firing up backup fossil fuel plants.</p>
    <p style="text-align:justify;">
    This web app is part of a capstone project for a Data Science Immersive course at <b><a href="https://generalassemb.ly/" target="_blank" >General Assembly</a></b> Sydney. The models are built using <b>XGBoost</b> algorithms by discovering the relationship between weather data and power data, both of which are publicly available through APIs. This web app is built with <b>Streamlit</b> and hosted on <b><a href="https://www.heroku.com/" target="_blank">Heroku</a></b>. You can check out the source code of this app <a href="https://github.com/hengwang322/wind_web_app" target="_blank">here</a>. If you're interested in the original datasets, Jupyter Notebooks and code, feel free to visit the repositories <a href="https://github.com/hengwang322/wind_power_prediction" target="_blank">here</a> and <a href="https://github.com/hengwang322/explainable-wind-power-forecast" target="_blank">here</a>.</p>
    <h3>The Author</h3>
    <p style="text-align:justify;">
    This web app is developed by Heng Wang. You can learn more about me on my <a href="https://github.com/hengwang322" target="_blank">GitHub</a> and <a href="https://hengwang322.github.io/" target="_blank">website</a>. Feel free to contact me on <a href="https://www.linkedin.com/in/hengwang322" target="_blank">LinkedIn</a> or through <a href="mailto:wangheng322<at>gmail.com">Email</a>.
    ''', unsafe_allow_html=True)
