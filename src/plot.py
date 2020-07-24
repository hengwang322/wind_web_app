import os

import pandas as pd
import numpy as np
import arrow
import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import FARM_LIST, FARM_NAME_LIST

tz = 'Australia/Sydney'
MAPBOX_TOKEN = os.environ['MAPBOX_TOKEN']


def format_title(string, acronym_list=['UV'], splitter='_'):
    format_words = []
    for word in string.split(splitter):
        if word in word in [a.lower() for a in acronym_list]:
            word = word.upper()
        else:
            word = word.title()
        format_words.append(word)

    format_string = ' '.join(format_words)

    return format_string


# @st.cache(persist=True, suppress_st_warning=True, ttl=600)
def plot_map(farms):
    lastest_time = farms.dropna().iloc[:, 3][0]
    lastest_time = arrow.get(lastest_time).to(tz).format("YYYY-MM-DD HH:mm")

    # Rectify negative values
    bad_index = farms[farms['Current Output (MW)'] < 0].index
    farms.loc[bad_index, 'Current Output (MW)'] = 0
    farms.fillna(0, inplace=True)
    farms['Power (MW)'] = farms['Current Output (MW)'].apply(
        lambda x: round(x, 1))

    px.set_mapbox_access_token(MAPBOX_TOKEN)
    fig = px.scatter_mapbox(farms.loc[FARM_LIST, :], lat="Lat", lon="Lon", zoom=5,
                            text='Station Name', size='Power (MW)',
                            color="Power (MW)",
                            color_continuous_scale=plotly.colors.sequential.haline,
                            center={'lat': -35.5, 'lon': 137.5},
                            title='Wind Power Generation in South Australia')

    fig.update_layout(annotations=[go.layout.Annotation(
                      showarrow=False,
                      xanchor='right',
                      x=1,
                      yanchor='top',
                      y=0,
                      text=f'Last Update: {lastest_time} AEST')],
                      title_x=0.5,
                      title_y=0.92,
                      margin=dict(l=20, r=20, t=60, b=80),
                      )

    return fig


# @st.cache(persist=True, suppress_st_warning=True, ttl=600)
def plot_forecast(df, farm):
    farm_name = FARM_NAME_LIST[FARM_LIST.index(farm)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.time, y=round(
        df['actual'], 2), name="Actual"))
    fig.add_trace(go.Scatter(x=df.time, y=round(
        df['prediction'], 2), name="Prediction"))

    title_text = f'Hourly Wind Power Prediction at {farm_name}'
    latest = df[df.actual.isna()].index[-1]

    fig.update_xaxes(nticks=12)
    fig.add_shape(
        dict(type="line",
             yref="paper",
             x0=df.loc[latest+1, 'time'],
             y0=0,
             x1=df.loc[latest+1, 'time'],
             y1=1,
             line=dict(color="Grey", width=1.5, dash="dash")))
    fig.add_annotation(x=df.loc[latest+10, 'time'],
                       yref="paper",
                       y=1.07,
                       showarrow=False,
                       text=f"Last Update {df.loc[latest,'time'][:16]} AEST")

    fig.update_layout(hovermode='x',
                      title_text=title_text,
                      title_x = 0.5,
                      title_y = 0.92,
                      yaxis_title="Power (MW)",
                      margin=dict(l=0, r=0, t=80, b=20)
                      )
    return fig


# @st.cache(persist=True, suppress_st_warning=True, ttl=600)
def plot_historical(df, farm):
    farm_name = FARM_NAME_LIST[FARM_LIST.index(farm)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.time, y=round(
        df['actual'], 2), name="Actual"))
    fig.add_trace(go.Scatter(x=df.time, y=round(
        df['prediction'], 2), name="Prediction"))

    title_text = f'Historical Prediction Data at {farm_name}'

    fig.update_layout(hovermode='x',
                      title_text=title_text,
                      title_x = 0.5,
                      title_y = 0.9,
                      yaxis_title="Power (MW)",
                      margin=dict(l=0, r=0, t=80, b=20))
    return fig


# //todo
# @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True, ttl=600)
def plot_error(df, farm):
    farm_name = FARM_NAME_LIST[FARM_LIST.index(farm)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=round(
        df['actual'], 2), name="Actual"))
    fig.add_trace(go.Scatter(x=df.date, y=round(
        df['prediction'], 2), name="Prediction"))
    fig.add_trace(go.Scatter(x=df.date, y=round(
        df['error'], 2), name="Prediction Error"))

    title_text = f'Average Prediction Error at {farm_name} (Last 3 Months)'

    fig.update_layout(hovermode='x',
                      yaxis_title="Power (MW)",
                      title_text=title_text,
                      title_x = 0.5,
                      title_y = 0.9,
                      margin=dict(l=0, r=0, t=80, b=20))
    return fig


# @st.cache(persist=True, suppress_st_warning=True, ttl=600)
def plot_weather(df, farm):
    fig = make_subplots(rows=8, cols=1)

    col_wind = ['wind_speed', 'wind_gust']
    for col in col_wind:
        fig.add_trace(go.Scatter(
            x=df.time, y=df[col], name=format_title(col)), row=1, col=1)

    col_other = ['temperature', 'pressure', 'precipitation',
                 'cloud_cover', 'humidity', 'dew_point', 'uv_index']
    for i, col in enumerate(col_other):
        fig.add_trace(go.Scatter(
            x=df.time, y=df[col], name=format_title(col)), row=i+2, col=1)

    for i in range(7):
        fig.update_xaxes(showticklabels=False, nticks=12, row=i+1, col=1)
    fig.update_xaxes(row=8, col=1, nticks=12)

    units = ["Speed (m/s)", "Temp (°C)", "Pressure (hPa)",
             "Precip (mm)", "Cloud", "Humidity", "Dew (°C)", "UV"]
    for i, unit in enumerate(units):
        fig.update_yaxes(title_text=unit, showgrid=False, row=i+1, col=1)

    fig.update_layout(hovermode='x', width=700, height=900, title="Weather Data",
                      margin=dict(l=0, r=0, t=60, b=20))

    return fig
