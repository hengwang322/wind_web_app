#!/usr/bin/env python

import os
import pickle
from pathlib import Path
import time

import arrow
import numpy as np
import pandas as pd
import streamlit as st

from src.models import MODEL_FILE, transform_data
from src.data import FARM_LIST, update_db, get_weather, get_power


def main():
    time_start = time.time()

    models = pickle.load(open(MODEL_FILE, 'rb'))

    DARKSKY_KEY = os.environ['DARKSKY_KEY']

    tz = 'Australia/Sydney'

    today = arrow.utcnow().to(tz).format('YYYY-MM-DD HH:00:00')
    yesterday = arrow.utcnow().to(tz).shift(days=-1).format('YYYY-MM-DD HH:00:00')
    dayafter = arrow.utcnow().to(tz).shift(days=+2).format('YYYY-MM-DD HH:00:00')

    for farm in FARM_LIST:
        print(f'Updating {farm}         ', end='\r', flush=True)
        weather_update = get_weather(farm, yesterday, dayafter)

        X, _ = transform_data(weather_update)

        model = models[farm]

        weather_update['prediction'] = model.predict(X)
        weather_update.prediction = weather_update.prediction.apply(float)
        weather_update.wind_bearing = weather_update.wind_bearing.apply(float)
        weather_update.uv_index = weather_update.uv_index.apply(float)

        bad_index = weather_update[weather_update.prediction < 0].index
        weather_update.loc[bad_index, 'prediction'] = 0

        update_db(farm, weather_update, upsert=True)

        power_update = get_power(farm, yesterday, today, offset='2D')
        update_db(farm, power_update, upsert=True)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    print(f'Done! Runtime: {runtime}')


if __name__ == '__main__':
    main()
    st.caching.clear_cache()
