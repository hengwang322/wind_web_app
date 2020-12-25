import os
import pickle
from pathlib import Path
import time

import arrow
import numpy as np
import pandas as pd
import streamlit as st

from src.models import MODEL_FILE, transform_data
from src.data import FARM_LIST, update_db, connect_db, get_weather, get_power
pd.options.mode.chained_assignment = None

MONGO_URI = os.environ['MONGO_URI']


def main():
    time_start = time.time()
    models = pickle.load(open(MODEL_FILE, 'rb'))
    tz = 'Australia/Sydney'
    dt_format = 'YYYY-MM-DD HH:00:00'  # round to hour
    today = arrow.utcnow().to(tz).format(dt_format)
    yesterday = arrow.utcnow().to(tz).shift(days=-1).format(dt_format)
    dayafter = arrow.utcnow().to(tz).shift(days=+2).format(dt_format)

    for farm in FARM_LIST:
        print(f'Updating {farm}         ', end='\r', flush=True)
        weather_update = get_weather(farm, yesterday, dayafter)
        X, _ = transform_data(weather_update)
        model = models[farm]
        weather_update['prediction'] = model.predict(X)

        # rectify negative values
        bad_index = weather_update[weather_update.prediction < 0].index
        weather_update.loc[bad_index, 'prediction'] = 0
        # convert the columns to type float64 so mongodb can take them
        for col in ['cloud_cover', 'dew_point', 'humidity', 'ozone',
                    'precipitation', 'pressure', 'temperature', 'uv_index',
                    'visibility', 'wind_bearing', 'wind_gust', 'wind_speed', 'prediction']:
            weather_update[col] = weather_update[col].apply(float)
        update_db(farm, weather_update, upsert=True)

        power_update = get_power(farm, yesterday, today)
        update_db(farm, power_update, upsert=True)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    print(f'Done! Runtime: {runtime}')


if __name__ == '__main__':
    main()
