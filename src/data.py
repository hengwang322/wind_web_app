import os
import urllib
import json
import time

import arrow
import numpy as np
import pandas as pd
from pymongo import MongoClient, UpdateOne

MONGO_URI = os.environ['MONGO_URI']
DARKSKY_KEY = os.environ['DARKSKY_KEY']

FARM_LIST = ['BLUFF1', 'CATHROCK', 'CLEMGPWF', 'HALLWF2', 'HDWF2', 
             'LKBONNY2', 'MTMILLAR', 'NBHWF1', 'SNOWNTH1', 'SNOWSTH1', 
             'STARHLWF', 'WATERLWF', 'WPWF']
FARM_NAME_LIST = ['Bluff Wind Farm', 'Cathedral Rocks Wind Farm', 
                  'Clements Gap Wind Farm','Hallett 2 Wind Farm', 
                  'Hornsdale Wind Farm 2', 'Lake Bonney Stage 2 Windfarm', 
                  'Mt Millar Wind Farm', 'North Brown Hill Wind Farm', 
                  'Snowtown Wind Farm Stage 2 North', 
                  'Snowtown South Wind Farm', 'Starfish Hill Wind Farm', 
                  'Waterloo Wind Farm', 'Wattle Point Wind Farm']


def connect_db(MONGO_URI):
    """Connect to MongoDB & return the client object."""
    return MongoClient(MONGO_URI)


def fetch_data(client, farm, limit):
    """Get the last N row of data."""
    time_start = time.time()
    db = client["wpp"]
    print(f"Fetching data for {farm}...", end="", flush=True)
    col = db[farm]
    if limit == None:
        df = pd.DataFrame(col.find({}, batch_size=10000).sort("_id", -1))
    else:
        df = pd.DataFrame(
            col.find({}, batch_size=1000).sort("_id", -1).limit(limit))

    if '_id' in df.columns:
        df = df.rename(columns={'_id': 'time'})

    runtime = round(time.time()-time_start, 2)

    print(f" Done! Fetched {len(df)} documents in {runtime} s")

    return df


def update_db(farm, update_df, upsert=True):
    """Update database via bulk write."""
    if 'time' in update_df.columns:
        update_df = update_df.rename(columns={'time': '_id'})
    client = connect_db(MONGO_URI)
    db = client["wpp"]
    ops = []
    for i in range(len(update_df)):
        _id = update_df.iloc[i]._id
        data = update_df.iloc[i].to_dict()
        ops.append(UpdateOne({'_id': _id}, {'$set': data}, upsert=upsert))
    db[farm].bulk_write(ops)


def fill_val(raw, offset):
    """Fill missing value with the mean of the -24h and +24h data.
    offset is the rows for the +24h/-24h, for 1h interval is 24, 
    for 5min interval is 288.
    """
    df = raw.copy(deep=True)
    for item in df.drop('time', axis=1).columns:
        for i in df[df.isna().any(1)].index:
            # Take into consideration if missing values don't have -24h and +24h data
            try:
                v_plus = df[item][i+offset]
            except:
                v_plus = np.nan
            try:
                v_minus = df[item][i-offset]
            except:
                v_minus = np.nan

            # fill with the with the mean of the -24h and +24h data if they both exist
            # otherwise, just fill with the one that exists
            if not pd.isnull(v_plus) and not pd.isnull(v_minus):
                v = 0.5 * (v_plus + v_minus)
            elif pd.isnull(v_plus):
                v = v_minus
            elif pd.isnull(v_minus):
                v = v_plus
            else:
                v = np.nan

            df.loc[i, item] = v
    return df


def get_power(farm, local_start_dt, local_end_dt):
    """Get power data & convert it to 1h format."""
    tz = 'Australia/Adelaide'
    utc_start_dt = pd.to_datetime(
        local_start_dt).tz_localize(tz).tz_convert('UTC')
    utc_end_dt = pd.to_datetime(local_end_dt).tz_localize(tz).tz_convert('UTC')

    offset = (arrow.now() - arrow.get(local_start_dt)).days + 2
    power_api = f'https://services.aremi.data61.io/aemo/v6/duidcsv/{farm}?offset={offset}D'

    raw = pd.read_csv(power_api)
    raw.columns = ['time', 'actual']
    raw['time'] = pd.to_datetime(raw['time'])

    # Ensure no vacant in the time series
    reference_idx = pd.date_range(start=raw.iloc[0].time,
                                  end=raw.iloc[-1].time,
                                  freq="5min",
                                  name='time')

    raw = raw.set_index('time').reindex(reference_idx).reset_index()
    raw = fill_val(raw, offset=288)

    # Slice the raw df to a desired range
    power_5min = raw[(raw['time'] >= utc_start_dt)
                     & (raw['time'] < utc_end_dt)]

    # rectify negative value
    neg_idx = power_5min[power_5min.actual < 0].index
    power_5min.loc[neg_idx, 'actual'] = 0

    # aggregate by the hour
    power_1h = power_5min.set_index('time')['actual'].resample(
        '60min', offset='30min', label='left').mean().reset_index()
    power_1h.time = power_1h.time.dt.strftime('%Y-%m-%d %H:%M:%S')

    return power_1h


def get_weather(farm, local_start_dt, local_end_dt):
    """Get weather data from Darksky.
    local_start_dt and local_end_dt are strings in format of %Y-%m-%d %H:%M:%S.
    Return a dataframe with hourly weather data.
    """
    overview = pd.read_csv('https://services.aremi.data61.io/aemo/v6/csv/wind')
    overview.set_index('DUID', inplace=True)
    location = f"{overview.loc[farm,'Lat']},{overview.loc[farm,'Lon']}"
    flags = '?exclude=currently,daily,flags&units=si'

    # Set a datetime range, call the api for each day's data in the dt range
    local_dt_range = pd.date_range(local_start_dt, local_end_dt, freq='1D')

    weather = pd.DataFrame()
    for dt in local_dt_range[:-1]:
        # Construct the API url for each day
        time = dt.strftime("%Y-%m-%d")+'T00:00:00'
        dsapi = f'https://api.darksky.net/forecast/{DARKSKY_KEY}/{location},{time}{flags}'
        with urllib.request.urlopen(dsapi) as url:
            data = json.loads(url.read().decode())
        try:
            df = pd.DataFrame(data['hourly']['data'])
        except:
            df = pd.DataFrame()
        weather = pd.concat([weather, df], axis=0, sort=True)

    weather['time'] = pd.to_datetime(weather['time'], unit='s')
    weather = weather[['time', 'cloudCover', 'dewPoint', 'humidity', 'ozone', 
                       'precipIntensity', 'pressure', 'icon', 'temperature',  
                       'uvIndex', 'visibility', 'windBearing', 'windGust', 
                       'windSpeed']]

    weather.columns = ['time', 'cloud_cover', 'dew_point', 'humidity', 'ozone', 
                       'precipitation', 'pressure', 'icon', 'temperature',  
                       'uv_index', 'visibility', 'wind_bearing', 'wind_gust', 
                       'wind_speed']
    # make sure there's no missing point in datetime range
    reference_idx = pd.date_range(start=weather.iloc[0].time,
                                  end=weather.iloc[-1].time,
                                  freq="1H",
                                  name='time')
    weather = weather.set_index('time').reindex(reference_idx).reset_index()
    weather = fill_val(weather, offset=24)

    weather.time = weather.time.dt.strftime('%Y-%m-%d %H:%M:%S')
    weather.wind_bearing = weather.wind_bearing.apply(float)
    weather.uv_index = weather.uv_index.apply(float)

    return weather
