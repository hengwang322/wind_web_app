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

FARM_LIST = ['BLUFF1', 'CATHROCK', 'CLEMGPWF', 'HALLWF2', 'HDWF2', 'LKBONNY2', 'MTMILLAR',
             'NBHWF1', 'SNOWNTH1', 'SNOWSTH1', 'STARHLWF', 'WATERLWF', 'WPWF']
FARM_NAME_LIST = ['Bluff Wind Farm', 'Cathedral Rocks Wind Farm', 'Clements Gap Wind Farm',
                  'Hallett 2 Wind Farm', 'Hornsdale Wind Farm 2', 'Lake Bonney Stage 2 Windfarm',
                  'Mt Millar Wind Farm', 'North Brown Hill Wind Farm', 'Snowtown Wind Farm Stage 2 North',
                  'Snowtown South Wind Farm', 'Starfish Hill Wind Farm', 'Waterloo Wind Farm', 'Wattle Point Wind Farm']


def connect_db():
    client = MongoClient(MONGO_URI)
    db = client["wpp"]
    return db


def fetch_data(farm, limit):
    time_start = time.time()

    db = connect_db()
    print(f"Fetching data for {farm}...", end="", flush=True)
    col = db[farm]
    if limit == None:
        df = pd.DataFrame(col.find({},batch_size=10000).sort("_id",-1))
    else:
        df = pd.DataFrame(col.find({},batch_size=1000).sort("_id",-1).limit(limit))

    if '_id' in df.columns:
        df = df.rename(columns={'_id': 'time'})

    runtime = round(time.time()-time_start, 2)

    print(f" Done! Fetched {len(df)} documents in {runtime} s")

    return df


def update_db(farm, update_df, upsert=True):
    if 'time' in update_df.columns:
        update_df = update_df.rename(columns={'time': '_id'})
    db = connect_db()
    ops = []
    for i in range(len(update_df)):
        _id = update_df.iloc[i]._id
        data = update_df.iloc[i].to_dict()
        ops.append(UpdateOne({'_id': _id}, {'$set': data}, upsert=upsert))
    db[farm].bulk_write(ops)


def fill_val(df, offset):
    '''
    Fill missing value with the mean of the -24h and +24h data
    offset is the rows for the +24h/-24h, for 1h interval is 24, for 5min interval is 288
    '''
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

            if not pd.isnull(v_plus) and not pd.isnull(v_minus):
                v = 0.5 * (v_plus + v_minus)
            elif pd.isnull(v_plus):
                v = v_minus
            elif pd.isnull(v_minus):
                v = v_plus
            else:
                v = np.nan

            df.loc[i, item] = v


def get_ref_dt_range(df, utc_start_dt, utc_end_dt, freq):
    '''
    Get a reference datetime range df given utc_start_dt, utc_end_dt, freq and merge it to original df
    This ensures df have no missing point in datetime 
    Missing values will show up as NaN, which can be filled later
    utc_start_dt and utc_end_dt are strings in the format of %Y-%m-%d %H:%M:%S
    freq is a string that specify frequency (e.g., 5Min,1H)
    output is a dataframe with no missing point in the desired datetime range
    '''
    ref_dt_range = pd.date_range(utc_start_dt, utc_end_dt, freq=freq)
    ref_dt_range = pd.DataFrame(ref_dt_range, columns=['time'])
    ref_dt_range['time'] = ref_dt_range['time'].apply(str)
    df = pd.merge(df, ref_dt_range, left_on='time',
                  right_on='time', how='outer', sort=True)

    return df


def get_power(farm, local_start_dt, local_end_dt, offset):
    '''
    Get the power data given datetime range
    local_start_dt and local_end_dt are strings in the format of %Y-%m-%d %H:%M:%S
    offset if a string corresponds to how far away from current time
    return a dataframe with power output in 1h interval, with no gap in datetime or missing value 
    '''
    tz = 'Australia/Adelaide'
    dt_format = 'YYYY-MM-DD HH:mm:ss'
    utc_start_dt = arrow.get(local_start_dt).replace(
        tzinfo=tz).to('UTC').format(dt_format)
    utc_end_dt = arrow.get(local_end_dt).replace(
        tzinfo=tz).to('UTC').format(dt_format)

    # The api only returns data from the offset time to current time. Offset value should be estimated properly.
    # Usually a larger window is chosen which is sliced to desired dt window later.
    power_api = f'https://services.aremi.data61.io/aemo/v6/duidcsv/{farm}?offset={offset}'
    power = pd.read_csv(power_api)

    # Convert the time format from ISO to dt_format
    power['time'] = power['Time (UTC)'].apply(
        lambda t: arrow.get(t).format(dt_format))
    power.drop(['Time (UTC)'], axis=1, inplace=True)

    # //todo: this only works when the between 30-59 minute of any hour if you're in a timezone that that has whole hour. Need to fix it.
    # Get the row of the start date and end date, slice the df to desired dt range
    start_index = 0
    if power['time'][0] > utc_start_dt:
        pass  # in case utc_start_time is after the data start time
    else:
        while not (utc_start_dt in str(power['time'][start_index])):
            start_index += 1

    # search from bottom to top for better performance
    end_index = len(power) - 1
    while not (utc_end_dt in str(power['time'][end_index])):
        end_index -= 1
    power = power[start_index:end_index]

    # Join the reference dt range. If there's missing time points the row will show as NaN.
    power = get_ref_dt_range(power, utc_start_dt, utc_end_dt, '5Min')
    power.drop_duplicates(subset='time', keep='first', inplace=True)
    power.reset_index(drop=True, inplace=True)

    # Fill missing value, if any
    fill_val(power, 288)

    # Set negative values to 0, if any
    for i in power[power['MW'] < 0].index:
        power.loc[i, 'MW'] = 0

    # Converts power output in 5min interval to 1h interval, the power is the mean of the power recorded within the hour.
    power1h = pd.DataFrame()
    for i in range(len(power)//12):
        power1h.loc[i, 'time'] = power[i*12:(i+1)*12]['time'][i*12]
        power1h.loc[i, 'actual'] = power[i*12:(i+1)*12]['MW'].mean()

    return power1h


def get_weather(farm, local_start_dt, local_end_dt):
    '''
    Get the weather data from Darksky, given datetime range
    local_start_dt and local_end_dt are strings in the format of %Y-%m-%d %H:%M:%S
    return a dataframe with hourly weather data 
    '''

    dt_format = 'YYYY-MM-DD HH:mm:ss'
    tz = 'Australia/Adelaide'
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

        # Load the json from api and turns to df, attach each day's df to the master weather df
        with urllib.request.urlopen(dsapi) as url:
            data = json.loads(url.read().decode())
        try:
            df = pd.DataFrame(data['hourly']['data'])
        except:
            df = pd.DataFrame()
        weather = pd.concat([weather, df], axis=0, sort=True)
    # Convert UNIX time to dt_format
    weather['time'] = weather['time'].apply(
        lambda t: arrow.get(int(t)).format(dt_format))

    weather = weather[['time', 'cloudCover', 'dewPoint', 'humidity', 'ozone', 'precipIntensity', 'pressure', 'icon',
                       'temperature',  'uvIndex', 'visibility', 'windBearing', 'windGust', 'windSpeed']]

    weather.columns = ['time', 'cloud_cover', 'dew_point', 'humidity', 'ozone', 'precipitation', 'pressure', 'icon',
                       'temperature',  'uv_index', 'visibility', 'wind_bearing', 'wind_gust', 'wind_speed']

    # make sure there's no missing point in datetime range
    weather = get_ref_dt_range(
        weather, weather.iloc[0, 0], weather.iloc[-1, 0], '1H')
    weather.drop_duplicates(subset='time', keep='first', inplace=True)
    weather.reset_index(drop=True, inplace=True)

    weather.wind_bearing = weather.wind_bearing.apply(float)
    weather.uv_index = weather.uv_index.apply(float)

    return weather
