import csv
import os
import time
import uuid
import pickle
from random import randint

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .data import fetch_data, FARM_LIST, connect_db

MODEL_FILE = os.path.join("models", "models.pkl")
TRAIN_LOG_FILE = os.path.join('models', 'train.log')
MONGO_URI = os.environ['MONGO_URI']

seed = randint(0, 10000)
space = {'max_depth': hp.quniform("max_depth", 3, 15, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 0, 200, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': hp.quniform("n_estimators", 100, 200, 25)
         }


def split_data(X, y, seed, test_size=0.1, val_size=0.1):
    X_train_, X_test, y_train_, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_, y_train_, test_size=val_size, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_data(original_df):
    X_COL = ['cloud_cover', 'dew_point', 'humidity', 'ozone',
             'precipitation', 'pressure', 'temperature',
             'uv_index', 'visibility', 'wind_gust', 'wind_speed',
             'wind_speed_^_2', 'wind_speed_^_3', 'wind_gust_^_2',
             'wind_gust_^_3', 'sin_wind_bearing', 'cos_wind_bearing']

    df = original_df.copy(deep=True)

    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['wind_speed_^_2'] = df['wind_speed']**2
    df['wind_speed_^_3'] = df['wind_speed']**3
    df['wind_gust_^_2'] = df['wind_gust']**2
    df['wind_gust_^_3'] = df['wind_gust']**3
    df['sin_wind_bearing'] = np.sin(df['wind_bearing'] * np.pi / 180.)
    df['cos_wind_bearing'] = np.cos(df['wind_bearing'] * np.pi / 180.)

    X = df[X_COL]

    if 'actual' in df.columns:
        y = df.actual
    else:
        y = None

    return X, y


def best_model_from_trials(trials):
    valid_trial_list = [trial for trial in trials
                        if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minimum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minimum_loss]

    return best_trial_obj['result']['model']


def optimize_model(farm, max_evals, timeout):
    time_start = time.time()
    # ingest data
    client = connect_db(MONGO_URI)
    df = fetch_data(client, farm, limit=None)
    df.dropna(inplace=True)
    X, y = transform_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, seed=seed)

    # tune paramaters
    def objective(space):
        model = XGBRegressor(n_estimators=int(space['n_estimators']),
                             max_depth=int(space['max_depth']),
                             gamma=space['gamma'],
                             reg_alpha=int(space['reg_alpha']),
                             min_child_weight=space['min_child_weight'],
                             colsample_bytree=space['colsample_bytree'],
                             random_state=seed)

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  eval_metric="rmse",
                  early_stopping_rounds=5,
                  verbose=False
                  )

        pred = model.predict(X_val)
        return {'loss': mse(y_val, pred), 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                timeout=timeout)
    model = best_model_from_trials(trials)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    dt_range = f"{df.time.iloc[0]}~{df.time.iloc[-1]}"
    trial_no = len(trials.trials)

    # logging
    header = ['unique_id', 'timestamp', 'model_name', 'runtime', 'trials',
              'dt_range_UTC', 'best_param', 'test_rmse', 'seed']
    write_header = False
    if not os.path.exists(TRAIN_LOG_FILE):
        write_header = True
    with open(TRAIN_LOG_FILE, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), int(time.time()), farm, runtime, trial_no,
                             dt_range, best, mse(model.predict(X_test), y_test, squared=False), seed])
        writer.writerow(to_write)

    return model


def train_models(train_list, max_evals=50, timeout=300, dump=True):
    models = dict()

    for farm in train_list:
        model = optimize_model(farm, max_evals=max_evals, timeout=timeout)
        models[farm] = model

    if dump:
        print('Dumping file...', end="", flush=True)
        pickle.dump(models, open(MODEL_FILE, "wb"))
        print(' Done!')

    return models
