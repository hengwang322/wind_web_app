import os
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.models import MODEL_FILE, transform_data
from src.data import FARM_LIST, update_db, connect_db, fetch_data
pd.options.mode.chained_assignment = None

MONGO_URI = os.environ['MONGO_URI']


def main():
    time_start = time.time()
    client = connect_db(MONGO_URI)
    models = pickle.load(open(MODEL_FILE, 'rb'))

    for farm in FARM_LIST:
        print(f'Updating {farm}         ', end='\r', flush=True)
        df = fetch_data(client, farm, limit=None)
        X, _ = transform_data(df)
        model = models[farm]

        update_df = df[['time', 'prediction']]
        update_df['prediction'] = model.predict(X)
        # convert the columns to type float64 so mongodb can take them
        update_df.prediction = update_df.prediction.apply(float)
        # rectify negative values
        bad_index = update_df[update_df.prediction < 0].index
        update_df.loc[bad_index, 'prediction'] = 0
        update_db(farm, update_df, upsert=True)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    print(f'Done! Runtime: {runtime}')


if __name__ == '__main__':
    main()
