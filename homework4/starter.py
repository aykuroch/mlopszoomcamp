#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip freeze | grep scikit-learn')

import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')



dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


import numpy as np


# STD of the predictions
np.std(y_pred)



df['year'] = df['tpep_pickup_datetime'].apply(lambda x: x.year)
df['month'] = df['tpep_pickup_datetime'].apply(lambda x: x.month)


# In[24]:


year = 2022
month = 2

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df['prediction'] = y_pred
output_file = '2022_02_results.parquet'

df[['ride_id','prediction']].to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)