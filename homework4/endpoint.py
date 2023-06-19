#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print('reading file')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(year, month):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    print('reading finished')
    df = df[categorical]

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print('prediction finished')   
    # df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['prediction'] = y_pred
    print('ready to print', df['prediction'].mean())  
    # output_file = f'{year:04d}_{month:02d}_results.parquet'
    avg = df['prediction'].mean()
    return avg

    # df[['ride_id','prediction']].to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )
    # return y_pred

app = Flask('taxi-prediction')

@app.route('/app', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    res = predict(data['year'], data['month'])
    result = {
        'predictions': res
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

