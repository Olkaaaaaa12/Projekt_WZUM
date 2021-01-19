import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple
import json

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:
    with Path('./models/reg.p').open('rb') as reg_file:
        reg = pickle.load(reg_file)

    with Path('./models/regValve.p').open('rb') as regV_file:
        regValve = pickle.load(regV_file)

    with open('./additional/additional_info.json') as f:
        add_data = json.load(f)

    devices = add_data['offices']['office_1']['devices']
    for d in devices:
        if d['description'] == "temperature_wall":
            wall = d['serialNumber']
        elif d['description'] == "temperature_window":
            window = d['serialNumber']
        elif d['description'] == "temperature_middle":
            mid = d['serialNumber']
        else:
            continue

    temperature.drop(columns=['unit'], inplace=True)
    target_temperature.drop(columns=['unit'], inplace=True)
    valve_level.drop(columns=['unit'], inplace=True)
    valve_level.rename(columns={'value': 'valve'}, inplace=True)
    target_temperature.rename(columns={'value': 'target'}, inplace=True)
    temperature.rename(columns={'value': 'temp'}, inplace=True)

    df_temp1 = temperature[temperature['serialNumber'] == wall]
    df_temp2 = temperature[temperature['serialNumber'] == window]
    df_temp3 = temperature[temperature['serialNumber'] == mid]

    df_temp1.rename(columns={'temp': 'tempWall'}, inplace=True)
    df_temp2.rename(columns={'temp': 'tempWindow'}, inplace=True)
    df_temp3.rename(columns={'temp': 'tempMid'}, inplace=True)

    df_temp1 = pd.concat([df_temp1, df_temp2, df_temp3, target_temperature, valve_level])

    df_temp = df_temp1.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    df_temp['gt'] = df_temp['tempMid'].shift(-1, fill_value=20)
    df_temp['gtValve'] = df_temp['valve'].shift(-1, fill_value=0)

    df_temp["day_of_week"] = df_temp.index.dayofweek
    df_temp["hour"] = df_temp.index.hour
    maskday = (df_temp["day_of_week"] < 5) & (df_temp["hour"] > 3) & (df_temp["hour"] < 17)
    df_temp = df_temp.loc[maskday]

    X_train = df_temp[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
    y_train = df_temp['gt'].to_numpy()[1:-1]

    X_test = [df_temp[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[-1]]

    X_trainV = df_temp[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
    y_trainV = df_temp['gtValve'].to_numpy()[1:-1]

    X_testV = [df_temp[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[-1]]

    reg.set_params(n_estimators=240, warm_start=True)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    regValve.set_params(n_estimators=160, warm_start=True)
    regValve.fit(X_trainV, y_trainV)

    y_predV = regValve.predict(X_testV)

    return y_pred, y_predV
