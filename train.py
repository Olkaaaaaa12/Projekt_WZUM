import pandas as pd
import json
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

df_temperature = pd.read_csv("./WZUM_project_2020.12.20/office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)
df_target_temperature = pd.read_csv("./WZUM_project_2020.12.20/office_1_targetTemperature_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)
df_valve = pd.read_csv("./WZUM_project_2020.12.20/office_1_valveLevel_supply_points_data_2020-03-05_2020-03-19.csv", index_col=0, parse_dates=True)

df_temperature1 = pd.read_csv("./WZUM_project_2020.12.20/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv", index_col=0, parse_dates=True)
df_target_temperature1 = pd.read_csv("./WZUM_project_2020.12.20/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv", index_col=0, parse_dates=True)
df_valve1 = pd.read_csv("./WZUM_project_2020.12.20/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv", index_col=0, parse_dates=True)
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
df_temperature = pd.concat([df_temperature, df_temperature1])
df_target_temperature = pd.concat([df_target_temperature, df_target_temperature1])
df_valve = pd.concat([df_valve, df_valve1])
df_temperature.drop(columns=['unit'], inplace=True)
df_target_temperature.drop(columns=['unit'], inplace=True)
df_valve.drop(columns=['unit'], inplace=True)
df_valve.rename(columns={'value':'valve'}, inplace=True)
df_target_temperature.rename(columns={'value':'target'}, inplace=True)
df_temperature.rename(columns={'value':'temp'}, inplace=True)

df_temp1 = df_temperature[df_temperature['serialNumber'] == wall]
df_temp2 = df_temperature[df_temperature['serialNumber'] == window]
df_temp3 = df_temperature[df_temperature['serialNumber'] == mid]

df_temp1.rename(columns={'temp':'tempWall'}, inplace=True)
df_temp2.rename(columns={'temp':'tempWindow'}, inplace=True)
df_temp3.rename(columns={'temp':'tempMid'}, inplace=True)

df_temp1 = pd.concat([df_temp1, df_temp2, df_temp3, df_target_temperature, df_valve])

df_temp = df_temp1.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

df_temp['gt'] = df_temp['tempMid'].shift(-1, fill_value=20)
df_temp['gtValve'] = df_temp['valve'].shift(-1, fill_value=0)

df_temp["day_of_week"] = df_temp.index.dayofweek
df_temp["hour"] = df_temp.index.hour
maskday = (df_temp["day_of_week"] < 5) & (df_temp["hour"] > 3) & (df_temp["hour"] < 17)
df_temp = df_temp.loc[maskday]

mask = (df_temp.index < '2020-10-30')
df_train = df_temp.loc[mask]

maskTest = (df_temp.index >= '2020-10-30')
df_test = df_temp.loc[maskTest]

X_train = df_train[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
y_train = df_train['gt'].to_numpy()[1:-1]
X_test = df_test[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
y_test = df_test['gt'].to_numpy()[1:-1]

reg = ensemble.RandomForestRegressor(n_estimators=120)
reg.fit(X_train, y_train)
pickle.dump(reg, open('./models/reg.p', 'wb'))

y_pred = reg.predict(X_test)
print(metrics.mean_absolute_error(y_test, y_pred))

X_trainV = df_train[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
y_trainV = df_train['gtValve'].to_numpy()[1:-1]
X_testV = df_test[['tempMid', 'tempWall', 'tempWindow', 'target', 'valve']].to_numpy()[1:-1]
y_testV = df_test['gtValve'].to_numpy()[1:-1]

regValve = ensemble.GradientBoostingRegressor(n_estimators=80)
regValve.fit(X_trainV, y_trainV)
pickle.dump(regValve, open('./models/regValve.p', 'wb'))

y_predValve = regValve.predict(X_testV)
print(metrics.mean_absolute_error(y_testV, y_predValve))

df_test = df_test.iloc[1:-1, :]
df_test['pred'] = y_pred.tolist()
df_test['predValve'] = y_predValve.tolist()

temp = df_test.drop(columns=['valve', 'tempWall','tempWindow', 'target', 'day_of_week', 'hour', 'predValve', 'gtValve'])
valve = df_test.drop(columns=['tempMid','tempWall','tempWindow', 'target', 'day_of_week', 'hour', 'pred', 'gt'])
temp.plot()
valve.plot()
plt.show()