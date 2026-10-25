import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote_plus


def prepare_df_date():
  df_date = pd.read_sql_query('SELECT * FROM DimDate', engine)
  df_date['Weekday'] = (df_date['Weekday']+5)%7

  df_date['Weekday_sin'] = np.sin(2 * np.pi *df_date['Weekday']/7)
  df_date['Weekday_cos'] = np.cos(2 * np.pi *df_date['Weekday']/7)


  df_date['Month_sin'] = np.sin(2 * np.pi *df_date['Month']/7)
  df_date['Month_cos'] = np.cos(2 * np.pi *df_date['Month']/7)

  df_date = df_date[['DateKey', 'DayOfYear', 'Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos']]


  return df_date

def prepare_df_weer():
  df_weer = pd.read_csv('aws_1day.csv')
  df_weer = df_weer[df_weer['the_geom'] == 'POINT (50.98 3.816)'] 
  df_weer['timestamp'] = df_weer['timestamp'].str[:4]+df_weer['timestamp'].str[5:7] + df_weer['timestamp'].str[8:10]
  df_weer['timestamp'] = pd.to_numeric(df_weer['timestamp'])
  df_weer['is_regen'] = df_weer['precip_quantity'] != 0

  df_weer = df_weer[['timestamp', 'sun_int_avg', 'sun_duration', 'temp_avg']]
  return df_weer

def prepare_df_activity():
  df_activity = pd.read_sql_query('SELECT * FROM DimActivity', engine)
  df_activity = df_activity[["ActivityKey", "CanonicalActivity", "IsCourse", 'IsPractical', 'IsExam','IsDigitalPossible']]
  df_activity = df_activity.drop_duplicates(['CanonicalActivity'])
  return df_activity

def prepare_test():
  #De csv werd door de docent meegeven. De volgende kollommen werden meegegeven: date, from, until, activity, subgroup, program, floor, roomcategory, capacity, area, credits
  test = pd.read_csv('lessen-subgroepen.csv', delimiter=";")
  test = test.merge(df_date, left_on='date', right_on='DateKey')
  test = test.merge(df_weer,left_on='date', right_on='timestamp' )
  test = test.merge(df_activity,left_on='activity', right_on='CanonicalActivity' )


  test['area'] = pd.to_numeric(test['area'].str.replace(',', '.'))
  test['from'] = pd.to_datetime(test['from'], format='%H%M')
  test['until'] = pd.to_datetime(test['until'], format='%H%M')

  test['ClassTime'] = (test['until'] - test['from']).dt.total_seconds() /60
  test['FromMinutes'] = test['from'].dt.hour * 60 + test['from'].dt.minute  
  test['UntillMinutes'] = test['until'].dt.hour * 60 + test['until'].dt.minute  

  test['FromHour_sin'] = np.sin(2 * np.pi *test['FromMinutes']/1440)
  test['FromHour_cos'] = np.cos(2 * np.pi * test['FromMinutes']/1440) 
  test['UntillHour_sin'] = np.sin(2 * np.pi * test['UntillMinutes']/1440)
  test['UntillHour_cos'] = np.cos(2*np.pi*test['UntillMinutes']/1440)   

  test['ClassTime'] = test['ClassTime'].astype('int64')
  test['DayOfYear_sin'] = np.sin(2 * np.pi * test['DayOfYear'] / 365)
  test['DayOfYear_cos'] = np.cos(2 * np.pi * test['DayOfYear'] / 365)

  test = test.drop(['DateKey', 'date', 'timestamp', 'DayOfYear', 'from', 'until', 'ActivityKey', 'activity'], axis=1)
  test = test.rename({'subgroup': 'SubgroupCode', 'program': 'ProgramName', 'floor': 'RoomFloor', 'roomcategory': 'Category',
                      'capacity': 'Capacity', 'area': 'Area', 'credits': 'ClassCredits'}, axis=1)

  test = test.reindex(['DayOfYear_sin', 'DayOfYear_cos','Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos', 'FromHour_sin', 'FromHour_cos', 'UntillHour_sin', 'UntillHour_cos','ClassTime' , 'CanonicalActivity', 'SubgroupCode', "IsCourse", 'IsPractical', 'IsExam', 'IsDigitalPossible',
                              'ProgramName', 'RoomFloor', 'Category', 'Capacity', 'Area', 'Ruimte', 'ClassCredits', 'is_regen', 'sun_int_avg', 'sun_duration', 'temp_avg', 'Aanwezigheidsgraad'], axis=1)
  return test


driver = '{ODBC Driver 17 for SQL Server}'
params = quote_plus(f"DRIVER={driver};SERVER=Pip-Boy\\MSSQLSERVER01;DATABASE=WifiDWH;Trusted_Connection=yes;")
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

df_date = prepare_df_date()
df_weer = prepare_df_weer()
df_activity = prepare_df_activity()
test =  prepare_test()

final_model = joblib.load("voting_regressor.pkl")
xgb = joblib.load("xgb_regressor.pkl")
rf = joblib.load("random_forest.pkl")

predict_final = final_model.predict(test)
test['resulaten'] = predict_final 
test['amount_students'] = test['resulaten'] * test['Capacity'] * 0.8 
print(test[['Capacity', 'resulaten', 'amount_students']])

predict_xgb = xgb.predict(test)
predict_rf = rf.predict(test)

print(predict_final)
#print(predict_xgb)
#print(predict_rf)

