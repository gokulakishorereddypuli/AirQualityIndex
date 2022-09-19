import os
import pandas as pd
import requests
import json
from datetime import date


df=pd.concat([pd.read_csv('9000_latest_aqi_reports_2022-09-17.csv'),
pd.read_csv('30000_latest_aqi_reports_2022-09-18.csv'),
pd.read_csv('31500-69001_latest_aqi_reports_2022-09-19.csv')

])
df.to_csv(os.path.join(str(df.shape)+"_latest_aqi_reports_"+str( date.today())+".csv"))

