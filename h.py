import os
import pandas as pd
import requests
import json
from datetime import date


df=pd.concat([pd.read_csv('a.csv'),
pd.read_csv('b.csv')
])
df.to_csv(str(df.shape)+"_latest_aqi_reports_"+str( date.today())+".csv")

