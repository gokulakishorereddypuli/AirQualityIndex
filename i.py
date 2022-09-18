import os
import pandas as pd
import requests
import json
from datetime import date

df1=pd.read_csv('files/datasets/13000_latest_aqi_reports_2022-09-18.csv')
df2=pd.read_csv('files/datasets/17000_latest_aqi_reports_2022-09-18.csv')
df3=pd.read_csv('files/datasets/18000_latest_aqi_reports_2022-09-18.csv')
df4=pd.read_csv('files/datasets/19000_latest_aqi_reports_2022-09-18.csv')
df5=pd.read_csv('files/datasets/20000_latest_aqi_reports_2022-09-18.csv')
df6=pd.read_csv('files/datasets/21500_latest_aqi_reports_2022-09-18.csv')
df7=pd.read_csv('files/datasets/23000_latest_aqi_reports_2022-09-18.csv')
df8=pd.read_csv('files/datasets/26000_latest_aqi_reports_2022-09-18.csv')
df9=pd.read_csv('files/datasets/27500_latest_aqi_reports_2022-09-18.csv')
df10=pd.read_csv('files/datasets/29000_latest_aqi_reports_2022-09-18.csv')
df11=pd.read_csv('files/datasets/30000_latest_aqi_reports_2022-09-18.csv')

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
save_path = 'files/datasets/'
df.to_csv(os.path.join(save_path,str("30000")+"_latest_aqi_reports_"+str( date.today())+".csv"))
