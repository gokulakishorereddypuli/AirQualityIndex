import os
import pandas as pd
import requests
import json
from datetime import date

pf=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/allindia_cities.csv')
pf=pf.dropna()
pf['SO2']=0.0
pf['NO2']=0.0
pf['O3']=0.0
pf['PM2.5']=0.0
pf['PM10']=0.0
pf['CO']=0.0
pf['Date']=''
print(pf.head(2))
c=33001
def get_data(pin,lat,lon):
    global pf, c
    pin=int(pin)
    print(c,date.today(),pin,lat,lon)
    url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"

    querystring = {"lat":lat,"lon": lon}
    headers = {
      "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
      "X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data=response.text
    data=json.loads(data)
    try:
      pf.loc[pf["postal_code"] == pin, "Date"] =  date.today()
      pf.loc[pf["postal_code"] == pin, "PM2.5"] = data["PM2.5"]['concentration'] 
      pf.loc[pf["postal_code"] == pin, "CO"] =  data["CO"]['concentration']
      pf.loc[pf["postal_code"] == pin, "SO2"] =  data["SO2"]['concentration']
      pf.loc[pf["postal_code"] == pin, "O3"] = data["O3"]['concentration']
      pf.loc[pf["postal_code"] == pin, "N02"] = data["NO2"]['concentration'] 
      pf.loc[pf["postal_code"] == pin, "AQI"] = data['overall_aqi']
      pf.loc[pf["postal_code"] == pin, "PM10"] = data["PM10"]['concentration']
    except  Exception as e:
      print(data)
      print("Error-",e)
    c=c+1
    if(c>=34000):
      save_path = 'files/datasets/'
      pf.to_csv(os.path.join(save_path,str(c)+"_latest_aqi_reports_"+str( date.today())+".csv"))
    

pf=pf.iloc[33001:34001] 
pf=pf[['postal_code','latitude','longitude']].apply(lambda x : get_data(*x),axis=1)

"""
df1=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/13000_latest_aqi_reports_2022-09-18.csv')
df2=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/17000_latest_aqi_reports_2022-09-18.csv')
df3=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/18000_latest_aqi_reports_2022-09-18.csv')
df4=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/19000_latest_aqi_reports_2022-09-18.csv')
df5=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/20000_latest_aqi_reports_2022-09-18.csv')
df6=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/21500_latest_aqi_reports_2022-09-18.csv')
df7=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/23000_latest_aqi_reports_2022-09-18.csv')
df8=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/26000_latest_aqi_reports_2022-09-18.csv')
df9=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/27500_latest_aqi_reports_2022-09-18.csv')
df10=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/29000_latest_aqi_reports_2022-09-18.csv')
df11=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/30000_latest_aqi_reports_2022-09-18.csv')

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
print(df.shape)
x=df.to_csv(str("30000")+"_latest_aqi_reports_"+str( date.today())+".csv")
print(x)
"""