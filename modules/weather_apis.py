import pandas as pd
import requests
from requests import request
import json
import os
import csv
from datetime import datetime
import datetime
def weatherbit(latitude,longitude):
    df=pd.read_csv('files/datasets/data.csv')
    print(df)
    df.to_csv("files/datasets/1.csv")
    return df

def rapidapi(location,latitude,longitude):
    url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"
    querystring = {"lat":latitude,"lon": longitude}
    headers = {
            "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
            "X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data=response.text
    data=json.loads(data)
    print(os.getcwd())
    json_data={"Date-Time":datetime.today(),"Location":location,"latitude":latitude,"longitude":longitude,
    	"CO": data['CO']['concentration'],"CO_aqi":   data['CO']['aqi'], 	
        "NO2": data['NO2']['concentration'], "NO2_aqi": data['NO2']['aqi'],	
        "O3" : data['O3']['concentration'], "O3_aqi":     data['O3']['aqi'],	
        "SO2": data['SO2']['concentration'], "SO2_aqi":    data['SO2']['aqi'],
        "PM2.5":data['PM2.5']['concentration'],"PM2.5_aqi":  data['PM2.5']['aqi'],	
        "PM10": data['PM10']['concentration'], "PM10_aqi":  data['PM10']['aqi'],	
        "overall_aqi": data['overall_aqi']
    }
    df= pd.DataFrame(json_data,index=[0])
    #df.to_csv('files/datasets/rapid.csv')
    #return data
    df1=pd.read_csv('files/datasets/rapid.csv')
    print(os.getcwd())
    df=pd.concat([df1,df])
    df=df[['Date-Time',	'Location'	,'latitude','longitude','CO','CO_aqi','NO2','NO2_aqi','O3','O3_aqi','SO2','SO2_aqi','PM2.5','PM2.5_aqi','PM10','PM10_aqi','overall_aqi']]
    #df=df.drop(['Unnamed: 0.1'],axis=0)
    os.remove("files/datasets/rapid.csv")
    df.to_csv('files/datasets/rapid.csv')
    return data

def histroy_rapidapi():
    return False
