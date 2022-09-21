import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from modules.weather_apis import *
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix 
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, session
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# importing modules


"""
dataset={'city_day':'https://drive.google.com/file/d/158j8UBocM-wzIF29fsiBVAmfwQA2JVIV/view?usp=sharing',
         'city_hour' :'https://drive.google.com/file/d/1vNRx81y6CehUR81t9oNiyirrE3F7Rwzj/view?usp=sharing',
         'station_day':'https://drive.google.com/file/d/1M6oxdKEjNflQh4euBTsFCslmXOjBi1wq/view?usp=sharing',
         'station_hour':'https://drive.google.com/file/d/1QyQY6vv4Ul_wsw69ZBsZrn69uD7Q8Vti/view?usp=sharing',
         'stations':'https://drive.google.com/file/d/1gP49xt_l-B2R3LNKBbPdCmje8fYmzr_T/view?usp=sharing'}

## defining dataset paths

PATH_STATION_DAY = 'https://drive.google.com/uc?id=' + dataset['station_day'].split('/')[-2]
PATH_STATION_HOUR = 'https://drive.google.com/uc?export=download&confirm=CONFIRM_CODE&id=1QyQY6vv4Ul_wsw69ZBsZrn69uD7Q8Vti'
PATH_CITY_HOUR = 'https://drive.google.com/uc?id=' + dataset['city_hour'].split('/')[-2]
PATH_CITY_DAY = 'https://drive.google.com/uc?id=' + dataset['city_day'].split('/')[-2]
PATH_STATIONS = 'https://drive.google.com/uc?id=' + dataset['stations'].split('/')[-2]
STATIONS = ["KL007", "KL008"]

## importing data and subsetting the station
df = pd.read_csv(PATH_STATION_HOUR, parse_dates = ["Datetime"])
stations = pd.read_csv(PATH_STATIONS)

df = df.merge(stations, on = "StationId")

df = df[df.StationId.isin(STATIONS)]
df.sort_values(["StationId", "Datetime"], inplace = True)
df["Date"] = df.Datetime.dt.date.astype(str)
df.Datetime = df.Datetime.astype(str)
df.fillna(0,inplace=True)

df["PM10_24hr_avg"] = df.groupby("StationId")["PM10"].rolling(window = 24, min_periods = 16).mean().values
df["PM2.5_24hr_avg"] = df.groupby("StationId")["PM2.5"].rolling(window = 24, min_periods = 16).mean().values
df["SO2_24hr_avg"] = df.groupby("StationId")["SO2"].rolling(window = 24, min_periods = 16).mean().values
df["NOx_24hr_avg"] = df.groupby("StationId")["NOx"].rolling(window = 24, min_periods = 16).mean().values
df["NH3_24hr_avg"] = df.groupby("StationId")["NH3"].rolling(window = 24, min_periods = 16).mean().values
df["CO_8hr_max"] = df.groupby("StationId")["CO"].rolling(window = 8, min_periods = 1).max().values
df["O3_8hr_max"] = df.groupby("StationId")["O3"].rolling(window = 8, min_periods = 1).max().values

## PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0
## PM10 Sub-Index calculation
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0
## SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0


## NH3 Sub-Index calculation
def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    else:
        return 0


## CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

## O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0


## AQI bucketing
def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN

df["SO2_SubIndex"] = df["SO2_24hr_avg"].apply(lambda x: get_SO2_subindex(x))
df["NOx_SubIndex"] = df["NOx_24hr_avg"].apply(lambda x: get_NOx_subindex(x))
df["O3_SubIndex"] = df["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))
df["CO_SubIndex"] = df["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))
df["PM10_SubIndex"] = df["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))
df["PM2.5_SubIndex"] = df["PM2.5_24hr_avg"].apply(lambda x: get_PM25_subindex(x))
df["NH3_SubIndex"] = df["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))

df["Checks"] = (df["PM2.5_SubIndex"] > 0).astype(int) + \
                (df["PM10_SubIndex"] > 0).astype(int) + \
                (df["SO2_SubIndex"] > 0).astype(int) + \
                (df["NOx_SubIndex"] > 0).astype(int) + \
                (df["NH3_SubIndex"] > 0).astype(int) + \
                (df["CO_SubIndex"] > 0).astype(int) + \
                (df["O3_SubIndex"] > 0).astype(int)

df["AQI_calculated"] = round(df[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NOx_SubIndex",
                                 "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))
df.loc[df["PM2.5_SubIndex"] + df["PM10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
df.loc[df.Checks < 3, "AQI_calculated"] = np.NaN

df["AQI_bucket_calculated"] = df["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))
df[~df.AQI_calculated.isna()].head(13)


df_station_hour = df
df_station_day = pd.read_csv(PATH_STATION_DAY)

df_station_day = df_station_day.merge(df.groupby(["StationId", "Date"])["AQI_calculated"].mean().reset_index(), on = ["StationId", "Date"])
df_station_day.AQI_calculated = round(df_station_day.AQI_calculated)


df_city_hour = pd.read_csv(PATH_CITY_HOUR)
df_city_day = pd.read_csv(PATH_CITY_DAY)

df_city_hour["Date"] = pd.to_datetime(df_city_hour.Datetime).dt.date.astype(str)

df_city_hour = df_city_hour.merge(df.groupby(["City", "Datetime"])["AQI_calculated"].mean().reset_index(), on = ["City", "Datetime"])
df_city_hour.AQI_calculated = round(df_city_hour.AQI_calculated)

df_city_day = df_city_day.merge(df_city_hour.groupby(["City", "Date"])["AQI_calculated"].mean().reset_index(), on = ["City", "Date"])
df_city_day.AQI_calculated = round(df_city_day.AQI_calculated)


df_check_station_hour = df_station_hour[["AQI", "AQI_calculated"]].dropna()
df_check_station_day = df_station_day[["AQI", "AQI_calculated"]].dropna()
df_check_city_hour = df_city_hour[["AQI", "AQI_calculated"]].dropna()
df_check_city_day = df_city_day[["AQI", "AQI_calculated"]].dropna()

df1=pd.read_csv(PATH_STATIONS)
df=pd.merge(df1,df,on='StationId')  

df = df.dropna()  

df=pd.read_csv('files/datasets/aqi_data.csv')
X=df[['PM2.5_SubIndex','PM10_SubIndex','SO2_SubIndex', 'NOx_SubIndex', 'NH3_SubIndex', 'CO_SubIndex','O3_SubIndex',]]
Y=df[['AQI_calculated']]
X.tail(10)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

RF=RandomForestRegressor().fit(X_train,Y_train)
train_preds1=RF.predict(X_train)
test_preds1=RF.predict(X_test)
RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train, train_preds1)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
print("RMSE TrainingData ", str(RMSE_train))
print("RMSE TestData", str(RMSE_test))
print('-'*50)
print('RSquared value on train:',RF.score (X_train, Y_train))
print('RSquared value on test:',RF.score (X_test, Y_test))

import requests
import json
url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"

querystring = {"city":"new delhi"}

headers = {
	"X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
	"X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)
data=response.text
data=json.loads(data)
print(data["CO"]['concentration'])

"""

#  df.to_csv("aqi_data.csv")

app = Flask(__name__)
@app.route('/')    
@app.route('/home')
def login():
    return render_template('index.html')
@app.route('/news')    
def news():
    return render_template('news.html')
@app.route('/contact')    
def contact():
    return render_template('contact.html')
@app.route('/live-cameras')    
def live_cameras():
    return render_template('live-cameras.html')
@app.route('/photos')    
def photos():
    return render_template('photos.html')
@app.route('/aqi')
def aqi():
    date = datetime.datetime.now()
    month=date.strftime('%b')
    da=date.strftime('%d')
 
    return render_template('aqi.html',data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109},month=month,date=da)
@app.route('/find-aqi-of-place',methods=['POST'])
def find_aqi():
    x = [x for x in request.form.values()]
    print(x)
    if(len(x[1])>0 and len(x[2])>0):
        location=x[0]
        latitude=x[1]
        longitude=x[2]
        #data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109}
        date = datetime.datetime.now()
        month=date.strftime('%b')
        da=date.strftime('%d')
        #print('-------------------========================================================-======',date,month,da)
        
    
        return render_template('aqi.html',data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109},month=month,date=da)
    return render_template('404.html')
@app.route('/404')
def notfound_404():
    return render_template('404.html')
@app.route('/graph-view')
def graph():

    df=weatherbit(1,2)
    fig = px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI of Delhi")
    fig1 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentrations of Delhi")
    graphJSON = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
    graph1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('graph.html', graphJSON=graphJSON,graph=graph1)

    

if __name__ == "__main__":
    app.run(debug=True)