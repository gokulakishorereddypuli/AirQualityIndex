import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime 
from datetime import timedelta
import datetime
from datetime import time
from datetime import date

# Flask app
import flask
from flask import Flask, render_template, request, redirect, url_for, session

# importing graphs
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# training models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix 



# importing modules
from modules.weather_apis import *
from modules.aqi_index_calculation import *
from modules.weather_prediction import *
from modules.aqi_api import *
#  from tempfile.timeseries import *
location1=''
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
# logo convertion
def weatherdays(test):
  j=0
  test['Day']=''
  for i in range(len(test['temp'])):
    if(j>7):
      break
    date=datetime.today()+timedelta(days=j)
    month=date.strftime('%b')
    da=date.strftime('%d')
    test['Day'][i]=month+" "+da
    if(test['temp'][i]<=24 and test['temp'][i]<=36):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=22 and test['temp'][i]<=34):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=21 and test['temp'][i]<=30):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=16 and test['temp'][i]<=32):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=16 and test['temp'][i]<=32):
      print(date," ",test['temp'][i],test['weather'][i])
    j+=1
  return test


app = Flask(__name__)
@app.route('/',methods=['GET','POST'])    
@app.route('/home')
def login():
    if flask.request.method == 'GET': 
        df=pd.read_csv('files/datasets/weather.csv')
        weather=weatherdays(df)
        df=df[['timestamp_local','temp']]
        df=df.rename(columns={'timestamp_local':'Timeline','temp':'Temperature'})
        fig = px.line(df, x="Timeline", y="Temperature",title="Weather Forecasting (Celsius)")
        graph_weather = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html',weather=weather.iloc[1:7],graph_weather=graph_weather,today=weather.iloc[0])
    else:
        print("x")




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



@app.route('/aqi',methods=['POST','GET'])
def aqi():
        global location1
    
        if flask.request.method == 'POST':  
            print(request.args.get('location'))
            x = [x for x in request.form.values()]
            print(x)
            location=x[0]
            location1=x[0]
            latitude=x[1]
            longitude=x[2]
            location=location.split(',')
            location=location[0]
            if(len(x[1])>0 and len(x[2])>0):
                location=x[0]
                latitude=x[1]
                longitude=x[2]
                #data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109}
                date = datetime.today()
                month=date.strftime('%b')
                da=date.strftime('%d')
                df=aqipredict(latitude,longitude)
                df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
                fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI "+location)
                fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration "+location)
                fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations "+location)
                fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations "+location)
                fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations "+location)
                fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations "+location)
                fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations "+location)
                graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
                graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
                graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
                graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
                graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
                graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
                graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
                # return render_template('graph.html',graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
                return render_template('aqi.html',data={'CO': {'max':df['CO'].max(), 'min':df['CO'].min(),'avg':df['CO'].mean() },
                                                         'NO2': {'max':df['NO2'].max(), 'min':df['NO2'].min(),'avg':df['NO2'].mean() }, 
                                                         'O3': {'max':df['O3'].max(), 'min':df['O3'].min(),'avg':df['O3'].mean() },
                                                         'SO2': {'max':df['SO2'].max(), 'min':df['SO2'].min(),'avg':df['SO2'].mean() },
                                                         'PM2.5': {'max':df['PM2.5'].max(), 'min':df['PM2.5'].min(),'avg':df['PM2.5'].mean() },
                                                         'PM10': {'max':df['PM10'].max(), 'min':df['PM10'].min(),'avg':df['PM10'].mean() } },
                                                         AQI={'max':df['AQI'].max(),'avg':df['AQI'].mean(),'min':df['AQI'].min()},month=month,date=da,graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
            else:
                return render_template('404.html')
        else:
            loc="Delhi"
            date = datetime.today()
            month=date.strftime('%b')
            da=date.strftime('%d')
            #df=pd.read_csv('files/datasets/aqi_predicted_hour_data.csv')
            # default delhi prediction
            df=aqipredict(28.7041,77.1025)
            df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
            fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI "+loc)
            fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration "+loc)
            fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations "+loc)
            fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations "+loc)
            fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations "+loc)
            fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations "+loc)
            fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations "+loc)
            graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
            graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
            graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
            # return render_template('graph.html',graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
            return render_template('aqi.html',location=loc,data={'CO': {'max':df['CO'].max(), 'min':df['CO'].min(),'avg':df['CO'].mean() },
                                                         'NO2': {'max':df['NO2'].max(), 'min':df['NO2'].min(),'avg':df['NO2'].mean() }, 
                                                         'O3': {'max':df['O3'].max(), 'min':df['O3'].min(),'avg':df['O3'].mean() },
                                                         'SO2': {'max':df['SO2'].max(), 'min':df['SO2'].min(),'avg':df['SO2'].mean() },
                                                         'PM2.5': {'max':df['PM2.5'].max(), 'min':df['PM2.5'].min(),'avg':df['PM2.5'].mean() },
                                                         'PM10': {'max':df['PM10'].max(), 'min':df['PM10'].min(),'avg':df['PM10'].mean() } },
                                                         AQI={'max':df['AQI'].max(),'avg':df['AQI'].mean(),'min':df['AQI'].min()},month=month,date=da,graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
"""
            print(e)
            loc=location1
            print(loc)
            loc=loc.split(',')
            loc=loc[0]
            date = datetime.today()
            month=date.strftime('%b')
            da=date.strftime('%d')
            print(e,"error occured")
            df=pd.read_csv('files\\datasets\\aqi_predicted_hour_data.csv')
            df=df.iloc[:99]
            df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
            fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI 1"+loc)
            fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration "+loc)
            fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations "+loc)
            fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations "+loc)
            fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations "+loc)
            fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations "+loc)
            fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations "+loc)
            graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
            graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
            graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('aqi.html',location=loc,data={'CO': {'max':df['CO'].max(), 'min':df['CO'].min(),'avg':df['CO'].mean() },
                                                         'NO2': {'max':df['NO2'].max(), 'min':df['NO2'].min(),'avg':df['NO2'].mean() }, 
                                                         'O3': {'max':df['O3'].max(), 'min':df['O3'].min(),'avg':df['O3'].mean() },
                                                         'SO2': {'max':df['SO2'].max(), 'min':df['SO2'].min(),'avg':df['SO2'].mean() },
                                                         'PM2.5': {'max':df['PM2.5'].max(), 'min':df['PM2.5'].min(),'avg':df['PM2.5'].mean() },
                                                         'PM10': {'max':df['PM10'].max(), 'min':df['PM10'].min(),'avg':df['PM10'].mean() } },
                                                         AQI={'max':df['AQI'].max(),'avg':df['AQI'].mean(),'min':df['AQI'].min()},month=month,date=da,graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
            
        """          
@app.route('/find-aqi-of-place',methods=['POST'])
def find_aqi():
    #print(request.get_data)
    x = [request.form['autocomplete'],request.form['latitude'],request.form['longitude']]
    print(x)
    if(len(x[1])>0 and len(x[2])>0):
        location=x[0]
        latitude=x[1]
        longitude=x[2]
        #data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109}
        date = datetime.today()
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
    df=pd.read_csv('files/datasets/aqi_predicted_hour_data.csv')
    df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
    fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI")
    fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration")
    fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations")
    fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations")
    fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations")
    fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations")
    fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations")
    graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
    graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
    graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
    graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
    graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
    graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
    graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('graph.html',graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)

if __name__ == "__main__":
    app.run(debug=True)