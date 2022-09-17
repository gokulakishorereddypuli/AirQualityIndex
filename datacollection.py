""""
import pandas as pd
import os
df=pd.read_csv('https://data.gov.in/sites/default/files/all_india_PO_list_without_APS_offices_ver2_lat_long.csv')

save_path = 'files/datasets/'
df.to_csv(os.path.join(save_path,"all_india_city_pincodes.csv"))

print("success")


import os
import pgeocode
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/all_india_city_pincodes.csv')
nomi = pgeocode.Nominatim('in')
li=df['pincode'].tolist()
li= map(str, li)
li=list(li)
print(len(li))
x=nomi.query_postal_code(li)

save_path = 'files/datasets/'
x.to_csv(os.path.join(save_path,"allindia_cities.csv"))  

import pandas as pd
import os
pin=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/allindia_cities.csv')
pin.drop(columns=['country_code','community_name','county_code','community_code','accuracy','state_code'],inplace=True)
save_path = 'files/datasets/'
pin.to_csv(os.path.join(save_path,"allindia_cities.csv")) """

import os
import pandas as pd
import requests
import json
from datetime import date

pf=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/allindia_cities.csv')
pf['SO2']=0.0
pf['NO2']=0.0
pf['O3']=0.0
pf['PM2.5']=0.0
pf['PM10']=0.0
pf['CO']=0.0
pf['Date']=''
def get_data(pin,lat,lon):
    global pf, c
    pin=int(pin)
    print(pin,lat,lon)
    url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"

    querystring = {"lat":lat,"lon": lon}
    headers = {
      "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
      "X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data=response.text
    data=json.loads(data)
    pf.loc[pf["postal_code"] == pin, "Date"] =  date.today()
    pf.loc[pf["postal_code"] == pin, "PM10"] = data["PM10"]['concentration']
    pf.loc[pf["postal_code"] == pin, "PM2.5"] = data["PM2.5"]['concentration'] 
    pf.loc[pf["postal_code"] == pin, "CO"] =  data["CO"]['concentration']
    pf.loc[pf["postal_code"] == pin, "SO2"] =  data["SO2"]['concentration']
    pf.loc[pf["postal_code"] == pin, "O3"] = data["O3"]['concentration']
    pf.loc[pf["postal_code"] == pin, "N02"] = data["NO2"]['concentration'] 
    pf.loc[pf["postal_code"] == pin, "AQI"] = data['overall_aqi']
    save_path = 'files/datasets/'
    pf.to_csv(os.path.join(save_path,"latest_aqi_reports.csv"))
    """
    # https://api.weatherbit.io/v2.0/history/daily?postal_code=27601&country=US&start_date=2022-09-12&end_date=2022-09-13&key=API_KEY
    url='https://api.weatherbit.io/v2.0/current/airquality?&lat='+str(lat)+'&lon='+str(lon)+'&key=f0defabbf503444c8e4892c942e3f0d1'
    response = requests.request("GET", url, headers=headers, params=querystring)
    print(response.text)
    d=json.loads(response.text) 

    update = pd.DataFrame({'postal_code' : int(pin) ,'AQI':d['data'][0]['aqi'] ,'PM10':d['data'][0]['pm10'], 'PM2.5': d['data'][0]['pm25'] , 'CO': d['data'][0]['co'], 'SO2': d['data'][0]['so2'], 'O3': d['data'][0]['o3'],'NO2':d['data'][0]['no2'] }, index=[0])
    pf.loc[pf.postal_code == pin].update(update)"""
    
pf=pf[['postal_code','latitude','longitude']].apply(lambda x : get_data(*x),axis=1)

save_path = 'files/datasets/'
pf.to_csv(os.path.join(save_path,"latest_aqi_reports.csv"))