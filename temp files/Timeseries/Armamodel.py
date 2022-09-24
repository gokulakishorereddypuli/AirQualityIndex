
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import datetime
from datetime import time
from datetime import date

df=pd.read_csv('files/datasets/weather.csv',index_col='datetime',parse_dates=True)
df=df.dropna()

"""
#print('Shape of data',df.shape)
#df.head()
import warnings
warnings.filterwarnings("ignore")
#training the data
train=df.iloc[:-30]
test=df.iloc[-30:]
model=ARIMA(df['temp'], order=(2,0,2))
model=model.fit()
model.summary()
from statsmodels.tsa.base import prediction
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,type='levels')

pred.plot( figsize=(12,4))
pred.plot(legend=True)
test['temp'].plot(legend=True)
test['temp']
j=0
for i in range(len(test['temp'])):
    if(j>7):
      break
    date=datetime.datetime.now()+timedelta(days=j)
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


"""
def weatherlog(test):
  j=0
  test['Day']=''
  for i in range(len(test['temp'])):
    if(j>7):
      break
    date=datetime.datetime.now()+timedelta(days=j)
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

d=weatherlog(df)
print(d)