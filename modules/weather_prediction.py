# codes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 

import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base import prediction
"""
df=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/data.csv',index_col='datetime',parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()
df
print(df['temp'].plot(figsize=(12,4)))

def ad_test(dataset):
  dftest=adfuller(dataset,autolag='AIC')
  print("1.ADF",dftest[0])
  print("2.P-value",dftest[1]) 
  print("3.Number of Lags:",dftest[2])
  print("4.Num of Observations used for adf  regression and critical values critical calculation:",dftest[3])
  print("5.Critical values :")
  for key,val in dftest[4].items():
    print('\t',key,":" ,val)

ad_test(df['temp'])

stepwise_fit=auto_arima(df["temp"],trace=True,supress_warnings=True)
stepwise_fit.summary()

print(df.shape)
train=df.iloc[:-30]
test=df.iloc[-30:]
print(train.shape,test.shape)


model=ARIMA(df['temp'], order=(2,0,2))
model=model.fit()
model.summary()

start=170
end=230
pred=model.predict(start=start,end=end,type='levels')
print(pred)

pred.plot(legend=True)
test['temp'].plot(legend=True)
 """