
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras.models import load_model
import streamlit as st

st.title('Stock Trend Predication')
#Getting any ticker is impossible right now due to the problem encounter with yfinance and pandas_datareader
#It is here for show currently
user_input = st.text_input('Enter Stock Ticker', 'GOOG')

#Change file directory for the stock data csv "GOOG" based off where it is located on your computer should be in the same file as LSTM model
myInfo = pd.read_csv ('c:\\Users\\Patron\\Desktop\\Stock Trend Prediction\\GOOG.csv')
myInfo.head()

st.subheader('Data from 2014 - 2024')
st.write(myInfo.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(myInfo.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = myInfo.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(myInfo.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = myInfo.Close.rolling(100).mean()
ma200 = myInfo.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(myInfo.Close)
st.pyplot(fig)

#Splittign data into test and train
training=pd.DataFrame(myInfo['Close'][0:int(len(myInfo)*0.70)])
testing=pd.DataFrame(myInfo['Close'][int(len(myInfo)*0.70): int(len(myInfo))])

scaler = MinMaxScaler(feature_range=(0,1))

trainingArray=scaler.fit_transform(training)
testingArray=scaler.fit_transform(testing)

model= load_model('keras_model.h5')

past_100_days = training.tail(100)
final_df=pd.concat([past_100_days, testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicated=model.predict(x_test)
scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicated, 'r', label = 'Predicated Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)