import numpy as np
import pandas as pd
# get data
import pandas_datareader as pdr
# visual
import matplotlib.pyplot as plt

#time
import datetime as datetime
#Prophet
from fbprophet import Prophet
from sklearn import metrics

start = datetime.datetime(2015,1,5)
end = datetime.datetime(2020,6,10)
# df_HSI = pdr.DataReader('0412.HK', 'yahoo', start=start, end=end)
df_HSI = pdr.DataReader('^HSI', 'yahoo', start=start, end=end)
# df_HSI = pdr.DataReader('^GSPC', 'yahoo', start=start, end=end)
df_HSI.head()

plt.style.use('ggplot')
df_HSI['Adj Close'].plot(figsize=(10, 6));

new_df_HSI = pd.DataFrame(df_HSI['Adj Close']).reset_index().rename(columns={'Date':'ds', 'Adj Close':'y'})
new_df_HSI.head()


# 定義模型
model = Prophet()

# 訓練模型
model.fit(new_df_HSI)

# 建構預測集
future = model.make_future_dataframe(periods=365) #forecasting for 1 year from now.

# 進行預測
forecast = model.predict(future)

forecast.head()

figure=model.plot(forecast)


df_HSI_close = pd.DataFrame(df_HSI['Adj Close'])
two_years = forecast.set_index('ds').join(df_HSI_close)
two_years = two_years[['Adj Close', 'yhat', 'yhat_upper', 'yhat_lower' ]].dropna().tail(800)
two_years['yhat']=np.exp(two_years.yhat)
two_years['yhat_upper']=np.exp(two_years.yhat_upper)
two_years['yhat_lower']=np.exp(two_years.yhat_lower)
two_years[['Adj Close', 'yhat']].plot(figsize=(8, 6));


two_years_AE = (two_years.yhat - two_years['Adj Close'])
two_years_AE.describe()

fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(two_years['Adj Close'])
ax1.plot(two_years.yhat)
ax1.plot(two_years.yhat_upper, color='black',  linestyle=':', alpha=0.5)
ax1.plot(two_years.yhat_lower, color='black',  linestyle=':', alpha=0.5)

ax1.set_title('Error between actual value and predit value')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')

plt.show()