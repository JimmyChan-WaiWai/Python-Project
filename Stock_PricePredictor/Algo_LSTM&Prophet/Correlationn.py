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

df_HSI = pdr.DataReader('^HSI', 'yahoo', start=start, end=end)
df_HSI.head()
df_GSPC = pdr.DataReader('^GSPC', 'yahoo', start=start, end=end)
df_GSPC.head()


plt.style.use('ggplot')
df_HSI['Adj Close'].plot(figsize=(10, 6));
df_GSPC['Adj Close'].plot(figsize=(10, 6));

new_df_HSI = pd.DataFrame(df_HSI['Adj Close']).reset_index().rename(columns={'Date':'ds', 'Adj Close':'HSI'})
new_df_HSI.head()
new_df_GSPC = pd.DataFrame(df_GSPC['Adj Close']).reset_index().rename(columns={'Date':'ds', 'Adj Close':'GSPC'})
new_df_GSPC.head()

df = pd.merge(new_df_HSI, new_df_GSPC, how='outer', on=['ds'])


corr_matrix = df[['HSI','GSPC']].corr()
print(corr_matrix)

plt.show()