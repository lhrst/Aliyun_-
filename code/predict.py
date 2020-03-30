import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt 
from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import time
from tqdm import tqdm

#取数据集
dataset = pd.DataFrame()
f = open('dataset_campus_competition.txt',mode='r')
s = f.readline()
while len(s)>0:
    w = [];
    for x in s[s.find('"')+1:s.find('\n')-1].split(','):
        if x == "NA":
            w.append(0)
        else:
            w.append(float(x));
    dataset[s[:s.find('"')-1]] = w
    s = f.readline()
f.close()

#设置时间项，假设是从2020年1月23日10am 武汉封城开始后的7天
b = datetime(2020,1,23, 10,0,0)
ind = []
for x in range(168):
    bi = b + pd.Timedelta(hours = x)
    ind.append(bi)
times = pd.DataFrame({'DS':ind})

#使用先知进行预测
def prophetprediction(s):
    myall = pd.DataFrame({'DS':ind,'Y':dataset.iloc[0:,s]})
    myall = myall.rename(columns={'DS':'ds', 'Y':'y'})
    mymean = myall['y'].mean()
    mystd = myall['y'].std()
    myall['y'] = (myall['y'] - mymean) / (mystd)
    train = myall.iloc[:,:]
    m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
    m.add_seasonality(name='daily', period=1, fourier_order=80)
    #m = Prophet(n_changepoints=1)
    m.fit(train)
    future = m.make_future_dataframe(periods=72, freq='H')
    forecast = m.predict(future)
    forecast['yhat'] = forecast['yhat']*mystd + mymean
    #fig1 = m.plot(forecast)
    out = pd.DataFrame({dataset.columns[s]:forecast['yhat'][-72:]})
    return out

#写入文件
file_handle = open('prediction.txt',mode='w')
for s in tqdm(range(120)):
    wanna = prophetprediction(s)
    file_handle.write(wanna.columns.tolist()[0]+ ' "')
    for i in wanna.iloc[:,0].tolist():
        if i < 0:
            i = 0
        if i != wanna.iloc[:,0].tolist()[len(wanna.iloc[:,0].tolist())-1]:
            file_handle.write('%.2f,'%(i))
        else:
            file_handle.write('%.2f'%(i))
    file_handle.write('"\n')
file_handle.close()