# ## installation
# conda create -n spider-env -y
# conda activate spider-env
# conda install pandas -y
# conda install spyder-kernels scikit-learn -y
# now change def env to this env by
# tools > preferences > change to the new env
# restart kernel
# -----------------------
# to convert "Year" from string to a date time for better time series forecasting
# df.index = pd.to_datetime(df['Year'], format='%Y')
# del df['Year']

import numpy as np
import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

wheatStr = 'Wheat'
df = pd.read_csv(r'C:\Users\sreya\Desktop\fcs-spyder\ind.csv')
df.columns = ['Entity','Code','Temp','Year','Wheat','Rice','Maize','Soybeans','Potatoes','Beans','Peas','Cassava','Cocoa','','Barley']

def plotYearProd(x, y):
    sns.set()
    plt.ylabel('production')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.plot(x, y, )
    plt.show()

def lrBasic():
    X = df[['Year', 'Temp']]
    y = df[wheatStr]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    lr = linear_model.LinearRegression()
    lr.fit(x_train, y_train)
    # predictedCO2 = regr.predict([[2021, 23], [2020, 24]])
    # print(regr.coef_)
    y_test_pred = lr.predict(x_test)
    y_dash = [*y_train, *y_test_pred]
    plotYearProd(df['Year'], y_dash)
    return lr.score(x_test, y_test)

def lrNormalised():
    sc = StandardScaler()
    X = df[['Year', 'Temp']]
    y = df[wheatStr]
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    lr = linear_model.LinearRegression()
    lr.fit(x_train, y_train)
    return lr.score(x_test, y_test)

df.index = pd.to_datetime(df['Year'], format='%Y')
del df['Year']
del df['Code']
del df['Entity']
for i in df.select_dtypes('object').columns:
   le = LabelEncoder().fit(df[i])
   df[i] = le.transform(df[i]) 
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(df[['Temp']])
Y_data = Y_scaler.fit_transform(df[[wheatStr]])


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
     X = []
     y = []
     start = start + window
     if end is None:
         end = len(dataset) - horizon
     for i in range(start, end):
         indices = range(i-window, i)
         X.append(dataset[indices])
         indicey = range(i+1, i+1+horizon)
         y.append(target[indicey])
     return np.array(X), np.array(y) 

hist_window = 48
horizon = 1
TRAIN_SPLIT = 30000
x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon) 

# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# print(reframed.head())

# plotYearProd(df['Year'], df[wheatStr])
# print("LR Basic : " + str(lrBasic()))
# print("LR Normalised : " + str(lrNormalised()))