import math
import numpy as np 
import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
wheatStr = 'Wheat | 00000015 || Yield | 005419 || tonnes per hectare'
df = pd.read_csv('ind.csv')

# to convert "Year" from string to a date time for better time series forecasting
df.index = pd.to_datetime(df['Year'], format='%Y')
del df['Year']
# print(df.head())

# to plot the relation between the temperature and the temperature
sns.set()
plt.ylabel('Temperature')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.plot(df.index, df['Temp'], )
plt.show()


# X = df[['Year', 'Temp']]
# y = df['Wheat | 00000015 || Yield | 005419 || tonnes per hectare']
# print(X)
# print(y)
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# predictedCO2 = regr.predict([[2021, 23]])
# print(predictedCO2)