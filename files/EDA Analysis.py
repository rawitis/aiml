import pandas as pd
import matplotlib.pyplot as py
import seaborn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("nyc.csv")
data

x = data[['Food']]
y = data[['Price']]

print(data.info())

print(data.corr())

py.boxplot(x)

py.boxplot(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

Lr = LinearRegression()
Lr.fit(x_train,y_train)
y_pred = Lr.predict(x_test)

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)
