import pandas as pd
import numpy as np
import matplotlib.pyplot as py

train = pd.read_csv("usa_house.csv")

x_train = train[['avg_area_income']]
y_train = train[['house_price']]

py.scatter(x_train,y_train,c='blue')
py.xlabel('House price')
py.ylabel('Average income')

def val(x_train,y_train):
    theta1=theta0=0
    a=len(x_train)
    itr=1000
    lr=0.0000000001
    for i in range(itr):
        y_pre = theta1*x_train +theta0
        cost=(1/a) *sum ( [val**2 for val in (y_train-y_pre)])
        md= -(2/a) *sum (x_train*(y_train-y_pre))
        cd= -(2/a) *sum(y_train-y_pre)
        theta1=theta1-lr*md
        theta0=theta0-lr*cd
        print("m {}, {}, cost{},itr {}".format(theta1, theta0 ,cost, i))
    return theta1,theta0

py.scatter(x_train,y_train,c='blue')
py.xlabel('house price')
py.ylabel('average income')

theta1,theta0=val(x_train,y_train)
predicted = theta0 + theta1*x_train
py.scatter(x_train,y_train,c='blue')
py.plot(x_train,predicted,'-y')
py.xlabel('house price')
py.ylabel('average income')
py.show()
