import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("Social_Network_Ads.csv")
data

x = data.iloc[:,[2,3]].values 
y = data.iloc[:,4].values
x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.25,random_state=1)


Scaler = StandardScaler()
x_train_scld = Scaler.fit_transform(x_train)
x_test_scld = Scaler.transform(x_test)

LR = LogisticRegression()
LR.fit(x_train_scld,y_train)
y_pred = LR.predict(x_test_scld)

Score = accuracy_score(y_test,y_pred)
print(Score)
