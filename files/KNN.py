import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("Diabetic.csv")
data.head()

data.info()

data.describe()

plt.hist(data)

y = data["Outcome"].values
x = data.drop(["Outcome"],axis=1)

SS = StandardScaler()
data = SS.fit_transform(data)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

train_score = []
test_score = []
k_value = []
accuracy = []
y_p = []

for k in range(1,21):
    k_value.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    
    y_pred = knn.predict(x_test)
    y_p.append(y_pred)
    
    a_score = accuracy_score(y_test,y_pred)
    accuracy.append(a_score)
    
    tr_score = knn.score(x_train,y_train)
    train_score.append(tr_score)
    
    te_score = knn.score(x_test,y_test)
    test_score.append(te_score)

plt.xlabel('Different Values of K')
plt.ylabel('Model score')
plt.plot(k_value, train_score, color = 'r', label = "training score")
plt.plot(k_value, test_score, color = 'b', label = 'test score')

plt.xlabel('Different Values of K')
plt.ylabel('Accuracy score')
plt.plot(k_value, accuracy, color = 'r', label = "accuracy")

print("Maximum training accuracy at: ",train_score.index(max(train_score))+1)
print("Accuracy: ",max(train_score))

print("Maximum testing accuracy at: ",test_score.index(max(test_score))+1)
print("Accuracy: ",max(test_score))

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))
