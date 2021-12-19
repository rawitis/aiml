import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3 ,4]].values

# Using the elbow method to find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
      kmeans = KMeans(n_clusters=i, init ='k-means++', n_init=10, max_iter =300 , random_state =None)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=5, init ='k-means++', n_init=10, max_iter =300 , random_state =None)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the Clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=10, c='red', label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=10, c='blue', label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=10, c='green', label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=10, c='cyan', label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=10, c='magenta', label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30, c='yellow', label='Centroid')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('spending Score(1-100)')
plt.legend()
plt.show()
