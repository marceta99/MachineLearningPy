# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

#select only two columns 3,4 for two variables Annual Income and Spending Score
#because we want to show this clusters visually in 2D, but I will not have to
# pick only two, this is only for teaching purposes so that we can plot 2D plot
X = dataset.iloc[:, [3, 4]].values
#and we dont have Y depending variable in clustering, so only X in clustering because
#in clustering we are grouping data in clusters and that is our outcome

# Using the elbow method to find the optimal number of clusters
# we will first create 10 KMeans models with number of clusters from 1-10 and then
# plot and see where elbow is, and choose best number of clusters to build KMeans model
# so here in for loop from 1-10 we will create 10 KMeans models with clusters from 1-10
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plot elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset with the number of clusters of 5 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

#we will use fit_predict method which not only trains K-Means model, but also returns
# clusters that are result of this model
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

 


