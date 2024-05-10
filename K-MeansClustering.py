from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generating Data
## make_blobs creates data to train on
## n_samples is how many rows of data that it generates ##
## n_features is the amount of columns of data that it generates. ##
## centers is to say how many groups you want your data to be split into. ##
## cluster_std is to determine how spread apart your data is. the closer it is to 0, the larger the gaps between the number of clusters will be. 
## This causes the points of both clusters to be closer together.##
X,y = make_blobs(n_samples = 100, n_features = 1, centers = 2, cluster_std = .7) 

# Splitting data
## This splits the data from the previous line into testing and training data. 
## It usually does this by putting 75% of the data from the previous line for training and the rest for testing.
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Model (Pick Algorithm)
model = KMeans(n_clusters = 2) # n_clusters is the amount of groups.

# Training/Learning/Fitting
model.fit(X_train) ## !!! K-Means Clustering is unsupervised machine learning algorithm, so it does not use y-train values (y-train already contains the real answers). !!! 

# Testing/Prediction
pred = model.predict(X_test) # prediction of main test data
trainpred = model.predict(X_train)
print(pred)
trainzeros = len(X_train) *[0]
testzeros = len(X_test) * [0]
clustercenters = model.cluster_centers_
centerzeros = len(clustercenters) * [0]

# Plotting
# You use c when you have a list of numbers for colors. You use color when you have a specific color.
plt.scatter(X_train, trainzeros, label = "Train Data", s = 30, ) # Training data is the circles.
plt.scatter(clustercenters, centerzeros , label = "Cluster Centers for Training Data", s = 200, color = "red", alpha = .5)
plt.scatter(X_test, testzeros, label = "Test Data", marker = "*", s = 80, c = pred, cmap = "cool") # Testing data is the stars.
plt.legend() # The legend shows that the stars are part of the test data and that the circles are part of the train data
