from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generating Data

X,y = make_blobs(n_samples = 100, n_features = 1, centers = 2)
X_train, X_test, y_train, y_test = train_test_split(X,y)
'''
# Data
X_train = np.array([[2],
             [3],
             [2.5],
             [8],
             [9],
             [11],
             [5],
             [7]])

X_test = np.array([[7.5],[1],[6.5],[-7], [9], [5], [-4], [6], [5.5]])
'''
# Model (Pick Algorithm)
model = KMeans(n_clusters = 2) # n_clusters is the amount of groups.

# Training/Learning/Fitting
model.fit(X_train)

# Testing/Prediction
pred = model.predict(X_test) # prediction of main test data
trainpred = model.predict(X_train)
print(pred)
#print(model.predict(X_test))
#print(model.predict(X_train))
trainzeros = len(X_train) *[0]
testzeros = len(X_test) * [0]
clustercenters = model.cluster_centers_
centerzeros = len(clustercenters) * [0]
# You use c when you have a list of numbers for colors. You use color when you have a specific color.
plt.scatter(X_train, trainzeros, label = "Train Data", s = 30, ) # Training data is the circles.
plt.scatter(clustercenters, centerzeros , label = "Cluster Centers for Training Data", s = 120, color = "red", alpha = .3)
plt.scatter(X_test, testzeros, label = "Test Data", marker = "*", s = 80, c = pred, cmap = "cool") # Testing data is the stars.
plt.legend() # The legend shows that the stars are part of the test data and that the circles are part of the train data
