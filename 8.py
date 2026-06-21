# Q: Write a program to perform unsupervised K-Means clustering technique.

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data  # all features

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Print cluster labels and centroids
print("K-means Labels:", labels.tolist())
print("\nK-means Centroids:\n", pd.DataFrame(centroids, columns=iris.feature_names))

# Plot (using 1st two features)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=60, c='red', marker='X', label='Centroids')  # smaller centroids
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.grid(True)
plt.show()
