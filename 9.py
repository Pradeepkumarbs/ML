# Q9: Write a program to perform agglomerative clustering based on single-linkage, complete-linkage criteria.

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # using all 4 features

# Apply Agglomerative Clustering
single = AgglomerativeClustering(n_clusters=3, linkage='single')
complete = AgglomerativeClustering(n_clusters=3, linkage='complete')

single_labels = single.fit_predict(X)
complete_labels = complete.fit_predict(X)

# Plotting both side by side
plt.figure(figsize=(12, 5))

# Single Linkage Plot
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=single_labels, cmap='viridis')
plt.title("Single Linkage")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid(True)

# Complete Linkage Plot
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=complete_labels, cmap='viridis')
plt.title("Complete Linkage")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid(True)

plt.tight_layout()
plt.show()
