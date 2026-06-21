# - Write a program to develop Linear Discriminant Analysis (LDA) algorithms.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def lda_analysis():
    data = load_iris()
    X = data.data
    y = data.target

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    print("Shape of Data:", X.shape)
    print("Shape of transformed Data:", X_lda.shape)
    

    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
    plt.title("LDA Projection")
    plt.show()

lda_analysis()
