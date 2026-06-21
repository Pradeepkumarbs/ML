import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Glass dataset
glass_df = pd.read_csv('glass.csv')

# Features and Target
X = glass_df.drop('Type', axis=1)
y = glass_df['Type']

# 70-30 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Distance Metrics
distance_metrics = [
    ('Euclidean', 'euclidean'),
    ('Manhattan', 'manhattan')
]

for name, metric in distance_metrics:

    # KNN Classifier with k=3
    knn = KNeighborsClassifier(
        n_neighbors=3,
        metric=metric
    )

    # Train Model
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- KNN with {name} Distance ---")
    print("Accuracy =", acc)
    print("Confusion Matrix:")
    print(cm)
