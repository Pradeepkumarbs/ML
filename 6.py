import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Select required columns
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Convert categorical data into numerical data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# ------------------- 90-10 Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy with 90-10 Split:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# ------------------- 70-30 Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy with 70-30 Split:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
