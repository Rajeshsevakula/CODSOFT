# Iris Flower Classification Project

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Species Column
df['species'] = iris.target

# Convert Numbers to Flower Names
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Display Dataset
print("First 5 Rows:\n")

print(df.head())

print("\nDataset Shape:")

print(df.shape)

# Features and Target
X = df.drop('species', axis=1)

y = df['species']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")

print(accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# Visualization

# Sepal Length vs Petal Length
plt.figure(figsize=(6,4))

for species in df['species'].unique():

    subset = df[df['species'] == species]

    plt.scatter(
        subset['sepal length (cm)'],
        subset['petal length (cm)'],
        label=species
    )

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Petal Length (cm)")

plt.title("Iris Flower Classification")

plt.legend()

plt.show()

print("\nIris Flower Classification Completed Successfully!")
