
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Load Dataset
df = pd.read_csv("D:\cod soft project\creditcard.csv")

# Display Dataset
print("First 5 Rows:\n")

print(df.head())

print("\nDataset Shape:")

print(df.shape)

print("\nDataset Info:\n")

print(df.info())

# Check Missing Values
print("\nMissing Values:\n")

print(df.isnull().sum())

# Class Distribution
print("\nClass Distribution:\n")

print(df['Class'].value_counts())

# Visualize Fraud vs Genuine
plt.figure(figsize=(6,4))

df['Class'].value_counts().plot(kind='bar')

plt.xticks([0,1], ['Genuine', 'Fraud'])

plt.title("Fraud vs Genuine Transactions")

plt.xlabel("Transaction Type")

plt.ylabel("Count")

plt.show()

# Normalize Amount Column
scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(
    df['Amount'].values.reshape(-1, 1)
)

# Features and Target
X = df.drop('Class', axis=1)

y = df['Class']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# Print Results
print("\nModel Evaluation:\n")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))
# Predict Single Transaction
sample_transaction = X_test.iloc[0:1]

prediction = model.predict(sample_transaction)

print("\nSample Prediction:")

if prediction[0] == 0:
    print("Genuine Transaction")
else:
    print("Fraudulent Transaction")

# Scatter Plot
plt.figure(figsize=(6,4))

plt.scatter(df['Time'], df['Amount'])

plt.xlabel("Time")

plt.ylabel("Amount")

plt.title("Transaction Amount over Time")

plt.show()

print("\nCredit Card Fraud Detection Completed Successfully!")
