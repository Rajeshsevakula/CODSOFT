# Movie Rating Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("D:\cod soft project\IMDb Movies India.csv", encoding='latin1')

# Display Dataset
print(df.head())

print("\nColumns:\n")
print(df.columns)

# Remove Missing Target Values
df = df.dropna(subset=['Rating'])

# Clean Year Column

df['Year'] = df['Year'].astype(str).str.extract('(\d+)')

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Clean Duration Column

df['Duration'] = df['Duration'].astype(str).str.replace('min', '')

df['Duration'] = df['Duration'].str.strip()

df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Clean Votes Column
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')

df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Fill Missing Numerical Values
df['Year'] = df['Year'].fillna(df['Year'].mean())

df['Duration'] = df['Duration'].fillna(df['Duration'].mean())

df['Votes'] = df['Votes'].fillna(df['Votes'].mean())

# Fill Missing Categorical Values
categorical_columns = [
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3'
]

for col in categorical_columns:

    df[col] = df[col].fillna('Unknown')

# Encode Categorical Columns
le = LabelEncoder()

for col in categorical_columns:

    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop(['Name', 'Rating'], axis=1)

y = df['Rating']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Evaluation:\n")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R2 Score:", r2_score(y_test, y_pred))

# Visualization

# Rating Distribution
plt.figure(figsize=(6,4))

plt.hist(df['Rating'], bins=20)

plt.title("Movie Rating Distribution")

plt.xlabel("Rating")

plt.ylabel("Count")

plt.show()

# Actual vs Predicted Ratings
plt.figure(figsize=(6,4))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Ratings")

plt.ylabel("Predicted Ratings")

plt.title("Actual vs Predicted Ratings")

plt.show()
print("\nMovie Rating Prediction Completed Successfully!")
