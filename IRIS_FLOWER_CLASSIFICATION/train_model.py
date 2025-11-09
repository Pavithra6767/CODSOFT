# train_model.py
# Iris Flower Classification - Model Training Script

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Dataset (from your IRIS.csv file)
df = pd.read_csv("IRIS.csv")

# Display dataset head
print("Dataset Loaded Successfully:")
print(df.head())

# Step 2: Split Features and Labels
X = df.drop("species", axis=1)
y = df["species"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build and Train Model
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# Step 8: Save Model and Scaler
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved: iris_model.pkl")
print("Saved: scaler.pkl")
print("\nTraining Completed Successfully!")
