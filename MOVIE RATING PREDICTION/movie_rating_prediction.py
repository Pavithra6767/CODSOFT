import os
print("Working dir:", os.getcwd())
print("Files here:", os.listdir('.'))

import pandas as pd

# Make sure this matches your file name exactly (including spaces and capitalization)
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

# Optional: print first few rows to verify it loaded correctly
print("Data loaded successfully!")
print(df.head())
print("\nColumns available:", df.columns.tolist())






import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Working dir:", os.getcwd())
print("Files here:", os.listdir('.'))

# --- Step 1: Load dataset safely ---
try:
    df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
    print("✅ Data loaded successfully!")
except Exception as e:
    print("❌ Error loading CSV:", e)
    exit()

print("\nColumns available:", list(df.columns))

# --- Step 2: Try to identify numeric columns ---
if 'Rating' not in df.columns:
    print("❌ Could not find 'Rating' column. Available:", df.columns)
    exit()

# Try to find something usable as numeric feature
possible_features = ['Votes', 'Duration', 'Year']
numeric_feature = None
for col in possible_features:
    if col in df.columns:
        numeric_feature = col
        break

if numeric_feature is None:
    print("❌ No numeric column found (like Votes, Duration, or Year).")
    exit()

print(f"Using '{numeric_feature}' as feature and 'Rating' as target.")

# --- Step 3: Clean numeric data ---
def clean_numeric(s):
    if isinstance(s, str):
        s = s.replace(",", "").replace("$", "").replace("M", "").strip()
    try:
        return float(s)
    except:
        return np.nan

df[numeric_feature] = df[numeric_feature].apply(clean_numeric)
df['Rating'] = df['Rating'].apply(clean_numeric)

# --- Step 4: Drop missing values ---
df = df.dropna(subset=[numeric_feature, 'Rating'])
print(f"✅ Rows after cleaning: {len(df)}")

if len(df) < 5:
    print("❌ Not enough data after cleaning. Please check your CSV values.")
    print(df[[numeric_feature, 'Rating']].head())
    exit()

# --- Step 5: Split data ---
X = df[[numeric_feature]]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 7: Predict and evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Step 8: Plot results ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual IMDb Rating")
plt.ylabel("Predicted IMDb Rating")
plt.title("Actual vs Predicted IMDb Ratings")
plt.grid(True)

plt.savefig("actual_vs_predicted.png")
print("\n✅ Plot saved as 'actual_vs_predicted.png'")



# --- Step 9: Save evaluation report ---
with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("MOVIE RATING PREDICTION REPORT\n")
    f.write("===============================\n")
    f.write(f"Total rows used: {len(df)}\n")
    f.write(f"Feature used: {numeric_feature}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n\n")
    f.write("✅ Model trained successfully!\n")
    f.write("Plot saved as: actual_vs_predicted.png\n")

print("✅ Evaluation report saved as 'evaluation_report.txt'")



# --- Step 10: Plot average rating per genre ---
import matplotlib.pyplot as plt

# Group by genre and calculate average rating
genre_rating = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
genre_rating.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Average IMDb Rating per Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save the plot
plt.savefig("genre_vs_rating.png")
plt.close()

print("✅ Genre vs Rating plot saved as 'genre_vs_rating.png'")



# --- Step 9: Save the trained model ---
import joblib

# Save the model to a file
joblib.dump(model, "movie_rating.model.pkl")

print("✅ Model saved as 'movie_rating.model.pkl'")

# --- Step 9: Save the trained model ---
import joblib

# Save the model to a file
joblib.dump(model, "movie_rating.model.pkl")

print("✅ Model saved as 'movie_rating.model.pkl'")

# --- Final confirmation message ---
print("\n🎯 All tasks completed successfully! ✅")













