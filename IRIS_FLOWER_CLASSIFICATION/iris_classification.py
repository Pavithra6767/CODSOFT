# iris_classification.py
# ✅ IRIS FLOWER CLASSIFICATION PROJECT

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1️⃣ Check working directory
print("Working dir:", os.getcwd())
print("Files here:", os.listdir('.'))

# 2️⃣ Load CSV safely
df = pd.read_csv("IRIS.csv", encoding="utf-8", skipinitialspace=True)
print("Data loaded. Columns:", list(df.columns))
print(df.head())

# 3️⃣ Normalize column names (fix spaces and case)
df.columns = df.columns.str.strip().str.lower()
print("\nNormalized column names:", df.columns.tolist())

# 4️⃣ Verify species column exists
if "species" not in df.columns:
    raise KeyError(f"❌ Column 'species' not found. Found: {df.columns.tolist()}")

# 5️⃣ Encode labels
le = LabelEncoder()
df["target"] = le.fit_transform(df["species"])

# 6️⃣ Define features and target
X = df.drop(columns=["species", "target"], errors="ignore")
y = df["target"]

# 7️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8️⃣ Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Predictions & metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc:.4f}")

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassification Report:\n", report)

# 🔟 Save evaluation report
with open("evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("🌸 IRIS FLOWER CLASSIFICATION REPORT 🌸\n\n")
    f.write(f"Model Accuracy: {acc:.4f}\n\n")
    f.write(report)
print("✅ Saved: evaluation_report.txt")

# 1️⃣1️⃣ Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Flower Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Saved: confusion_matrix.png")

# 1️⃣2️⃣ Save model
joblib.dump(model, "iris_flower_model.pkl")
print("✅ Saved: iris_flower_model.pkl")

# ✅ Final message
print("\n🎉 All tasks completed successfully!")
print("Generated files:")
print(" - evaluation_report.txt")
print(" - confusion_matrix.png")
print(" - iris_flower_model.pkl")





