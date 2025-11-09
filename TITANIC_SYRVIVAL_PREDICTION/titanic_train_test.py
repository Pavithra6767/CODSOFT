# ---------------------------------------------------------------
# TITANIC SURVIVAL PREDICTION (Train + Test, Auto CSV Detection)
# ---------------------------------------------------------------
# Place these files in the same folder:
#   - Titanic-Dataset.csv
#   - Test.csv
# ---------------------------------------------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle

# ---------------------------------------------------------------
# STEP 1: File paths
# ---------------------------------------------------------------
TRAIN_PATH = r"C:\Users\sarit\TITANIC_SYRVIVAL_PREDICTION\Titanic-Dataset.csv"
TEST_PATH = r"C:\Users\sarit\TITANIC_SYRVIVAL_PREDICTION\Test.csv"

# ---------------------------------------------------------------
# STEP 2: Auto CSV Reader (handles ; , \t etc.)
# ---------------------------------------------------------------
def read_csv_auto(path):
    """Reads a CSV file with auto delimiter detection."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {os.path.abspath(path)}")

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        # try semicolon
        df = pd.read_csv(path, sep=';')
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        raise
    return df

train_df = read_csv_auto(TRAIN_PATH)
test_df = read_csv_auto(TEST_PATH)

print("✅ Files loaded successfully!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ---------------------------------------------------------------
# STEP 3: Preprocessing Function
# ---------------------------------------------------------------
def preprocess_data(df, is_train=True):
    df = df.copy()

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Fill missing values
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categorical data
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    if "Embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        df = pd.concat([df.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    if is_train:
        target_col = None
        for col in ["Survived", "survived", "Survive"]:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("❌ Target column 'Survived' not found in training data!")
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        return X, y
    else:
        return df

# ---------------------------------------------------------------
# STEP 4: Prepare Train/Test Data
# ---------------------------------------------------------------
X, y = preprocess_data(train_df, is_train=True)
test_processed = preprocess_data(test_df, is_train=False)

# Ensure test has same columns
for col in X.columns:
    if col not in test_processed.columns:
        test_processed[col] = 0
test_processed = test_processed[X.columns]

print("✅ Preprocessing complete!")
print("Train features:", len(X.columns))

# ---------------------------------------------------------------
# STEP 5: Train Model
# ---------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ---------------------------------------------------------------
# STEP 6: Evaluate Model
# ---------------------------------------------------------------
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)

print("\n🔹 Model Evaluation 🔹")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# ---------------------------------------------------------------
# STEP 7: Predict on Test Data
# ---------------------------------------------------------------
test_predictions = model.predict(test_processed)

if "PassengerId" in test_df.columns:
    output = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predictions
    })
else:
    output = pd.DataFrame({"Survived": test_predictions})

# ---------------------------------------------------------------
# STEP 8: Save Outputs
# ---------------------------------------------------------------
os.makedirs("titanic_output", exist_ok=True)

# Save model
with open("titanic_output/titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save predictions
output.to_csv("titanic_output/final_predictions.csv", index=False)
print("\n✅ All tasks completed successfully!")
print("📁 Outputs saved in 'titanic_output' folder:")
print(" - titanic_model.pkl")
print(" - final_predictions.csv")
