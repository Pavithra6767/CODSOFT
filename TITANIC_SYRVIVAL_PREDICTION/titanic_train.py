# titanic_train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 1. Load Dataset
# ===============================
DATA_PATH = r"C:\Users\sarit\TITANIC_SYRVIVAL_PREDICTION\Titanic-Dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

print("✅ Titanic dataset loaded successfully.")
df = pd.read_csv(DATA_PATH)

# ===============================
# 2. Basic Data Cleaning
# ===============================
df.drop(["Cabin", "Ticket", "Name"], axis=1, inplace=True, errors="ignore")

# Fill missing Age with median, Embarked with mode
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert categorical data
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# ===============================
# 3. Feature Engineering
# ===============================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ===============================
# 4. Split Features and Target
# ===============================
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 5. Train Model
# ===============================
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# ===============================
# 6. Evaluate
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n🎯 Model Evaluation:")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ===============================
# 7. Visualizations (EDA)
# ===============================
print("\n📊 Generating visualizations...")

sns.countplot(x='Survived', data=df)
plt.title("Overall Survival Count")
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

sns.histplot(df, x='Age', hue='Survived', multiple='stack')
plt.title("Age Distribution by Survival")
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# 8. Feature Importance
# ===============================
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

print("\n✅ Titanic Survival Prediction Completed Successfully!")


def add_features(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(['Mlle','Ms'], 'Miss')
    df["Title"] = df["Title"].replace(['Mme'], 'Mrs')
    df["Title"] = df["Title"].replace(['Don','Dona','Lady','Countess','Sir','Jonkheer','Capt','Col','Major','Rev','Dr'], 'Rare')
    return df


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f}")


    plt.show(block=False)
plt.pause(2)
plt.close()



# ===============================
# Confusion Matrix Visualization
# ===============================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix (raw values):")
print(cm)

# Plot confusion matrix (method 1 — built-in display)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Titanic Survival Prediction")
plt.show()

# Plot confusion matrix (method 2 — seaborn heatmap, optional)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Pred: No', 'Pred: Yes'], 
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title("Confusion Matrix - Titanic Survival Prediction (Seaborn)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("\n✅ Titanic Survival Prediction Task Completed Successfully!\n")







