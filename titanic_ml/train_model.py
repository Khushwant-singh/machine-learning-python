import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ---------------------------
# Load Data
# ---------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ---------------------------
# Data Cleaning
# ---------------------------
# Age
train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(train["Age"].mean(), inplace=True)

# Fare
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# Embarked
train["Embarked"].fillna("S", inplace=True)
test["Embarked"].fillna("S", inplace=True)

# Drop Cabin
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

# Feature Engineering
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Encode categorical
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Drop unused columns
columns_to_drop = ["Name", "Ticket", "PassengerId", "SibSp", "Parch"]
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

# ---------------------------
# Prepare ML Data
# ---------------------------
X = train.drop("Survived", axis=1)
y = train["Survived"]

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ---------------------------
# Save Model
# ---------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/titanic_model.pkl")
joblib.dump(X.columns.tolist(), "model/titanic_features.pkl")

print("✅ Model and features saved successfully")
