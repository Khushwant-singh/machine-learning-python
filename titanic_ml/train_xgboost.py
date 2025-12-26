import pandas as pd
import joblib
import os
from xgboost import XGBClassifier

# ---------------------------
# Load Data
# ---------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ---------------------------
# Data Cleaning
# ---------------------------
train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(train["Age"].mean(), inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)

train["Embarked"].fillna("S", inplace=True)
test["Embarked"].fillna("S", inplace=True)

train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

# ---------------------------
# Feature Engineering
# ---------------------------
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

columns_to_drop = ["Name", "Ticket", "PassengerId", "SibSp", "Parch"]
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

# ---------------------------
# Prepare ML Data
# ---------------------------
X = train.drop("Survived", axis=1)
y = train["Survived"]

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

os.makedirs("model/v2", exist_ok=True)

joblib.dump(model, "model/v2/titanic_model.pkl")
joblib.dump(X.columns.tolist(), "model/v2/titanic_features.pkl")

print("✅ XGBoost model saved as v2")
# ---------------------------