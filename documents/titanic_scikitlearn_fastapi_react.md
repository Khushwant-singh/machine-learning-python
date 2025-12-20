🛳️ Titanic End-to-End Machine Learning Project

This document describes the complete workflow for building an end-to-end machine learning system using the Titanic dataset — from data preparation to serving predictions via FastAPI and consuming them in a React frontend.

📌 Project Goals

Learn data cleaning and feature engineering

Train a machine learning model

Save and version the trained model

Serve predictions via FastAPI

Add validation and preprocessing in the API

Build a simple React frontend

Prepare for future model improvements

📁 Project Structure
titanic_ml/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   ├── v1/
│   │   ├── titanic_model.pkl
│   │   └── titanic_features.pkl
│   └── metadata.json
│
├── train_model.py
├── app.py
├── requirements.txt
└── requests.http


Frontend (separate project):

titanic-ui/
├── src/
│   ├── App.js
│   └── PredictionForm.js
└── package.json

🧹 1. Data Preparation & Cleaning
1.1 Load Data
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

1.2 Handle Missing Values

Age → fill with mean

Fare → fill with median (test set)

Embarked → fill with most common value (S)

Cabin → drop column (too many missing values)

train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(train["Age"].mean(), inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)

train["Embarked"].fillna("S", inplace=True)
test["Embarked"].fillna("S", inplace=True)

train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

1.3 Feature Engineering

Create FamilySize:

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

1.4 Encode Categorical Features
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

1.5 Drop Unused Columns
columns_to_drop = ["Name", "Ticket", "PassengerId", "SibSp", "Parch"]
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

🤖 2. Model Training (Baseline)
2.1 Prepare Features & Target
X = train.drop("Survived", axis=1)
y = train["Survived"]

2.2 Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X, y)

2.3 Save Model & Feature Order
import joblib
import os

os.makedirs("model/v1", exist_ok=True)

joblib.dump(model, "model/v1/titanic_model.pkl")
joblib.dump(X.columns.tolist(), "model/v1/titanic_features.pkl")

🏷️ 3. Model Versioning
3.1 Metadata File
{
  "current_version": "v1",
  "description": "Baseline Logistic Regression model",
  "created_at": "2025-01-01"
}

3.2 Benefits

Safe upgrades (v2, v3, …)

Clear visibility of deployed model

No API code changes for new versions

🚀 4. FastAPI Backend
4.1 Initialize App
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Titanic Survival API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

4.2 Load Versioned Model
import json
from pathlib import Path
import joblib

with open("model/metadata.json") as f:
    metadata = json.load(f)

MODEL_VERSION = metadata["current_version"]
MODEL_PATH = Path("model") / MODEL_VERSION

model = joblib.load(MODEL_PATH / "titanic_model.pkl")
features = joblib.load(MODEL_PATH / "titanic_features.pkl")

🧾 5. Input Validation (Pydantic)
from pydantic import BaseModel, Field, validator

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., gt=0, le=100)
    Fare: float = Field(..., ge=0)
    Embarked: str
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)

    @validator("Sex")
    def validate_sex(cls, v):
        if v.lower() not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'")
        return v.lower()

    @validator("Embarked")
    def validate_embarked(cls, v):
        if v.lower() not in {"s", "c", "q"}:
            raise ValueError("Embarked must be 'S', 'C', or 'Q'")
        return v.lower()

🔄 6. Preprocessing Inside API
import pandas as pd

def preprocess_input(p: Passenger) -> pd.DataFrame:
    sex_map = {"male": 0, "female": 1}
    embarked_map = {"s": 0, "c": 1, "q": 2}

    family_size = p.SibSp + p.Parch + 1

    return pd.DataFrame([[
        p.Pclass,
        sex_map[p.Sex],
        p.Age,
        p.Fare,
        embarked_map[p.Embarked],
        family_size
    ]], columns=features)

🔮 7. Prediction Endpoint
@app.post("/predict")
def predict_survival(passenger: Passenger):
    X_input = preprocess_input(passenger)

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    return {
        "survived": int(prediction),
        "survival_probability": round(float(probability), 3)
    }

🖥️ 8. React Frontend (Summary)

Simple form (PredictionForm.js)

Sends POST request to /predict

Displays survival result and probability

Backend handles all logic & validation

🏁 Current Status

✔ End-to-end ML pipeline
✔ Backend API with validation
✔ Model versioning
✔ Frontend integration
✔ Ready for improvements

🔜 Next Steps

Add another ML algorithm (e.g. XGBoost)

Compare models

Upgrade version to v2

Optional: Docker & deployment

✅ END OF DOCUMENT