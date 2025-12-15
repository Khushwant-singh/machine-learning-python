from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title = "Titanic Survival Prediction API ")

model = joblib.load("model/titanic_model.pkl")
features = joblib.load("model/titanic_features.pkl") 

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    Fare: float
    Embarked: int
    FamilySize: int


@app.post("/predict")
def predict_survival(passenger: Passenger):
    input_df = pd.DataFrame(
        [[
            passenger.Pclass,
            passenger.Sex,
            passenger.Age,
            passenger.Fare,
            passenger.Embarked,
            passenger.FamilySize
        ]],
        columns=features
    )

#make prediction
 # Get prediction (0 or 1)
    prediction = model.predict(input_df)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_df)[0]
    survival_probability = probabilities[1]  # probability of Survived = 1

    return {
        "survived": int(prediction),
        "survival_probability": round(float(survival_probability), 3)
    }


#Health checkup
@app.get("/")
def health_check():
    return {"status": "API is running"}

