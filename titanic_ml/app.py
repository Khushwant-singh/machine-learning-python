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
    input_df = pd.DateFrame(
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
    prediction = model.predict(input_df)

    return{
        "survived": bool(prediction[0])
    }


#Health checkup
@app.get("/")
def health_check():
    return {"status": "API is running"}

