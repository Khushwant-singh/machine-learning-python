from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import json 
from pathlib import Path

app = FastAPI(title = "Titanic Survival Prediction API ")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load the model and features
with open(Path("model/metadata.json")) as f:
    metadata = json.load(f)

MODEL_VERSION = metadata["current_version"]
MODEL_PATH = Path("model") / MODEL_VERSION

model = joblib.load(MODEL_PATH / "titanic_model.pkl")
features = joblib.load(MODEL_PATH / "titanic_features.pkl")

# class Passenger(BaseModel):
#     Pclass: int
#     Sex: int
#     Age: float
#     Fare: float
#     Embarked: int
#     FamilySize: int

# class Passenger(BaseModel):
#     Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
#     Sex: int = Field(..., ge=0, le=1, description="0 = male, 1 = female")
#     Age: float = Field(..., gt=0, le=100, description="Age in years")
#     Fare: float = Field(..., ge=0, le=600, description="Ticket fare")
#     Embarked: int = Field(..., ge=0, le=2, description="0=S, 1=C, 2=Q")
#     FamilySize: int = Field(..., ge=1, le=11, description="Family size (1–11)")

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., gt=0, le=100)
    Fare: float = Field(..., ge=0)
    Embarked: str
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)

    @validator("Sex")
    def validate_sex(cls, value):
        allowed = {"male", "female"}
        if value.lower() not in allowed:
            raise ValueError("Sex must be 'male' or 'female'")
        return value.lower()

    @validator("Embarked")
    def validate_embarked(cls, value):
        allowed = {"s", "c", "q"}
        if value.lower() not in allowed:
            raise ValueError("Embarked must be 'S', 'C', or 'Q'")
        return value.lower()


def preprocess_input(passenger: Passenger) -> pd.DataFrame:
    # Convert categorical variables

    # Endcode Sex
    sex_map = {"male": 0, "female": 1}
    sex_encoded = sex_map[passenger.Sex.lower()]

    # Endcode Embarked
    embarked_map = {"s": 0, "c": 1, "q": 2}
    embarked_encoded = embarked_map[passenger.Embarked.lower()]

    # Create FamilySize
    family_size = passenger.SibSp + passenger.Parch + 1


    # Create DataFrame
    input_df = pd.DataFrame(
        [[
            passenger.Pclass,
            sex_encoded,
            passenger.Age,
            passenger.Fare,
            embarked_encoded,
            family_size
        ]],
        columns=features
    )
    return input_df


@app.post("/predict")
def predict_survival(passenger: Passenger):
    input_df = preprocess_input(passenger)

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


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "description": metadata.get("description", ""),
        "created_at": metadata.get("created_at", "")
    }