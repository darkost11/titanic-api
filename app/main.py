from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class Passenger(BaseModel):
    Age: float = Field(ge=0, le=120)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    SibSp: int = Field(ge=0)
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Embarked: Literal["C", "S", "Q"]


class Prediction(BaseModel):
    survived: Literal[0, 1]
    label: Literal["Survived", "Did not survive"]
    survival_probability: float = Field(ge=0, le=1)


MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Titanic Survival API")


@app.get("/")
def root():
    return {"message": "hello, it works"}


@app.post("/predict")
def predict(passenger: Passenger, response_model=Prediction):
    row = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(row)[0])

    result = {
        "survived": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive",
        "surival_probability": round(float(model.predict_proba(row)[0][1]), 4),
    }

    return result
