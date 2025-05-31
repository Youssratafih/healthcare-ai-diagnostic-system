from fastapi import FastAPI
from schemas.DiabetsInput import DiabetesInput  
import numpy as np
import joblib
app = FastAPI()

# Charger le modèle une seule fois
model = joblib.load("/datafiles/diabetes_model.pkl")
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

# Autorise les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou ["*"] pour tout autoriser (dev uniquement)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predictdiabete")
async def predict_diabetes(data: DiabetesInput):
    input_array = np.array([[
        data.pregnancies,
        data.glucose,
        data.bloodPressure,
        data.skinThickness,
        data.insulin,
        data.bmi,
        data.diabetesPedigreeFunction,
        data.age
    ]])

    prediction = model.predict(input_array)[0]

    if prediction == 1:
        return {"result": "Ce patient est diabétique."}
    else:
        return {"result": "Ce patient n'est pas diabétique."}
