from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained models
logistic_model = joblib.load('logistic_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    recency_woe: float
    frequency: float
    monetary: float
    rfms_score: float

# Define the prediction endpoint for Logistic Regression
@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    input_data = np.array([[data.recency_woe, data.frequency, data.monetary, data.rfms_score]])
    prediction = logistic_model.predict(input_data)
    return {"prediction": int(prediction[0])}

# Define the prediction endpoint for Random Forest
@app.post("/predict/random-forest")
def predict_random_forest(data: InputData):
    input_data = np.array([[data.recency_woe, data.frequency, data.monetary, data.rfms_score]])
    prediction = random_forest_model.predict(input_data)
    return {"prediction": int(prediction[0])}