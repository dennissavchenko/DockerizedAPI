from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

# Initialize FastAPI app
app = FastAPI()


# Define the request body
class PredictionRequest(BaseModel):
    features: list


@app.get("/")
def root():
    return {"message": "Welcome to the Iris Predictive Model API!"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        input_data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
