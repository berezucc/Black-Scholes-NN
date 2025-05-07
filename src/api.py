from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

app = FastAPI()
model = load_model("models/model.pth")

class InputData(BaseModel):
    S: float
    K: float
    q: float
    r: float
    sigma: float
    t: float

@app.post("/predict")
def predict_endpoint(data: InputData):
    features = [data.S, data.K, data.q, data.r, data.sigma, data.t]
    price = predict(model, features)
    return {"predicted_price": float(price)}