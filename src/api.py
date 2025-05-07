import redis
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

app = FastAPI()
model = load_model("models/model.pth")

# Connect to Redis
redis_host = os.environ.get("REDIS_HOST", "localhost")
r = redis.Redis(host=redis_host, port=6379, db=0)

class InputData(BaseModel):
    S: float
    K: float
    q: float
    r: float
    sigma: float
    t: float

@app.post("/predict")
def predict_endpoint(data: InputData):
    # Serialize input as a string key
    cache_key = json.dumps(data.dict(), sort_keys=True)
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    features = [data.S, data.K, data.q, data.r, data.sigma, data.t]
    price = predict(model, features)
    result = {"predicted_price": float(price)}
    r.set(cache_key, json.dumps(result))
    return result