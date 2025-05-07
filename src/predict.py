import torch
import numpy as np
from src.model import baseline_model

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = baseline_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, input_features):
    # Ensure input_features is already normalized/scaled
    input_tensor = torch.tensor([input_features], dtype=torch.float32)  # Add batch dimension
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
    return pred_scaled[0][0]  # Return the predicted value