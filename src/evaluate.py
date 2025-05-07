import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model import baseline_model

def evaluate_model(csv_path, model_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    X = df[['S', 'K', 'q', 'r', 'sigma', 't']].values
    y_true = df[['price']].values

    # Load the model
    checkpoint = torch.load(model_path)
    model = baseline_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare input tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    # Compute evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")