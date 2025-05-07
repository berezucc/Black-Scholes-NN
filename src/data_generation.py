import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from src.black_scholes import black_scholes_price

def generate_data(n_samples=100_000, batch_size=16, option_type='call', csv_path="data/data_train.csv"):
    # Generate features
    S = torch.rand(n_samples) * 100 + 1
    K = torch.rand(n_samples) * 100 + 1
    T = torch.rand(n_samples) * 2 + 0.01
    r = torch.rand(n_samples) * 0.1
    sigma = torch.rand(n_samples) * 0.5 + 0.01
    q = torch.rand(n_samples) * 0.05

    # Calculate prices using Black-Scholes
    prices = black_scholes_price(S, K, T, r, sigma, q, option_type).unsqueeze(1)

    # Stack and scale features
    features = torch.stack([S, K, T, r, sigma, q], dim=1)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features.numpy())
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Save unscaled data + price to CSV
    df = pd.DataFrame({
        'S': S.numpy(),
        'K': K.numpy(),
        't': T.numpy(),
        'r': r.numpy(),
        'sigma': sigma.numpy(),
        'q': q.numpy(),
        'price': prices.numpy().flatten()
    })
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic dataset to {csv_path}")

    # Create dataset and loaders
    dataset = TensorDataset(features_tensor, prices)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler