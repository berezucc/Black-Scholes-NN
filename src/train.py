import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.model import baseline_model

def train_model(csv_path, save_path):
    df = pd.read_csv(csv_path)
    X = df[['S', 'K', 'q', 'r', 'sigma', 't']].values
    y = df[['price']].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    n_samples = len(dataset)
    train_size = int(0.8 * n_samples)
    val_size = int(0.2 * n_samples)
    test_size = n_samples - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    model = baseline_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 120
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    # Save only model weights (no scalers)
    torch.save({'model_state_dict': model.state_dict()}, save_path)

    return model