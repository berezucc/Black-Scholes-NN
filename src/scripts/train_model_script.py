from src.train import train_model

# Path to the dataset CSV file
csv_path = "data/data_train_500000.csv"

# Path to save the trained model
save_path = "models/model.pth"

# Train the model and save it
train_model(csv_path, save_path)