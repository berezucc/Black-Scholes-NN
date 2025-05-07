from src.data_generation import generate_data
from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    print("Generating dataset...")
    generate_data()
    print("Training model...")
    train_model("data/data_train.csv", "models/model.pth")
    print("Model training complete.")
    print("Evaluating model...")
    evaluate_model("data/data_train.csv", "models/model.pth")