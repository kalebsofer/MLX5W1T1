"""
Loading the trained model and making predictions
"""

from models.predictor import Predictor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 63642  # len(vocab_to_int) + 1  # Add 1 for <PAD>
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
OUTPUT_DIM = 1

predictor_path = "models/predictor.pth"
model = Predictor(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(predictor_path, weights_only=True))

model.to(device)


@torch.inference_mode()
def predict(list_features: list):
    print(f"Predicting")
    tokens = torch.LongTensor(list_features)
    prediction = model(tokens)
    return prediction.item()
