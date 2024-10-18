import os
import sys
import wandb
import torch
import psycopg2
import argparse
import subprocess
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Add the parent directory and current directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetch import fetch, preprocess, vocab_to_int
from model.predictor import Predictor
from w2v_model import Word2Vec

def parse_args():
    parser = argparse.ArgumentParser(description="Train a predictor model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=20, help="Length to which sequences will be padded")
    parser.add_argument("--window", type=int, default=1000, help="Length to which sequences will be padded")
    parser.add_argument("--w2v-path", type=str, default="./w2v_epoch_11.pth", help="Length to which sequences will be padded")
    parser.add_argument("--iterations", type=int, default=2, help="Device to train on") # -1 for all data
    parser.add_argument("--data-src", type=str, default="parquet", help="Path to the data")
    return parser.parse_args()

args = parse_args()

# Hyperparameters
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
OUTPUT_DIM = 1
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
SEQ_LENGTH = args.seq_length # Length to which sequences will be padded
ITERATIONS = args.iterations
DATA_SRC = args.data_src
window = args.window

# Parameters
vocab_size = 63642 #len(vocab_to_int) + 1  # Add 1 for <PAD>

torch.manual_seed(2)
torch.cuda.manual_seed(2)
# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model setup
# Load the w2v weights via the model
w2v_path = args.w2v_path
w2v = Word2Vec(EMBEDDING_DIM, vocab_size)
w2v.load_state_dict(torch.load(w2v_path, map_location=device, weights_only=True))
print("W2V loaded")

model = Predictor(vocab_size, 
                  EMBEDDING_DIM, 
                  HIDDEN_DIM, 
                  OUTPUT_DIM, 
                  emb_weights=w2v.center_embed.weight)
model.to(device)
# # Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Predictor initialized")

# # Training loop with data iteration
wandb.init(project="mlx-w1-upvote-prediction")
try:
    for epoch in range(EPOCHS):
        if DATA_SRC == "sql":
            connection = psycopg2.connect(
                'postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
            )
            print("Connected to the database successfully")
        model.train()
        epoch_loss = 0.0

        # # Iteratively fetch data and train the model
        # offset = 0
        # while True:
        #     if DATA_SRC == "sql":
        #         if ITERATIONS > -1:
        #             print(f"Fetching data: {offset}/{ITERATIONS} times")
        #         else: 
        #             print(f"Fetching ALL the data.")
        #         data_chunk = fetch(offset, window, connection)  # Fetch data starting from the current offset

        #         if data_chunk is None or len(data_chunk) == 0:
        #             break
        #         if ITERATIONS != 0 and offset == ITERATIONS: # remove this to train on the whole dataset
        #             break
        #         # Convert fetched data into DataFrame
        #         df = pd.DataFrame(data_chunk, columns=['title', 'score'])

        #         df['tkn_title'] = df['title'].apply(preprocess)
        #         df['tkn_title_id'] = df['tkn_title'].apply(
        #             lambda x: [vocab_to_int.get(word, 0) for word in x])  # 0 for unknown words, which is <PAD>
        #     elif DATA_SRC == "parquet":
        df = pd.read_parquet("../data/tokenized_titles.parquet")

        # Pad sequences to the desired length
        padded_titles = torch.zeros((len(df), SEQ_LENGTH), dtype=torch.long).to(device)
        # Put vectors into the padded tensor
        for i, row in enumerate(df['tkn_title_id']):
            length = min(len(row), SEQ_LENGTH)
            padded_titles[i, :length] = torch.tensor(row[:length]).to(device)

        # Prepare target labels
        targets = torch.tensor(df['score'].values, dtype=torch.float32).unsqueeze(1).to(device)

        # Create DataLoader for training
        train_dataset = TensorDataset(padded_titles, targets) # TODO: replace padded_titles with df['tkn_title_id'] as a tensor
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                labels = labels.squeeze(-1)

                # Compute the loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item() * inputs.size(0)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
            wandb.log({"loss": loss.item()})

            # offset += 1
        
        # Average epoch loss
        # avg_epoch_loss = epoch_loss / offset
        # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")
        if epoch == EPOCHS % 5 or epoch == EPOCHS - 1:
            commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"./params/model_{timestamp}_epoch_{epoch+1}_{commit}.pth"
            torch.save(model.state_dict(), model_path)
            # wandb.save(model_path)
            # print(f"Model saved to wandb: {model_path}")
            
            # # Log the model file to wandb artifacts
            # artifact = wandb.Artifact(f"model-epoch-{epoch+1}", type="model")
            # artifact.add_file(model_path)
            # wandb.log_artifact(artifact)
            # print(f"Model logged as artifact in wandb")

    wandb.finish()
    # if connection is not None:
        # connection.close()
        # print("Connection closed")
except Exception as e:
    print(f"Error: {e}")

print("Training completed.")