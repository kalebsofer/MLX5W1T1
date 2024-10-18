import os
import sys
import wandb
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add the parent directory and current directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetch import fetch, preprocess, vocab_to_int
from model.predictor import Predictor
from w2v_model import Word2Vec

# Hyperparameters
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
OUTPUT_DIM = 1
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
SEQ_LENGTH = 20  # Length to which sequences will be padded

# Parameters
vocab_size = 63642 #len(vocab_to_int) + 1  # Add 1 for <PAD>

torch.manual_seed(2)
torch.cuda.manual_seed(2)
# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model setup
# Load the w2v weights via the model
w2v_path = "./w2v_epoch_11.pth"
w2v = Word2Vec(EMBEDDING_DIM, vocab_size)
w2v.load_state_dict(torch.load(w2v_path, map_location=device, weights_only=True))
print("W2V loaded")

model = Predictor(vocab_size, 
                  EMBEDDING_DIM, 
                  HIDDEN_DIM, 
                  OUTPUT_DIM, 
                  emb_weights=w2v.center_embed.weight)
# # Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Predictor initialized")

# # Training loop with data iteration
wandb.init(project="mlx-w1-upvote-prediction")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    # Iteratively fetch data and train the model
    offset = 0
    window = 1000 # How can we figure out the window size that can 
                  # be supported by the GPU? Does this matter much? 
    while True:
        data_chunk = fetch(offset, window)  # Fetch data starting from the current offset
        print(offset)
        if data_chunk is None or len(data_chunk) == 0:
            break
        if offset == 4: # remove this to train on the whole dataset
            break
        # Convert fetched data into DataFrame
        df = pd.DataFrame(data_chunk, columns=['title', 'score'])

        df['tkn_title'] = df['title'].apply(preprocess)
        df['tkn_title_id'] = df['tkn_title'].apply(
            lambda x: [vocab_to_int.get(word, 0) for word in x])  # 0 for unknown words, which is <PAD>

        # Pad sequences to the desired length
        padded_titles = torch.zeros((len(df), SEQ_LENGTH), dtype=torch.long)
        for i, row in enumerate(df['tkn_title_id']):
            length = min(len(row), SEQ_LENGTH)
            padded_titles[i, :length] = torch.tensor(row[:length])

        # Prepare target labels
        targets = torch.tensor(df['score'].values, dtype=torch.float32).unsqueeze(1)

        # Create DataLoader for training
        train_dataset = TensorDataset(padded_titles, targets)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
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

        # Update offset for the next chunk
        offset += 1

    # Average epoch loss
    avg_epoch_loss = epoch_loss / offset
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")
wandb.finish()
print("Training completed.")
