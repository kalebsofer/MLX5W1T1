
# Define torch predictor model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Pseudocode


class Predictor(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 embedding_matrix: torch.Tensor = None):
        super(Predictor, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = torch.nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        # Activation function ?
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        emb = self.embedding(x)
        pooled = torch.mean(emb, dim=1)

        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)
        return output.squeeze(-1)


if __name__ == "__main__":
    # Example usage
    vocab_size = 5000  # Replace with actual vocab size
    embedding_dim = 100
    hidden_dim = 64
    output_dim = 1  # Predicting a single value (upvote count)

    model = Predictor(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Sample input: batch of tokenized and padded titles
    sample_input = torch.randint(0, vocab_size, (32, 20))  # (batch_size, seq_length)
    output = model(sample_input)
    print("Model output shape:", output.shape)  # Should be (batch_size, output_dim)

