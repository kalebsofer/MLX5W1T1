
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
                 emb_weights: torch.Tensor = None,
                 pretrained_emb_path: str = None):
                #  embedding_matrix: torch.Tensor = None):
        super(Predictor, self).__init__()

        self.pad_idx = 0
        self.unk_idx = emb_weights.shape[0] # is this right? 
        self.embeddings = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=self.pad_idx)

        if emb_weights is not None:
            # Create embeddings matrix, plus pad and unknown vectors
                # TODO: We don't need this!
            pad_unk_embeds = torch.zeros(2, embedding_dim)
            extended_weights = torch.cat([emb_weights, pad_unk_embeds], dim=0)
            self.embeddings.weight = nn.Parameter(extended_weights)
        else:
            # TODO: Load pretrained embeddings from file path
                # Requires embeddings matrix pth file from w2v
            pass
            # Load pretrained embeddings from file path
            # self.load_pretrained_embeddings(pretrained_emb_path)

        # if embedding_matrix is not None:
        #     self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        # else:
        #     self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = torch.nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(0.3)

    def load_pretrained_embeddings(self, path: str):
        pretrained_emb = torch.load(path)
        # Add padding and unknown token embeddings
        pad_unk_embeds = torch.zeros(2, pretrained_emb.size(1))
        extended_emb = torch.cat([pad_unk_embeds, pretrained_emb], dim=0)
        self.embeddings.weight.data.copy_(extended_emb)

    def forward(self, x):

        x = torch.where(x < self.embeddings.num_embeddings, x, torch.tensor(self.unk_idx))
        emb = self.embeddings(x)
        pooled = torch.mean(emb, dim=1)

        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)
        return output.squeeze(-1)


if __name__ == "__main__":
    # Example usage
    vocab_size = 63642  # Replace with actual vocab size
    embedding_dim = 50
    hidden_dim = 64
    output_dim = 1  # Predicting a single value (upvote count)

    model = Predictor(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Sample input: batch of tokenized and padded titles
    sample_input = torch.randint(0, vocab_size, (32, 20))  # (batch_size, seq_length)
    output = model(sample_input)
    print("Model output shape:", output.shape)  # Should be (batch_size, output_dim)

