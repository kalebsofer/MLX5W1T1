import torch
import torch.nn as nn
import torch.nn.functional as F
from predictor.model.w2v_model import Word2Vec
import pickle
import more_itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

# device = "cpu"

print("Loading corpus")
with open("dataset/processed.pkl", "rb") as f:
    (
        corpus,
        tokens,  # corpus as tokens
        words_to_ids,
        ids_to_words,
    ) = pickle.load(f)

words = [
    "king",
    "man",
    "football",
    "love",
    "the",
    "space",
    "umbrella",
    "rain",
    "blood",
    "arm",
    "verb",
    "walk",
    "blue",
    "happy",
]


def validate_model(model: Word2Vec):
    print("Validation is as follows: \n")
    embeddings = model.center_embed.weight.data

    for word in words:
        word_embed = embeddings[words_to_ids[word]]

        all_cos_sims = F.cosine_similarity(embeddings, word_embed.unsqueeze(0), dim=1)

        sorted_sims, sorted_ids = torch.sort(all_cos_sims, descending=True)
        top_n_sims = [
            (ids_to_words[sorted_ids[i].item()], sorted_sims[i].item())
            for i in range(1, 6)
        ]
        print(
            f"""
              Similar words to {word}: {top_n_sims}
              """
        )


if __name__ == "__main__":
    embed_dim = 50

    print("Loading model...")
    model = Word2Vec(embed_dim, len(words_to_ids)).to(device)
    model_path = "checkpoints/w2v_epoch_3.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    print("Model loaded")

    validate_model(model=model)

    king_embed = model.center_embed(
        torch.tensor(words_to_ids["king"], dtype=torch.long).to(device)
    ).unsqueeze(0)
    queen_embed = model.center_embed(
        torch.tensor(words_to_ids["queen"], dtype=torch.long).to(device)
    ).unsqueeze(0)

    k_q_sim = F.cosine_similarity(king_embed, queen_embed)

    print(f"Similarity between king and queen: {k_q_sim.item()}")
