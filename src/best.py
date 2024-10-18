import pickle

import more_itertools
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from w2v_model import Word2Vec

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("dataset/processed.pkl", "rb") as f:
    (
        corpus,
        tokens,  # corpus as tokens
        words_to_ids,
        ids_to_words,
    ) = pickle.load(f)


model = Word2Vec(20, len(tokens)).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.003)

window_size = 2

wandb.init(project="word2vec", name="BCEWithLogits")
for epoch in range(20):
    wins = more_itertools.windowed(tokens[:1000], (window_size * 2) + 1)
    prgs = tqdm(
        enumerate(wins), total=len(tokens[:1000]), desc=f"Epoch {epoch+1}", leave=False
    )

    for i, tks in prgs:
        # TODO Insert subsampling here

        inpt = torch.LongTensor([tks[1]]).to(device)
        trgs = torch.LongTensor(tks[:window_size] + tks[: window_size + 1 :]).to(device)

        # TODO Insert unigram sampling here
        neg_rand = torch.randint(0, len(words_to_ids), (2,)).to(device)
        optimizer.zero_grad()

        loss = model.get_loss(inpt, trgs, neg_rand)

        loss.backward()

        optimizer.step()
        wandb.log({"loss": loss.item()})
    # TODO add model checkpoints
    # save_path = f"checkpoints/w2v_epoch_{epoch}.pth"
    # torch.save(model.state_dict(), save_path)
    # wandb.save(save_path)
wandb.finish()
