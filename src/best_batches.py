import pickle

import more_itertools
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from w2v_model import Word2Vec

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading corpus")
with open("dataset/processed.pkl", "rb") as f:
    (
        corpus,
        tokens,  # corpus as tokens
        words_to_ids,
        ids_to_words,
    ) = pickle.load(f)
print("Loaded corpus")


print("Init model...")
embed_dim = 50
model = Word2Vec(embed_dim, len(words_to_ids)).to(device)
print("Model initialised")

optimizer = optim.Adam(model.parameters(), lr=0.003)

context_window = 2
batch_size = 10_000


wandb.init(
    project="word2vec",
    name="corrected",
    config={
        "batch_size": batch_size,
        "context_window": context_window,
        "embed_dims": embed_dim,
    },
)
for epoch in range(30):
    wins = more_itertools.windowed(tokens[:500_000], (context_window * 2) + 1)

    batches = more_itertools.chunked(wins, batch_size)

    prgs = tqdm(
        enumerate(batches),
        total=len(tokens[:500_000]) // batch_size,
        desc=f"Epoch {epoch+1}",
        leave=False,
    )

    for batch_id, batch in prgs:
        inputs = []
        targets = []
        neg_samples = []
        # TODO Insert subsampling here

        for tks in batch:

            inpt = torch.LongTensor([tks[1]]).to(device)
            trgs = torch.LongTensor(
                tks[:context_window] + tks[: context_window + 1 :]
            ).to(device)

            # TODO Insert unigram sampling here
            neg_rand = torch.randint(0, len(words_to_ids), (20,)).to(device)

            inputs.append(inpt)
            targets.append(trgs)
            neg_samples.append(neg_rand)

        inputs_stack = torch.stack(inputs)
        targets_stack = torch.stack(targets)
        neg_samples_stack = torch.stack(neg_samples)

        optimizer.zero_grad()

        loss = model.get_loss(inputs_stack, targets_stack, neg_samples_stack)

        loss.backward()

        optimizer.step()
        wandb.log({"loss": loss.item()})
    # TODO add model checkpoints
    if not (epoch + 1) % 5:
        save_path = f"checkpoints/w2v_epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
    # wandb.save(save_path)

    # Kinda validation
    # val_loss = 1- torch.cosine_similarity(model.embed(words_to_ids['house']), model...['home'])

    # For 5 words
    # for every vocab word calculate cosine sim to target word, order by desc, pick 5. display
wandb.finish()
