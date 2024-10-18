#
#
#
import collections

import more_itertools
import torch
import tqdm
import wandb


class SkipGramOne(torch.nn.Module):
    def __init__(self, voc, emb, _):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc)
        self.max = torch.nn.Softmax(dim=1)

    def forward(self, inpt, trgs):
        emb = self.emb(inpt)
        out = self.ffw(emb)
        sft = self.max(out)
        return -(sft[0, trgs]).log().mean()


#
#
#
class SkipGramTwo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=ctx * voc)
        self.max = torch.nn.Softmax(dim=1)

    def forward(self, inpt, trgs):
        emb = self.emb(inpt)
        hid = self.ffw(emb)
        lgt = hid.view(self.ctx, -1)
        sft = self.max(lgt)
        arg = torch.arange(sft.size(0))
        foo = sft[arg, trgs]
        return -foo.log().mean()


#
#
#
class SkipGramTre(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs]
        lgt = torch.mm(ctx, emb.T)
        sig = self.sig(lgt)
        return -sig.log().mean()


#
#
#
class SkipGramFoo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs]
        rnd = self.ffw.weight[rand]
        out = torch.mm(ctx, emb.T)
        rnd = torch.mm(rnd, emb.T)
        out = self.sig(out)
        rnd = self.sig(rnd)
        pst = -out.log().mean()
        ngt = -(1 - rnd).log().mean()
        return pst + ngt


#
#
#
args = (len(words_to_ids), 64, 2)
mOne = SkipGramOne(*args)
mTwo = SkipGramTwo(*args)
mTre = SkipGramTre(*args)
mFoo = SkipGramFoo(*args)


#
#
#
print("mOne", sum(p.numel() for p in mOne.parameters()))
print("mTwo", sum(p.numel() for p in mTwo.parameters()))
print("mTre", sum(p.numel() for p in mTre.parameters()))
print("mFoo", sum(p.numel() for p in mFoo.parameters()))


#
#
#
opOne = torch.optim.Adam(mOne.parameters(), lr=0.003)
opTwo = torch.optim.Adam(mTwo.parameters(), lr=0.003)
opTre = torch.optim.Adam(mTre.parameters(), lr=0.003)
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)


# #
# #
# #
# wandb.init(project='skip-gram', name='mOne')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     opOne.zero_grad()
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     loss = mOne(inpt, trgs)
#     loss.backward()
#     opOne.step()
#     wandb.log({'loss': loss.item()})
# wandb.finish()


# #
# #
# #
# wandb.init(project='skip-gram', name='mTwo')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(wins, desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     opTwo.zero_grad()
#     loss = mTwo(inpt, trgs)
#     loss.backward()
#     opTwo.step()
#     wandb.log({'loss': loss.item()})
# wandb.finish()


# #
# #
# #
# wandb.init(project='skip-gram', name='mTre')
# for epoch in range(10):
#   wins = more_itertools.windowed(tokens[:10000], 3)
#   prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
#   for i, tks in prgs:
#     inpt = torch.LongTensor([tks[1]])
#     trgs = torch.LongTensor([tks[0], tks[2]])
#     opTre.zero_grad()
#     loss = mTre(inpt, trgs)
#     loss.backward()
#     opTre.step()
#     wandb.log({'loss': loss.item()})
# wandb.finish()


#
#
#
wandb.init(project="skip-gram", name="mFoo")
for epoch in range(10):
    wins = more_itertools.windowed(tokens[:10000], 3)
    prgs = tqdm.tqdm(
        enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False
    )
    for i, tks in prgs:
        inpt = torch.LongTensor([tks[1]])
        trgs = torch.LongTensor([tks[0], tks[2]])
        rand = torch.randint(0, len(words_to_ids), (2,))
        opFoo.zero_grad()
        loss = mFoo(inpt, trgs, rand)
        loss.backward()
        opFoo.step()
        wandb.log({"loss": loss.item()})
wandb.finish()
