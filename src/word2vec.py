import collections

import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

wandb.login()

device = "cuda" if torch.cuda.is_available() else "cpu"
device

with open("dataset/text8") as f:
    text8: str = f.read()


#
#
#
def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace(":", " <COLON> ")
    words = text.split()
    stats = collections.Counter(words)
    words = [word for word in words if stats[word] > 5]
    return words


corpus: list[str] = preprocess(text8)


def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = "<PAD>"
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab, word_counts


words_to_ids, ids_to_words, word_counts = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]

total_vocab_size = len(corpus)


def get_context_words(words, center_id, context_window):
    total_len = len(words)
    start = max(0, center_id)
    end = min(total_len, center_id + context_window + 1)
    return words[start:center_id] + words[center_id + 1 : end]


def negative_pair_generator(words, context_window, number_of_samples):
    total_len = len(words)
    for i, word in enumerate(words):
        neg_samples = []
        for i in range(number_of_samples):
            sample_index = i
            while (
                sample_index > i - context_window and sample_index < i + context_window
            ):
                sample_index = random.randint(0, total_len - 1)
            neg_samples.append(words[sample_index])
        for neg_sample in neg_samples:
            yield word, neg_sample


def get_negative_samples(words, center_id, context_window, number_of_samples):
    total_len = len(words)
    neg_samples = []
    for i in range(number_of_samples):
        sample_index = center_id
        while sample_index > i - context_window and sample_index < i + context_window:
            sample_index = random.randint(0, total_len - 1)
        neg_samples.append(words[sample_index])
    return neg_samples


training_split_ratio = 0.8
traing_test_cutoff = int(len(corpus) * training_split_ratio)
training_words = corpus[:traing_test_cutoff]
test_words = corpus[traing_test_cutoff:]


class Word2Vec(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embedding_dim)
        self.context_projection_embed = nn.Embedding(vocab_size, embedding_dim)

    def similarity(self, ids1, ids2):
        if ids1 is int:
            ids1 = torch.tensor(ids1)
        if ids2 is int:
            ids2 = torch.tensor(ids2)

        center_embed = self.center_embed(ids1)
        context_proj_embed = self.context_proj_embed(ids2)
        dot_product = torch.matmul(center_embed, context_proj_embed).sum(1)
        return dot_product

    def forward(self, id):
        if id is not torch.Tensor:
            id = torch.tensor(id)
        return self.center_embed(id)


# %%
from itertools import islice


def generate_batch(generator, batch_size):
    result = list(islice(generator, batch_size))
    exhausted = len(result) < batch_size
    return result, exhausted


# %%
subsampling_threshold = 10**-5


def word_seq_generator_enumerated(words):
    for i, word in enumerate(words):
        word_freq = word_counts[word] / len(corpus)
        drop_prob = np.sqrt(subsampling_threshold / word_freq)
        if random.ranf() < drop_prob:
            continue
        yield i, word


# %%
epochs = 10

batch_size = 3_000

embedding_dim = 100

neg_sample_count = 20

context_window = 2

samples_per_words = neg_sample_count + 2 * context_window

words_per_batch = np.floor(batch_size / samples_per_words).astype(int)

model = Word2Vec(embedding_dim=embedding_dim, vocab_size=total_vocab_size).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

wandb.init(project="word2vec")
optimizer.zero_grad()
for epoch in range(epochs):
    word_seq = word_seq_generator_enumerated(corpus)

    exhausted = False

    batch_counter = 0

    while not exhausted:
        batch_counter += 1

        batch_inputs_list = []
        batch_labels_list = []
        for _ in range(words_per_batch):
            # Select word to train
            index, center_word = next(word_seq_generator_enumerated)

            context_words = get_context_words(
                corpus, words_to_ids[center_word], context_window
            )

            pos_pair_inputs = torch.tensor(
                [
                    [words_to_ids[center_word], words_to_ids[context_word]]
                    for context_word in context_words
                ]
            )
            pos_labels = torch.ones_like(pos_pair_inputs)

            # Get all negative samples
            neg_words = get_negative_samples(
                corpus, words_to_ids[center_word], context_window, neg_sample_count
            )

            neg_pair_inputs = torch.tensor(
                [
                    [words_to_ids[center_word], words_to_ids[context_word]]
                    for context_word in neg_words
                ]
            )
            neg_labels = torch.zeros_like(neg_pair_inputs)

            word_pairs = torch.cat(pos_pair_inputs)
            word_labels = torch.cat(pos_labels, neg_labels)

            batch_inputs_list.append(word_pairs)
            batch_labels_list.append(word_labels)

        # Batch everything
        batch_inputs = torch.cat(batch_inputs_list)
        batch_labels = torch.cat(batch_labels_list)

        pred = model.similarity(batch_inputs)

        loss = criterion(pred, batch_labels)

        optimizer.step()

        optimizer.zero_grad()

        wandb.log({"loss": loss.item()})
wandb.finish()
