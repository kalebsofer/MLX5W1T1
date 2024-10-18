"""
Data preprocessing and feature extraction
"""

import numpy as np
from collections import Counter

# replace with the path of trained word2vec model
# from .models.word2vec import word_vectors


def preprocess(api_request: dict, field: str) -> list[str]:
    text = api_request.get(field, "")
    text = text.lower()

    replacements = {
        ".": " <PERIOD> ",
        ",": " <COMMA> ",
        '"': " <QUOTATION_MARK> ",
        ";": " <SEMICOLON> ",
        "!": " <EXCLAMATION_MARK> ",
        "?": " <QUESTION_MARK> ",
        "(": " <LEFT_PAREN> ",
        ")": " <RIGHT_PAREN> ",
        "--": " <HYPHENS> ",
        ":": " <COLON> ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    words = text.split()

    word_counts = Counter(words)

    filtered_words = [word for word in words if word_counts[word] > 5]

    return filtered_words


def get_title_embedding(filtered_words: list[str], model, embedding_dim=50):
    # Get the embeddings for each word in the filtered words list
    embeddings = [
        model.wv[word] if word in model.wv else np.zeros(embedding_dim)
        for word in filtered_words
    ]

    # If we have word embeddings, compute the average; otherwise, return a zero vector
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)


if __name__ == "__main__":
    sample_json = {
        "title": "This is a sample title. It contains some words, some punctuation, and repeated words words words.",
        "author": "John Doe",
        "url": "http://example.com",
        "date": "2023-05-20",
    }

    result = preprocess(sample_json, "title")
    print(result)
