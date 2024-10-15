import os
from gensim.models import Word2Vec

def load_models(directory='models'):
    """
    Load all Word2Vec models from the given directory.
    
    :param directory: Directory where models are saved.
    :return: Dictionary with model names as keys and loaded Word2Vec models as values.
    """
    models = {}
    for filename in os.listdir(directory):
        if filename.endswith(".model"):
            model_path = os.path.join(directory, filename)
            models[filename] = Word2Vec.load(model_path)
    return models

def evaluate_model(model, test_words):
    """
    Evaluate a Word2Vec model using similarity between word pairs.
    
    :param model: Trained Word2Vec model.
    :param test_words: List of word pairs to test similarity.
    :return: Dictionary of word pairs and their cosine similarity.
    """
    results = {}
    for word1, word2 in test_words:
        try:
            similarity = model.wv.similarity(word1, word2)
            results[(word1, word2)] = similarity
        except KeyError as e:
            print(f"Word '{e.args[0]}' not in vocabulary.")
            results[(word1, word2)] = None
    return results

if __name__ == "__main__":
    # Load all trained models from the 'models' directory
    models = load_models()

    # Define some word pairs for similarity testing
    test_words = [
        ("king", "queen"),
        ("man", "woman"),
        ("cat", "dog"),
        ("car", "bicycle"),
        ("computer", "keyboard")
    ]

    # Evaluate each model and print results
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        results = evaluate_model(model, test_words)
        for word_pair, similarity in results.items():
            if similarity is not None:
                print(f"Similarity between {word_pair[0]} and {word_pair[1]}: {similarity:.4f}")
            else:
                print(f"Could not evaluate similarity for {word_pair[0]} and {word_pair[1]}")
        print("\n")
