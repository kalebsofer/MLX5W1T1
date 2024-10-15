import gensim
from gensim.models import Word2Vec
from tokenizer import tokenize_text

def train_word2vec(filename, method="simple", vector_size=100, window=5, min_count=5, workers=4, epochs=5, negative=5, alpha=0.025, sample=1e-3, hs=0, cbow_mean=1):
    """
    Train a Word2Vec model using the tokenized text from the specified file.
    
    :param filename: Path to the input text file.
    :param method: Tokenization method ('simple', 'regex', 'nltk', 'spacy').
    :param vector_size: Dimensionality of the word vectors.
    :param window: Maximum distance between the current and predicted word within a sentence.
    :param min_count: Ignores all words with total frequency lower than this.
    :param workers: Number of worker threads to train the model.
    :param epochs: Number of iterations (epochs) over the corpus.
    :param negative: Number of negative samples for negative sampling.
    :param alpha: Initial learning rate.
    :param sample: Threshold for downsampling high-frequency words.
    :param hs: If 1, hierarchical softmax will be used for model training; if 0, negative sampling will be used.
    :param cbow_mean: If 1, use the mean of the context word vectors; if 0, use the sum (only applies if CBOW is used).
    :return: Trained Word2Vec model.
    """
    # Tokenize the text using the specified method
    tokens = tokenize_text(filename, method=method)
    
    # Train the Word2Vec model using skip-gram
    model = Word2Vec(sentences=[tokens], vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1, epochs=epochs, negative=negative, alpha=alpha, sample=sample, hs=hs, cbow_mean=cbow_mean)
    
    # Save the model to a file
    import os
    # Create models subdirectory if it does not exist
    if not os.path.exists('models'):
        os.makedirs('models')
    model_name = f"word2vec_skipgram_vs{vector_size}_w{window}_mc{min_count}_e{epochs}_neg{negative}_alpha{alpha}_hs{hs}_cbow{cbow_mean}.model"
    model_path = os.path.join('models', model_name)
    model.save(model_path)
    
    return model

if __name__ == "__main__":
    import sys
    # Check if user passed filename and tokenization method via command line
    import argparse

    print('Initializing training of word2vec')
    parser = argparse.ArgumentParser(description="Train a Word2Vec model with custom hyperparameters.")
    parser.add_argument('filename', type=str, help='Path to the input text file.')
    parser.add_argument('method', type=str, help='Tokenization method (simple, regex, nltk, spacy).')
    parser.add_argument('--vector_size', type=int, default=100, help='Dimensionality of the word vectors.')
    parser.add_argument('--window', type=int, default=5, help='Maximum distance between current and predicted word within a sentence.')
    parser.add_argument('--min_count', type=int, default=5, help='Ignores all words with total frequency lower than this.')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads to train the model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of iterations (epochs) over the corpus.')
    parser.add_argument('--negative', type=int, default=10, help='Number of negative samples for negative sampling.')
    parser.add_argument('--alpha', type=float, default=0.03, help='Initial learning rate.')
    parser.add_argument('--sample', type=float, default=1e-4, help='Threshold for downsampling high-frequency words.')
    parser.add_argument('--hs', type=int, default=1, help='If 1, hierarchical softmax will be used; if 0, negative sampling will be used.')
    parser.add_argument('--cbow_mean', type=int, default=0, help='If 1, use the mean of the context word vectors; if 0, use the sum (only applies if CBOW is used).')
    
    args = parser.parse_args()
    filename = args.filename
    method = args.method
    vector_size = args.vector_size
    window = args.window
    min_count = args.min_count
    workers = args.workers
    epochs = args.epochs
    negative = args.negative
    alpha = args.alpha
    sample = args.sample
    hs = args.hs
    cbow_mean = args.cbow_mean
    # Train the Word2Vec model using the provided file and tokenization method
    model = train_word2vec(filename, method=method, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs, negative=negative, alpha=alpha, sample=sample, hs=hs, cbow_mean=cbow_mean)
    print(f"Word2Vec model trained and saved in 'models/word2vec_skipgram_vs{vector_size}_w{window}_mc{min_count}_e{epochs}_neg{negative}_alpha{alpha}_hs{hs}_cbow{cbow_mean}.model' using {method} tokenization.")
    