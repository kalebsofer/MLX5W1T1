import os
import psycopg2
from gensim.models import Word2Vec
from tokenizer import tokenize_text

def fetch_titles_from_db(db_url, table_name):
    """
    Fetch titles from the given database and table.
    
    :param db_url: Connection URL for the PostgreSQL database.
    :param table_name: Name of the table containing titles.
    :return: List of titles.
    """
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    cursor.execute(f"SELECT title FROM hacker_news.{table_name} LIMIT 100;")
    titles = [row[0] for row in cursor.fetchall()]
    print(titles)
    connection.close()
    return titles

def tokenize_titles(titles, method="simple"):
    """
    Tokenize a list of titles using the specified method.
    
    :param titles: List of text titles.
    :param method: Tokenization method ('simple', 'regex', 'nltk', 'spacy').
    :return: List of tokenized titles.
    """
    all_tokens = []
    for title in titles:
        tokens = tokenize_text(title, method=method)
        all_tokens.append(tokens)
    return all_tokens

def fine_tune_word2vec(db_path, table_name, model_path, method="simple", epochs=5):
    """
    Fine-tune an existing Word2Vec model with new data from a database.
    
    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table containing titles.
    :param model_path: Path to the pre-trained Word2Vec model.
    :param method: Tokenization method ('simple', 'regex', 'nltk', 'spacy').
    :param epochs: Number of epochs to fine-tune the model.
    :return: Fine-tuned Word2Vec model.
    """
    # Load the existing model
    model = Word2Vec.load(model_path)
    
    # Fetch titles from the database
    titles = fetch_titles_from_db(db_url, table_name)
    
    # Tokenize the titles
    tokenized_titles = tokenize_titles(titles, method=method)
    
    # Build the vocabulary using new data
    model.build_vocab(tokenized_titles, update=True)
    
    # Train the model with new data
    model.train(tokenized_titles, total_examples=len(tokenized_titles), epochs=epochs)
    
    # Save the updated model
    model.save(model_path)
    
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utility functions for interacting with the PostgreSQL database and fine-tuning a Word2Vec model.")
    parser.add_argument('function', type=str, choices=['fetch_titles', 'fine_tune'], help='Function to execute: fetch_titles or fine_tune.')
    parser.add_argument('db_url', type=str, help='Connection URL for the PostgreSQL database.')
    parser.add_argument('table_name', type=str, help='Name of the table containing titles.')
    parser.add_argument('--model_path', type=str, help='Path to the pre-trained Word2Vec model (required for fine-tuning).')
    parser.add_argument('--method', type=str, default='simple', help='Tokenization method (simple, regex, nltk, spacy).')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to fine-tune the model.')
    
    args = parser.parse_args()
    db_url = args.db_url
    table_name = args.table_name
    function = args.function

    if function == 'fetch_titles':
        titles = fetch_titles_from_db(db_url, table_name)
        print("Fetched Titles:")
        for title in titles[:10]:  # Print the first 10 titles for verification
            print(title)
    elif function == 'fine_tune':
        if not args.model_path:
            raise ValueError("--model_path is required for fine-tuning.")
        model_path = args.model_path
        method = args.method
        epochs = args.epochs

        # Fine-tune the Word2Vec model
        model = fine_tune_word2vec(db_url, table_name, model_path, method=method, epochs=epochs)
        print(f"Word2Vec model fine-tuned and saved in '{model_path}' using {method} tokenization.")