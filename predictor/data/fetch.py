import psycopg2
import collections
import pickle
import os
import sys

# Set the base directory for locating resources
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Add the '../data' directory to the Python path to import necessary modules
sys.path.append(os.path.join(BASE_DIR, '../data'))


def load_lookups():
    """
    Load lookup tables (vocab_to_int and int_to_vocab) from the pickle files stored in the same directory.
    """
    with open(os.path.join(BASE_DIR, 'vocab_to_int.pkl'), 'rb') as f:
        vocab_to_int = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'int_to_vocab.pkl'), 'rb') as f:
        int_to_vocab = pickle.load(f)
    return vocab_to_int, int_to_vocab

# Load lookup tables
vocab_to_int, int_to_vocab = load_lookups()

# def create_lookups(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
#   word_counts = collections.Counter(words)
#   vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
#   int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
#   int_to_vocab[0] = '<PAD>'
#   vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
#   return vocab_to_int, int_to_vocab


def fetch(i, window=10, connection=None):    
    # Connect to PostgreSQL database
    try:
        preconnection = True
        if connection is None:
            preconnection = False
            connection = psycopg2.connect(
                'postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
            )
            print("Connected to the database successfully")
        cursor = connection.cursor()
        
        # SQL query to fetch relevant data
        query = f"""
            SELECT title, score
            FROM hacker_news.items
            WHERE title IS NOT NULL 
            AND score IS NOT NULL
            AND type IN ('story')
            LIMIT {window}
            OFFSET {i * window};
        """
        print(f'Fetching next {window} records: {i * window} to {i * window + window}')
        cursor.execute(query)
        records = cursor.fetchall()
        return records
        
    except Exception as error:
        print(f"Error connecting to the database: {error}")
    
    finally:
        if not preconnection:
            connection.close()    
        print("Data fetch completed")

def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  words = text.split()
  words = [word for word in words]
  return words
