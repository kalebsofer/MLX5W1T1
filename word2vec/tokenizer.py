import re
import nltk
from nltk.tokenize import word_tokenize
import spacy

def read_file(filename):
    """
    Reads the content of a text file and returns it as a string.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return ""

def simple_tokenize(filename):
    """
    Simple tokenization by splitting on whitespace.
    """
    text = read_file(filename)
    tokens = text.split()
    return tokens

def regex_tokenize(filename):
    """
    Tokenization using regex to handle punctuation and more complex patterns.
    """
    text = read_file(filename)
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def nltk_tokenize(filename):
    """
    Tokenization using NLTK's word tokenizer.
    """

    nltk.download('punkt')  # Needed if using nltk for tokenization
    nltk.download('punkt_tab')  # Additional download needed
    text = read_file(filename)
    tokens = word_tokenize(text)
    return tokens

# If you want to use spaCy (optional)
# def spacy_tokenize(filename):
#     """
#     Tokenization using spaCy with batch processing for large texts.
#     """
#     try:
#         nlp = spacy.load('en_core_web_sm')
#         nlp.max_length = 15000000  # Increase the max length to handle larger texts
#     except OSError:
#         print("SpaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
#         sys.exit(1)
    
#     text = read_file(filename)
#     tokens = []
#     chunk_size = 1000000  # Process in chunks of 1 million characters
    
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i:i+chunk_size]
#         doc = nlp(chunk)
#         tokens.extend([token.text for token in doc])
    
#     return tokens

def preprocess(text, lower=True, remove_punctuation=True):
    """
    Preprocess text: lowercasing, punctuation removal, etc.
    """
    if lower:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    return text

def tokenize_text(filename, method="simple"):
    """
    Tokenize the input text using the specified method.
    Available methods: 'simple', 'regex', 'nltk', 'spacy'
    """
    text = read_file(filename)
    text = preprocess(text)
    
    if method == "simple":
        return simple_tokenize(filename)
    elif method == "regex":
        return regex_tokenize(filename)
    elif method == "nltk":
        return nltk_tokenize(filename)
    # elif method == "spacy":
    #     return spacy_tokenize(filename)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")

# Testing the function
if __name__ == "__main__":
    import sys
    # Check if user passed filename and method via command line
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        method = sys.argv[2]
        tokens = tokenize_text(filename, method=method)
    else:
        print("Please provide a filename and tokenization method as arguments.")
    