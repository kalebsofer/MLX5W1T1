import nltk
from urllib.parse import urlparse
import pandas as pd

nltk.download('punkt')

def text_feature_engineering(df):
    # Tokenize title to get the length in words
    df['title_length_words'] = df['title_raw'].apply(lambda x: len(nltk.word_tokenize(x)) if pd.notnull(x) else 0)

    # Extract domain from URL and encode it
    df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc if pd.notnull(x) else None)
    df['domain_encoded'] = df['domain'].factorize()[0]  # Use pandas' factorize for label encoding

    return df
