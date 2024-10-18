import os
import sys
import psycopg2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add the parent directory and current directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetch import preprocess, vocab_to_int

def save_all_data(connection):
    # SQL query to fetch relevant data
    query = f"""
        SELECT title, score
        FROM hacker_news.items
        WHERE title IS NOT NULL 
        AND score IS NOT NULL
        AND type IN ('story');
    """
    try:
        preconnection = True
        if connection is None:
            preconnection = False
            connection = psycopg2.connect(
                'postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki'
            )
            print("Connected to the database successfully")
        cursor = connection.cursor()

        cursor.execute(query)
        records = cursor.fetchall()

        # Convert fetched data into DataFrame
        df = pd.DataFrame(records, columns=['title', 'score'])
        print("Data fetched!", df.head())
        df['tkn_title'] = df['title'].apply(preprocess)
        df['tkn_title_id'] = df['tkn_title'].apply(
            lambda x: [vocab_to_int.get(word, 0) for word in x])  # 0 for unknown words, which is <PAD>

        # Save to Parquet
        df.to_parquet('tokenized_titles.parquet', engine='pyarrow', compression='snappy')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if not preconnection:
            connection.close()
            print("Disconnected from the database")

if __name__ == "__main__":
    save_all_data(None)