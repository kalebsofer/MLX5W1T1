import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docker/.env'))
load_dotenv(dotenv_path)

read_database_url = os.getenv("READ_DB_URL")
write_user = os.getenv('POSTGRES_USER')
write_password = os.getenv('POSTGRES_PASSWORD')
write_db = os.getenv('POSTGRES_DB')
print(read_database_url, write_user, write_password, write_db)
# SQL query to extract data from remote database

def main():

    # Create engine for remote and local databases

    read_engine = create_engine(read_database_url)
    write_engine = create_engine(f'postgresql://{write_user}:{write_password}@localhost:5432/{write_db}')

    # Execute setup script to ensure tables exist
    setup_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../init_db/setup_db.sql'))
    with write_engine.connect() as connection:
        with open(setup_file_path, 'r') as setup_file:
            setup_sql = setup_file.read()
            connection.execute(text(setup_sql))


    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sql/denormalized_query.sql'))
    with open(file_path, 'r') as file:
        query = file.read()

    # Extract data from the remote database
    with read_engine.connect() as connection:
        df = pd.read_sql(query, connection)
    print(df.head())
    # Insert the data into the local denormalized_data table

    # Normalize the data 

    df.to_sql('denormalized_data', con=write_engine, if_exists='replace', index=False)
    print("Data loaded successfully into denormalized_data table.")

if __name__ == "__main__":
    main()
