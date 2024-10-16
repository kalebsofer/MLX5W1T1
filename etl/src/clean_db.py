import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docker/.env'))
load_dotenv(dotenv_path)

write_user = os.getenv('POSTGRES_USER')
write_password = os.getenv('POSTGRES_PASSWORD')
write_db = os.getenv('POSTGRES_DB')
# SQL query to extract data from remote database


# SQL commands to drop tables if they exist
drop_tables_sql = """
DROP TABLE IF EXISTS denormalized_data;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS users;
"""

def clean_database():
    # Create engine for local database
    local_engine = create_engine(f'postgresql://{write_user}:{write_password}@localhost:5432/{write_db}')

    try:
    # Connect to the database and execute the drop tables script
        with local_engine.connect() as connection:
            connection.execute(text(drop_tables_sql))
            print("Database cleaned successfully. All tables dropped.")
            
            # Check if the table still exists
            result = connection.execute(text("SELECT to_regclass('denormalized_data');"))
            if result.fetchone()[0] is None:
                print("Confirmed: denormalized_data table does not exist.")
            else:
                print("Warning: denormalized_data table still exists.")
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_database()