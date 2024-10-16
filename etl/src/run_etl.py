import os
import psycopg2
from sqlalchemy import create_engine

def run_sql_file(engine, file_path):
    with open(file_path, 'r') as file:
        query = file.read()
    with engine.connect() as connection:
        connection.execute(query)

def main():
    # Database connection
    engine = create_engine('postgresql://etl_user:secret@localhost:5432/etl_db')
    
    # Run SQL scripts in order
    sql_files = [
        '../sql/extract_items.sql',
        '../sql/extract_users.sql',
        '../sql/create_joined_data.sql',
        '../sql/transform_temporal_features.sql'
    ]
    
    for file_path in sql_files:
        run_sql_file(engine, file_path)
    
    print("SQL extraction and transformation complete. Proceeding to Python-based transformations...")

    # Call user transformation functions and text engineering
    from transform_user_features import transform_user_features
    df = transform_user_features(engine)

    from text_feature_engineering import text_feature_engineering
    df = text_feature_engineering(df)

    # Save the final dataframe to the database
    df.to_sql('training_data', con=engine, if_exists='replace', index=False)
    print("ETL process complete and data loaded to training_data table.")

if __name__ == "__main__":
    main()
