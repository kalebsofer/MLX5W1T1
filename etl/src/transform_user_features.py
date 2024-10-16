import pandas as pd

def transform_user_features(engine):
    query = "SELECT * FROM enriched_data;"
    df = pd.read_sql(query, con=engine)
    
    # Calculate user account age in days
    df['user_account_age_days'] = (pd.Timestamp.now() - pd.to_datetime(df['user_created'])).dt.days
    
    # Calculate user submission count
    df['user_submission_count'] = df['user_submitted'].apply(lambda x: len(x) if x is not None else 0)
    
    return df
