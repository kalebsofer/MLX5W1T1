CREATE TABLE IF NOT EXISTS denormalized_data (
    item_id SERIAL PRIMARY KEY,
    title_raw TEXT,
    time TIMESTAMP,
    initial_score INTEGER,
    url TEXT,
    item_domain TEXT,
    type TEXT,
    user_id TEXT,
    user_karma INTEGER,
    user_created TIMESTAMP,
    submitted TEXT[],
    item_day_of_week INTEGER,
    item_day_of_year INTEGER,
    item_hour_of_day INTEGER,
    item_minute_of_hour INTEGER,
    item_age_in_minutes DOUBLE PRECISION
);
