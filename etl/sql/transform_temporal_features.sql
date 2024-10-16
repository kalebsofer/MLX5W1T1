-- Add temporal features to the joined data
CREATE TEMP TABLE enriched_data AS
SELECT *,
    EXTRACT(DOW FROM created_at) AS item_day_of_week,
    EXTRACT(DOY FROM created_at) AS item_day_of_year,
    EXTRACT(HOUR FROM created_at) AS item_hour_of_day,
    EXTRACT(MINUTE FROM created_at) AS item_minute_of_hour,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 60 AS item_age_in_minutes
FROM
    joined_data;
