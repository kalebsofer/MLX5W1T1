WITH normalized_users AS (
    SELECT
        id AS user_id,
        COALESCE(karma, 0) AS user_karma,
        created AS user_created,
        COALESCE(array_length(submitted, 1), 0) AS user_submitted_count,
        submitted,
        -- Normalize karma: (karma - min) / (max - min) using actual min/max values from the table
        COALESCE((karma - (SELECT MIN(karma) FROM hacker_news.users))::DOUBLE PRECISION / 
        (SELECT MAX(karma) - MIN(karma) FROM hacker_news.users), 0)::DOUBLE PRECISION AS normalized_karma
    FROM
        hacker_news.users
),
normalized_items AS (
    SELECT
        id AS item_id,
        COALESCE(title, '') AS title_raw,
        time,
        COALESCE(score, 0) AS initial_score,
        url,
        COALESCE(regexp_replace(url, '^https?://([^/]+).*$', '\1'), 'unknown_domain') as item_domain,
        type,
        by AS user_id,
        -- Normalize score: (score - min) / (max - min) using actual min/max values from the table
        COALESCE((score - (SELECT MIN(score) FROM hacker_news.items))::DOUBLE PRECISION / 
        (SELECT MAX(score) - MIN(score) FROM hacker_news.items)::DOUBLE PRECISION, 0) AS normalized_score
    FROM
        hacker_news.items
)
SELECT 
    normalized_items.item_id,
    normalized_items.title_raw,
    normalized_items.time,
    normalized_items.initial_score,
    normalized_items.normalized_score,
    normalized_items.url,
    normalized_items.item_domain,
    normalized_items.type,
    normalized_users.user_id,
    normalized_users.user_karma,
    normalized_users.normalized_karma,
    normalized_users.user_created,
    normalized_users.submitted,
    normalized_users.user_submitted_count,
    EXTRACT(DOW FROM normalized_items.time) AS item_day_of_week,
    EXTRACT(DOY FROM normalized_items.time) AS item_day_of_year,
    EXTRACT(HOUR FROM normalized_items.time) AS item_hour_of_day,
    EXTRACT(MINUTE FROM normalized_items.time) AS item_minute_of_hour,
    COALESCE(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - normalized_items.time)) / 60, 0) AS item_age_in_minutes
FROM 
    normalized_items
JOIN 
    normalized_users
ON 
    normalized_items.user_id = normalized_users.user_id
WHERE
    normalized_items.type = 'story'
AND normalized_users.user_id IN (
    SELECT user_id FROM normalized_users
    WHERE user_submitted_count > 5
    ORDER BY RANDOM()
    LIMIT 20
);
