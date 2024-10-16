-- Step 1: Fetch a limited number of items from hacker_news.items
WITH limited_items AS (
    SELECT *
    FROM hacker_news.items
    WHERE type = 'story'
    ORDER BY id ASC
    LIMIT 1000  -- Adjust this value as per memory and performance requirements
),

-- Step 2: Fetch relevant users that match the limited items
relevant_users AS (
    SELECT *
    FROM hacker_news.users
    WHERE id IN (SELECT DISTINCT by FROM limited_items)
),

-- Step 3: Normalize and perform feature extraction
normalized_users AS (
    SELECT
        id AS user_id,
        COALESCE(karma, 0) AS user_karma,
        created AS user_created,
        array_length(submitted, 1) AS user_submitted_count,
        submitted,
        -- Normalize karma: (karma - min) / (max - min) using actual min/max values from the table
        (karma - (SELECT MIN(karma) FROM relevant_users))::DOUBLE PRECISION / 
        (SELECT (MAX(karma) - MIN(karma)) FROM relevant_users)::DOUBLE PRECISION AS normalized_karma
    FROM
        relevant_users
),
normalized_items AS (
    SELECT
        id AS item_id,
        COALESCE(title, '') AS title_raw,
        time,
        COALESCE(score, 0) AS initial_score,
        url,
        regexp_replace(url, '^https?://([^/]+).*$', '\1') as item_domain,
        type,
        by AS user_id,
        -- Normalize score: (score - min) / (max - min) using actual min/max values from the table
        (score - (SELECT MIN(score) FROM limited_items))::DOUBLE PRECISION / 
        (SELECT (MAX(score) - MIN(score)) FROM limited_items)::DOUBLE PRECISION AS normalized_score
    FROM
        limited_items
)

-- Step 4: Select the denormalized data for further processing
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
    normalized_items.user_id = normalized_users.user_id;

-- Note: To implement a rolling window, you can use an iterative process in Python or another language to adjust the OFFSET in the limited_items CTE
-- and execute this query repeatedly, shifting the OFFSET value to effectively iterate over the entire dataset in manageable chunks.