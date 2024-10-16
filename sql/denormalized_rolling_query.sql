-- A query to fetch denormalized data, prepared to feed into the predictor model
-- Step 1: Fetch a limited number of items from hacker_news.items
WITH limited_items AS (
    SELECT *
    FROM hacker_news.items
    WHERE type = 'story'
    ORDER BY id ASC	
    LIMIT 1000  -- Adjust this value as per memory and performance requirements
	OFFSET 0    -- This will be used to implement a rolling window
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
),

-- Step 4: Aggregate prior occurrences of the item_domain
item_domain_aggregates AS (
    SELECT
        item_id,
        item_domain,
        (
            SELECT COUNT(*)
            FROM hacker_news.items AS previous_items
            WHERE previous_items.id < normalized_items.item_id
            AND regexp_replace(previous_items.url, '^https?://([^/]+).*$', '\1') = normalized_items.item_domain
        ) AS prior_domain_count
    FROM
        normalized_items
)

-- Step 5: Select the denormalized data for further processing
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
    -- normalized_users.submitted, -- This is a list of item_ids, and is very large. Probably not necessary. 
    normalized_users.user_submitted_count, -- This leaks information from the future .... probably can't use this really
	-- normalized_users.user_submitted_count_to_date, -- to implement: rolling count of prior submissions
	-- also, get average prior upvotes for each post? 
    EXTRACT(DOW FROM normalized_items.time) AS item_day_of_week,
    EXTRACT(DOY FROM normalized_items.time) AS item_day_of_year,
    EXTRACT(HOUR FROM normalized_items.time) AS item_hour_of_day,
    EXTRACT(MINUTE FROM normalized_items.time) AS item_minute_of_hour,
    COALESCE(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - normalized_items.time)) / 60, 0) AS item_age_in_minutes,
    item_domain_aggregates.prior_domain_count
FROM 
    normalized_items
JOIN 
    normalized_users
ON 
    normalized_items.user_id = normalized_users.user_id
JOIN 
    item_domain_aggregates
ON
    normalized_items.item_id = item_domain_aggregates.item_id;

-- Note: To implement a rolling window, we'll use an iterative process in Python or another language to adjust the OFFSET in the limited_items CTE
-- and execute this query repeatedly, shifting the OFFSET value to effectively iterate over the entire dataset in manageable chunks.