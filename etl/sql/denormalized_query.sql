SELECT 
    items.id AS item_id,
    items.title AS title_raw,
    items.time,
    items.score AS initial_score,
    items.url,
    regexp_replace(items.url, '^https?://([^/]+).*$', '\1') as item_domain,
    items.type,
    users.id AS user_id,
    users.karma AS user_karma,
    users.created AS user_created,
    users.submitted,
    EXTRACT(DOW FROM items.time) AS item_day_of_week,
    EXTRACT(DOY FROM items.time) AS item_day_of_year,
    EXTRACT(HOUR FROM items.time) AS item_hour_of_day,
    EXTRACT(MINUTE FROM items.time) AS item_minute_of_hour,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - items.time)) / 60 AS item_age_in_minutes
FROM 
    hacker_news.items as items
JOIN 
    hacker_news.users as users
ON 
    items.by = users.id
WHERE 
    items.type = 'story'
    AND array_length(users.submitted, 1) > 5
AND users.id IN (
    SELECT id FROM hacker_news.users
    WHERE array_length(submitted, 1) > 5
    ORDER BY RANDOM()
    LIMIT 10
);