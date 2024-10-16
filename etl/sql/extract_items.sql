-- Extract core fields from items table
CREATE TEMP TABLE items_temp AS
SELECT 
    id AS item_id,
    title AS title_raw,
    created_at,
    score AS initial_score,
    url,
    type,
    by AS user_id
FROM 
    items;
