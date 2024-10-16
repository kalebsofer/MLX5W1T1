-- Join the temporary tables for items and users
CREATE TEMP TABLE joined_data AS
SELECT 
    items_temp.item_id,
    items_temp.title_raw,
    items_temp.created_at,
    items_temp.initial_score,
    items_temp.url,
    items_temp.type,
    users_temp.user_id,
    users_temp.user_karma,
    users_temp.user_created,
    users_temp.user_submitted
FROM 
    items_temp
JOIN 
    users_temp
ON 
    items_temp.user_id = users_temp.user_id;
