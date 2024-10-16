-- Extract core fields from users table
CREATE TEMP TABLE users_temp AS
SELECT 
    id AS user_id,
    karma AS user_karma,
    created AS user_created,
    submitted AS user_submitted
FROM 
    users;
