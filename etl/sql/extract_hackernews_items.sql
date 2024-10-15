SELECT 
    EXTRACT(YEAR FROM time) AS year,
    EXTRACT(MONTH FROM time) AS month,
    EXTRACT(DAY FROM time) AS day,
	EXTRACT(HOUR FROM time) AS hour,
	EXTRACT(MINUTE FROM time) AS minute,
	EXTRACT( EPOCH FROM NOW() - time) / 60 AS age_in_minutes,
	EXTRACT(DOW FROM time) AS day_of_week,
	EXTRACT(WEEK FROM time) AS week_of_year,
	EXTRACT(DOY FROM time) AS day_of_year	
FROM hacker_news.items LIMIT 1;