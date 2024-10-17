import psycopg2

# Set up a connection to the PostgreSQL database
connection = psycopg2.connect(
    "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
)

# Query to select the titles
query = """
    SELECT title
    FROM hacker_news.items
    WHERE type = 'story'
	AND title IS NOT NULL
    ORDER BY id;
"""

# Create a cursor to execute the query
cursor = connection.cursor()
cursor.execute(query)

# Fetch all titles
titles = cursor.fetchall()

# Write titles to a text file, each title on a new line
with open('titles.txt', 'w') as f:
    for title in titles:
        f.write(f"{title[0]}\n")

# Close the cursor and connection
cursor.close()
connection.close()
