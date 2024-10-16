# Use the official PostgreSQL image from Docker Hub
FROM postgres:14

# Environment variables for PostgreSQL were set in .env

# Copy initialization scripts (optional) if needed
COPY ../sql/create_tables.sql /docker-entrypoint-initdb.d/

# Expose the default PostgreSQL port
EXPOSE 5432
