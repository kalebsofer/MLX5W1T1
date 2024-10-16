# Use the official PostgreSQL image from Docker Hub
FROM postgres:14

# Environment variables for PostgreSQL were set in .env

# Install additional dependencies to connect to a remote Postgres database
RUN apt-get update && apt-get install -y postgresql-client python3-pip

# Install Python dependencies
RUN pip3 install psycopg2-binary sqlalchemy

# Copy initialization scripts (optional) if needed
# COPY ../sql/create_tables.sql /docker-entrypoint-initdb.d/

# Expose the default PostgreSQL port
EXPOSE 5432
