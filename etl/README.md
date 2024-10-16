
# ETL Pipeline for Neural Network Training

This project provides an ETL (Extract, Transform, Load) pipeline to move data from an origin database to a PostgreSQL instance running in Docker. The goal is to prepare the data for training a neural network. The ETL process follows industry best practices to ensure that data extraction, transformation, and loading are performed in a reliable and replicable way.

## Overview

- **Extract**: Pull data from the origin database.
- **Transform**: Normalize, clean, and prepare data for machine learning.
- **Load**: Store transformed data into a Dockerized PostgreSQL instance for future use.

## Key Features
- **Dockerized Setup**: Easily replicate the environment with Docker and Docker Compose.
- **Data Preparation**: Data is transformed and normalized to be ready for training neural networks.
- **Scalable and Reusable**: Designed to be reused and adapted for various data extraction and transformation needs.

## Repository Structure
- **`/etl/docker`**: Docker-related files for setting up PostgreSQL.
- **`/etl/sql`**: SQL scripts used to create tables and manage the database.
- **`/etl/src`**: Python scripts for extraction, transformation, and loading.
- **`/etl/notebooks`**: Jupyter notebooks for data exploration and experiments.

This README will be updated with more detailed information on how to set up, run, and deploy the ETL process.

### Environment Setup
Copy the `.env.example` file inside the `/etl/docker` into `.env` and update with relevant credentials.

