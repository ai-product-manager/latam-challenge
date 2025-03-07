# syntax=docker/dockerfile:1.2
#FROM python:latest
# Using this image due the application was developed with this version and is a lightweight official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y gcc g++ python3-dev libpq-dev

# Copy the project files into the container
COPY challenge /app/challenge
COPY data /app/data
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r /app/requirements.txt

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]