# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update GPG keys (if needed) and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5003

# Expose the port for MLflow
EXPOSE 5004

# Set environment variables for MLflow
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT=/mlruns

# Run the application
CMD ["python", "run.py"]
