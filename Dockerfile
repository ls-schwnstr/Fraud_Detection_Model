# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update GPG keys
RUN apt-get update && apt-get install -y --no-install-recommends dirmngr gnupg

# Add this line to update the keys
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Git
RUN apt-get update && apt-get install -y git

# Expose the port the app runs on
EXPOSE 5003

# Expose the port for MLflow
EXPOSE 5004

# Set environment variables for MLflow
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT=/mlruns

# Run the application
CMD ["python", "run.py"]



