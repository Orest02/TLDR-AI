# Use the official Python image as a parent image
FROM python:3.11

# Set environment variables
ENV POETRY_VERSION=1.4.0

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl git

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
        pip install tldrai && \
        pip install pytest pytest-mock  # Install testing dependencies

# Install Ollama
RUN curl -sSL https://ollama.com/install.sh | bash

# Run tests
CMD ["tldrai", "apply function to pandas column"]
