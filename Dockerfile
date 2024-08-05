# Use the official Python image as a parent image
FROM python:3.11

# Set environment variables
ENV POETRY_VERSION=1.4.0

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl git && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    poetry config virtualenvs.create false

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    poetry lock && poetry install --with test

# Install Ollama
RUN curl -sSL https://ollama.com/install.sh | bash

# Run tests
CMD ["poetry", "run", "pytest"]
