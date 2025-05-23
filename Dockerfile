# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that uvicorn will run on
EXPOSE 8000

# Add the script directly into the Dockerfile
CMD /bin/bash -c "python main.py && uvicorn src.api:app --host 0.0.0.0 --port 8000"