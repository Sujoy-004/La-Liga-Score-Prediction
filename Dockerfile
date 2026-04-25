# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 and 8000 available to the world outside this container
EXPOSE 7860
EXPOSE 8000

# Make start.sh executable
RUN chmod +x /app/start.sh

# Run start.sh when the container launches
CMD ["/app/start.sh"]
