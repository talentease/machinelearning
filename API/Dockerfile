# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# Use a base image with Python pre-installed
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose any necessary ports (if applicable)
# EXPOSE <port_number>

# Define the command to run the application
CMD [ "python", "main.py" ]
