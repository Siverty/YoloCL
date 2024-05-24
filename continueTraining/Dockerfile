# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Install torch and torchvision separately because it is not available in the default pip repository
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Exposing port 9000 as a check to make sure it runs
EXPOSE 9000

# Run main.py when the container launches
CMD ["python", "main.py"]
