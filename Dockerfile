# Use the official Python image as a base image
FROM python:3.13-slim

# Expose the port that Streamlit runs on (default Streamlit port)
EXPOSE 8501

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . ./

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures that pip does not store downloaded packages in a cache
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Streamlit application
# We specify chef.py, port 8501, and disable CORS/XSRF protection for compatibility
ENTRYPOINT ["streamlit", "run", "chef.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
