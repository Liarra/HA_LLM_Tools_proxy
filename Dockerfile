FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary project files
COPY front.py .
COPY embedding.py .
COPY tools_storage.py .

# Create logs directory
RUN mkdir -p /app/logs
RUN mkdir -p /app/data

# Set environment variables
ENV OPENAI_API_KEY=your_openai_api_key_here
ENV OPENAI_API_URL=https://api.openai.com/v1
ENV TOOLS_TO_KEEP=3

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "front:app", "--host", "0.0.0.0", "--port", "8000"]

# For loading environment variables from .env file
# We install python-dotenv in requirements.txt and the app loads them with dotenv.load_dotenv()