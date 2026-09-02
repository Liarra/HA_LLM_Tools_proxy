FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary project files
COPY front.py .
COPY diagnostics.py .
COPY embedding.py .
COPY smart_tokenizer.py .
COPY tools_storage.py .

RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV OPENAI_API_KEY=your_openai_api_key_here
ENV OPENAI_API_URL=https://api.openai.com/v1
ENV TOOLS_TO_KEEP=3
ENV MIN_TOOL_SIMILARITY=0.75

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "front:app", "--host", "0.0.0.0", "--port", "8000"]

# For loading environment variables from .env file
# We install python-dotenv in requirements.txt and the app loads them with dotenv.load_dotenv()
