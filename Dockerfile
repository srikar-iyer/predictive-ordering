FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories for outputs
RUN mkdir -p /app/static/images/inventory /app/static/images/profit /app/static/images/time_series \
    /app/models/time_series /app/test_output /app/output /app/csv-data

# Set proper permissions
RUN chmod -R 777 /app/static /app/models /app/test_output /app/output /app/csv-data

# Create directory for CSV data
RUN mkdir -p /app/csv-data

# Copy CSV data files to the csv-data directory
COPY *.csv /app/csv-data/

# Create symlinks for backward compatibility
RUN for file in /app/csv-data/*.csv; do ln -sf "$file" /app/; done

# Copy Python files
COPY *.py /app/

# Copy static files
COPY static/ /app/static/

# Copy models directory
COPY models/ /app/models/

# Copy test_output directory if it exists
# Copy test_output directory
COPY test_output/ /app/test_output/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8050
ENV HOST=0.0.0.0

# Make port 8050 available for Plotly Dash
EXPOSE 8050

# Healthcheck to verify the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
  CMD curl -f http://localhost:8050/ || exit 1

# Start the application using the main dashboard
CMD ["python", "plotly_dashboard.py"]