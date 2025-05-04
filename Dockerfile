FROM python:3.11-slim

# Install system dependencies including git and wget
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Remove Git Clone steps
# RUN git clone https://github.com/McGill-MMA-EnterpriseAnalytics/Employee-Attrition-2.git . && \
#    git checkout main

# Copy local project files to the container
COPY . /app

# Install Poetry and dependencies
RUN pip install --upgrade pip && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    /root/.local/bin/poetry config virtualenvs.create false && \
    /root/.local/bin/poetry install --without dev --no-interaction

# Set environment variables including PYTHONPATH
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Start FastAPI with Uvicorn using the path relative to PYTHONPATH
CMD ["uvicorn", "src.employee_attrition_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]