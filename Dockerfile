# 1) pick a stable, common base
FROM python:3.11-slim

# 2) make Python output unbuffered (better logs in Docker)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3) set working directory inside the container
WORKDIR /app

# 4) install OS deps (pandas/numpy sometimes need these;
#    add others here if your requirements need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5) copy ONLY requirements first (better build cache)
COPY requirements.txt /app/requirements.txt

# 6) install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7) now copy the rest of the repo
#    this will bring in src/, including src/data/*.csv
COPY . /app

# 8) default command: run your pipeline
CMD ["python", "src/python_code/main.py"]
