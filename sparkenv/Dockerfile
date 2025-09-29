
FROM python:3.13-slim

WORKDIR /app

# Prevents interactive prompts when installing packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

COPY pyproject.toml ./
RUN pip install uv && \
    uv pip install --no-cache --system .

COPY src/ ./src
    
CMD ["spark-submit", "main.py"]
