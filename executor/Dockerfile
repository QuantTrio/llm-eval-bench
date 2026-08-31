FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 llmbench

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir '.[executor]'

USER llmbench
EXPOSE 8765
CMD ["llmbench", "executor", "serve", "--config", "/config/executor.yaml"]
