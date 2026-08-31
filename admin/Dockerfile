# Admin Streamlit app image (project-plan.md's Dockerfiles slice, split
# into separate per-service Dockerfiles). Build from the repo root -- see
# mcp-servers/Dockerfile's header comment for why this shares the full
# locked dependency set rather than a trimmed one.
#   docker build -f admin/Dockerfile -t fraud-triage-admin .
# Or use infra/docker-compose.yml, which builds and wires all 4 together.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "admin/app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
