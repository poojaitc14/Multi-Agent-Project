# Customer chat Streamlit app image (project-plan.md Q78 / Dockerfiles
# slice, split into separate per-service Dockerfiles). Build from the repo
# root -- see mcp-servers/Dockerfile's header comment for why this shares
# the full locked dependency set rather than a trimmed one.
#   docker build -f chat/Dockerfile -t fraud-triage-chat .
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

EXPOSE 8503
CMD ["uv", "run", "streamlit", "run", "chat/app.py", "--server.headless", "true", "--server.port", "8503", "--server.address", "0.0.0.0"]
