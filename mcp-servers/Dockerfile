# MCP server image (project-plan.md's Dockerfiles slice, split into
# separate per-service Dockerfiles). Build from the repo root -- this
# service needs agents/, ml/, etc., not just this directory:
#   docker build -f mcp-servers/Dockerfile -t fraud-triage-mcp-server .
# Or use infra/docker-compose.yml, which builds and wires all 4 together.
#
# The 4 services (mcp-server, backend, admin, chat) each get their own
# Dockerfile rather than one shared multi-stage file, by explicit request --
# they still install the exact same locked dependency set as local dev
# (`uv sync` already respects pyproject.toml's default-groups =
# ["dev", "ml"]), deliberately not trimmed per-service: mcp-server is the
# only one that actually needs the "ml" group's heavy packages (mlflow,
# xgboost, etc. -- it imports ml/fraud_attribution.py at module level), but
# splitting dependency groups per container is a real refactor nobody's
# asked for, not a free optimization just because the Dockerfiles split.

FROM python:3.12-slim

# libgomp1: xgboost/lightgbm/catboost's real runtime dependency (OpenMP).
# opencv-python-headless deliberately avoids needing libGL -- do not add
# libgl1 "just in case"; the headless package choice was specifically to
# avoid that system dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first, in their own layer, so a code-only change doesn't
# invalidate the (large, slow-to-rebuild) uv sync layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV PYTHONUTF8=1
# Without this, Python fully block-buffers stdout when it isn't attached to
# a TTY (true for every container here) -- print()-based startup logs (the
# FastMCP banner, "Warming up FraudAttributor...", etc.) would sit in an
# internal buffer and never appear in `docker logs`, confirmed for real.
ENV PYTHONUNBUFFERED=1

EXPOSE 8001
CMD ["uv", "run", "python", "mcp-servers/orchestrator_server.py"]
