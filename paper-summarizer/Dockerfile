# syntax=docker/dockerfile:1.7
# One Dockerfile, two targets:
#   runtime  mock-llm, and the Day 4 api service (FastAPI + LiteLLM + Chroma)
#   dev      runtime + pytest, ruff, ipykernel/nbclient (headless notebook runs); package installed editable
# There is no Jupyter server: notebooks are run in Google Colab.
# Multi-arch: python:3.11-slim publishes linux/amd64 and linux/arm64, and every locked
# dependency ships wheels for both, so no compiler is needed in either stage.

ARG PYTHON_IMAGE=python:3.11.17-slim-trixie

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ARG UID=1000
ARG GID=1000
# Corporate TLS inspection: any certs/*.crt is added to the system trust store (see README).
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates >/dev/null \
    && groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app

# ---------------------------------------------------------------------------
FROM base AS deps-runtime
COPY requirements.lock /tmp/requirements.lock
RUN python -m venv /opt/venv && pip install -r /tmp/requirements.lock
# Bake tokenizer files into the image so tiktoken works offline (Day 1 instructor demo D3.2).
RUN python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('gpt2', 'r50k_base', 'cl100k_base', 'o200k_base')]"

FROM deps-runtime AS deps-dev
COPY requirements-dev.lock /tmp/requirements-dev.lock
RUN pip install -r /tmp/requirements-dev.lock

# The runtime never installs packages: drop pip and setuptools (~28 MB).
FROM deps-runtime AS deps-runtime-slim
RUN python -m pip uninstall -y -q pip setuptools

# ---------------------------------------------------------------------------
FROM base AS runtime
COPY --from=deps-runtime-slim /opt/venv /opt/venv
COPY --from=deps-runtime /opt/tiktoken /opt/tiktoken
WORKDIR /app
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app paper_agent ./paper_agent
COPY --chown=app:app mock_llm ./mock_llm
ENV PYTHONPATH=/app
# The api service keeps uploaded papers and its index here (a named volume inherits this owner).
RUN mkdir -p /data && chown app:app /data
USER app
EXPOSE 8000
CMD ["uvicorn", "mock_llm.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
FROM base AS dev
COPY --from=deps-dev /opt/venv /opt/venv
COPY --from=deps-runtime /opt/tiktoken /opt/tiktoken
WORKDIR /workspace
COPY --chown=app:app . .
# compat mode writes a .pth pointing at /workspace, so the bind-mounted source is what gets imported.
RUN pip install --no-deps -e . --config-settings editable_mode=compat \
    && chown -R app:app /workspace
USER app
CMD ["pytest", "-q"]
