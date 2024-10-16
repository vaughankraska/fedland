# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    FEDN_NUM_DATA_SPLITS=5

WORKDIR /workspace

ARG UID=10001
RUN adduser --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY . .

RUN pip install --no-cache-dir fedn==0.16.1

RUN fedn package create --path client && \
    fedn run build --path client --keep-venv

USER appuser

CMD ["fedn", "client", "start", "--init", "settings-client.yaml"]
