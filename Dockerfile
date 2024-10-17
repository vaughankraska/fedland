# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    FEDN_NUM_DATA_SPLITS=5

RUN if [ ! -z "$GRPC_HEALTH_PROBE_VERSION" ]; then \
        apt-get install -y wget && \
        wget -qO /bin/grpc_health_probe \
        https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
        chmod +x /bin/grpc_health_probe && \
        apt-get remove -y wget && \
        apt autoremove -y; \
    else \
        echo "No grpc_health_probe version specified, skipping installation"; \
    fi

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
RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["fedn", "client", "start", "--init", "client-setting.yaml"]
