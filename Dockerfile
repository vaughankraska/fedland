# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace

RUN pip install --no-cache-dir poetry==1.8.3

COPY . .

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# TODO: optimize and minimize

CMD ["sh", "-c", "fedn client start -n $(hostname) --init client-settings.yaml" ]
