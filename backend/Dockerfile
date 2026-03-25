FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY requirements.txt /build/requirements.txt
RUN pip wheel --wheel-dir /build/wheels -r /build/requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY app /app/app
COPY api /app/api
COPY ingestion /app/ingestion
COPY voice /app/voice

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
