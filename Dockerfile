# Stage 1: Build stage
FROM python:3.13-slim AS builder
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt      


# Stage 2: Runtime stage
FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .
RUN rm -rf /root/.cache

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888" ]
