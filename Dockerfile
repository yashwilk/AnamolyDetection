# --- builder: install dependencies into a venv, isolated from build tools ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- runtime: copy only the venv and application code ---
FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY app ./app
COPY configs ./configs
# models/ ships whatever has been trained into the build context (only
# .gitkeep by default). Mount a real trained models/ dir as a volume, or
# run `python -m src.models.train` before building, before serving traffic.
COPY models ./models

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
