FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY feedback_agent ./feedback_agent
# editable install keeps data/ and samples/ next to the code, where the app looks for them
RUN pip install --no-cache-dir -e ".[dev]"
COPY data ./data
COPY samples ./samples
COPY eval ./eval
COPY tests ./tests
RUN useradd -m app && mkdir -p /app/state && chown -R app /app
USER app
ENV DATABASE_PATH=/app/state/feedback_agent.db TRACE_DIR=/app/state/traces
VOLUME /app/state
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["python", "-m", "feedback_agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
