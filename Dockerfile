# Public read-only demo (e.g. Hugging Face Spaces, Docker SDK; port 7860).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app && mkdir -p /home/app/regressiontrace && chown app:app /home/app/regressiontrace
WORKDIR /home/app/regressiontrace
COPY --chown=app pyproject.toml README.md ./
COPY --chown=app regressiontrace ./regressiontrace
COPY --chown=app benchmark ./benchmark
COPY --chown=app scripts ./scripts
RUN pip install --no-cache-dir -e .
USER app

ENV REGRESSIONTRACE_HOME=/home/app/regressiontrace REGRESSIONTRACE_N_JOBS=2 MLFLOW_DISABLE_AGENT_HINT=1
# Build the demo at image build time: champion, regressed candidate, and a repaired candidate.
RUN PY=python ./scripts/demo.sh

ENV REGRESSIONTRACE_READ_ONLY=1
EXPOSE 7860
CMD ["uvicorn", "regressiontrace.api:app", "--host", "0.0.0.0", "--port", "7860"]
