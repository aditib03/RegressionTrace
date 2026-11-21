# Public read-only demo (e.g. Hugging Face Spaces, Docker SDK; port 7860).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app && mkdir -p /home/app/batchsleuth && chown app:app /home/app/batchsleuth
WORKDIR /home/app/batchsleuth
COPY --chown=app pyproject.toml README.md ./
COPY --chown=app batchsleuth ./batchsleuth
COPY --chown=app benchmark ./benchmark
COPY --chown=app scripts ./scripts
RUN pip install --no-cache-dir -e .
USER app

ENV BATCHSLEUTH_HOME=/home/app/batchsleuth BATCHSLEUTH_N_JOBS=2 MLFLOW_DISABLE_AGENT_HINT=1
# Build the demo at image build time: champion, regressed candidate, and a repaired candidate.
RUN PY=python ./scripts/demo.sh

ENV BATCHSLEUTH_READ_ONLY=1
EXPOSE 7860
CMD ["uvicorn", "batchsleuth.api:app", "--host", "0.0.0.0", "--port", "7860"]
