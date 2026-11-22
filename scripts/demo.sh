#!/bin/bash
# End-to-end demo: init -> regression check -> diagnose -> repair -> promote.
set -euo pipefail
cd "$(dirname "$0")/.."
BS="${PY:-.venv/bin/python} -m regressiontrace.cli"
export MLFLOW_DISABLE_AGENT_HINT=1
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
quiet() { grep -v -E "INFO mlflow|Registered model|Created version|Successfully registered" || true; }

echo "== 1. ingest batches, train champion (v1) and refresh candidate (v2)"
$BS init --force 2>&1 | quiet | tee "$TMP/init.txt"
RUN=$(grep "candidate run" "$TMP/init.txt" | awk '{print $3}')

echo; echo "== 2. diagnose the regressed candidate"
$BS diagnose "$RUN" 2>&1 | quiet

echo; echo "== 3. counterfactual repair: drop suspect batches, retrain, run promotion gate"
$BS repair "$RUN" --action drop 2>&1 | quiet | tee "$TMP/repair.txt"
NEW=$(grep -- "-> version" "$TMP/repair.txt" | awk '{print $NF}')

echo; echo "== 4. promote (only if every gate check passes)"
$BS promote "$NEW" 2>&1 | quiet | tail -1

echo; echo "== ground truth (hidden from diagnosis)"
$BS faults 2>&1 | quiet
