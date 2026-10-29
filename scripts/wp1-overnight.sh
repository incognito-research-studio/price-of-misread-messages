#!/bin/zsh
# WP1, the fresh-stream replication (docs/research/replication-protocol.md), unattended:
# generate and paraphrase the fresh split, run it, analyse it, then export Laya's output.
# Run from the checkout of tag icores-replication-freeze-v2 with Ollama serving:
#   zsh scripts/wp1-overnight.sh
# Each step logs to runs/; a failed step does not stop the next one. Safe to start again after a crash: existing streams are kept, paraphrases
# that already exist are kept, and the run resumes instead of starting over.
cd "${0:A:h}/.." || exit 1
CDD=.venv/bin/cdd
PY=.venv/bin/python
OLLAMA=${OLLAMA:-/Applications/Ollama.app/Contents/Resources/ollama}
RUN=icores-replication-para
step() { echo "== $(date +%H:%M:%S) $1"; }
mkdir -p runs/paraphrase

step "commit $(git rev-parse --short HEAD), tag $(git describe --exact-match --tags HEAD 2>/dev/null || echo NONE)"
if [[ "$(git describe --exact-match --tags HEAD 2>/dev/null)" != "icores-replication-freeze-v2" ]]; then
  echo "STOP: not on tag icores-replication-freeze-v2 (git switch --detach icores-replication-freeze-v2)"; exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: the checkout has changes; every run would be marked dirty"; git status --short; exit 1
fi
# importlib.metadata, not pip: a venv made by uv has no pip.
LAYA=$($PY -c "import importlib.metadata as m; print(m.version('laya'))" 2>/dev/null)
step "laya ${LAYA:-not installed}"
if [[ "$LAYA" != "0.4.1" ]]; then echo "STOP: laya must be 0.4.1"; exit 1; fi

step "generate the fresh split (configs/benchmark/confirm.yaml, once)"
if [[ -d data/v2/confirm ]]; then
  echo "data/v2/confirm exists; kept"
else
  $CDD generate --config configs/benchmark/confirm.yaml --out data/v2/confirm; echo "exit $?"
fi

step "paraphrase with Phi-4"
count() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }
if [[ $(count data/v2/confirm-para) -gt 0 && $(count data/v2/confirm-para) -eq $(count data/v2/confirm) ]]; then
  echo "data/v2/confirm-para complete; kept"
else
  $CDD -v paraphrase --streams data/v2/confirm --out data/v2/confirm-para --model phi4 \
    > runs/paraphrase/wp1-confirm.log 2>&1; echo "exit $?"
fi

step "replication run $RUN"
if [[ -d runs/$RUN ]]; then
  $CDD -v run --config configs/experiments/icores-replication.yaml --run-name $RUN --resume \
    >> runs/$RUN.log 2>&1; echo "exit $?"
else
  $CDD -v run --config configs/experiments/icores-replication.yaml --run-name $RUN \
    > runs/$RUN.log 2>&1; echo "exit $?"
fi
if [[ "$(cat runs/$RUN/failed.json 2>/dev/null)" != "[]" ]]; then
  step "failed pairs: one resume (protocol: rerun once, then report as missing)"
  $CDD -v run --config configs/experiments/icores-replication.yaml --run-name $RUN --resume \
    >> runs/$RUN.log 2>&1; echo "exit $?"
fi
{ $OLLAMA --version; $OLLAMA list; } > runs/$RUN/ollama.txt 2>&1
$PY scripts/scan_run.py runs/$RUN data/v2/confirm-para > runs/$RUN.scan 2>&1; tail -3 runs/$RUN.scan

step "analysis (primary and secondary)"
$PY scripts/replication_analysis.py runs/$RUN data/v2/confirm-para > runs/$RUN.analysis 2>&1
sed -n '1,12p' runs/$RUN.analysis

step "Laya export for WP3 (no reply cache: asks Laya again)"
$CDD -v run --config configs/experiments/icores-export-laya.yaml --run-name icores-export-laya-para \
  > runs/icores-export-laya-para.log 2>&1; echo "exit $?"
$PY scripts/check_export.py runs/icores-export-laya-para runs/icores-final-readers-para-r3-shapley \
  > runs/icores-export-laya-para.gate 2>&1; tail -3 runs/icores-export-laya-para.gate
step "done"
