#!/bin/zsh
# The pilot runs on benchmark b2: readers, model deciders and baselines. Start `ollama serve` first.
# Each step logs to runs/<name>.log and is scanned with scripts/scan_run.py; a failed step does
# not stop the next one, and any run can be resumed with --resume.
cd "${0:A:h}/.." || exit 1
CDD=.venv/bin/cdd
PY=.venv/bin/python
OLLAMA=${OLLAMA:-/Applications/Ollama.app/Contents/Resources/ollama}
step() { echo "== $(date +%H:%M:%S) $1"; }
run() {  # run NAME CONFIG STREAMS
  step "$1"
  $CDD -v run --config $2 --streams $3 --run-name $1 > runs/$1.log 2>&1; echo "exit $?"
  $PY scripts/scan_run.py runs/$1 $3 > runs/$1.scan 2>&1; tail -3 runs/$1.scan
  { $OLLAMA --version; $OLLAMA list; } > runs/$1/ollama.txt 2>&1
}
step "commit $(git rev-parse --short HEAD), dirty: $(git status --porcelain | wc -l | tr -d ' ') files"
run icores-final-readers-para configs/experiments/local-readers.yaml data/v2/pilot-para
run icores-final-deciders-para configs/experiments/local-deciders.yaml data/v2/pilot-para
run icores-final-readers-template configs/experiments/local-readers.yaml data/v2/pilot
run icores-final-baselines-dev configs/experiments/local-baselines.yaml data/v2/dev-para
step "train paraphrase"
$CDD -v paraphrase --streams data/v2/train --out data/v2/train-para --model phi4 > runs/paraphrase/v2-train.log 2>&1; echo "exit $?"
step "numbers"
$PY scripts/paper_numbers.py icores-final-readers-para icores-final-readers-template icores-final-deciders-para > runs/icores-final-numbers.txt 2>&1
step "done"
