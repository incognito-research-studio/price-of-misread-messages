#!/bin/zsh
# Paraphrase the b2 dev and train splits with Phi-4. The pilot was paraphrased first, on its own.
cd "${0:A:h}/.." || exit 1
for split in dev train; do
  echo "== $(date +%H:%M:%S) $split"
  .venv/bin/cdd -v paraphrase --streams data/v2/$split --out data/v2/$split-para --model phi4 > runs/paraphrase/v2-$split.log 2>&1
  echo "exit $?"
done
echo "== $(date +%H:%M:%S) done"
