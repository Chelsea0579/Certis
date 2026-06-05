#!/usr/bin/env bash
# Certis: end-to-end reproduction pipeline.
# Run from the repository root: `bash run.sh`.
# Stages run in order; each script reads the previous stage's outputs and uses
# its own argparse defaults (override on the command line if needed).
#
# Prerequisites:
#   - Python >= 3.9 and the packages in requirements.txt
#   - The datasets in data/data_links.txt downloaded under data/
#   - A GPU and access to the four instruction-tuned LLMs used for the
#     semantic channel (see the paper, Experimental Setup)

set -euo pipefail
cd "$(dirname "$0")"

SEEDS="42 1 7 100"

echo "[0/4] Install dependencies"
python -m pip install -r requirements.txt

echo "[1/4] Prepare datasets (CoDEx-M, FB15k-237)"
python code/tools/prep_codex_m.py
python code/tools/prep_fb15k237.py

echo "[2/4] Candidate generation, TransE structural features, LLM logit features"
python code/candidates_generation/triple_gen.py
python code/candidates_filtering/embedding/train_model.py
python code/candidates_filtering/embedding/get_emb_transe.py
python code/experiment/prep_llm.py

echo "[3/4] Leakage-free per-seed protocol: split -> freeze -> certify -> locked test"
python code/tools/redesign/01_build_protocol_ledger.py --seeds $SEEDS
python code/tools/redesign/06_assemble_features.py
python code/tools/redesign/09b_freeze_and_certify_perseed.py
python code/tools/redesign/10_test_perseed.py

echo "[4/4] Certificates, baselines, ablations, and F1 reporting"
python code/tools/redesign/cp_onetime_test.py
python code/tools/redesign/positive_commit_cert.py
python code/tools/redesign/per_channel_cert.py
python code/tools/redesign/rotate_selective.py
python code/tools/redesign/08b_controls_ablations_perseed.py
python code/tools/redesign/perseed_f1.py

echo "Done. Selective-risk and false-insertion-rate bounds are printed above;"
echo "detailed per-seed outputs are written under code/tools/redesign/."
