# Certis

*Certis: Risk-Certified Fact Insertion for
Knowledge Graph Construction*.

Certis is a calibrated commit / reject / review gate for LLM-based knowledge
graph completion. It fuses LLM semantic confidences, structural graph features,
and an optional KGE score, calibrates them into a probability, and attaches
distribution-free, finite-sample certificates (selective risk/coverage and a
false-insertion rate) over a fixed candidate workload.

## Repository layout

| Path | Contents |
|------|----------|
| `code/candidates_generation/` | candidate triple generation |
| `code/candidates_filtering/`  | structural filtering and TransE embeddings |
| `code/experiment/`            | LLM feature capture and result aggregation |
| `code/tools/`                 | dataset preparation and helpers |
| `code/tools/redesign/`        | leakage-free per-seed protocol: splits, selective-risk and false-insertion-rate certificates, locked-test evaluation, baselines, and ablations |
| `data/data_links.txt`         | dataset sources |
| `requirements.txt`            | Python dependencies |

## Requirements

Python >= 3.9. Install dependencies with:

```
pip install -r requirements.txt
```

Feature capture for the semantic channel requires a GPU and access to the four
instruction-tuned LLMs listed in the paper.

## Reproduce

1. Download the datasets listed in `data/data_links.txt` into `data/`.
2. Run the full pipeline:

```
bash run.sh
```

Alternatively, run the stages in `code/tools/redesign/` individually in their
numeric order. Each script accepts `--seeds` (default `42 1 7 100`) and writes
its outputs under `code/tools/redesign/`.
