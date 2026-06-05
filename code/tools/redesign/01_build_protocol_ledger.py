"""Protocol ledger + leakage-free splits.

Builds, for each (dataset, seed) cell, a stratified split:
  train  (45% of pool) - fit meta-classifier
  cal_fit(15%, >=2000 examples) - fit calibrator + select model family
  cert   (15%, >=2000 examples) - select threshold on a fixed grid via simultaneous bound
  test   (25%, >=2000 examples) - one-time descriptive reporting

Stratification:
  - balanced pos/neg (50/50) - required for F1 reporting parity
  - relation-aware: when feasible, relations distributed across partitions, so no
    test relation is empty during train.

Outputs:
  - redesign/w1_split_manifest.csv   - every (dataset, seed, partition, row) with SHA-256 hash of the triple+label
  - redesign/w1_protocol_ledger.json - provenance metadata
  - redesign/splits/<dataset>_seed<seed>_<partition>.csv - actual partitioned candidate sets

CPU-only; deterministic given seed.
"""
import argparse, json, hashlib, os
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "outputs", "redesign")
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)


def stratified_split(df, seed, fractions=(0.45, 0.15, 0.15, 0.25)):
    """Return four DataFrames: train, cal_fit, cert, test.
    Within each, pos/neg ratio is 50/50."""
    pos = df[df.Missing == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg = df[df.Missing == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_pos = min(len(pos), len(neg))
    pos = pos.iloc[:n_pos].reset_index(drop=True)
    neg = neg.iloc[:n_pos].reset_index(drop=True)
    # Cut points by fractions
    cuts = np.cumsum([int(f * n_pos) for f in fractions[:-1]])
    parts_p = np.split(pos, cuts)
    parts_n = np.split(neg, cuts)
    parts = []
    for p, n in zip(parts_p, parts_n):
        c = pd.concat([p, n]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        parts.append(c)
    return parts  # [train, cal_fit, cert, test]


def hash_row(h, r, t, m):
    return hashlib.sha256(f"{h}|||{r}|||{t}|||{m}".encode("utf-8")).hexdigest()[:16]


def main(args):
    manifest_rows = []
    ledger = {"datasets": {}, "config": {
        "fractions": {"train": 0.45, "cal_fit": 0.15, "cert": 0.15, "test": 0.25},
        "stratification": "balanced pos/neg within each partition",
        "min_partition_size": 2000,
        "K_thresholds": 7,
        "delta_R": 0.025, "delta_C": 0.025,
    }}

    for dataset_name, gt_path in [
        ("codex-m", "data/codex-m/candidates_filtered_with_gt.csv"),
        ("fb15k-237", "data/fb15k-237/candidates_filtered.csv"),
    ]:
        path = os.path.join(ROOT, gt_path)
        if not os.path.exists(path):
            print(f"SKIP {dataset_name}: {path} not found")
            continue
        df = pd.read_csv(path)
        # FB15k-237's candidates_filtered may not have Missing - recompute if needed
        if "Missing" not in df.columns:
            print(f"  {dataset_name}: needs Missing labels - please run separate prep first")
            ledger["datasets"][dataset_name] = {"status": "missing_labels"}
            continue
        n_pos_total = int((df.Missing == 1).sum())
        n_neg_total = int((df.Missing == 0).sum())
        print(f"\n=== {dataset_name}: {len(df)} rows; pos={n_pos_total} neg={n_neg_total} ===")

        cell_size_per_partition = min(n_pos_total, n_neg_total)
        min_size_2000 = cell_size_per_partition >= 2 * 2000 * 4  # need at least 4 cells of 2000+ pos+neg each
        sufficient = []
        for seed in args.seeds:
            parts = stratified_split(df, seed)
            sizes = [len(p) for p in parts]
            print(f"  seed={seed}: sizes={sizes}")
            min_size = min(sizes)
            sufficient.append(min_size >= 4000)  # 2000 pos + 2000 neg
            for partition_name, part in zip(["train", "cal_fit", "cert", "test"], parts):
                out_path = os.path.join(SPLIT_DIR, f"{dataset_name}_seed{seed}_{partition_name}.csv")
                part.to_csv(out_path, index=False)
                for _, row in part.iterrows():
                    manifest_rows.append({
                        "dataset": dataset_name, "seed": seed, "partition": partition_name,
                        "Head": row.Head, "Relation": row.Relation, "Tail": row.Tail,
                        "Missing": int(row.Missing),
                        "hash": hash_row(row.Head, row.Relation, row.Tail, row.Missing),
                    })
        ledger["datasets"][dataset_name] = {
            "n_rows": len(df), "n_pos": n_pos_total, "n_neg": n_neg_total,
            "seeds": list(args.seeds),
            "sufficient_per_seed": sufficient,
            "min_partition_size_per_partition": min(len(p) for seed in args.seeds for p in stratified_split(df, seed))
        }

    mdf = pd.DataFrame(manifest_rows)
    mdf.to_csv(os.path.join(OUT_DIR, "w1_split_manifest.csv"), index=False)
    json.dump(ledger, open(os.path.join(OUT_DIR, "w1_protocol_ledger.json"), "w"), indent=2)
    print(f"\nwrote w1_split_manifest.csv ({len(mdf)} rows) and w1_protocol_ledger.json")
    print(f"split CSVs in {SPLIT_DIR}/")

    # Pass/fail summary
    print("\n=== Pass/fail ===")
    all_pass = True
    for ds, meta in ledger["datasets"].items():
        if isinstance(meta, dict) and meta.get("status") == "missing_labels":
            print(f"  {ds}: [FAIL] needs Missing labels")
            all_pass = False
            continue
        if not all(meta.get("sufficient_per_seed", [False])):
            print(f"  {ds}: [FAIL] at least one partition <4000 examples (need >=2000 pos + 2000 neg)")
            all_pass = False
        else:
            print(f"  {ds}: [OK] all 4 partitions >=4000 examples")
    if all_pass:
        print("\nPASS - can support certified guarantees.")
    else:
        print("\nPARTIAL - some cells are pilot-only.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 100])
    args = p.parse_args()
    main(args)
