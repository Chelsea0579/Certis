"""Download CoDEx-M from the official repo, assemble the data CSV the CERTIS
pipeline expects, then run:
  1. random 80/20 train/missing split (matches paper's protocol)
  2. candidate generation (Algorithms 1+2)
  3. TransE training + threshold-based filtering
  4. balanced 500-row evaluation sample (250 true + 250 false)

Outputs at <out_dir>:
  data_sample.csv      -- the 80% retained KG, columns: Head,Relation,Tail
  cand_sample_500.csv  -- 500 evaluation candidates (250 missing + 250 non-missing)
                          with a "Missing" column for ground truth.
"""
import argparse, os, sys, random, subprocess, json
import pandas as pd
import numpy as np

CODEX_URL_BASE = "https://raw.githubusercontent.com/tsafavi/codex/master"

# Allow importing the local certis repo modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Replace the O(n^2) candidate generator with the vectorized version
import fast_triple_gen
fast_triple_gen.install()
import fast_dist
fast_dist.install()
# Also patch the filtering module which has its own get_list_dist import
import candidates_filtering.triple_filter as _tf
from candidates_filtering.embedding.get_emb_transe import get_list_dist as _gld
_tf.get_list_dist = _gld
from experiment import filtering as _flt
_flt.get_list_dist = _gld


def download(url: str, dst: str):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # use curl with already-exported proxy env
    rc = subprocess.call(["curl", "-sSL", "-o", dst, url])
    if rc != 0 or os.path.getsize(dst) == 0:
        raise RuntimeError(f"failed to download {url}")


def load_codex_m(work_dir: str) -> pd.DataFrame:
    files = ["train.txt", "valid.txt", "test.txt"]
    dfs = []
    for f in files:
        local = os.path.join(work_dir, "codex_m_raw", f)
        url = f"{CODEX_URL_BASE}/data/triples/codex-m/{f}"
        download(url, local)
        df = pd.read_csv(local, sep="\t", header=None, names=["h", "r", "t"])
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    print(f"[prep] loaded {len(full)} raw codex-m triples")

    # Optionally remap entity IDs (Q123) and relations (P31) to surface labels
    ent_url = f"{CODEX_URL_BASE}/data/entities/en/entities.json"
    rel_url = f"{CODEX_URL_BASE}/data/relations/en/relations.json"
    ent_local = os.path.join(work_dir, "codex_m_raw", "entities.json")
    rel_local = os.path.join(work_dir, "codex_m_raw", "relations.json")
    try:
        download(ent_url, ent_local)
        download(rel_url, rel_local)
        ent = json.load(open(ent_local))
        rel = json.load(open(rel_local))
        full["Head"] = full["h"].map(lambda k: ent.get(k, {}).get("label", k))
        full["Tail"] = full["t"].map(lambda k: ent.get(k, {}).get("label", k))
        full["Relation"] = full["r"].map(lambda k: rel.get(k, {}).get("label", k))
        print("[prep] mapped IDs -> labels")
    except Exception as e:
        print(f"[prep] WARN: label mapping failed ({e}); falling back to raw IDs")
        full["Head"], full["Relation"], full["Tail"] = full["h"], full["r"], full["t"]

    return full[["Head", "Relation", "Tail"]].drop_duplicates().reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/codex-m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_proportion", type=float, default=0.8,
                   help="fraction of triples kept as the observed graph (paper: 0.8)")
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--true_ratio", type=float, default=0.5)
    p.add_argument("--n_subset", type=int, default=0,
                   help="if >0, subsample the full KG to this size (for faster runs)")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_codex_m(args.out_dir)

    if args.n_subset and args.n_subset < len(df):
        df = df.sample(n=args.n_subset, random_state=args.seed).reset_index(drop=True)
        print(f"[prep] subsampled to {len(df)} triples")

    # === paper protocol: remove 20% uniformly at random; the kept 80% is data_sample ===
    sample_size = int(args.sample_proportion * len(df))
    sample_df = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    missing_df = df[~df.apply(tuple, axis=1).isin(sample_df.apply(tuple, axis=1))].reset_index(drop=True)
    print(f"[prep] data_sample={len(sample_df)}  missing={len(missing_df)}")

    sample_path = os.path.join(args.out_dir, "data_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[prep] wrote {sample_path}")

    # === Algorithms 1+2: candidate generation on sample_df ===
    cand_csv = os.path.join(args.out_dir, "candidates_full.csv")
    if os.path.exists(cand_csv) and os.path.getsize(cand_csv) > 1000:
        print(f"[prep] loading existing candidates from {cand_csv}")
        cand_df = pd.read_csv(cand_csv)
        print(f"[prep] loaded {len(cand_df)} candidates")
    else:
        from candidates_generation import triple_gen
        cand_df = triple_gen.generate_all_candidates(sample_df)
        print(f"[prep] generated {len(cand_df)} candidates")
        cand_df.to_csv(cand_csv, index=False)

    # Label candidates with ground truth (Missing=1 if it exists in missing_df)
    miss_keys = set(map(tuple, missing_df.values))
    cand_df["Missing"] = cand_df.apply(lambda r: 1 if (r["Head"], r["Relation"], r["Tail"]) in miss_keys else 0, axis=1)
    n_true = (cand_df["Missing"] == 1).sum()
    n_false = (cand_df["Missing"] == 0).sum()
    print(f"[prep] candidate ground truth: true={n_true} false={n_false}")

    # === TransE filter ===
    from experiment import filtering
    # Use a smaller missing_df for the filtering step (just to compute coverage); keep all candidates
    print("[prep] training TransE for filtering...")
    filtred_df = filtering.create_filtred_df(sample_df, cand_df[["Head", "Relation", "Tail"]].copy(), missing_df)
    print(f"[prep] after TransE filter: {len(filtred_df)} candidates kept")
    filtred_df.to_csv(os.path.join(args.out_dir, "candidates_filtered.csv"), index=False)

    # Tag filtered with the same Missing labels (recompute since DataFrame may have lost it)
    filtred_df["Missing"] = filtred_df.apply(
        lambda r: 1 if (r["Head"], r["Relation"], r["Tail"]) in miss_keys else 0, axis=1)
    n_t = (filtred_df["Missing"] == 1).sum()
    n_f = (filtred_df["Missing"] == 0).sum()
    print(f"[prep] filtered ground truth: true={n_t} false={n_f}")

    # === Balanced 500-row evaluation sample ===
    n_per = int(args.n_eval * args.true_ratio)
    pool_t = filtred_df[filtred_df["Missing"] == 1]
    pool_f = filtred_df[filtred_df["Missing"] == 0]
    take_t = min(n_per, len(pool_t))
    take_f = min(args.n_eval - take_t, len(pool_f))
    eval_df = pd.concat([
        pool_t.sample(n=take_t, random_state=args.seed),
        pool_f.sample(n=take_f, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out_eval = os.path.join(args.out_dir, "cand_sample_500.csv")
    eval_df.to_csv(out_eval, index=False)
    print(f"[prep] wrote {out_eval}: true={take_t} false={take_f}")


if __name__ == "__main__":
    main()
