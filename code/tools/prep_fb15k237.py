"""Build FB15k-237 CERTIS pipeline inputs:
  - data_sample.csv (80% retained KG)
  - candidates_full.csv (via fast_triple_gen)
  - candidates_filtered.csv (via TransE)
  - cand_sample_500.csv (250+250 balanced)

FB15k-237 uses raw Freebase mids (`/m/0d_b14`) for entities and relation paths
(`/people/person/profession`) for relations. We keep them as-is - they're already
mid-obfuscated since LLMs barely memorize Freebase mids - which makes FB15k-237
a useful "naturally obfuscated" complement to CoDEx-M's Wikidata-label dataset.

Uses PyKEEN's FB15k237 dataset class.
"""
import argparse, os, sys, random
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fast_triple_gen
fast_triple_gen.install()
import fast_dist
fast_dist.install()


def main(args):
    np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from pykeen.datasets import FB15k237
    dataset = FB15k237()
    # The PyKEEN dataset is a TriplesFactory - extract labels
    full_triples = []
    for tf in [dataset.training, dataset.validation, dataset.testing]:
        labeled = tf.triples  # numpy array of (head_label, relation_label, tail_label)
        full_triples.append(pd.DataFrame(labeled, columns=["Head", "Relation", "Tail"]))
    df = pd.concat(full_triples, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"[fb] loaded {len(df)} triples from FB15k-237")

    if args.n_subset and args.n_subset < len(df):
        df = df.sample(n=args.n_subset, random_state=args.seed).reset_index(drop=True)
        print(f"[fb] subsampled to {len(df)}")

    sample_size = int(args.sample_proportion * len(df))
    sample_df = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    missing_df = df[~df.apply(tuple, axis=1).isin(sample_df.apply(tuple, axis=1))].reset_index(drop=True)
    print(f"[fb] data_sample={len(sample_df)} missing={len(missing_df)}")
    sample_df.to_csv(os.path.join(args.out_dir, "data_sample.csv"), index=False)

    cand_csv = os.path.join(args.out_dir, "candidates_full.csv")
    if os.path.exists(cand_csv) and os.path.getsize(cand_csv) > 1000:
        cand_df = pd.read_csv(cand_csv)
        print(f"[fb] loaded {len(cand_df)} cached candidates")
    else:
        from candidates_generation import triple_gen
        cand_df = triple_gen.generate_all_candidates(sample_df)
        print(f"[fb] generated {len(cand_df)} candidates")
        cand_df.to_csv(cand_csv, index=False)

    miss_keys = set(map(tuple, missing_df.values))
    cand_df["Missing"] = cand_df.apply(
        lambda r: 1 if (r.Head, r.Relation, r.Tail) in miss_keys else 0, axis=1)
    print(f"[fb] candidates: true={int((cand_df.Missing==1).sum())} false={int((cand_df.Missing==0).sum())}")

    # TransE filter
    from experiment import filtering
    print("[fb] training TransE...")
    filtred_df = filtering.create_filtred_df(sample_df, cand_df[["Head", "Relation", "Tail"]].copy(), missing_df)
    filtred_df["Missing"] = filtred_df.apply(
        lambda r: 1 if (r.Head, r.Relation, r.Tail) in miss_keys else 0, axis=1)
    print(f"[fb] after TransE: {len(filtred_df)} candidates (true={int((filtred_df.Missing==1).sum())})")
    filtred_df.to_csv(os.path.join(args.out_dir, "candidates_filtered.csv"), index=False)

    # 500-balanced eval sample
    half = args.n_eval // 2
    pos = filtred_df[filtred_df.Missing == 1]
    neg = filtred_df[filtred_df.Missing == 0]
    take_t = min(half, len(pos)); take_f = min(args.n_eval - take_t, len(neg))
    eval_df = pd.concat([pos.sample(take_t, random_state=args.seed),
                         neg.sample(take_f, random_state=args.seed)])\
                .sample(frac=1, random_state=args.seed).reset_index(drop=True)
    eval_df.to_csv(os.path.join(args.out_dir, "cand_sample_500.csv"), index=False)
    print(f"[fb] wrote cand_sample_500.csv: pos={take_t} neg={take_f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/fb15k-237")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_proportion", type=float, default=0.8)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--n_subset", type=int, default=0)
    args = p.parse_args()
    main(args)
