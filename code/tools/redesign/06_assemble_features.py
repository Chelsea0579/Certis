"""Capture multi-seed LLM logit confidences on the new larger splits.

For each (dataset, seed, partition) cell in the  manifest, capture P('1' | prompt)
from each of 4 LLMs using the cluster-evidence prompt (same as exp_e18_logits.py).

Time-conservative pass: only capture the cells we actually need for fitting +
certifying + testing. Structural features are CPU-only and computed in-line.

Outputs:
  outputs/redesign/features/<dataset>_seed<seed>_<partition>_<llm>.csv
"""
import argparse, os, sys, time
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_llm_patch
local_llm_patch.patch_prep_llm()
import local_llm_logits as L


PROMPT = """You are evaluating whether a knowledge graph triple represents a correct fact.

Triple to evaluate: ({h}, {r}, {t})

In our knowledge graph, the following entities are confirmed to have the same relation '{r}' with '{t}':
{evidence}

These entities form a cohort - they share the structural pattern (*, {r}, {t}).
Question: is '{h}' plausibly a member of this same cohort?

Reply with a single line starting with 'Score:' followed by 1 (correct), 0 (incorrect)."""


def score_one_cell(obs_df, cand_df, max_ev=8, log_every=200):
    rt = defaultdict(set)
    for h, r, t in zip(obs_df.Head, obs_df.Relation, obs_df.Tail):
        rt[(r, t)].add(h)
    confs = []
    t0 = time.time()
    for i, row in cand_df.iterrows():
        h, rl, t = row.Head, row.Relation, row.Tail
        heads = rt.get((rl, t), set()) - {h}
        if not heads:
            ev = "(none - no known cohort)"
        else:
            ev = "\n".join("- " + e for e in sorted(heads, key=lambda x: hash(x))[:max_ev])
        prompt = PROMPT.format(h=h, r=rl, t=t, evidence=ev)
        c = L.score_confidence(prompt)
        confs.append(c)
        if (i + 1) % log_every == 0:
            print(f"  {i+1}/{len(cand_df)} elapsed={time.time()-t0:.0f}s", flush=True)
    return confs


def main(args):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    OUT = os.path.join(ROOT_DIR, "outputs", "redesign", "features")
    os.makedirs(OUT, exist_ok=True)

    # Dataset -> cohort source path
    cohort_sources = {
        "codex-m": os.path.join(ROOT_DIR, "data/codex-m/data_sample.csv"),
        "fb15k-237": os.path.join(ROOT_DIR, "data/fb15k-237/data_sample.csv"),
    }

    for ds in args.datasets:
        obs = pd.read_csv(cohort_sources[ds])
        for seed in args.seeds:
            for partition in args.partitions:
                src = os.path.join(ROOT_DIR, "outputs/redesign/splits",
                                    f"{ds}_seed{seed}_{partition}.csv")
                if not os.path.exists(src):
                    print(f"  skip missing: {src}")
                    continue
                out_path = os.path.join(OUT, f"{ds}_seed{seed}_{partition}_{args.tag}.csv")
                if os.path.exists(out_path):
                    print(f"  SKIP {os.path.basename(out_path)} (exists)")
                    continue
                cand = pd.read_csv(src)
                if args.cap > 0 and len(cand) > args.cap:
                    half = args.cap // 2
                    pos = cand[cand.Missing == 1].sample(n=min(half, int((cand.Missing == 1).sum())), random_state=seed)
                    neg = cand[cand.Missing == 0].sample(n=min(args.cap - len(pos), int((cand.Missing == 0).sum())), random_state=seed)
                    cand = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
                print(f"\n=== {ds} seed={seed} {partition} (n={len(cand)}, cap={args.cap}) tag={args.tag} ===")
                confs = score_one_cell(obs, cand, args.max_ev)
                cand[f"{args.tag}_conf"] = confs
                cand.to_csv(out_path, index=False)
                print(f"  wrote {out_path}; mean conf={sum(confs)/len(confs):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["codex-m"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 100])
    p.add_argument("--partitions", nargs="+", default=["cal_fit", "cert", "test"],
                   help="train is huge; capture only cal_fit/cert/test initially")
    p.add_argument("--tag", required=True, help="e.g. Q35, L, M, Q")
    p.add_argument("--max_ev", type=int, default=8)
    p.add_argument("--cap", type=int, default=2000, help="Max candidates per partition (balanced)")
    args = p.parse_args()
    main(args)
