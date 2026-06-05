"""CERTIS-V - verbalized feature capture for fb15k-237 (entity mids -> names[+desc]).

Same cohort-evidence prompt as 06_assemble_features.py, but every Freebase mid
(candidate head/tail AND cohort-evidence heads) is replaced by its canonical name;
optionally a short description is appended for the candidate head and tail.

Purpose: test whether restoring the semantic channel lifts the LLM features on
fb15k-237 (mids -> near-chance). Writes features/<ds>_seed<seed>_<part>_<tag>.csv
with column "<tag>_conf"; use a distinct tag (e.g. Mv) so mid features are kept.
"""
import argparse, os, sys, time, re
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)  # for `import experiment` (used by local_llm_patch)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_llm_patch
local_llm_patch.patch_prep_llm()
import local_llm_logits as L

NAME_FILE = os.path.join(ROOT_DIR, "data/fb15k-237/entity2text.txt")
DESC_FILE = os.path.join(ROOT_DIR, "data/fb15k-237/entity2description.txt")

PROMPT = """You are evaluating whether a knowledge graph triple represents a correct fact.

Triple to evaluate: ({h}, {r}, {t})

In our knowledge graph, the following entities are confirmed to have the same relation '{r}' with '{t}':
{evidence}

These entities form a cohort - they share the structural pattern (*, {r}, {t}).
Question: is '{h}' plausibly a member of this same cohort?

Reply with a single line starting with 'Score:' followed by 1 (correct), 0 (incorrect)."""


def load_names(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                d[parts[0]] = parts[1].replace("_", " ").strip()
    return d


def load_desc(path, max_chars=160):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                txt = parts[1].strip().strip('"')
                txt = re.sub(r"@en$", "", txt).strip().strip('"')
                first = txt.split(". ")[0]
                d[parts[0]] = first[:max_chars]
    return d


def clean_rel(r):
    # readable relation: keep path but drop leading slash; underscores->spaces
    return r.lstrip("/").replace("_", " ")


def verbalize(mid, names, descs, with_desc):
    nm = names.get(mid, mid)
    if with_desc and mid in descs and descs[mid]:
        return f"{nm} ({descs[mid]})"
    return nm


def score_one_cell(obs_df, cand_df, names, descs, with_desc, max_ev=8, log_every=200):
    rt = defaultdict(set)
    for h, r, t in zip(obs_df.Head, obs_df.Relation, obs_df.Tail):
        rt[(r, t)].add(h)
    confs = []
    t0 = time.time()
    for i, row in cand_df.iterrows():
        h, rl, t = row.Head, row.Relation, row.Tail
        heads = rt.get((rl, t), set()) - {h}
        rl_v = clean_rel(rl)
        t_v = verbalize(t, names, descs, with_desc)
        h_v = verbalize(h, names, descs, with_desc)
        if not heads:
            ev = "(none - no known cohort)"
        else:
            ev = "\n".join("- " + names.get(e, e) for e in sorted(heads, key=lambda x: hash(x))[:max_ev])
        prompt = PROMPT.format(h=h_v, r=rl_v, t=t_v, evidence=ev)
        confs.append(L.score_confidence(prompt))
        if (i + 1) % log_every == 0:
            print(f"  {i+1}/{len(cand_df)} elapsed={time.time()-t0:.0f}s", flush=True)
    return confs


def main(args):
    OUT = os.path.join(ROOT_DIR, "outputs", "redesign", "features")
    os.makedirs(OUT, exist_ok=True)
    obs = pd.read_csv(os.path.join(ROOT_DIR, "data/fb15k-237/data_sample.csv"))
    names = load_names(NAME_FILE)
    descs = load_desc(DESC_FILE) if args.desc else {}
    print(f"[verbalize] names={len(names)} descs={len(descs)} with_desc={args.desc}")

    for seed in args.seeds:
        for partition in args.partitions:
            src = os.path.join(ROOT_DIR, "outputs/redesign/splits", f"fb15k-237_seed{seed}_{partition}.csv")
            if not os.path.exists(src):
                print(f"  skip missing: {src}"); continue
            out_path = os.path.join(OUT, f"fb15k-237_seed{seed}_{partition}_{args.tag}.csv")
            if os.path.exists(out_path):
                print(f"  SKIP {os.path.basename(out_path)} (exists)"); continue
            cand = pd.read_csv(src)
            if args.cap > 0 and len(cand) > args.cap:
                half = args.cap // 2
                pos = cand[cand.Missing == 1].sample(n=min(half, int((cand.Missing == 1).sum())), random_state=seed)
                neg = cand[cand.Missing == 0].sample(n=min(args.cap - len(pos), int((cand.Missing == 0).sum())), random_state=seed)
                cand = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
            cov = np.mean([1 if h in names else 0 for h in cand.Head]) if len(cand) else 0
            print(f"\n=== fb15k-237 seed={seed} {partition} (n={len(cand)}) tag={args.tag} head-name-cov={cov:.3f} ===")
            confs = score_one_cell(obs, cand, names, descs, args.desc, args.max_ev)
            cand[f"{args.tag}_conf"] = confs
            cand.to_csv(out_path, index=False)
            print(f"  wrote {out_path}; mean conf={sum(confs)/len(confs):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--partitions", nargs="+", default=["cal_fit"])
    p.add_argument("--tag", required=True, help="verbalized tag e.g. Mv, Lv, Q35v")
    p.add_argument("--desc", action="store_true", help="append short entity description for head/tail")
    p.add_argument("--max_ev", type=int, default=8)
    p.add_argument("--cap", type=int, default=2000)
    args = p.parse_args()
    main(args)
