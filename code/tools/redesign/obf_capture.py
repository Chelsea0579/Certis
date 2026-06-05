"""Contamination control: capture LLM cohort-evidence confidences with entities AND
relations replaced by deterministic abstract IDs (entity 0042 / relation 07). If the
LLM's signal survives obfuscation, gains are structural/reasoning, not memorized recall.
Mirrors the cohort-evidence prompt exactly; only surface labels change. str-safe.
Output: features/<split_tag>_seed<seed>_<part>_<tag>.csv (tag e.g. Lo/Mo/Qo/Q35o).
"""
import argparse, os, sys, time
import pandas as pd, numpy as np
from collections import defaultdict
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_llm_patch; local_llm_patch.patch_prep_llm()
import local_llm_logits as L
SPLIT_DIR = os.path.join(ROOT_DIR, "outputs/redesign/splits")
OUT = os.path.join(ROOT_DIR, "outputs/redesign/features")
STR = {"Head": str, "Relation": str, "Tail": str}
PROMPT = """You are evaluating whether a knowledge graph triple represents a correct fact.

Triple to evaluate: ({h}, {r}, {t})

In our knowledge graph, the following entities are confirmed to have the same relation '{r}' with '{t}':
{evidence}

These entities form a cohort - they share the structural pattern (., {r}, {t}).
Question: is '{h}' plausibly a member of this same cohort?

Reply with a single line starting with 'Score:' followed by 1 (correct), 0 (incorrect)."""


def build_maps(obs):
    ents = sorted(set(obs.Head) | set(obs.Tail)); rels = sorted(set(obs.Relation))
    e2o = {e: f"entity {i:05d}" for i, e in enumerate(ents)}
    r2o = {r: f"relation {i:03d}" for i, r in enumerate(rels)}
    return e2o, r2o


def score(obs, cand, e2o, r2o, max_ev=8, log_every=400):
    rt = defaultdict(set)
    for h, r, t in zip(obs.Head, obs.Relation, obs.Tail):
        rt[(r, t)].add(h)
    def eo(x): return e2o.get(x, "entity unknown")
    def ro(x): return r2o.get(x, "relation unknown")
    confs = []; t0 = time.time()
    for i, row in cand.iterrows():
        h, rl, t = row.Head, row.Relation, row.Tail
        heads = rt.get((rl, t), set()) - {h}
        ev = "\n".join("- " + eo(e) for e in sorted(heads, key=lambda x: hash(x))[:max_ev]) if heads else "(none - no known cohort)"
        confs.append(L.score_confidence(PROMPT.format(h=eo(h), r=ro(rl), t=eo(t), evidence=ev)))
        if (i + 1) % log_every == 0: print(f"  {i+1}/{len(cand)} elapsed={time.time()-t0:.0f}s", flush=True)
    return confs


def main(a):
    os.makedirs(OUT, exist_ok=True)
    obs = pd.read_csv(os.path.join(ROOT_DIR, a.obs_path), dtype=STR)
    e2o, r2o = build_maps(obs)
    print(f"[obf] {a.split_tag} entities={len(e2o)} relations={len(r2o)} tag={a.tag}", flush=True)
    for seed in a.seeds:
        for part in a.partitions:
            src = os.path.join(SPLIT_DIR, f"{a.split_tag}_seed{seed}_{part}.csv")
            if not os.path.exists(src): print(f"  skip {src}"); continue
            outp = os.path.join(OUT, f"{a.split_tag}_seed{seed}_{part}_{a.tag}.csv")
            if os.path.exists(outp): print(f"  SKIP {os.path.basename(outp)}"); continue
            cand = pd.read_csv(src, dtype=STR); cand["Missing"] = cand["Missing"].astype(int)
            if a.cap > 0 and len(cand) > a.cap:
                half = a.cap // 2
                pos = cand[cand.Missing == 1].sample(n=min(half, int((cand.Missing == 1).sum())), random_state=seed)
                neg = cand[cand.Missing == 0].sample(n=min(a.cap - len(pos), int((cand.Missing == 0).sum())), random_state=seed)
                cand = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"\n=== {a.split_tag} seed={seed} {part} n={len(cand)} tag={a.tag} OBFUSCATED ===", flush=True)
            cand[f"{a.tag}_conf"] = score(obs, cand, e2o, r2o, a.max_ev)
            cand.to_csv(outp, index=False); print(f"  wrote {outp}; mean conf={cand[f'{a.tag}_conf'].mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split_tag", required=True); p.add_argument("--obs_path", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 100])
    p.add_argument("--partitions", nargs="+", default=["cal_fit", "cert"])
    p.add_argument("--max_ev", type=int, default=8); p.add_argument("--cap", type=int, default=4000)
    a = p.parse_args(); main(a)
