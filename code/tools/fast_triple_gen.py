"""Drop-in replacement for candidates_generation.triple_gen.generate_all_candidates,
implementing the same paper semantics (Algorithms 1+2) with O(sum m_k * p_k) cost.

Semantics (preserved from original code, which is what the paper Fig. 2 actually shows):
  - C_(r,t) = set of heads that connect to t via r (head clusters)
  - For each cluster with |C|>1: P_C = union over h in C of {(rel, tail) such that (h, rel, tail) in graph}
  - Candidates: {(h, rel, tail) | h in C, (rel, tail) in P_C}  minus observed triples
"""
import pandas as pd
from collections import defaultdict


def generate_all_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Head", "Relation", "Tail"]]
    rels = df["Relation"].to_numpy()
    tails = df["Tail"].to_numpy()
    heads = df["Head"].to_numpy()

    # head -> set of (rel, tail) pairs it participates in
    head_to_rt = defaultdict(set)
    for h, r, t in zip(heads, rels, tails):
        head_to_rt[h].add((r, t))

    # (rel, tail) -> set of heads (only keep clusters with >1 head)
    rt_to_heads = defaultdict(set)
    for h, r, t in zip(heads, rels, tails):
        rt_to_heads[(r, t)].add(h)
    cluster_items = [(k, hs) for k, hs in rt_to_heads.items() if len(hs) > 1]
    print(f"[fast_triple_gen] {len(cluster_items)} clusters (>1 head)")

    observed = set(zip(heads, rels, tails))

    candidates = set()
    for i, (anchor, head_set) in enumerate(cluster_items):
        # Union of (rel, tail) pairs across all heads in this cluster
        rt_union = set()
        for h in head_set:
            rt_union.update(head_to_rt[h])
        # Cross product head_set x rt_union
        for h in head_set:
            for (r, t) in rt_union:
                if (h, r, t) not in observed:
                    candidates.add((h, r, t))
        if (i + 1) % 5000 == 0:
            print(f"[fast_triple_gen] processed {i+1}/{len(cluster_items)} clusters, {len(candidates)} candidates so far")

    print(f"[fast_triple_gen] total candidates: {len(candidates)}")
    return pd.DataFrame(list(candidates), columns=["Head", "Relation", "Tail"])


def install():
    """Monkey-patch the original triple_gen.generate_all_candidates."""
    from candidates_generation import triple_gen
    triple_gen.generate_all_candidates = generate_all_candidates
    print("[fast_triple_gen] generate_all_candidates patched")
