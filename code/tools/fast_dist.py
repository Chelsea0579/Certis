"""Vectorized TransE distance computation for very large candidate sets."""
import numpy as np
import pandas as pd


def get_list_dist(df: pd.DataFrame, model, triple_factory) -> np.ndarray:
    """Vectorized: distance = || h_emb + r_emb - t_emb ||_2 for each row."""
    # Entity / relation embeddings as numpy arrays
    ent_emb = model.entity_representations[0](indices=None).detach().cpu().numpy()
    rel_emb = model.relation_representations[0](indices=None).detach().cpu().numpy()

    e2i = triple_factory.entity_to_id
    r2i = triple_factory.relation_to_id

    # vectorized id lookup via mapping (note: candidates may contain unseen entities)
    h_ids = df["Head"].map(e2i).to_numpy()
    r_ids = df["Relation"].map(r2i).to_numpy()
    t_ids = df["Tail"].map(e2i).to_numpy()
    # Any unmapped (NaN) -> mark as inf distance (will be filtered out)
    bad = np.asarray(pd.isna(h_ids) | pd.isna(r_ids) | pd.isna(t_ids))
    h_ids = np.where(pd.isna(h_ids), 0, h_ids).astype(np.int64)
    r_ids = np.where(pd.isna(r_ids), 0, r_ids).astype(np.int64)
    t_ids = np.where(pd.isna(t_ids), 0, t_ids).astype(np.int64)

    # batch in chunks to avoid blowing memory on 56M rows
    out = np.empty(len(df), dtype=np.float32)
    CHUNK = 1_000_000
    for s in range(0, len(df), CHUNK):
        e = min(s + CHUNK, len(df))
        diff = ent_emb[h_ids[s:e]] + rel_emb[r_ids[s:e]] - ent_emb[t_ids[s:e]]
        out[s:e] = np.linalg.norm(diff, axis=1).astype(np.float32)
    out[bad] = np.inf
    return out


def install():
    from candidates_filtering.embedding import get_emb_transe as orig
    orig.get_list_dist = get_list_dist
    print("[fast_dist] get_list_dist vectorized")
