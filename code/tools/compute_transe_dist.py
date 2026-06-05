"""Retrain TransE on data_sample then compute the transe_dist feature for every
per-seed split candidate. Save to a cand-level CSV (Head,Relation,Tail,transe_dist).
"""
import os, sys, glob
import pandas as pd, numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_dist
fast_dist.install()

from candidates_filtering.embedding import train_model
from candidates_filtering.embedding.get_emb_transe import get_list_dist

obs = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"))
print(f"obs={len(obs)} triples")

train_factory = train_model.create_dataset(obs)
test_factory = train_model.create_dataset(obs.sample(n=min(50, len(obs)), random_state=42))

print("training TransE dim=5 ...")
res = train_model.create_pipeline(
    train_factory, test_factory, "TransE", {"embedding_dim": 5}, "transe_for_meta", num_epochs=50,
)

# Score the per-seed split candidates (produced by 01_build_protocol_ledger.py) so
# transe_dist covers every evaluated triple; fall back to the candidate sample.
sources = sorted(glob.glob(os.path.join(ROOT, "outputs/redesign/splits/codex-m_seed*_*.csv")))
if not sources:
    sources = [os.path.join(ROOT, "data/codex-m/cand_sample_500.csv")]
unique_triples = pd.DataFrame()
for p in sources:
    if os.path.exists(p):
        df = pd.read_csv(p)
        unique_triples = pd.concat([unique_triples, df[["Head", "Relation", "Tail"]]], ignore_index=True)
unique_triples = unique_triples.drop_duplicates().reset_index(drop=True)
print(f"unique candidates to score: {len(unique_triples)}")

dist = get_list_dist(unique_triples, res.model, train_factory)
unique_triples["transe_dist"] = dist
out_path = os.path.join(ROOT, "data/codex-m/cand_transe_dist.csv")
unique_triples.to_csv(out_path, index=False)
print(f"wrote {out_path}")
print(unique_triples.describe()[["transe_dist"]])
