"""Add Missing column to candidates_filtered.csv for multi-seed E2/E3a use."""
import os, json, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
src = os.path.join(ROOT, "data/codex-m/candidates_filtered.csv")
dst = os.path.join(ROOT, "data/codex-m/candidates_filtered_with_gt.csv")
if os.path.exists(dst) and os.path.getsize(dst) > 100:
    print("already exists:", dst)
    exit(0)

obs = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"))
sample_keys = set(zip(obs.Head, obs.Relation, obs.Tail))
ent = json.load(open(os.path.join(ROOT, "data/codex-m/codex_m_raw/entities.json")))
rel = json.load(open(os.path.join(ROOT, "data/codex-m/codex_m_raw/relations.json")))
full_parts = []
for x in ["train.txt", "valid.txt", "test.txt"]:
    p = os.path.join(ROOT, "data/codex-m/codex_m_raw", x)
    df = pd.read_csv(p, sep="\t", header=None, names=["h", "r", "t"])
    full_parts.append(df)
full = pd.concat(full_parts, ignore_index=True)
full["Head"] = full.h.map(lambda k: ent.get(k, {}).get("label", k))
full["Tail"] = full.t.map(lambda k: ent.get(k, {}).get("label", k))
full["Relation"] = full.r.map(lambda k: rel.get(k, {}).get("label", k))
full = full[["Head", "Relation", "Tail"]].drop_duplicates()
full_keys = set(zip(full.Head, full.Relation, full.Tail))
missing_keys = full_keys - sample_keys
print(f"missing keys: {len(missing_keys)}")

chunks = []
for ch in pd.read_csv(src, chunksize=500000):
    ch["Missing"] = ch.apply(lambda r: 1 if (r.Head, r.Relation, r.Tail) in missing_keys else 0, axis=1)
    chunks.append(ch)
out = pd.concat(chunks, ignore_index=True)
out.to_csv(dst, index=False)
print(f"wrote {dst} rows={len(out)} pos={int((out.Missing==1).sum())}")
