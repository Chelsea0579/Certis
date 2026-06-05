"""one-time locked-test evaluation + descriptive bounds.

Reads the per-seed locked models (locked_perseed_<ds>_seed<seed>.pkl) and the
family-certified threshold tau* from w3b_preregistered_perseed.json (written by 09b).
Applies each seed's frozen (model, isotonic, tau*) to that seed's TEST partition
(opened ONCE here; never used before). Reports realized risk/coverage and a
one-sided Hoeffding CI on test, and checks the certified U_R/L_C still hold.

This is the only place test is touched.
"""
import os, sys, json, math, pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FEATURE_DIR = os.path.join(ROOT, "outputs/redesign/features")
DIST_PATH = os.path.join(ROOT, "data/codex-m/cand_transe_dist.csv")
sys.path.insert(0, ROOT)

SEEDS = [42, 1, 7, 100]
K = 7
N_SEEDS = len(SEEDS)
DELTA_R = 0.025
DELTA_C = 0.025
feat_all = ["L_conf", "Q35_conf", "M_conf", "Q_conf",
            "log_cohort", "log_h_deg", "log_t_ind", "log_rel_freq", "log_shared_n", "transe_dist"]


def hoeffding_eps(delta, n, k_union):
    return 1.0 if n == 0 else math.sqrt(math.log(k_union / delta) / (2 * n))


def load_cell(dataset, seed, partition):
    parts = {}
    for tag in ["L", "M", "Q", "Q35"]:
        p = os.path.join(FEATURE_DIR, f"{dataset}_seed{seed}_{partition}_{tag}.csv")
        parts[tag] = pd.read_csv(p)
    base = parts["L"][["Head", "Relation", "Tail", "Missing"]].reset_index(drop=True)
    for tag in ["L", "M", "Q", "Q35"]:
        base[f"{tag}_conf"] = parts[tag][f"{tag}_conf"].values
    return base


def add_structural(df, obs_df, dist_lookup):
    rt = defaultdict(set); h2rt = defaultdict(set); t2hr = defaultdict(set)
    rf = Counter(); hn = defaultdict(set); tn = defaultdict(set)
    for h, r, t in zip(obs_df.Head, obs_df.Relation, obs_df.Tail):
        rt[(r, t)].add(h); h2rt[h].add((r, t)); t2hr[t].add((h, r))
        rf[r] += 1; hn[h].add(t); tn[t].add(h)
    df["log_cohort"] = np.log1p(df.apply(lambda x: len(rt.get((x.Relation, x.Tail), set())) - 1, axis=1))
    df["log_h_deg"] = np.log1p(df.Head.map(lambda h: len(h2rt.get(h, set()))))
    df["log_t_ind"] = np.log1p(df.Tail.map(lambda t: len(t2hr.get(t, set()))))
    df["log_rel_freq"] = np.log1p(df.Relation.map(lambda r: rf.get(r, 0)))
    df["log_shared_n"] = np.log1p(df.apply(
        lambda x: len(hn.get(x.Head, set()) & tn.get(x.Tail, set())), axis=1))
    if dist_lookup:
        med = np.median(list(dist_lookup.values()))
        df["transe_dist"] = df.apply(lambda x: dist_lookup.get((x.Head, x.Relation, x.Tail), med), axis=1)
    else:
        df["transe_dist"] = 1.7
    return df


def main():
    obs_c = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"))
    obs_f = pd.read_csv(os.path.join(ROOT, "data/fb15k-237/data_sample.csv"))
    dist = pd.read_csv(DIST_PATH)
    dist_lookup = {(r.Head, r.Relation, r.Tail): r.transe_dist for r in dist.itertuples()}
    prereg = json.load(open(os.path.join(ROOT, "outputs/redesign/w3b_preregistered_perseed.json")))
    finals = prereg["final_thresholds_perseed"]

    rows = []
    for ds, obs_df, dl in [("codex-m", obs_c, dist_lookup), ("fb15k-237", obs_f, {})]:
        ft = finals.get(ds)
        if not ft or ft.get("level") != "family":
            print(f"[{ds}] no family-certified tau*; skipping locked test (cert level={ft.get('level') if ft else None}).")
            continue
        a, b = ft["t_lo"], ft["t_hi"]
        print(f"\n=== {ds}: locked test at family-certified tau*=({a:.2f},{b:.2f}) ===")
        for seed in SEEDS:
            clf, iso = pickle.load(open(
                os.path.join(ROOT, f"outputs/redesign/locked_perseed_{ds}_seed{seed}.pkl"), "rb"))
            te = add_structural(load_cell(ds, seed, "test"), obs_df, dl)
            X = te[feat_all].values.astype(float); y = te.Missing.values
            probs = iso.transform(clf.predict_proba(X)[:, 1])
            m = len(probs)
            accept = (probs <= a) | (probs >= b)
            n_t = int(accept.sum())
            pred = (probs[accept] >= 0.5).astype(int)
            err = int((pred != y[accept]).sum())
            rhat = err / n_t if n_t else 0.0
            chat = n_t / m
            epsR = hoeffding_eps(DELTA_R, n_t, K * N_SEEDS)
            epsC = hoeffding_eps(DELTA_C, m, K * N_SEEDS)
            rows.append({"dataset": ds, "seed": seed, "t_lo": a, "t_hi": b,
                         "n_t": n_t, "m": m, "rhat": rhat, "chat": chat,
                         "U_R_test": min(1.0, rhat + epsR), "L_C_test": max(0.0, chat - epsC)})
            print(f"  seed {seed:3d}: test n={m} accept={n_t} r_hat={rhat:.3f} (U_R<= {min(1.0,rhat+epsR):.3f})  "
                  f"cov={chat:.3f} (L_C>= {max(0.0,chat-epsC):.3f})")

    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("dataset").agg(
            rhat_mean=("rhat", "mean"), rhat_max=("rhat", "max"),
            chat_mean=("chat", "mean"), chat_min=("chat", "min"),
            U_R_test_max=("U_R_test", "max"), L_C_test_min=("L_C_test", "min")).reset_index()
        print("\n=== TEST summary (mean / worst over 4 seeds) ===")
        for _, r in agg.iterrows():
            ok = (r.U_R_test_max <= 0.08) and (r.L_C_test_min >= 0.90)
            print(f"  {r.dataset}: r_hat[mean={r.rhat_mean:.3f} max={r.rhat_max:.3f}]  "
                  f"cov[mean={r.chat_mean:.3f} min={r.chat_min:.3f}]  "
                  f"worst U_R<= {r.U_R_test_max:.3f}  worst L_C>= {r.L_C_test_min:.3f}  "
                  f"{'HOLDS on test' if ok else 'does NOT hold on test'}")
        df.to_csv(os.path.join(ROOT, "outputs/redesign/w4_test_perseed.csv"), index=False)
        print("\nwrote w4_test_perseed.csv")
    else:
        print("\nNo dataset had a family-certified threshold; nothing tested.")


if __name__ == "__main__":
    main()
