"""selective controls + mechanism ablations.

Leakage fix: NO leave-one-seed-out (cross-seed splits overlap ~19%). Instead,
for each seed independently we evaluate on cal_fit via cross-fitted calibrated
predictions: nested 5-fold where, in each outer fold, LR+isotonic are fit on the
other folds and predict the held-out fold. Every prediction comes from a model
(LR + isotonic) that never saw that example -> leakage-free, and cert/test are
never touched here. Naive controls have no fit, evaluated directly on cal_fit.

Reports mean +/- std over the 4 independent seeds.
"""
import os, sys, json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FEATURE_DIR = os.path.join(ROOT, "outputs/redesign/features")
DIST_PATH = os.path.join(ROOT, "data/codex-m/cand_transe_dist.csv")
sys.path.insert(0, ROOT)
from experiment import result as R
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEEDS = [42, 1, 7, 100]
THRESHOLD_GRID = [(0.50, 0.50), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65),
                  (0.30, 0.70), (0.25, 0.75), (0.20, 0.80)]


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


def compute_ece(probas, y, n_bins=15):
    probas = np.asarray(probas).astype(float); y = np.asarray(y).astype(int)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0; n = len(probas)
    for i in range(n_bins):
        mask = ((probas >= bins[i]) & (probas <= bins[i+1])) if i == n_bins - 1 else ((probas >= bins[i]) & (probas < bins[i+1]))
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - probas[mask].mean())
    return ece


def selective_f1(probas, y, t_lo, t_hi):
    accept = [i for i, p in enumerate(probas) if not (t_lo < p < t_hi)]
    if not accept: return 0.0, 0.0
    pred = [int(probas[i] >= 0.5) for i in accept]
    yi = [int(y[i]) for i in accept]
    tp = sum(1 for p, gt in zip(pred, yi) if p == 1 and gt == 1)
    fp = sum(1 for p, gt in zip(pred, yi) if p == 1 and gt == 0)
    fn = sum(1 for p, gt in zip(pred, yi) if p == 0 and gt == 1)
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    return f1, len(accept) / len(probas)


def crossfit_calibrated(base_cls, X, y, seed):
    """Nested 5-fold: outer fold predicted by LR+isotonic fit on other folds.
    Returns calibrated out-of-fold probabilities (leakage-free)."""
    out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = clone(base_cls).fit(X[tr], y[tr])
        inner = cross_val_predict(clone(base_cls), X[tr], y[tr], cv=5, method="predict_proba")[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(inner, y[tr])
        out[te] = iso.transform(m.predict_proba(X[te])[:, 1])
    return out


def main():
    obs_codex = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"))
    obs_fb = pd.read_csv(os.path.join(ROOT, "data/fb15k-237/data_sample.csv"))
    dist = pd.read_csv(DIST_PATH)
    dist_lookup = {(r.Head, r.Relation, r.Tail): r.transe_dist for r in dist.itertuples()}

    feat_all = ["L_conf", "Q35_conf", "M_conf", "Q_conf",
                "log_cohort", "log_h_deg", "log_t_ind", "log_rel_freq", "log_shared_n", "transe_dist"]
    feat_struct = ["log_cohort", "log_h_deg", "log_t_ind", "log_rel_freq", "log_shared_n", "transe_dist"]
    feat_sem = ["L_conf", "Q35_conf", "M_conf", "Q_conf"]

    out_records = []
    for ds, obs_df in [("codex-m", obs_codex), ("fb15k-237", obs_fb)]:
        dl = dist_lookup if ds == "codex-m" else {}
        for seed in SEEDS:
            cell = add_structural(load_cell(ds, seed, "cal_fit"), obs_df, dl)
            y = cell.Missing.values

            for model_name, base_cls, feats in [
                ("struct_only_RF", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42), feat_struct),
                ("sem_only_RF",    RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42), feat_sem),
                ("full_RF",        RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42), feat_all),
                ("full_LR",        LogisticRegression(max_iter=3000, C=1.0), feat_all),
            ]:
                X = cell[feats].values.astype(float)
                probs = crossfit_calibrated(base_cls, X, y, seed)
                row = {"dataset": ds, "seed": seed, "model": model_name}
                pred05 = (probs >= 0.5).astype(int)
                _, f1, rec, prec = R.compute_score(pred05.tolist(), y.tolist())
                row["F1_full"] = f1
                row["ECE"] = compute_ece(probs, y)
                for t_lo, t_hi in THRESHOLD_GRID:
                    f1_t, cov = selective_f1(probs, y, t_lo, t_hi)
                    row[f"F1_t{t_lo}_{t_hi}"] = f1_t
                    row[f"Cov_t{t_lo}_{t_hi}"] = cov
                out_records.append(row)

            for ctrl_name, score_fn in [
                ("max_LLM_conf", lambda d: np.max(d[feat_sem].values, axis=1)),
                ("Q35_only",     lambda d: d["Q35_conf"].values),
                ("Llama_only",   lambda d: d["L_conf"].values),
                ("mean_LLM",     lambda d: np.mean(d[feat_sem].values, axis=1)),
            ]:
                probs = score_fn(cell)
                row = {"dataset": ds, "seed": seed, "model": ctrl_name}
                pred05 = (probs >= 0.5).astype(int)
                _, f1, rec, prec = R.compute_score(pred05.tolist(), y.tolist())
                row["F1_full"] = f1
                row["ECE"] = compute_ece(probs, y)
                for t_lo, t_hi in THRESHOLD_GRID:
                    f1_t, cov = selective_f1(probs, y, t_lo, t_hi)
                    row[f"F1_t{t_lo}_{t_hi}"] = f1_t
                    row[f"Cov_t{t_lo}_{t_hi}"] = cov
                out_records.append(row)

    df = pd.DataFrame(out_records)
    agg = df.groupby(["dataset", "model"]).agg(
        F1_full_mean=("F1_full", "mean"), F1_full_std=("F1_full", "std"),
        ECE_mean=("ECE", "mean"), ECE_std=("ECE", "std"),
        F1_30_mean=("F1_t0.3_0.7", "mean"), Cov_30_mean=("Cov_t0.3_0.7", "mean"),
        F1_20_mean=("F1_t0.2_0.8", "mean"), Cov_20_mean=("Cov_t0.2_0.8", "mean"),
    ).reset_index()
    print("\n=== Per-model summary (per-seed self-contained, mean+/-std over 4 seeds, cal_fit cross-fit) ===\n")
    for ds in ["codex-m", "fb15k-237"]:
        sub = agg[agg.dataset == ds].sort_values("F1_full_mean", ascending=False)
        print(f"--- {ds} ---")
        for _, r in sub.iterrows():
            print(f"  {r.model:16s}: F1={r.F1_full_mean:.3f}+/-{r.F1_full_std:.3f}  "
                  f"ECE={r.ECE_mean:.3f}+/-{r.ECE_std:.3f}  "
                  f"F1@(.30,.70)={r.F1_30_mean:.3f} cov={r.Cov_30_mean:.3f}  "
                  f"F1@(.20,.80)={r.F1_20_mean:.3f} cov={r.Cov_20_mean:.3f}")
        print()

    OUT = os.path.join(ROOT, "outputs/redesign")
    df.to_csv(os.path.join(OUT, "w3b_ablation_perseed_full.csv"), index=False)
    agg.to_csv(os.path.join(OUT, "w3b_ablation_perseed_summary.csv"), index=False)
    print("wrote w3b_ablation_perseed_full.csv + w3b_ablation_perseed_summary.csv")


if __name__ == "__main__":
    main()
