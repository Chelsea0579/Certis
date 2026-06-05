"""freeze + certify with NO cross-seed leakage.

Bug fixed: W1 builds independent per-seed splits (random_state=seed shuffles), so
partitions overlap ~19% ACROSS seeds. The old 09 trained the locked model on
union(4 cal_fit) and certified per-seed cert -> 18.5% of each cert set was in
training. This version is ENTIRELY within-seed (within-seed partitions are disjoint
by construction):

  for each seed s:
    fit LR(C=1.0) on seed-s.cal_fit
    5-fold OOF isotonic on seed-s.cal_fit
    certify K=7 threshold grid on seed-s.cert   (disjoint from cal_fit -> zero leak)

Model spec FROZEN (no tuning on cert/test). Grid frozen. Does NOT touch test.

Two certificates reported:
  - per-seed   : union over K thresholds, delta=0.025          (k_union = K)
  - family     : union over K thresholds AND N_SEEDS seeds     (k_union = K * N_SEEDS)
                 a threshold qualifies at family level iff worst seed passes
                 with the inflated epsilon.
"""
import os, sys, json, hashlib
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FEATURE_DIR = os.path.join(ROOT, "outputs/redesign/features")
DIST_PATH = os.path.join(ROOT, "data/codex-m/cand_transe_dist.csv")
sys.path.insert(0, ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

THRESHOLD_GRID = [(0.50, 0.50), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65),
                  (0.30, 0.70), (0.25, 0.75), (0.20, 0.80)]
K = len(THRESHOLD_GRID)
SEEDS = [42, 1, 7, 100]
N_SEEDS = len(SEEDS)
DELTA_R = 0.025
DELTA_C = 0.025
U_R_TARGET = 0.08
L_C_TARGET = 0.90


def hoeffding_eps(delta, n, k_union):
    if n == 0:
        return 1.0
    return math.sqrt(math.log(k_union / delta) / (2 * n))


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


feat_all = ["L_conf", "Q35_conf", "M_conf", "Q_conf",
            "log_cohort", "log_h_deg", "log_t_ind", "log_rel_freq", "log_shared_n", "transe_dist"]


def fit_perseed(cal):
    Xtr = cal[feat_all].values.astype(float); ytr = cal.Missing.values
    base = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    oof = cross_val_predict(clone(base), Xtr, ytr, cv=5, method="predict_proba")[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr)
    coef_bytes = pickle.dumps((base.coef_.tobytes(), base.intercept_.tobytes()))
    h = hashlib.sha256(coef_bytes).hexdigest()[:16]
    return base, iso, h


def main():
    obs_c = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"))
    obs_f = pd.read_csv(os.path.join(ROOT, "data/fb15k-237/data_sample.csv"))
    dist = pd.read_csv(DIST_PATH)
    dist_lookup = {(r.Head, r.Relation, r.Tail): r.transe_dist for r in dist.itertuples()}

    rows = []
    locked = {}
    for ds, obs_df, dl in [("codex-m", obs_c, dist_lookup), ("fb15k-237", obs_f, {})]:
        locked[ds] = {}
        for seed in SEEDS:
            cal = add_structural(load_cell(ds, seed, "cal_fit"), obs_df, dl)
            clf, iso, h = fit_perseed(cal)
            locked[ds][seed] = h
            pickle.dump((clf, iso), open(
                os.path.join(ROOT, f"outputs/redesign/locked_perseed_{ds}_seed{seed}.pkl"), "wb"))

            te = add_structural(load_cell(ds, seed, "cert"), obs_df, dl)
            Xte = te[feat_all].values.astype(float); yte = te.Missing.values
            probs = iso.transform(clf.predict_proba(Xte)[:, 1])
            m = len(probs)
            for (a, b) in THRESHOLD_GRID:
                accept = (probs <= a) | (probs >= b)
                n_t = int(accept.sum())
                if n_t == 0:
                    continue
                pred = (probs[accept] >= 0.5).astype(int)
                err = int((pred != yte[accept]).sum())
                rhat = err / n_t
                chat = n_t / m
                rows.append({"dataset": ds, "seed": seed, "t_lo": a, "t_hi": b,
                             "n_t": n_t, "m": m, "err": err, "rhat": rhat, "chat": chat})

    cert = pd.DataFrame(rows)

    # epsilons
    def add_eps(df, k_union, suffix):
        df[f"epsR_{suffix}"] = df.apply(lambda r: hoeffding_eps(DELTA_R, r.n_t, k_union), axis=1)
        df[f"epsC_{suffix}"] = df.apply(lambda r: hoeffding_eps(DELTA_C, r.m, k_union), axis=1)
        df[f"U_R_{suffix}"] = (df.rhat + df[f"epsR_{suffix}"]).clip(upper=1.0)
        df[f"L_C_{suffix}"] = (df.chat - df[f"epsC_{suffix}"]).clip(lower=0.0)
        df[f"pass_{suffix}"] = (df[f"U_R_{suffix}"] <= U_R_TARGET) & (df[f"L_C_{suffix}"] >= L_C_TARGET)
    add_eps(cert, K, "seed")            # per-seed certificate
    add_eps(cert, K * N_SEEDS, "fam")   # family certificate (union over seeds too)
    cert.to_csv(os.path.join(ROOT, "outputs/redesign/w3b_cert_bounds_perseed.csv"), index=False)

    final = {}
    for ds in ["codex-m", "fb15k-237"]:
        sub = cert[cert.dataset == ds]
        print(f"\n=== {ds} : per-seed self-contained certification ===")
        # per-seed: does each seed certify SOME threshold?
        print("  -- per-seed certificate (union over K=7) --")
        for seed in SEEDS:
            ss = sub[(sub.seed == seed) & sub.pass_seed]
            if len(ss):
                best = ss.sort_values("chat", ascending=False).iloc[0]
                print(f"    seed {seed:3d}: PASS  tau=({best.t_lo:.2f},{best.t_hi:.2f}) "
                      f"r_hat={best.rhat:.3f} U_R={best.U_R_seed:.3f} chat={best.chat:.3f} L_C={best.L_C_seed:.3f}")
            else:
                # report least-bad
                tmp = sub[sub.seed == seed].copy()
                tmp["slack"] = (tmp.U_R_seed - U_R_TARGET).clip(lower=0) + (L_C_TARGET - tmp.L_C_seed).clip(lower=0)
                b = tmp.sort_values("slack").iloc[0]
                print(f"    seed {seed:3d}: fail  best tau=({b.t_lo:.2f},{b.t_hi:.2f}) "
                      f"U_R={b.U_R_seed:.3f} L_C={b.L_C_seed:.3f}")
        # family: a single threshold where WORST seed passes with inflated eps
        print("  -- family certificate (union over K*N_SEEDS=28, worst-seed) --")
        byt = sub.groupby(["t_lo", "t_hi"]).agg(
            rhat_mean=("rhat", "mean"), rhat_max=("rhat", "max"),
            chat_mean=("chat", "mean"), chat_min=("chat", "min"),
            U_R_fam_max=("U_R_fam", "max"), L_C_fam_min=("L_C_fam", "min"),
            pass_fam_all=("pass_fam", "all"),
            n_t_mean=("n_t", "mean"),
        ).reset_index()
        for _, r in byt.iterrows():
            tag = "PASS" if r.pass_fam_all else "fail"
            print(f"    tau=({r.t_lo:.2f},{r.t_hi:.2f}): r_hat[mean={r.rhat_mean:.3f} max={r.rhat_max:.3f}] "
                  f"c_hat[mean={r.chat_mean:.3f} min={r.chat_min:.3f}]  "
                  f"U_R<= {r.U_R_fam_max:.3f}  L_C>= {r.L_C_fam_min:.3f}  {tag}")
        q = byt[byt.pass_fam_all]
        if len(q):
            chosen = q.sort_values("chat_mean", ascending=False).iloc[0]
            final[ds] = {"level": "family", "t_lo": float(chosen.t_lo), "t_hi": float(chosen.t_hi),
                         "U_R_max": float(chosen.U_R_fam_max), "L_C_min": float(chosen.L_C_fam_min),
                         "chat_mean": float(chosen.chat_mean)}
            print(f"  -> FAMILY-CERTIFIED tau* = ({chosen.t_lo:.2f},{chosen.t_hi:.2f})")
        else:
            # fall back: per-seed pass count
            persd = {}
            for seed in SEEDS:
                persd[seed] = bool(len(sub[(sub.seed == seed) & sub.pass_seed]))
            n_pass = sum(persd.values())
            final[ds] = {"level": "per-seed", "seeds_passing": n_pass, "detail": persd}
            print(f"  -> NO family threshold. Per-seed passes: {n_pass}/{N_SEEDS}")

    _plan_path = os.path.join(ROOT, "outputs/redesign/w3_preregistered_analysis.json")
    if not os.path.exists(_plan_path):
        _plan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w3_preregistered_analysis.json")
    plan = json.load(open(_plan_path))
    plan["protocol"] = "per-seed self-contained (leakage-fixed 2026-05-28)"
    plan["model_hashes_perseed"] = locked
    plan["final_thresholds_perseed"] = final
    plan["eps_note"] = ("per-seed eps uses k_union=K=7; family eps uses k_union=K*N_SEEDS=28; "
                        "delta_R=delta_C=0.025")
    json.dump(plan, open(os.path.join(ROOT, "outputs/redesign/w3b_preregistered_perseed.json"), "w"), indent=2)
    print("\nwrote w3b_cert_bounds_perseed.csv + w3b_preregistered_perseed.json")


if __name__ == "__main__":
    main()
