"""E1: end-to-end fact-insertion VIGNETTE + review-load / certified-risk FRONTIER.
Repackages the existing certified-commit object as ONE deployment flow (NOT a new system):
a frozen verifier receives a candidate workload (the held-out test split), and routes each
candidate to COMMIT (p>=t_hi, inserted), REJECT (p<=1-t_hi, dropped), or REVIEW (middle ->
human queue). We report, per dataset:
  - VIGNETTE at the paper's frozen commit threshold t_hi: workload size, #committed, commit
    coverage (fraction of true positives captured), CERTIFIED false-insertion rate (CP upper,
    worst-seed family, from cert) vs REALIZED test FIR, and the human-review load.
  - FRONTIER over the t_hi grid: review-load vs certified FIR vs commit coverage (the
    "review-load <-> certified-risk" curve that is the deployment selling point).
CPU only, cached features, per-seed (no pooling). Certified numbers are worst-seed; realized
are mean over seeds.
"""
import os, sys, math
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.stats import beta as betadist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); FD = os.path.join(ROOT, "outputs/redesign/features")
SEEDS = [42, 1, 7, 100]; K = 7; N = 4; DELTA = 0.025
STRUCT = ["log_cohort", "log_h_deg", "log_t_ind", "log_rel_freq", "log_shared_n", "transe_dist"]; STR = {"Head": str, "Relation": str, "Tail": str}
# (tags, has_transe, frozen commit t_hi from the paper's positive-commit cert)
CFG = {"codex-m": (["L", "Q35", "M", "Q"], True, 0.65), "fb15k-237": (["Lvn", "Q35vn", "Mvn", "Qvn"], False, 0.80)}
THI_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]


def cp_upper(k, n, d):
    if n == 0: return 1.0
    if k >= n: return 1.0
    return float(betadist.ppf(1 - d, k + 1, n - k))


def add_struct(df, obs, dl):
    rt = defaultdict(set); h2rt = defaultdict(set); t2hr = defaultdict(set); rf = Counter(); hn = defaultdict(set); tn = defaultdict(set)
    for h, r, t in zip(obs.Head, obs.Relation, obs.Tail):
        rt[(r, t)].add(h); h2rt[h].add((r, t)); t2hr[t].add((h, r)); rf[r] += 1; hn[h].add(t); tn[t].add(h)
    df["log_cohort"] = np.log1p(df.apply(lambda x: max(0, len(rt.get((x.Relation, x.Tail), set())) - 1), axis=1))
    df["log_h_deg"] = np.log1p(df.Head.map(lambda h: len(h2rt.get(h, set()))))
    df["log_t_ind"] = np.log1p(df.Tail.map(lambda t: len(t2hr.get(t, set()))))
    df["log_rel_freq"] = np.log1p(df.Relation.map(lambda r: rf.get(r, 0)))
    df["log_shared_n"] = np.log1p(df.apply(lambda x: len(hn.get(x.Head, set()) & tn.get(x.Tail, set())), axis=1))
    df["transe_dist"] = df.apply(lambda x: dl.get((x.Head, x.Relation, x.Tail), 1.7), axis=1) if dl else 1.7
    return df


def lc(ds, seed, part, tags):
    b = pd.read_csv(os.path.join(FD, f"{ds}_seed{seed}_{part}_{tags[0]}.csv"), dtype=STR)[["Head", "Relation", "Tail", "Missing"]].copy(); b["Missing"] = b["Missing"].astype(int)
    for t in tags:
        b[f"{t}_conf"] = pd.read_csv(os.path.join(FD, f"{ds}_seed{seed}_{part}_{t}.csv"), dtype=STR)[f"{t}_conf"].astype(float).values
    return b


def main():
    obs_c = pd.read_csv(os.path.join(ROOT, "data/codex-m/data_sample.csv"), dtype=STR)
    obs_f = pd.read_csv(os.path.join(ROOT, "data/fb15k-237/data_sample.csv"), dtype=STR)
    dist = pd.read_csv(os.path.join(ROOT, "data/codex-m/cand_transe_dist.csv")); dl_c = {(r.Head, r.Relation, r.Tail): r.transe_dist for r in dist.itertuples()}
    dfam = DELTA / (K * N)
    for ds, (tags, ht, thi_star) in CFG.items():
        obs = obs_c if ds == "codex-m" else obs_f; dl = dl_c if ht else {}
        feat = [f"{t}_conf" for t in tags] + STRUCT
        perseed = []; frontier = defaultdict(list)
        for s in SEEDS:
            cal = add_struct(lc(ds, s, "cal_fit", tags), obs, dl); ce = add_struct(lc(ds, s, "cert", tags), obs, dl); te = add_struct(lc(ds, s, "test", tags), obs, dl)
            X = np.nan_to_num(cal[feat].values.astype(float)); y = cal.Missing.values
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(X, y)
            oof = cross_val_predict(clone(clf), X, y, cv=5, method="predict_proba")[:, 1]; iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
            pcert = iso.transform(clf.predict_proba(np.nan_to_num(ce[feat].values.astype(float)))[:, 1]); ycert = ce.Missing.values
            ptest = iso.transform(clf.predict_proba(np.nan_to_num(te[feat].values.astype(float)))[:, 1]); ytest = te.Missing.values
            npos = int((ytest == 1).sum())
            # vignette at frozen t_hi
            cm = pcert >= thi_star; cfir = cp_upper(int(((ycert == 0) & cm).sum()), int(cm.sum()), dfam)
            tm = ptest >= thi_star; ntm = int(tm.sum()); rfir = float(((ytest == 0) & tm).sum() / ntm) if ntm else 0.0
            ccov = float(((ytest == 1) & tm).sum() / npos) if npos else 0.0
            tlo = 1 - thi_star
            review = (ptest > tlo) & (ptest < thi_star); reject = ptest <= tlo; commit = ptest >= thi_star
            perseed.append(dict(s=s, Nin=len(ytest), ncommit=ntm, nreject=int(reject.sum()), nreview=int(review.sum()),
                                cfir=cfir, rfir=rfir, ccov=ccov, rl=float(review.mean())))
            for thi in THI_GRID:
                cm2 = pcert >= thi; cfir2 = cp_upper(int(((ycert == 0) & cm2).sum()), int(cm2.sum()), dfam)
                tm2 = ptest >= thi; ccov2 = float(((ytest == 1) & tm2).sum() / npos) if npos else 0.0
                rl2 = float(((ptest > 1 - thi) & (ptest < thi)).mean())
                frontier[thi].append((cfir2, ccov2, rl2))
        wc = max(p["cfir"] for p in perseed); mr = float(np.mean([p["rfir"] for p in perseed]))
        mcov = float(np.mean([p["ccov"] for p in perseed])); mrl = float(np.mean([p["rl"] for p in perseed]))
        nin = int(np.mean([p["Nin"] for p in perseed])); ncom = int(np.mean([p["ncommit"] for p in perseed]))
        nrej = int(np.mean([p["nreject"] for p in perseed])); nrev = int(np.mean([p["nreview"] for p in perseed]))
        print(f"\n{'='*76}\nE1 INSERTION VIGNETTE :: {ds}  (frozen commit t_hi={thi_star}; per-seed; worst-seed certified)\n{'='*76}")
        print(f"  workload (test) ~{nin}/seed  ->  COMMIT {ncom}  |  REJECT(p<= {1-thi_star:.2f}) {nrej}  |  REVIEW(human) {nrev}")
        print(f"  certified false-insertion rate (CP, worst-seed) <= {wc:.3f}   realized test FIR = {mr:.3f}   commit-coverage = {mcov:.3f}   review-load = {mrl:.3f}")
        print(f"  --- review-load / certified-risk FRONTIER (commit threshold t_hi) ---")
        print(f"   t_hi |  certFIR(worst) | commit-cov(mean) | review-load(mean)")
        for thi in THI_GRID:
            arr = frontier[thi]; cf = max(a[0] for a in arr); cc = float(np.mean([a[1] for a in arr])); rl = float(np.mean([a[2] for a in arr]))
            star = " *" if abs(thi - thi_star) < 1e-9 else ""
            print(f"   {thi:.2f} |     {cf:.3f}     |     {cc:.3f}      |     {rl:.3f}{star}")


if __name__ == "__main__":
    main()
