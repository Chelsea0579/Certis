"""E2: same-protocol text-KGC baseline = TextCE-KGC, a KG-BERT-style cross-encoder verifier,
trained + calibrated + certified under the IDENTICAL leakage-free per-seed protocol as CertiS.
Spec (codex-locked, pre-declared; no per-dataset tuning):
  base = bert-base-uncased (HF cache, offline). input "head: H [SEP] relation: R [SEP] tail: T", maxlen 64.
  Stage A (graph-supervised, seed-shared): positives = observed graph (data_sample.csv); 4 filtered
    corruptions/positive (2 head, 2 tail); 2 epochs, lr 2e-5, bs 64, AdamW wd 0.01, 10% warmup, BCE, grad-clip 1.0.
  Stage B (workload adaptation, PER SEED): fine-tune on labeled candidate TRAIN split; 3 epochs, lr 1e-5.
  Wiring: train on TRAIN only (graph + train candidates); score cal_fit/cert/test; isotonic-calibrate on
  cal_fit; K=7 grid selection on cert; test opened once; CP + Hoeffding worst-seed family certificate.
Pre-committed claims: (i) certifies too -> "text can also be certifiable; CertiS is model-agnostic + fusion
  improves frontier"; (ii) no cert -> "strong text-only verifier fails the same certificate"; (iii) certifies
  only at lower coverage -> "fusion lowers human-review burden at equal FIR budget."
Outputs the cert-set + test certification for TextCE-KGC, to drop beside CertiS / RotatE-only.
"""
import os, sys, math, time, random, argparse
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta as betadist
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); SPL = os.path.join(ROOT, "outputs/redesign/splits")
SEEDS = [42, 1, 7, 100]
K_GRID = [(.50, .50), (.45, .55), (.40, .60), (.35, .65), (.30, .70), (.25, .75), (.20, .80)]
K = 7; N = 4; DR = DC = 0.025; UR_T = 0.08; LC_T = 0.90
BASE = "bert-base-uncased"; MAXLEN = 64; DEV = "cuda"
STR = {"Head": str, "Relation": str, "Tail": str}


def setseed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def verb(h, r, t):
    return f"head: {h} [SEP] relation: {r} [SEP] tail: {t}"


def cp_upper(k, n, d):
    if n == 0: return 1.0
    if k >= n: return 1.0
    return float(betadist.ppf(1 - d, k + 1, n - k))


def heps(n, ku, d):
    return math.sqrt(math.log(ku / d) / (2 * n)) if n else 1.0


def build_stageA(obs, cap, negper, seed=0):
    rng = np.random.RandomState(seed)
    pos = set(zip(obs.Head, obs.Relation, obs.Tail))
    ents = pd.unique(pd.concat([obs.Head, obs.Tail], ignore_index=True))
    sub = obs.sample(n=min(cap, len(obs)), random_state=seed).reset_index(drop=True)
    texts, labels = [], []
    nh = negper // 2; nt = negper - nh
    for h, r, t in zip(sub.Head, sub.Relation, sub.Tail):
        texts.append(verb(h, r, t)); labels.append(1.0)
        for _ in range(nt):
            for _try in range(8):
                t2 = ents[rng.randint(len(ents))]
                if (h, r, t2) not in pos: break
            texts.append(verb(h, r, t2)); labels.append(0.0)
        for _ in range(nh):
            for _try in range(8):
                h2 = ents[rng.randint(len(ents))]
                if (h2, r, t) not in pos: break
            texts.append(verb(h2, r, t)); labels.append(0.0)
    return texts, labels


def train(model, tok, texts, labels, epochs, lr, bs):
    model.train()
    idx = np.arange(len(texts))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(texts) // bs + 1) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    labels = np.asarray(labels, dtype=np.float32)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]
            enc = tok([texts[j] for j in bi], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            labs = torch.tensor(labels[bi], device=DEV)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**enc).logits[:, 0]
                loss = lossf(out, labs)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    return model


@torch.no_grad()
def score(model, tok, texts, bs=256):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        enc = {k: v.to(DEV) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model(**enc).logits[:, 0].float()
        out.append(torch.sigmoid(lo).cpu().numpy())
    return np.concatenate(out)


def load_split(ds, seed, part):
    d = pd.read_csv(os.path.join(SPL, f"{ds}_seed{seed}_{part}.csv"), dtype=STR)
    d["Missing"] = d["Missing"].astype(int)
    return d


def texts_of(d):
    return [verb(h, r, t) for h, r, t in zip(d.Head, d.Relation, d.Tail)]


def main(a):
    t0 = time.time()
    ds = a.dataset
    obs = pd.read_csv(os.path.join(ROOT, f"data/{ds}/data_sample.csv"), dtype=STR)
    tok = AutoTokenizer.from_pretrained(BASE)
    capA = 3000 if a.smoke else a.capA
    epA, epB = (1, 1) if a.smoke else (2, 3)
    seeds = [42] if a.smoke else SEEDS
    print(f"[E2] {ds} smoke={a.smoke} capA={capA} epA={epA} epB={epB} seeds={seeds}", flush=True)
    sa_path = os.path.join(ROOT, f"outputs/redesign/textce_stageA_{ds}{'_smoke' if a.smoke else ''}.pt")
    if a.skip_stageA and os.path.exists(sa_path):
        print(f"[E2] reuse cached Stage-A {sa_path}", flush=True)
    else:
        print("[E2] building Stage-A graph data...", flush=True)
        setseed(0)
        txtA, labA = build_stageA(obs, capA, 4)
        print(f"[E2] Stage-A examples={len(txtA)} (pos+4neg). training (shared)...", flush=True)
        setseed(0)
        mA = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
        mA = train(mA, tok, txtA, labA, epA, 2e-5, 64)
        torch.save(mA.state_dict(), sa_path)
        print(f"[E2] Stage-A done t={time.time()-t0:.0f}s saved {sa_path}", flush=True)

    rows = []; f1s = []; accs = []
    for s in seeds:
        setseed(s)
        m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
        m.load_state_dict(torch.load(sa_path));
        tr = load_split(ds, s, "train")
        m = train(m, tok, texts_of(tr), tr.Missing.values.astype(float), epB, 1e-5, 64)
        sc = {}
        for part in ["cal_fit", "cert", "test"]:
            d = load_split(ds, s, part); sc[part] = (score(m, tok, texts_of(d)), d.Missing.values)
        # calibrate isotonic on cal_fit (scorer never saw cal_fit)
        pcf, ycf = sc["cal_fit"]; iso = IsotonicRegression(out_of_bounds="clip").fit(pcf, ycf)
        pce = iso.transform(sc["cert"][0]); yce = sc["cert"][1]
        pte = iso.transform(sc["test"][0]); yte = sc["test"][1]
        pred = (pte >= 0.5).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0; rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0); accs.append(float((pred == yte).mean()))
        for (lo, hi) in K_GRID:
            for tag, (p, y) in [("cert", (pce, yce)), ("test", (pte, yte))]:
                m_ = len(p); acc = (p <= lo) | (p >= hi); nt = int(acc.sum())
                if nt == 0: continue
                err = int(((p[acc] >= 0.5).astype(int) != y[acc]).sum())
                rhat = err / nt; chat = nt / m_
                rows.append(dict(seed=s, split=tag, lo=lo, hi=hi, nt=nt, m=m_, err=err, rhat=rhat, chat=chat))
        print(f"[E2] seed {s} done t={time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, f"outputs/redesign/e2_textce_{ds}{'_smoke' if a.smoke else ''}_raw.csv"), index=False)
    print(f"\n[E2] {ds} TextCE-KGC STRONG-BASELINE CHECK (full-coverage TEST): F1={np.mean(f1s):.3f}+-{np.std(f1s):.3f}  acc={np.mean(accs):.3f}", flush=True)

    # family certificate (worst-seed, union over K*N) on CERT, and TEST eval
    print(f"\n{'='*72}\nE2 TextCE-KGC certificate :: {ds}  (worst-seed family, delta/(K*N))\n{'='*72}")
    for split in ["cert", "test"]:
        sub = df[df.split == split].copy()
        # full coverage point (.50,.50)
        fc = sub[(sub.lo == .50) & (sub.hi == .50)]
        if len(fc):
            urs = [r.rhat + heps(r.nt, K * N, DR) for r in fc.itertuples()]
            urs_cp = [cp_upper(r.err, r.nt, DR / (K * N)) for r in fc.itertuples()]
            print(f"  [{split}] FULL-COVERAGE (.50,.50): worst-seed U_R(Hoeff)={max(urs):.3f}  U_R(CP)={max(urs_cp):.3f}  (target<= {UR_T})")
        # best family op point: worst-seed passes both, max mean coverage
        best = None
        for (lo, hi), g in sub.groupby(["lo", "hi"]):
            if g.seed.nunique() < len(seeds): continue
            urmax = max(cp_upper(r.err, r.nt, DR / (K * N)) for r in g.itertuples())
            lcmin = min(r.chat - heps(r.m, K * N, DC) for r in g.itertuples())
            covmean = float(g.chat.mean())
            ok = (urmax <= UR_T) and (lcmin >= LC_T)
            if ok and (best is None or covmean > best[3]):
                best = (lo, hi, urmax, covmean, lcmin)
        if best:
            print(f"  [{split}] CERTIFIES: tau=({best[0]:.2f},{best[1]:.2f}) U_R(CP)<= {best[2]:.3f} cov~{best[3]:.3f} L_C>= {best[4]:.3f}")
        else:
            print(f"  [{split}] NO certified operating point at U_R<= {UR_T}/L_C>= {LC_T} (worst-seed family)")
    print(f"[E2] total t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="codex-m")
    p.add_argument("--capA", type=int, default=60000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip_stageA", action="store_true")
    main(p.parse_args())
