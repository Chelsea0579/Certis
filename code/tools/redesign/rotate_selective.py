"""Codex-mandated key experiment: RotatE-selective-certified vs CERTIS vs CERTIS+RotatE.

For each dataset, train RotatE-dim128 on the observed graph, score the EXACT
candidates in our cap=4000 feature cells (aligned by H/R/T), then run the SAME
per-seed family Hoeffding selective certification (isotonic on cal_fit, K=7 grid,
union over K*N_SEEDS=28) for three models:
  - RotatE_only   : isotonic-calibrated RotatE score
  - CERTIS         : the locked LLM-fusion (mid LLMs for codex-m, verbalized for fb15k) + 6 struct
  - CERTIS+RotatE  : CERTIS features + z-scored RotatE score (11th feature)
Reports, per model: full-cov F1, family-certified tau*, worst U_R, worst L_C, coverage.
Answers: does a strong KGE certify as well as CERTIS, and does the LLM add value on
top of a modern KGE?
"""
import os, sys, math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FEATURE_DIR = os.path.join(ROOT, "outputs/redesign/features")
sys.path.insert(0, ROOT)
from experiment import result as R
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

SEEDS = [42, 1, 7, 100]
GRID = [(0.50,0.50),(0.45,0.55),(0.40,0.60),(0.35,0.65),(0.30,0.70),(0.25,0.75),(0.20,0.80)]
K = 7; N = 4; DR = DC = 0.025
STRUCT = ["log_cohort","log_h_deg","log_t_ind","log_rel_freq","log_shared_n","transe_dist"]


def eps(n, ku, d=0.025): return math.sqrt(math.log(ku/d)/(2*n)) if n else 1.0


def add_struct(df, obs, dl):
    rt=defaultdict(set);h2rt=defaultdict(set);t2hr=defaultdict(set);rf=Counter();hn=defaultdict(set);tn=defaultdict(set)
    for h,r,t in zip(obs.Head,obs.Relation,obs.Tail):
        rt[(r,t)].add(h);h2rt[h].add((r,t));t2hr[t].add((h,r));rf[r]+=1;hn[h].add(t);tn[t].add(h)
    df["log_cohort"]=np.log1p(df.apply(lambda x:len(rt.get((x.Relation,x.Tail),set()))-1,axis=1))
    df["log_h_deg"]=np.log1p(df.Head.map(lambda h:len(h2rt.get(h,set()))))
    df["log_t_ind"]=np.log1p(df.Tail.map(lambda t:len(t2hr.get(t,set()))))
    df["log_rel_freq"]=np.log1p(df.Relation.map(lambda r:rf.get(r,0)))
    df["log_shared_n"]=np.log1p(df.apply(lambda x:len(hn.get(x.Head,set())&tn.get(x.Tail,set())),axis=1))
    df["transe_dist"]=df.apply(lambda x:dl.get((x.Head,x.Relation,x.Tail),1.7),axis=1) if dl else 1.7
    return df


def load_cell(ds, seed, part, tags):
    base=pd.read_csv(os.path.join(FEATURE_DIR,f"{ds}_seed{seed}_{part}_{tags[0]}.csv"))[["Head","Relation","Tail","Missing"]].reset_index(drop=True)
    for t in tags:
        d=pd.read_csv(os.path.join(FEATURE_DIR,f"{ds}_seed{seed}_{part}_{t}.csv"))
        base[f"{t}_conf"]=d[f"{t}_conf"].values
    return base


def family_cert(per):  # per: dict[(a,b)] -> list of (rhat, chat, n_t, m)
    best=None; table=[]
    for (a,b),vals in per.items():
        rs=[v[0] for v in vals]; cs=[v[1] for v in vals]; nts=[v[2] for v in vals]; ms=[v[3] for v in vals]
        URmax=max(rs[i]+eps(nts[i],K*N,DR) for i in range(len(vals)))
        LCmin=min(cs[i]-eps(ms[i],K*N,DC) for i in range(len(vals)))
        ok=(URmax<=0.08) and (LCmin>=0.90)
        table.append((a,b,np.mean(rs),np.mean(cs),URmax,LCmin,ok))
        if ok and (best is None or np.mean(cs)>best[3]): best=(a,b,np.mean(rs),np.mean(cs),URmax,LCmin)
    return table, best


def selective_per(probs_by_seed, y_by_seed):
    per={(a,b):[] for (a,b) in GRID}
    for seed in SEEDS:
        probs=probs_by_seed[seed]; y=y_by_seed[seed]; m=len(probs)
        for a,b in GRID:
            acc=(probs<=a)|(probs>=b); nt=int(acc.sum())
            if nt==0: per[(a,b)].append((0.0,0.0,0,m)); continue
            err=int(((probs[acc]>=0.5).astype(int)!=y[acc]).sum())
            per[(a,b)].append((err/nt, nt/m, nt, m))
    return per


def main():
    obs_c=pd.read_csv(os.path.join(ROOT,"data/codex-m/data_sample.csv"))
    obs_f=pd.read_csv(os.path.join(ROOT,"data/fb15k-237/data_sample.csv"))
    dist=pd.read_csv(os.path.join(ROOT,"data/codex-m/cand_transe_dist.csv"))
    dl_c={(r.Head,r.Relation,r.Tail):r.transe_dist for r in dist.itertuples()}

    configs=[("codex-m",obs_c,dl_c,["L","Q35","M","Q"]),
             ("fb15k-237",obs_f,{},["Mv","Lv","Qv","Q35v"])]

    for ds, obs, dl, llm_tags in configs:
        print(f"\n########## {ds} (LLM tags={llm_tags}) ##########", flush=True)
        # train RotatE on observed graph
        tf=TriplesFactory.from_labeled_triples(obs[["Head","Relation","Tail"]].astype(str).values)
        e2i,r2i=tf.entity_to_id,tf.relation_to_id
        res=pipeline(training=tf,testing=tf,model="RotatE",model_kwargs=dict(embedding_dim=128),
                     training_kwargs=dict(num_epochs=50,batch_size=2048),random_seed=42,
                     device="cuda" if torch.cuda.is_available() else "cpu")
        model=res.model; dev=next(model.parameters()).device
        def rotate_score(df):
            ids=[];mask=[]
            for h,r,t in zip(df.Head.astype(str),df.Relation.astype(str),df.Tail.astype(str)):
                if h in e2i and r in r2i and t in e2i: ids.append([e2i[h],r2i[r],e2i[t]]);mask.append(True)
                else: ids.append([0,0,0]);mask.append(False)
            with torch.no_grad():
                s=model.score_hrt(torch.tensor(ids,dtype=torch.long).to(dev)).squeeze(-1).cpu().numpy()
            return np.where(np.array(mask),s,np.min(s)-1.0)

        feat10=[f"{t}_conf" for t in llm_tags]+STRUCT
        # build per-seed frames + rotate scores
        cal={}; cert={}
        for seed in SEEDS:
            c=add_struct(load_cell(ds,seed,"cal_fit",llm_tags),obs,dl); c["rot"]=rotate_score(c)
            t=add_struct(load_cell(ds,seed,"cert",llm_tags),obs,dl); t["rot"]=rotate_score(t)
            cal[seed]=c; cert[seed]=t

        def run_model(name, fit_fn):
            pby={}; yby={}; f1s=[]
            for seed in SEEDS:
                c=cal[seed]; t=cert[seed]
                probs_cert, f1_full = fit_fn(c, t)
                pby[seed]=probs_cert; yby[seed]=t.Missing.values; f1s.append(f1_full)
            per=selective_per(pby,yby); table,best=family_cert(per)
            print(f"\n--- {ds} / {name}: full-cov F1={np.mean(f1s):.3f} ---")
            for a,b,rm,cm,UR,LC,ok in table:
                print(f"    tau=({a:.2f},{b:.2f}): r_mean={rm:.3f} c_mean={cm:.3f} worstU_R={UR:.3f} worstL_C={LC:.3f} {'PASS' if ok else 'fail'}")
            print(f"    => {'FAMILY-CERT tau*=('+format(best[0],'.2f')+','+format(best[1],'.2f')+') cov_mean='+format(best[3],'.3f') if best else 'NO family cert'}")
            return best

        # RotatE-only: isotonic on cal_fit rot-score
        def fit_rot(c,t):
            iso=IsotonicRegression(out_of_bounds="clip").fit(c["rot"].values,c.Missing.values)
            p_t=iso.transform(t["rot"].values)
            f1=R.compute_score((iso.transform(c["rot"].values)>=0.5).astype(int).tolist(),c.Missing.values.tolist())[1]
            return p_t,f1
        # CERTIS-10
        def fit_certis(c,t,feats=feat10):
            X=c[feats].values.astype(float); y=c.Missing.values
            clf=LogisticRegression(max_iter=3000,C=1.0).fit(X,y)
            oof=cross_val_predict(clone(clf),X,y,cv=5,method="predict_proba")[:,1]
            iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
            p_t=iso.transform(clf.predict_proba(t[feats].values.astype(float))[:,1])
            f1=R.compute_score((p_t>=0.5).astype(int).tolist(),t.Missing.values.tolist())[1]
            return p_t,f1
        # CERTIS+RotatE-11 (z-score rot on cal_fit)
        def fit_certis_rot(c,t):
            mu,sd=c["rot"].mean(),c["rot"].std()+1e-9
            c=c.copy();t=t.copy(); c["rotz"]=(c["rot"]-mu)/sd; t["rotz"]=(t["rot"]-mu)/sd
            return fit_certis(c,t,feats=feat10+["rotz"])

        run_model("RotatE_only", fit_rot)
        run_model("CERTIS", fit_certis)
        run_model("CERTIS+RotatE", fit_certis_rot)


if __name__=="__main__":
    main()
