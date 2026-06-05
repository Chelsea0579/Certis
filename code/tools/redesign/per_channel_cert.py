"""Per-channel CERTIFICATE ablation (closes 'is the LLM load-bearing?' major).
For a split_tag, run per-seed family Hoeffding cert for: struct_only (6 feats),
llm_only (LLM confs), full (LLM+struct). Reports full-cov F1 + family cert each.
No RotatE/KGE needed -> pure CPU on cached features. str-safe.
Usage: python per_channel_cert.py --split_tag codex-m --obs_path data/codex-m/data_sample.csv --llm_tags L Q35 M Q [--has_transe]
"""
import os, sys, math, argparse
import numpy as np, pandas as pd
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from experiment import result as R
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); FEATURE_DIR = os.path.join(ROOT, "outputs/redesign/features")
SEEDS=[42,1,7,100]; GRID=[(0.50,0.50),(0.45,0.55),(0.40,0.60),(0.35,0.65),(0.30,0.70),(0.25,0.75),(0.20,0.80)]
K=7; N=4; DR=DC=0.025
STRUCT=["log_cohort","log_h_deg","log_t_ind","log_rel_freq","log_shared_n","transe_dist"]
STR={"Head":str,"Relation":str,"Tail":str}


def eps(n,ku,d=0.025): return math.sqrt(math.log(ku/d)/(2*n)) if n else 1.0


def add_struct(df,obs,dl):
    rt=defaultdict(set);h2rt=defaultdict(set);t2hr=defaultdict(set);rf=Counter();hn=defaultdict(set);tn=defaultdict(set)
    for h,r,t in zip(obs.Head,obs.Relation,obs.Tail):
        rt[(r,t)].add(h);h2rt[h].add((r,t));t2hr[t].add((h,r));rf[r]+=1;hn[h].add(t);tn[t].add(h)
    df["log_cohort"]=np.log1p(df.apply(lambda x:max(0,len(rt.get((x.Relation,x.Tail),set()))-1),axis=1))
    df["log_h_deg"]=np.log1p(df.Head.map(lambda h:len(h2rt.get(h,set()))))
    df["log_t_ind"]=np.log1p(df.Tail.map(lambda t:len(t2hr.get(t,set()))))
    df["log_rel_freq"]=np.log1p(df.Relation.map(lambda r:rf.get(r,0)))
    df["log_shared_n"]=np.log1p(df.apply(lambda x:len(hn.get(x.Head,set())&tn.get(x.Tail,set())),axis=1))
    df["transe_dist"]=df.apply(lambda x:dl.get((x.Head,x.Relation,x.Tail),1.7),axis=1) if dl else 1.7
    return df


def load_cell(tag,seed,part,llm_tags):
    base=pd.read_csv(os.path.join(FEATURE_DIR,f"{tag}_seed{seed}_{part}_{llm_tags[0]}.csv"),dtype=STR)[["Head","Relation","Tail","Missing"]].copy()
    base["Missing"]=base["Missing"].astype(int)
    for t in llm_tags:
        d=pd.read_csv(os.path.join(FEATURE_DIR,f"{tag}_seed{seed}_{part}_{t}.csv"),dtype=STR)
        base[f"{t}_conf"]=d[f"{t}_conf"].astype(float).values
    return base


def cert(cal,cert_,feats):
    pby={};yby={};f1s=[]
    for s in SEEDS:
        X=np.nan_to_num(cal[s][feats].values.astype(float)); y=cal[s].Missing.values
        clf=LogisticRegression(max_iter=3000,C=1.0).fit(X,y)
        oof=cross_val_predict(clone(clf),X,y,cv=5,method="predict_proba")[:,1]
        iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
        p=iso.transform(clf.predict_proba(np.nan_to_num(cert_[s][feats].values.astype(float)))[:,1])
        pby[s]=p; yby[s]=cert_[s].Missing.values; f1s.append(R.compute_score((p>=0.5).astype(int).tolist(),yby[s].tolist())[1])
    best=None
    for a,b in GRID:
        URs=[];LCs=[]
        for s in SEEDS:
            pr=pby[s];y=yby[s];m=len(pr);acc=(pr<=a)|(pr>=b);nt=int(acc.sum())
            err=int(((pr[acc]>=0.5).astype(int)!=y[acc]).sum()) if nt else 0
            URs.append((err/nt if nt else 0)+eps(nt,K*N,DR)); LCs.append(nt/m-eps(m,K*N,DC))
        if max(URs)<=0.08 and min(LCs)>=0.90:
            cov=np.mean([((pby[s]<=a)|(pby[s]>=b)).mean() for s in SEEDS])
            if best is None or cov>best[2]: best=(a,b,cov)
    return np.mean(f1s),best


def main(a):
    obs=pd.read_csv(os.path.join(ROOT,a.obs_path),dtype=STR)
    dl={}
    if a.has_transe:
        dist=pd.read_csv(os.path.join(ROOT,"data/codex-m/cand_transe_dist.csv"))
        dl={(r.Head,r.Relation,r.Tail):r.transe_dist for r in dist.itertuples()}
    cal={s:add_struct(load_cell(a.split_tag,s,"cal_fit",a.llm_tags),obs,dl) for s in SEEDS}
    crt={s:add_struct(load_cell(a.split_tag,s,"cert",a.llm_tags),obs,dl) for s in SEEDS}
    llm=[f"{t}_conf" for t in a.llm_tags]
    print(f"=== {a.split_tag} per-channel certificate ablation ===")
    for name,feats in [("struct_only",STRUCT),("llm_only",llm),("full",llm+STRUCT)]:
        f1,best=cert(cal,crt,feats)
        tag=f"FAMILY-CERT tau*=({best[0]:.2f},{best[1]:.2f}) cov={best[2]:.3f}" if best else "NO family cert"
        print(f"  {name:12s}: full-cov F1={f1:.3f}  {tag}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--split_tag",required=True);p.add_argument("--obs_path",required=True)
    p.add_argument("--llm_tags",nargs="+",required=True);p.add_argument("--has_transe",action="store_true")
    a=p.parse_args();main(a)
