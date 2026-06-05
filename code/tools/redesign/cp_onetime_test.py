"""CP-primary one-time TEST certificate for CERTIS (LLM+struct, no RotatE => CPU).
Fit cal_fit, select tau* on cert via CP family bound, eval test with CP bound.
codex-m + fb15k name-only. Gives the CP-primary test headline (expect full coverage)."""
import os, sys, math
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.stats import beta as betadist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from experiment import result as R
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); FD=os.path.join(ROOT,"outputs/redesign/features")
SEEDS=[42,1,7,100]; GRID=[(0.50,0.50),(0.45,0.55),(0.40,0.60),(0.35,0.65),(0.30,0.70),(0.25,0.75),(0.20,0.80)]
K=7;N=4;DR=DC=0.025; STRUCT=["log_cohort","log_h_deg","log_t_ind","log_rel_freq","log_shared_n","transe_dist"]; STR={"Head":str,"Relation":str,"Tail":str}
CFG={"codex-m":(["L","Q35","M","Q"],True),"fb15k-237":(["Lvn","Q35vn","Mvn","Qvn"],False)}
def cp_up(k,n,d): return 1.0 if k>=n else float(betadist.ppf(1-d,k+1,n-k))
def cp_lo(k,n,d): return 0.0 if k<=0 else float(betadist.ppf(d,k,n-k+1))
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
def lc(ds,seed,part,tags):
    b=pd.read_csv(os.path.join(FD,f"{ds}_seed{seed}_{part}_{tags[0]}.csv"),dtype=STR)[["Head","Relation","Tail","Missing"]].copy();b["Missing"]=b["Missing"].astype(int)
    for t in tags: b[f"{t}_conf"]=pd.read_csv(os.path.join(FD,f"{ds}_seed{seed}_{part}_{t}.csv"),dtype=STR)[f"{t}_conf"].astype(float).values
    return b
def cpcert(pby,yby):  # CP family bound; return best tau (highest cov) passing
    best=None
    for a,b in GRID:
        URs=[];LCs=[];covs=[]
        for s in SEEDS:
            pr=pby[s];y=yby[s];m=len(pr);acc=(pr<=a)|(pr>=b);nt=int(acc.sum());err=int(((pr[acc]>=0.5).astype(int)!=y[acc]).sum())
            URs.append(cp_up(err,nt,DR/(K*N)) if nt else 1.0);LCs.append(cp_lo(nt,m,DC/(K*N)));covs.append(nt/m)
        if max(URs)<=0.08 and min(LCs)>=0.90 and (best is None or np.mean(covs)>best[2]): best=(a,b,np.mean(covs),max(URs),min(LCs))
    return best
def main():
    obs_c=pd.read_csv(os.path.join(ROOT,"data/codex-m/data_sample.csv"),dtype=STR);obs_f=pd.read_csv(os.path.join(ROOT,"data/fb15k-237/data_sample.csv"),dtype=STR)
    dist=pd.read_csv(os.path.join(ROOT,"data/codex-m/cand_transe_dist.csv"));dl_c={(r.Head,r.Relation,r.Tail):r.transe_dist for r in dist.itertuples()}
    for ds,(tags,has_t) in CFG.items():
        obs=obs_c if ds=="codex-m" else obs_f; dl=dl_c if has_t else {}; feat=[f"{t}_conf" for t in tags]+STRUCT
        pc={};yc={};pt={};yt={}
        for s in SEEDS:
            cal=add_struct(lc(ds,s,"cal_fit",tags),obs,dl);ce=add_struct(lc(ds,s,"cert",tags),obs,dl);te=add_struct(lc(ds,s,"test",tags),obs,dl)
            X=np.nan_to_num(cal[feat].values.astype(float));y=cal.Missing.values
            clf=LogisticRegression(max_iter=3000,C=1.0).fit(X,y);oof=cross_val_predict(clone(clf),X,y,cv=5,method="predict_proba")[:,1];iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
            pc[s]=iso.transform(clf.predict_proba(np.nan_to_num(ce[feat].values.astype(float)))[:,1]);yc[s]=ce.Missing.values
            pt[s]=iso.transform(clf.predict_proba(np.nan_to_num(te[feat].values.astype(float)))[:,1]);yt[s]=te.Missing.values
        certtau=cpcert(pc,yc)
        if not certtau: print(f"{ds} CERTIS CP: no cert on cert-set"); continue
        a,b,_,_,_=certtau
        # apply tau* to TEST with CP
        rs=[];URs=[];LCs=[];covs=[]
        for s in SEEDS:
            pr=pt[s];y=yt[s];m=len(pr);acc=(pr<=a)|(pr>=b);nt=int(acc.sum());err=int(((pr[acc]>=0.5).astype(int)!=y[acc]).sum())
            rs.append(err/nt if nt else 0);URs.append(cp_up(err,nt,DR/(K*N)) if nt else 1.0);LCs.append(cp_lo(nt,m,DC/(K*N)));covs.append(nt/m)
        hold=max(URs)<=0.08 and min(LCs)>=0.90
        print(f"=== {ds} CERTIS CP-primary one-time TEST: tau*=({a:.2f},{b:.2f}) ===")
        print(f"  TEST risk[mean={np.mean(rs):.3f} max={max(rs):.3f}] cov[mean={np.mean(covs):.3f} min={min(covs):.3f}]  worstU_R(CP)={max(URs):.3f} worstL_C(CP)={min(LCs):.3f}  strict-recert={'HOLD' if hold else 'no'}")
if __name__=="__main__": main()
