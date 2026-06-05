"""Round-1 review addition: POSITIVE-COMMIT certificate (the metric a KGC pipeline
actually cares about): among triples we would COMMIT to the graph (predicted positive,
p >= t_hi), what is the certified false-INSERTION rate (FDR)? CP upper bound, per-seed
family (union over the K t_hi grid x N seeds). Also error-localization: accepted-error
rate by score-confidence quantile (why RotatE fails despite high F1). CERTIS (LLM+struct)
=> CPU. codex-m + fb15k name-only, cert split."""
import os, sys, math
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from scipy.stats import beta as betadist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")); FD=os.path.join(ROOT,"outputs/redesign/features")
SEEDS=[42,1,7,100]; THI=[0.55,0.60,0.65,0.70,0.75,0.80,0.90]; K=len(THI); N=4; DR=0.025
STRUCT=["log_cohort","log_h_deg","log_t_ind","log_rel_freq","log_shared_n","transe_dist"]; STR={"Head":str,"Relation":str,"Tail":str}
CFG={"codex-m":(["L","Q35","M","Q"],True),"fb15k-237":(["Lvn","Q35vn","Mvn","Qvn"],False)}
def cp_up(k,n,d): return 1.0 if k>=n else float(betadist.ppf(1-d,k+1,n-k))
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
def main():
    obs_c=pd.read_csv(os.path.join(ROOT,"data/codex-m/data_sample.csv"),dtype=STR);obs_f=pd.read_csv(os.path.join(ROOT,"data/fb15k-237/data_sample.csv"),dtype=STR)
    dist=pd.read_csv(os.path.join(ROOT,"data/codex-m/cand_transe_dist.csv"));dl_c={(r.Head,r.Relation,r.Tail):r.transe_dist for r in dist.itertuples()}
    for ds,(tags,has_t) in CFG.items():
        obs=obs_c if ds=="codex-m" else obs_f; dl=dl_c if has_t else {}; feat=[f"{t}_conf" for t in tags]+STRUCT
        pby={};yby={}
        for s in SEEDS:
            cal=add_struct(lc(ds,s,"cal_fit",tags),obs,dl);ce=add_struct(lc(ds,s,"cert",tags),obs,dl)
            X=np.nan_to_num(cal[feat].values.astype(float));y=cal.Missing.values
            clf=LogisticRegression(max_iter=3000,C=1.0).fit(X,y);oof=cross_val_predict(clone(clf),X,y,cv=5,method="predict_proba")[:,1];iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
            pby[s]=iso.transform(clf.predict_proba(np.nan_to_num(ce[feat].values.astype(float)))[:,1]);yby[s]=ce.Missing.values
        print(f"\n=== {ds} CERTIS positive-commit certificate (CP, family over {K} t_hi x {N} seeds) ===")
        print("  commit if p>=t_hi; false-insertion rate (FDR) among committed + CP upper bound; commit-coverage = frac of true positives committed")
        npos_total=sum(int((yby[s]==1).sum()) for s in SEEDS)
        for th in THI:
            fdrs=[];ncommit=[];ncap=[]
            tp_tot=0;committed_tp=0
            for s in SEEDS:
                p=pby[s];y=yby[s];commit=p>=th;ncm=int(commit.sum());fp=int((commit&(y==0)).sum())
                fdrs.append(cp_up(fp,ncm,DR/(K*N)) if ncm else 0.0);ncommit.append(ncm)
                tp_tot+=int((y==1).sum());committed_tp+=int((commit&(y==1)).sum())
            worst_fdr=max(fdrs);commit_cov=committed_tp/tp_tot
            ok="cert FDR<=0.05" if worst_fdr<=0.05 else ("cert FDR<=0.08" if worst_fdr<=0.08 else "")
            print(f"  t_hi={th:.2f}: committed/seed~{int(np.mean(ncommit))}  worst CP-FDR={worst_fdr:.3f}  commit-coverage(of true pos)={commit_cov:.3f}  {ok}")
        # error-localization: accepted-error rate by predicted-confidence quartile (pooled)
        allp=np.concatenate([pby[s] for s in SEEDS]);ally=np.concatenate([yby[s] for s in SEEDS])
        conf=np.abs(allp-0.5)*2  # 0..1 confidence
        print("  error-localization (CERTIS): error rate by confidence quartile (high conf should be low error):")
        q=np.quantile(conf,[0,.25,.5,.75,1.0])
        for i in range(4):
            m=(conf>=q[i])&(conf<=q[i+1] if i==3 else conf<q[i+1]);pred=(allp[m]>=0.5).astype(int);err=(pred!=ally[m]).mean()
            print(f"    Q{i+1} conf[{q[i]:.2f},{q[i+1]:.2f}]: err={err:.3f} (n={int(m.sum())})")
if __name__=="__main__": main()
