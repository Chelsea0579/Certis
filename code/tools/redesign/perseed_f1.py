"""Per-seed test F1 (mean+-std) for CERTIS and CERTIS+RotatE - for honest variance reporting."""
import os,sys,math
import numpy as np,pandas as pd,torch
from collections import defaultdict,Counter
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from experiment import result as R
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."));FD=os.path.join(ROOT,"outputs/redesign/features")
SEEDS=[42,1,7,100];STRUCT=["log_cohort","log_h_deg","log_t_ind","log_rel_freq","log_shared_n","transe_dist"];STR={"Head":str,"Relation":str,"Tail":str}
CFG={"codex-m":(["L","Q35","M","Q"],True),"fb15k-237":(["Lvn","Q35vn","Mvn","Qvn"],False)}
def add_struct(df,obs,dl):
    rt=defaultdict(set);h2rt=defaultdict(set);t2hr=defaultdict(set);rf=Counter();hn=defaultdict(set);tn=defaultdict(set)
    for h,r,t in zip(obs.Head,obs.Relation,obs.Tail):
        rt[(r,t)].add(h);h2rt[h].add((r,t));t2hr[t].add((h,r));rf[r]+=1;hn[h].add(t);tn[t].add(h)
    df["log_cohort"]=np.log1p(df.apply(lambda x:max(0,len(rt.get((x.Relation,x.Tail),set()))-1),axis=1))
    df["log_h_deg"]=np.log1p(df.Head.map(lambda h:len(h2rt.get(h,set()))));df["log_t_ind"]=np.log1p(df.Tail.map(lambda t:len(t2hr.get(t,set()))))
    df["log_rel_freq"]=np.log1p(df.Relation.map(lambda r:rf.get(r,0)));df["log_shared_n"]=np.log1p(df.apply(lambda x:len(hn.get(x.Head,set())&tn.get(x.Tail,set())),axis=1))
    df["transe_dist"]=df.apply(lambda x:dl.get((x.Head,x.Relation,x.Tail),1.7),axis=1) if dl else 1.7
    return df
def lc(ds,seed,part,tags):
    b=pd.read_csv(os.path.join(FD,f"{ds}_seed{seed}_{part}_{tags[0]}.csv"),dtype=STR)[["Head","Relation","Tail","Missing"]].copy();b["Missing"]=b["Missing"].astype(int)
    for t in tags: b[f"{t}_conf"]=pd.read_csv(os.path.join(FD,f"{ds}_seed{seed}_{part}_{t}.csv"),dtype=STR)[f"{t}_conf"].astype(float).values
    return b
def main():
    oc=pd.read_csv(os.path.join(ROOT,"data/codex-m/data_sample.csv"),dtype=STR);of=pd.read_csv(os.path.join(ROOT,"data/fb15k-237/data_sample.csv"),dtype=STR)
    dist=pd.read_csv(os.path.join(ROOT,"data/codex-m/cand_transe_dist.csv"));dlc={(r.Head,r.Relation,r.Tail):r.transe_dist for r in dist.itertuples()}
    for ds,(tags,ht) in CFG.items():
        obs=oc if ds=="codex-m" else of;dl=dlc if ht else {};feat=[f"{t}_conf" for t in tags]+STRUCT
        tf=TriplesFactory.from_labeled_triples(obs[["Head","Relation","Tail"]].astype(str).values);e2i,r2i=tf.entity_to_id,tf.relation_to_id
        m=pipeline(training=tf,testing=tf,model="RotatE",model_kwargs=dict(embedding_dim=128),training_kwargs=dict(num_epochs=50,batch_size=2048),random_seed=42,device="cuda" if torch.cuda.is_available() else "cpu").model
        dev=next(m.parameters()).device
        def rot(df):
            ids=[[e2i.get(h,0),r2i.get(r,0),e2i.get(t,0)] for h,r,t in zip(df.Head,df.Relation,df.Tail)]
            with torch.no_grad(): s=m.score_hrt(torch.tensor(ids,device=dev)).squeeze(-1).cpu().numpy()
            mk=np.array([(h in e2i and r in r2i and t in e2i) for h,r,t in zip(df.Head,df.Relation,df.Tail)]);return np.where(mk,s,np.min(s)-1.0)
        f1o=[];f1or=[]
        for s in SEEDS:
            cal=add_struct(lc(ds,s,"cal_fit",tags),obs,dl);te=add_struct(lc(ds,s,"test",tags),obs,dl);cal["rot"]=rot(cal);te["rot"]=rot(te)
            for name,feats,store in [("CERTIS",feat,f1o),("CERTIS+RotatE",feat+["rotz"],f1or)]:
                c=cal.copy();t=te.copy()
                if "rotz" in feats:
                    mu,sd=c["rot"].mean(),c["rot"].std()+1e-9;c["rotz"]=(c["rot"]-mu)/sd;t["rotz"]=(t["rot"]-mu)/sd
                X=np.nan_to_num(c[feats].values.astype(float));y=c.Missing.values
                clf=LogisticRegression(max_iter=3000,C=1.0).fit(X,y);oof=cross_val_predict(clone(clf),X,y,cv=5,method="predict_proba")[:,1];iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
                p=iso.transform(clf.predict_proba(np.nan_to_num(t[feats].values.astype(float)))[:,1]);store.append(R.compute_score((p>=0.5).astype(int).tolist(),t.Missing.values.tolist())[1])
        print(f"{ds}: CERTIS test F1 per-seed={[round(x,3) for x in f1o]} mean={np.mean(f1o):.3f}+-{np.std(f1o):.3f} | CERTIS+RotatE={[round(x,3) for x in f1or]} mean={np.mean(f1or):.3f}+-{np.std(f1or):.3f}")
if __name__=="__main__": main()
