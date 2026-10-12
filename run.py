import pandas as pd, numpy as np, time, pickle, os, warnings, sys
warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb, shap

DATA="/mnt/user-data/uploads/academic_integrity_records_demo.csv"
SEEDS=[42,123,456,789,101112,131415,161718,192021,222324,252627]
W=1.75; THR=0.5; CUM=0.85; OUT="results_chapter.pkl"
d=pd.read_csv(DATA); y=d.violation.values; groups=d.student_id.values
X=d.drop(columns=["violation","student_id"])
X=pd.get_dummies(X,columns=["device_os_version"],dtype=float)
X["device_type"]=(X["device_type"]=="mobile").astype(float); X["enrollment_type"]=(X["enrollment_type"]=="part-time").astype(float)
X=X.rename(columns={"device_type":"device_type_mobile","enrollment_type":"enrollment_part_time"}).astype(float)
FEATS=list(X.columns); Xv=X.values; D=len(FEATS)
BIN=[i for i,c in enumerate(FEATS) if set(np.unique(Xv[:,i]))<= {0.0,1.0}]
NUM=[i for i in range(D) if i not in BIN]
REF=dict(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,gamma=5.0,reg_lambda=1.0,reg_alpha=0.5)

def mk(name,p,seed):
    if name in("xgb_full","xgb_pruned"): return xgb.XGBClassifier(**{**p,"n_estimators":300},scale_pos_weight=W,random_state=seed,n_jobs=1,tree_method="hist",verbosity=0,eval_metric="logloss")
    if name=="rf": return RandomForestClassifier(**p,class_weight={0:1,1:W},random_state=seed,n_jobs=1)
    if name=="gb": return GradientBoostingClassifier(**p,random_state=seed)
    if name=="svm": return SVC(kernel="rbf",**p,class_weight={0:1,1:W},random_state=seed)
    if name=="lr": return LogisticRegression(**p,class_weight={0:1,1:W},max_iter=2000,random_state=seed)
def fit(name,m,X,y):
    if name=="gb": m.fit(X,y,sample_weight=np.where(y==1,W,1.0))
    else: m.fit(X,y)
    return m
def suggest(name,t):
    if name in("xgb_full","xgb_pruned"): return dict(max_depth=t.suggest_int("max_depth",3,8),learning_rate=t.suggest_float("learning_rate",0.01,0.3,log=True),subsample=t.suggest_float("subsample",0.5,1.0),colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),gamma=t.suggest_float("gamma",0,10),reg_lambda=t.suggest_float("reg_lambda",0.1,10),reg_alpha=t.suggest_float("reg_alpha",0,10))
    if name=="rf": return dict(n_estimators=t.suggest_int("n_estimators",100,300),max_depth=t.suggest_int("max_depth",3,20),min_samples_leaf=t.suggest_int("min_samples_leaf",1,10),max_features=t.suggest_float("max_features",0.3,1.0))
    if name=="gb": return dict(n_estimators=t.suggest_int("n_estimators",50,150),learning_rate=t.suggest_float("learning_rate",0.01,0.3,log=True),max_depth=t.suggest_int("max_depth",2,4),subsample=t.suggest_float("subsample",0.5,1.0),min_samples_leaf=t.suggest_int("min_samples_leaf",1,10))
    if name=="svm": return dict(C=t.suggest_float("C",0.1,100,log=True),gamma=t.suggest_float("gamma",1e-4,1,log=True))
    if name=="lr": return dict(C=t.suggest_float("C",1e-3,100,log=True))
def score_of(name,m,X):
    return m.decision_function(X) if name=="svm" else m.predict_proba(X)[:,1]
def metr(yt,pred,s):
    tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
    return dict(acc=accuracy_score(yt,pred),prec=precision_score(yt,pred,zero_division=0),rec=recall_score(yt,pred),f1=f1_score(yt,pred),roc=roc_auc_score(yt,s),ap=average_precision_score(yt,s),fpr=fp/(fp+tn),spec=tn/(tn+fp),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

res=pickle.load(open(OUT,"rb")) if os.path.exists(OUT) else {}
t00=time.time()
for r,seed in enumerate(SEEDS):
    if r in res: continue
    o=StratifiedGroupKFold(5,shuffle=True,random_state=seed); rest,te=next(o.split(Xv,y,groups))
    i=StratifiedGroupKFold(4,shuffle=True,random_state=seed); a,b=next(i.split(Xv[rest],y[rest],groups[rest])); tr,va=rest[a],rest[b]
    sc=StandardScaler().fit(Xv[tr][:,NUM])
    def prep(ix):
        Z=Xv[ix].copy(); Z[:,NUM]=sc.transform(Z[:,NUM]); return Z
    Xtr,Xva,Xte=prep(tr),prep(va),prep(te); ytr,yva,yte=y[tr],y[va],y[te]
    # SHAP pruning on training partition
    tp=time.time(); ref=mk("xgb_full",REF,seed); ref.set_params(n_estimators=300); ref.fit(Xtr,ytr)
    sv=np.abs(shap.TreeExplainer(ref).shap_values(Xtr)).mean(0); order=np.argsort(-sv); cum=np.cumsum(sv[order])/sv.sum()
    k=int(np.searchsorted(cum,CUM)+1); keep=sorted(order[:k].tolist()); prune_t=time.time()-tp
    run=dict(seed=seed,n=(len(tr),len(va),len(te)),pos=(int(ytr.sum()),int(yva.sum()),int(yte.sum())),k=k,keep=[FEATS[j] for j in keep],ref_imp=sv.tolist(),te=te,yte=yte,models={})
    for name,cols in [("xgb_pruned",keep),("xgb_full",list(range(D))),("rf",list(range(D))),("gb",list(range(D))),("svm",list(range(D))),("lr",list(range(D)))]:
        ntr=50 if name=="lr" else 100
        def obj(t):
            p=suggest(name,t); m=fit(name,mk(name,p,seed),Xtr[:,cols],ytr)
            return f1_score(yva,m.predict(Xva[:,cols]))
        st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(obj,n_trials=ntr)
        bp=st.best_params
        t1=time.time(); m=fit(name,mk(name,bp,seed),Xtr[:,cols],ytr); ft=time.time()-t1+(prune_t if name=="xgb_pruned" else 0)
        t2=time.time(); s=score_of(name,m,Xte[:,cols]); pred=m.predict(Xte[:,cols]); it=time.time()-t2
        sva=score_of(name,m,Xva[:,cols])
        mm=metr(yte,pred,s); mm.update(train_s=ft,infer_us=1e6*it/len(te),val_f1=st.best_value,params=bp,score=s,pred=pred,val_score=sva,val_f1_at_best=f1_score(yva,m.predict(Xva[:,cols])))
        if name in("xgb_pruned","xgb_full"):
            ts=time.time(); ex=shap.TreeExplainer(m).shap_values(Xte[:,cols]); mm["shap_s"]=time.time()-ts
            full=np.zeros(D); full[cols]=np.abs(ex).mean(0); mm["shap"]=full
        if name=="xgb_full":
            pi=permutation_importance(m,Xte,yte,scoring="f1",n_repeats=5,random_state=seed,n_jobs=1); mm["perm"]=pi.importances_mean
        run["models"][name]=mm
    run["pdiff"]=float(np.mean(np.abs(run["models"]["xgb_full"]["score"]-run["models"]["xgb_pruned"]["score"])))
    res[r]=run; pickle.dump(res,open(OUT,"wb"))
    print(f"run {r} seed {seed} k={k} F1 "+" ".join(f"{n}={run['models'][n]['f1']:.3f}" for n in run['models'])+f" t={time.time()-t00:.0f}s",flush=True)
print("DONE",flush=True)
