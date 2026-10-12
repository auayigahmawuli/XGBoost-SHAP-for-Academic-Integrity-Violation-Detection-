import pickle,numpy as np,pandas as pd,json,subprocess,sys
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
res=pickle.load(open("results_chapter.pkl","rb")); R=sorted(res); J=len(R)
d=pd.read_csv("/mnt/user-data/uploads/academic_integrity_records_demo.csv"); y=d.violation.values; groups=d.student_id.values
M=["xgb_pruned","xgb_full","rf","gb","svm","lr"]
LAB={"xgb_pruned":"Proposed (SHAP-pruned XGBoost)","xgb_full":"Ablated (XGBoost, no pruning)","rf":"Random forest","gb":"Gradient boosting","svm":"SVM (RBF)","lr":"Logistic regression"}
COL={"xgb_pruned":"#d95f02","xgb_full":"#55a868","rf":"#8172b3","gb":"#937860","svm":"#c44e52","lr":"#4c72b0"}
FEATS=pd.get_dummies(d.drop(columns=["violation","student_id"]),columns=["device_os_version"],dtype=float).rename(columns={"device_type":"device_type_mobile","enrollment_type":"enrollment_part_time"}).columns.tolist()
A=lambda m,k:np.array([res[r]["models"][m][k] for r in R])
out={"J":J,"n":res[R[0]]["n"],"pos":[res[r]["pos"] for r in R]}
out["mean"]={m:{k:(A(m,k).mean(),A(m,k).std(ddof=1)) for k in ["acc","prec","rec","f1","roc","ap","fpr","spec","train_s","infer_us"]} for m in M}
out["cnt"]={m:{k:(A(m,k).mean(),A(m,k).std(ddof=1)) for k in ["tn","fp","fn","tp"]} for m in M}
n_tr,n_va,n_te=res[R[0]]["n"]; ratio=n_te/n_tr
# stats on F1
comp={}
for o in M[1:]:
    dd=A("xgb_pruned","f1")-A(o,"f1"); md=dd.mean(); sd=dd.std(ddof=1); se=np.sqrt((1/J+ratio)*sd**2); t=md/se; p=2*stats.t.sf(abs(t),J-1); tc=stats.t.ppf(.975,J-1)
    comp[o]=dict(md=md,sd=sd,t=t,p=p,pb=min(1,5*p),ci=(md-tc*se,md+tc*se),dz=md/sd,sw=stats.shapiro(dd),wins=int((dd>0).sum()))
out["comp"]=comp; out["ratio"]=ratio
# pruning
ks=np.array([res[r]["k"] for r in R]); out["k"]=(ks.mean(),ks.std(ddof=1),ks.min(),ks.max())
sel=np.zeros(len(FEATS))
for r in R:
    for f in res[r]["keep"]: sel[FEATS.index(f)]+=1
out["sel"]=sel; sm=np.array([res[r]["models"]["xgb_pruned"]["shap"] for r in R]); out["shap"]=(sm.mean(0),sm.std(0,ddof=1))
# verification: ablated model SHAP rank & permutation importance of removed vs retained
rk_ret,rk_rem,pi_ret,pi_rem=[],[],[],[]
for r in R:
    sh=res[r]["models"]["xgb_full"]["shap"]; rank=(-sh).argsort().argsort()+1; pi=res[r]["models"]["xgb_full"]["perm"]
    keep=[FEATS.index(f) for f in res[r]["keep"]]; rem=[i for i in range(len(FEATS)) if i not in keep]
    rk_ret.append(rank[keep].mean()); rk_rem.append(rank[rem].mean()); pi_ret.append(pi[keep].mean()); pi_rem.append(pi[rem].mean())
out["ver"]=dict(rk_ret=(np.mean(rk_ret),np.std(rk_ret,ddof=1)),rk_rem=(np.mean(rk_rem),np.std(rk_rem,ddof=1)),pi_ret=(np.mean(pi_ret),np.std(pi_ret,ddof=1)),pi_rem=(np.mean(pi_rem),np.std(pi_rem,ddof=1)),pdiff=(np.mean([res[r]["pdiff"] for r in R]),np.std([res[r]["pdiff"] for r in R],ddof=1)))
# thresholds (proposed) and validation-optimised threshold
grid=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]; thr={t:[] for t in grid}; opt=[]; optt=[]
for r in R:
    seed=res[r]["seed"]; o=StratifiedGroupKFold(5,shuffle=True,random_state=seed); rest,te=next(o.split(d.values,y,groups))
    i=StratifiedGroupKFold(4,shuffle=True,random_state=seed); a,b=next(i.split(d.values[rest],y[rest],groups[rest])); va=rest[b]; yva=y[va]
    mm=res[r]["models"]["xgb_pruned"]; s=mm["score"]; sv=mm["val_score"]; yt=res[r]["yte"]
    for t in grid:
        pr=(s>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yt,pr,labels=[0,1]).ravel(); thr[t].append((fp/(fp+tn),tp/(tp+fn),tp/max(tp+fp,1),f1_score(yt,pr)))
    ts=np.linspace(0.05,0.95,91); bt=ts[np.argmax([f1_score(yva,(sv>=t).astype(int)) for t in ts])]; optt.append(bt)
    pr=(s>=bt).astype(int); tn,fp,fn,tp=confusion_matrix(yt,pr,labels=[0,1]).ravel(); opt.append((fp/(fp+tn),tp/(tp+fn),f1_score(yt,pr)))
out["thr"]={t:np.array(v) for t,v in thr.items()}; out["opt"]=np.array(opt); out["optt"]=(np.mean(optt),np.std(optt,ddof=1))
# subgroups
sub={}
for name,col,lv in [("device_type","device_type",["desktop","mobile"]),("enrollment_type","enrollment_type",[None,None])]:
    vals=sorted(d[col].unique()); sub[name]={}
    for v in vals:
        L=[]
        for r in R:
            te=res[r]["te"]; yt=res[r]["yte"]; pr=res[r]["models"]["xgb_pruned"]["pred"]; m=(d[col].values[te]==v)
            tn,fp,fn,tp=confusion_matrix(yt[m],pr[m],labels=[0,1]).ravel(); L.append((m.sum(),f1_score(yt[m],pr[m]),fp/(fp+tn),tp/(tp+fn)))
        sub[name][v]=np.array(L)
out["sub"]=sub
# figures
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
g=np.linspace(0,1,201); fig,ax=plt.subplots(figsize=(4.8,3.9)); curves={}
for m in ["xgb_pruned","xgb_full","lr"]:
    cs=[]
    for r in R:
        s=res[r]["models"][m]["score"]; pr,rc,_=precision_recall_curve(res[r]["yte"],s); o=np.argsort(rc); cs.append(np.interp(g,rc[o],pr[o]))
    cs=np.array(cs); curves[m]=cs.mean(0)
    ax.plot(g,cs.mean(0),color=COL[m],lw=1.8,ls={"xgb_pruned":"-","xgb_full":"--","lr":":"}[m],label=LAB[m]); ax.fill_between(g,cs.mean(0)-cs.std(0,ddof=1),cs.mean(0)+cs.std(0,ddof=1),color=COL[m],alpha=.15,lw=0)
ax.axhline(y.mean(),color="grey",lw=.8,ls=":"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(loc="upper right",fontsize=7.5,frameon=False)
fig.tight_layout(); fig.savefig("c_fig4_pr_curves.png",dpi=300); fig.savefig("c_fig4_pr_curves.pdf"); plt.close()
dd=curves["xgb_pruned"]-curves["xgb_full"]; out["pr_cross"]=g[np.where(np.diff((dd>0).astype(int))!=0)[0]]; out["pr_start_sign"]=np.sign(dd[1:6])
fig,ax=plt.subplots(figsize=(6.0,3.8)); rs=np.random.RandomState(0)
for i,m in enumerate(M):
    v=A(m,"f1"); ax.boxplot(v,positions=[i],widths=.5,showfliers=False,medianprops=dict(color="black"),boxprops=dict(color=COL[m]),whiskerprops=dict(color=COL[m]),capprops=dict(color=COL[m]))
    ax.scatter(i+rs.uniform(-.15,.15,len(v)),v,s=14,color=COL[m],alpha=.8,zorder=3)
ax.set_xticks(range(len(M))); ax.set_xticklabels(["Proposed","Ablated","RF","GB","SVM","LR"]); ax.set_ylabel("F1 (violation class, τ = 0.5)")
lo=min(A(m,"f1").min() for m in M); ax.set_ylim(lo-0.06,max(A(m,"f1").max() for m in M)+0.02)
for i,m in enumerate(M): ax.text(i,lo-0.055,f"{A(m,'f1').mean():.3f}\n± {A(m,'f1').std(ddof=1):.3f}",ha="center",fontsize=7.5)
fig.tight_layout(); fig.savefig("c_fig5_f1_distribution.png",dpi=300); fig.savefig("c_fig5_f1_distribution.pdf"); plt.close()
mm_,ss_=out["shap"]; top=[i for i in np.argsort(-mm_) if mm_[i]>0][:10]
fig,ax=plt.subplots(figsize=(5.2,3.8)); ax.barh(range(len(top))[::-1],mm_[top],xerr=ss_[top],color=COL["xgb_pruned"],ecolor="black",error_kw=dict(lw=.8,capsize=2))
ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels([FEATS[i] for i in top],fontsize=8); ax.set_xlabel("Mean |SHAP value| (log-odds), held-out test partitions")
fig.tight_layout(); fig.savefig("c_fig6_shap_importance.png",dpi=300); fig.savefig("c_fig6_shap_importance.pdf"); plt.close()
fig,axs=plt.subplots(1,2,figsize=(6.2,3.0))
for ax,m in zip(axs,["xgb_full","xgb_pruned"]):
    c=np.array([[out["cnt"][m]["tn"][0],out["cnt"][m]["fp"][0]],[out["cnt"][m]["fn"][0],out["cnt"][m]["tp"][0]]]); n=c/c.sum(1,keepdims=True); ax.imshow(n,cmap="Blues",vmin=0,vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{n[i,j]:.3f}\n({c[i,j]:.1f})",ha="center",va="center",color="white" if n[i,j]>.5 else "black",fontsize=8)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["No violation","Violation"]); ax.set_yticklabels(["No violation","Violation"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(["Ablated","Proposed"][m=="xgb_pruned"],fontsize=9)
    for s_ in ax.spines.values(): s_.set_visible(True)
fig.tight_layout(); fig.savefig("c_fig7_confusion.png",dpi=300); fig.savefig("c_fig7_confusion.pdf"); plt.close()
fig,ax=plt.subplots(figsize=(4.8,3.6)); T=np.array(grid)
for idx,(lab,c) in enumerate([("False-positive rate","#c44e52"),("Recall","#4c72b0"),("F1","#d95f02")]):
    col={0:0,1:1,2:3}[idx]; mu=np.array([out["thr"][t][:,col].mean() for t in grid]); sd=np.array([out["thr"][t][:,col].std(ddof=1) for t in grid])
    ax.plot(T,mu,color=c,marker="o",ms=3.5,label=lab); ax.fill_between(T,mu-sd,mu+sd,color=c,alpha=.15,lw=0)
ax.axvline(0.5,color="grey",lw=.8,ls=":"); ax.set_xlabel("Decision threshold"); ax.set_ylabel("Rate or score (test partitions)"); ax.set_ylim(0,1.02); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig("c_fig8_threshold.png",dpi=300); fig.savefig("c_fig8_threshold.pdf"); plt.close()
# memory
code='''
import resource,sys,pandas as pd,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
d=pd.read_csv("/mnt/user-data/uploads/academic_integrity_records_demo.csv");y=d.violation.values
X=pd.get_dummies(d.drop(columns=["violation","student_id"]),dtype=float).values
k=sys.argv[1]
if k=="xgb_pruned": X=X[:,:8]
m={"lr":LogisticRegression(max_iter=2000),"rf":RandomForestClassifier(n_estimators=200,n_jobs=1),"gb":GradientBoostingClassifier(n_estimators=100),"svm":SVC(),"xgb_full":xgb.XGBClassifier(n_estimators=300,max_depth=6,n_jobs=1,verbosity=0),"xgb_pruned":xgb.XGBClassifier(n_estimators=300,max_depth=6,n_jobs=1,verbosity=0)}[k]
m.fit(X[:2220],y[:2220]);m.predict(X[2960:])
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
'''
out["mem"]={m:float(subprocess.check_output([sys.executable,"-c",code,m]).decode().strip()) for m in M}
pickle.dump(out,open("summary_chapter.pkl","wb"))
def P(*a): print(*a)
for m in M: P(m,{k:f"{a:.3f}±{b:.3f}" for k,(a,b) in out["mean"][m].items()})
for m in M: P(m,"cnt",{k:f"{a:.1f}±{b:.1f}" for k,(a,b) in out["cnt"][m].items()})
P("n",out["n"],"pos",out["pos"][:3],"ratio",ratio)
for o,c in comp.items(): P(o,{k:(np.round(v,4) if not isinstance(v,tuple) else tuple(np.round(v,4))) for k,v in c.items() if k!="sw"},"SW",tuple(np.round(c["sw"],3)))
P("k",out["k"]); P("sel",[(FEATS[i],int(sel[i])) for i in np.argsort(-sel) if sel[i]>0])
P("shap",[(FEATS[i],round(mm_[i],3),round(ss_[i],3)) for i in top])
P("ver",out["ver"]); P("thr",{t:np.round(v.mean(0),3) for t,v in out["thr"].items()}); P("opt",out["opt"].mean(0),out["opt"].std(0,ddof=1),out["optt"])
for n_,dct in sub.items():
    for v,a in dct.items(): P(n_,v,"n",a[:,0].mean(),"F1",a[:,1].mean().round(3),a[:,1].std(ddof=1).round(3),"FPR",a[:,2].mean().round(3),a[:,2].std(ddof=1).round(3),"rec",a[:,3].mean().round(3),a[:,3].std(ddof=1).round(3))
P("mem",out["mem"]); P("cross",out["pr_cross"],out["pr_start_sign"])
for m in M: P(m,"params(run0)",res[R[0]]["models"][m]["params"])
P("val f1",{m:round(float(A(m,"val_f1").mean()),3) for m in M})
