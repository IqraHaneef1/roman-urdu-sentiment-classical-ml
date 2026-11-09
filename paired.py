import re, numpy as np, pandas as pd, sys, platform
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
SEED=42
df=pd.read_csv("data.csv",header=None,usecols=[0,1],names=["text","label"],on_bad_lines="skip")
df["label"]=df["label"].astype(str).str.strip().str.capitalize()
labels=["Positive","Negative","Neutral"]
df=df[df["label"].isin(labels)].dropna(subset=["text"]).copy()
def clean(t):
    t=str(t).lower(); t=re.sub(r"https?://\S+|www\.\S+"," ",t); t=re.sub(r"@\w+"," ",t)
    t=re.sub(r"[^a-z0-9\s]"," ",t); t=re.sub(r"(.)\1{2,}",r"\1\1",t); return re.sub(r"\s+"," ",t).strip()
df["clean"]=df["text"].map(clean); df=df[df["clean"].str.len()>0].drop_duplicates(subset="clean").reset_index(drop=True)
df["y"]=df["label"].map({l:i for i,l in enumerate(labels)})
train,test=train_test_split(df,test_size=0.2,stratify=df["y"],random_state=SEED)
print("sizes",len(train),len(test),"python",platform.python_version())
yte=test["y"].values
vw=TfidfVectorizer(analyzer="word",ngram_range=(1,1),min_df=2,sublinear_tf=True)
vc=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=3,sublinear_tf=True,max_features=100000)
Xw=vw.fit_transform(train["clean"]); Xwt=vw.transform(test["clean"])
Xc=vc.fit_transform(train["clean"]); Xct=vc.transform(test["clean"])
P={}
P["svm_char"]=LinearSVC(C=0.3).fit(Xc,train["y"]).predict(Xct)
P["svm_word"]=LinearSVC(C=0.3).fit(Xw,train["y"]).predict(Xwt)
P["lr_char"]=LogisticRegression(C=3,max_iter=2000).fit(Xc,train["y"]).predict(Xct)
P["lr_word"]=LogisticRegression(C=3,max_iter=2000).fit(Xw,train["y"]).predict(Xwt)
for k,v in P.items(): print(k, round(f1_score(yte,v,average="macro"),4))
rng=np.random.default_rng(SEED); B=2000; n=len(yte); idx=rng.integers(0,n,size=(B,n))
def boot(p): return np.array([f1_score(yte[i],p[i],average="macro") for i in idx])
bs={k:boot(v) for k,v in P.items()}
def cmp(a,b):
    d=bs[a]-bs[b]; print(f"{a} - {b}: mean {d.mean():+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] frac>0 {(d>0).mean():.3f}")
cmp("svm_char","svm_word"); cmp("lr_char","lr_word"); cmp("svm_char","lr_char")
