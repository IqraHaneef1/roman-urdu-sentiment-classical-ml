# Roman Urdu sentiment: classical classifiers x TF-IDF feature sets (runs on CPU, no downloads)
import os, re, sys, time, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
np.random.seed(SEED)
SRC = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "run_log.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

P("sklearn", sklearn.__version__, "numpy", np.__version__, "pandas", pd.__version__)
df = pd.read_csv(SRC, header=None, usecols=[0, 1], names=["text", "label"], on_bad_lines="skip")
P("Rows read:", len(df))
P("Raw label counts:\n" + df["label"].astype(str).str.strip().value_counts(dropna=False).to_string())
df["label"] = df["label"].astype(str).str.strip().str.capitalize()
labels = ["Positive", "Negative", "Neutral"]
n0 = len(df)
df = df[df["label"].isin(labels)].dropna(subset=["text"]).copy()
P("Dropped (bad label / missing text):", n0 - len(df))

def clean(t):
    t = str(t).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["text"].map(clean)
n1 = len(df)
df = df[df["clean"].str.len() > 0]
n_empty = n1 - len(df)
n2 = len(df)
# duplicates: report how many, and how many have conflicting labels
dup_groups = df.groupby("clean")["label"].nunique()
conflict = int((dup_groups > 1).sum())
df = df.drop_duplicates(subset="clean").reset_index(drop=True)
P(f"Empty after cleaning: {n_empty}; duplicates removed: {n2 - len(df)}; duplicate texts with conflicting labels: {conflict}")
P("Final size:", len(df))
cc = df["label"].value_counts().reindex(labels)
P("Final class counts:\n" + cc.to_string())
cc.to_csv(os.path.join(OUT, "class_counts.csv"))
lens = df["clean"].str.split().str.len()
P(f"Words per sentence: mean {lens.mean():.2f}, median {lens.median():.0f}, max {lens.max()}")
df["y"] = df["label"].map({l: i for i, l in enumerate(labels)})

train, test = train_test_split(df, test_size=0.20, stratify=df["y"], random_state=SEED)
P(f"Train {len(train)} | Test {len(test)}")
ytr, yte = train["y"].values, test["y"].values

def macro(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return accuracy_score(y, p), pr, rc, f

# baseline
dm = DummyClassifier(strategy="most_frequent").fit(train["clean"], ytr)
bm = macro(yte, dm.predict(test["clean"]))
P("Majority-class baseline (acc, P, R, F1):", [round(x, 4) for x in bm])

feats = {
    "Word 1-gram": dict(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True),
    "Word 1-2-gram": dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    "Char 2-5-gram": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000),
}
grids = {
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.3, 1.0]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000), {"C": [0.3, 1, 3, 10]}),
    "Linear SVM": (LinearSVC(), {"C": [0.1, 0.3, 1, 3]}),
    "Random Forest": (RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=SEED), None),
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
rows, preds, vocab = [], {}, {}
for fname, fkw in feats.items():
    vec = TfidfVectorizer(**fkw)
    Xtr = vec.fit_transform(train["clean"]); Xte = vec.transform(test["clean"])
    vocab[fname] = (Xtr.shape[1], vec)
    P(f"\n== {fname}: {Xtr.shape[1]} features")
    for mname, (est, grid) in grids.items():
        t0 = time.time()
        if grid:
            gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=2)
            gs.fit(Xtr, ytr); model = gs.best_estimator_
            best = gs.best_params_; cvf1 = gs.best_score_
            cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
        else:
            model = est.fit(Xtr, ytr); best = {"n_estimators": 300}; cvf1 = np.nan; cv_std = np.nan
        secs = time.time() - t0
        p = model.predict(Xte); preds[(mname, fname)] = p
        acc, pr, rc, f1 = macro(yte, p)
        rows.append(dict(Features=fname, Model=mname, Accuracy=acc, MacroP=pr, MacroR=rc, MacroF1=f1,
                         CV_MacroF1=cvf1, CV_std=cv_std, Params=json.dumps(best), Seconds=round(secs, 1)))
        P(f"{mname:20s} F1={f1:.4f} acc={acc:.4f} best={best} cvF1={cvf1:.4f} ({secs:.1f}s)")
        if mname == "Logistic Regression":
            vocab[(fname, "lr")] = model

res = pd.DataFrame(rows)
res.round(4).to_csv(os.path.join(OUT, "results_all.csv"), index=False)

best_row = res.sort_values("MacroF1", ascending=False).iloc[0]
bn, bf = best_row["Model"], best_row["Features"]
P(f"\nBest on test: {bn} + {bf}  macro-F1={best_row['MacroF1']:.4f}")
bp = preds[(bn, bf)]
rep = classification_report(yte, bp, target_names=labels, digits=4, zero_division=0)
P(rep); open(os.path.join(OUT, "best_report.txt"), "w").write(f"{bn} + {bf}\n{rep}")
pr_, rc_, f_, sup = precision_recall_fscore_support(yte, bp, zero_division=0)
pd.DataFrame({"Class": labels, "Precision": pr_, "Recall": rc_, "F1": f_, "Support": sup}).round(4).to_csv(
    os.path.join(OUT, "per_class_best.csv"), index=False)

# bootstrap CI for the best model and paired difference vs. the runner-up (different model name)
rng = np.random.default_rng(SEED)
B = 2000; n = len(yte)
idx = rng.integers(0, n, size=(B, n))
def boot(p):
    return np.array([f1_score(yte[i], p[i], average="macro") for i in idx])
bb = boot(bp)
lo, hi = np.percentile(bb, [2.5, 97.5])
P(f"Bootstrap 95% CI macro-F1 of best: [{lo:.4f}, {hi:.4f}]")
ci_rows = []
for (m, f), p in preds.items():
    b = boot(p); ci_rows.append(dict(Model=m, Features=f, Lo=np.percentile(b, 2.5), Hi=np.percentile(b, 97.5)))
pd.DataFrame(ci_rows).round(4).to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)
ru = res[(res["Model"] != bn)].sort_values("MacroF1", ascending=False).iloc[0]
rp = preds[(ru["Model"], ru["Features"])]
diff = bb - boot(rp)
P(f"Runner-up (other model): {ru['Model']} + {ru['Features']} F1={ru['MacroF1']:.4f}; "
  f"paired bootstrap diff mean={diff.mean():.4f} 95% CI [{np.percentile(diff,2.5):.4f}, {np.percentile(diff,97.5):.4f}]; "
  f"P(best>runner-up)={(diff>0).mean():.3f}")
# same-model feature gap
same = res[res["Model"] == bn].sort_values("MacroF1", ascending=False)
P("Feature sets for best model:\n" + same[["Features", "MacroF1"]].round(4).to_string(index=False))

# figures
fig, ax = plt.subplots(figsize=(7.5, 4))
mods = list(grids.keys()); fl = list(feats.keys()); w = 0.26
cols = ["#4a6fa5", "#d98c3f", "#5a9e78"]
for j, fn in enumerate(fl):
    vals = [res[(res.Model == m) & (res.Features == fn)]["MacroF1"].iloc[0] for m in mods]
    ax.bar(np.arange(len(mods)) + (j - 1) * w, vals, w, label=fn, color=cols[j])
ax.axhline(bm[3], color="gray", ls="--", lw=1, label="Majority baseline")
ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods)
ax.set_ylabel("Macro-F1 (test set)"); ax.set_ylim(0, 1)
ax.legend(fontsize=8, ncol=2, loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_macro_f1.png"), dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay(confusion_matrix(yte, bp), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"{bn}, {bf}", fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_confusion_best.png"), dpi=300); plt.close()
cmn = confusion_matrix(yte, bp)
pd.DataFrame(cmn, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]).to_csv(
    os.path.join(OUT, "confusion_best.csv"))
P("Confusion matrix (rows true, cols pred):\n" + str(cmn))

# top terms from Logistic Regression (word 1-gram) for interpretation
lr = vocab[("Word 1-gram", "lr")]; vec = vocab["Word 1-gram"][1]
fn_ = np.array(vec.get_feature_names_out()); coefs = lr.coef_
tops = {}
for i, l in enumerate(labels):
    tops[l] = list(fn_[np.argsort(-coefs[i])[:12]])
    P(f"Top LR terms for {l}: {tops[l]}")
json.dump(tops, open(os.path.join(OUT, "top_terms.json"), "w"))

# error examples from the best model
err = test.assign(pred=[labels[i] for i in bp])
err = err[err["label"] != err["pred"]][["text", "label", "pred"]]
err.sample(min(60, len(err)), random_state=SEED).to_csv(os.path.join(OUT, "error_examples.csv"), index=False)
P("Error rate:", round(len(err) / len(test), 4))
# agreement between classes by length
tl = test["clean"].str.split().str.len().values
for lo_, hi_, nm in [(0, 3, "1-3 words"), (4, 8, "4-8 words"), (9, 10**6, "9+ words")]:
    m = (tl >= lo_) & (tl <= hi_)
    P(f"Accuracy {nm}: n={m.sum()} acc={accuracy_score(yte[m], bp[m]):.4f} F1={f1_score(yte[m], bp[m], average='macro'):.4f}")
P("DONE")
