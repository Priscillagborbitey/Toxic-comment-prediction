# train.py — offline training pipeline
import os, json, re, string, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import FastText
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, make_scorer

# --- NLTK setup (offline only) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
for r in ["stopwords", "wordnet", "omw-1.4"]:
    try: nltk.data.find(f"corpora/{r}")
    except LookupError: nltk.download(r)

# ---------- CONFIG ----------
DATA_PATH = Path("project/data/train.csv")
ARTI = Path("project/artifacts"); ARTI.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
TEXT_COL  = "comment_text"
CLEAN_COL = "cleaned_text"
RANDOM_SEED = 42

# ---------- CLEANING ----------
stop_words = set(stopwords.words("english"))
stop_words.update({"article","wikipedia","page","edit","talk","user","please","thanks","thank"})
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+","", text)
    text = re.sub(r"\d+","", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+"," ", text).strip()
    text = re.sub(r"[^a-z\s]","", text)
    toks = [w for w in text.split() if w.isalpha() and w not in stop_words]
    out = []
    for w in toks:
        try: out.append(stemmer.stem(w))
        except RecursionError: continue
    return " ".join(out)

# ---------- LOAD & VALIDATE ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Place Jigsaw `train.csv` there.")
df = pd.read_csv(DATA_PATH)

missing = [c for c in [TEXT_COL] + LABEL_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in LABEL_COLS:
    try:
        df[c] = df[c].astype(int)
    except Exception:
        df[c] = df[c].map({True:1, False:0, "1":1, "0":0}).astype(int)
    u = set(df[c].unique().tolist())
    if not u.issubset({0,1}):
        raise ValueError(f"Label `{c}` has non-binary values: {sorted(u)}")

# ---------- PREPROCESS ----------
df[CLEAN_COL] = df[TEXT_COL].astype(str).apply(clean_text)
df = df[df[CLEAN_COL].str.strip() != ""].drop_duplicates(subset=CLEAN_COL)
y = df[LABEL_COLS].astype(int)

# ---------- TRAIN FASTTEXT ----------
sentences = df[CLEAN_COL].astype(str).apply(str.split).tolist()
ft = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)

def embed_row(s: str) -> np.ndarray:
    toks = s.split()
    vecs = [ft.wv[w] for w in toks if w in ft.wv]
    if not vecs: return np.zeros(ft.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

X = np.vstack([embed_row(s) for s in df[CLEAN_COL]])

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# ---------- MODELS & TUNING ----------
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
f1_macro = make_scorer(f1_score, average="macro")

# Logistic Regression (OvR)
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
lr_grid = GridSearchCV(
    lr,
    param_grid=[
        {"estimator__solver":["lbfgs"], "estimator__penalty":["l2"], "estimator__C":[0.1,1,3,10],
         "estimator__class_weight":[None,"balanced"]},
        {"estimator__solver":["liblinear"], "estimator__penalty":["l1","l2"], "estimator__C":[0.1,1,3,10],
         "estimator__class_weight":[None,"balanced"]},
    ],
    scoring=f1_macro, cv=cv, n_jobs=-1, refit=True, verbose=0
)
lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_

# Random Forest (multioutput)
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED),
    param_grid={
        "n_estimators":[200,400],
        "max_depth":[None,10,20],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1,2],
    },
    scoring=f1_macro, cv=cv, n_jobs=-1, refit=True, verbose=0
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# ---------- PROBABILITIES & METRICS ----------
def proba_lr(model, X):
    if hasattr(model,"predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model,"decision_function"):
        s = model.decision_function(X)
        return 1/(1+np.exp(-np.asarray(s)))
    return np.zeros((X.shape[0], len(LABEL_COLS)))

def proba_rf(model, X):
    if hasattr(model,"predict_proba"):
        plist = model.predict_proba(X)
        cols = []
        for p in plist:
            if p is None: cols.append(np.zeros((X.shape[0],)))
            else: cols.append(p[:,1] if p.shape[1]==2 else np.zeros((X.shape[0],)))
        return np.column_stack(cols)
    return np.zeros((X.shape[0], len(LABEL_COLS)))

def macro_auc(y_true, y_score):
    aucs=[]
    yt = y_true.values
    for j in range(yt.shape[1]):
        if len(np.unique(yt[:,j]))<2: continue
        try:
            aucs.append(roc_auc_score(yt[:,j], y_score[:,j]))
        except Exception: pass
    return float(np.mean(aucs)) if aucs else float("nan")

lr_p = proba_lr(lr_best, X_test)
rf_p = proba_rf(rf_best, X_test)
lr_pred = (lr_p >= 0.5).astype(int)
rf_pred = (rf_p >= 0.5).astype(int)

report = {
    "lr": {
        "f1_macro": float(f1_score(y_test, lr_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, lr_pred, average="weighted")),
        "auc_macro": float(macro_auc(y_test, lr_p)),
        "classification_report": classification_report(y_test, lr_pred, zero_division=0, output_dict=True),
        "best_params": lr_grid.best_params_
    },
    "rf": {
        "f1_macro": float(f1_score(y_test, rf_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, rf_pred, average="weighted")),
        "auc_macro": float(macro_auc(y_test, rf_p)),
        "classification_report": classification_report(y_test, rf_pred, zero_division=0, output_dict=True),
        "best_params": rf_grid.best_params_
    }
}

# ---------- THRESHOLD TUNING ----------
def tune_thresholds(y_true, scores, grid=np.linspace(0.1,0.9,17)):
    th = {}
    for i, lbl in enumerate(LABEL_COLS):
        best_f1, best_t = -1.0, 0.5
        yt = y_true.values[:,i]
        if len(np.unique(yt)) < 2:
            th[lbl] = 0.5
            continue
        for t in grid:
            yhat = (scores[:,i] >= t).astype(int)
            f1 = f1_score(yt, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        th[lbl] = float(best_t)
    return th

thresholds = {"lr": tune_thresholds(y_test, lr_p), "rf": tune_thresholds(y_test, rf_p)}

# ---------- SAVE ARTIFACTS ----------
def md5p(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()

# save models
ft.save(str(ARTI/"fasttext_model.bin"))
joblib.dump(lr_best, ARTI/"logistic_model.pkl")
joblib.dump(rf_best, ARTI/"random_forest_model.pkl")
joblib.dump(stop_words, ARTI/"stop_words.pkl")
joblib.dump(stemmer, ARTI/"stemmer.pkl")

# metadata
(ARTI/"label_cols.json").write_text(json.dumps(LABEL_COLS))
(ARTI/"thresholds.json").write_text(json.dumps(thresholds, indent=2))

# metrics & versions
versions = {
    "numpy": __import__("numpy").__version__,
    "pandas": __import__("pandas").__version__,
    "sklearn": __import__("sklearn").__version__,
    "gensim": __import__("gensim").__version__,
    "nltk": __import__("nltk").__version__,
}
(ARTI/"metrics_report.json").write_text(json.dumps(report, indent=2))
(ARTI/"versions.json").write_text(json.dumps(versions, indent=2))
(ARTI/"artifact_meta.json").write_text(json.dumps({
    "model_id": "ft_lr_rf_v1",
    "created_by": "train.py",
    "params": {"ft_dim": 100, "cv_folds": 3},
}, indent=2))

# checksums
files = ["fasttext_model.bin","logistic_model.pkl","random_forest_model.pkl",
         "label_cols.json","thresholds.json","metrics_report.json",
         "versions.json","artifact_meta.json","stop_words.pkl","stemmer.pkl"]
checks = {f: md5p(ARTI/f) for f in files}
(ARTI/"checksums.json").write_text(json.dumps(checks, indent=2))

print("Artifacts saved to", ARTI.resolve())
