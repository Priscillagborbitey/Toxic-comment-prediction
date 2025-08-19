# app.py — inference-only Streamlit app
import json, re, string, hashlib
from pathlib import Path
import numpy as np
import streamlit as st
from gensim.models import FastText
import joblib

ARTI = Path("project/artifacts")

# ---------- Load artifacts ----------
required = ["fasttext_model.bin","label_cols.json","thresholds.json","logistic_model.pkl","random_forest_model.pkl","stop_words.pkl","stemmer.pkl"]
missing = [f for f in required if not (ARTI/f).exists()]
if missing:
    st.error(f"Missing artifacts: {missing}. Run `python train.py` first.")
    st.stop()

LABEL_COLS = json.loads((ARTI/"label_cols.json").read_text())
THRESHOLDS = json.loads((ARTI/"thresholds.json").read_text())
STOP_WORDS = joblib.load(ARTI/"stop_words.pkl")
STEMMER = joblib.load(ARTI/"stemmer.pkl")

ft = FastText.load(str(ARTI/"fasttext_model.bin"))
lr_model = joblib.load(ARTI/"logistic_model.pkl")
rf_model = joblib.load(ARTI/"random_forest_model.pkl")

# Optional: verify checksums
try:
    checksums = json.loads((ARTI/"checksums.json").read_text())
    def md5p(p: Path) -> str:
        h=hashlib.md5(); h.update(p.read_bytes()); return h.hexdigest()
    bad = []
    for fname, md5 in checksums.items():
        p = ARTI/fname
        if p.exists() and md5p(p) != md5:
            bad.append(fname)
    if bad:
        st.warning(f"Checksum mismatch for: {bad}. Artifacts may be corrupted.")
except Exception:
    pass

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+|https\S+","", s)
    s = re.sub(r"\d+","", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+"," ", s).strip()
    s = re.sub(r"[^a-z\s]","", s)
    toks = [w for w in s.split() if w.isalpha() and w not in STOP_WORDS]
    toks = [STEMMER.stem(w) for w in toks]
    return " ".join(toks)

def embed_avg(comment: str) -> np.ndarray:
    toks = comment.split()
    vecs = [ft.wv[w] for w in toks if w in ft.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(ft.vector_size, dtype=np.float32)

def proba_lr(model, X):
    if hasattr(model,"predict_proba"): return np.asarray(model.predict_proba(X))
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

st.set_page_config(page_title="Toxic Comment Detection", layout="wide")
st.title("🔍 Toxic Comment Detection (Inference Only)")

model_choice = st.sidebar.radio("Model", ["Logistic Regression","Random Forest"])
txt = st.text_area("Enter a comment:", height=140)

if st.button("Predict"):
    cleaned = clean_text(txt)
    emb = embed_avg(cleaned).reshape(1,-1)
    if not np.any(emb):
        st.error("No recognizable tokens in FastText vocabulary.")
    else:
        if model_choice == "Logistic Regression":
            scores = proba_lr(lr_model, emb)
            th = THRESHOLDS["lr"]
        else:
            scores = proba_rf(rf_model, emb)
            th = THRESHOLDS["rf"]

        n = len(LABEL_COLS)
        if scores.shape != (1, n):
            st.error(f"Score shape mismatch: {scores.shape} vs expected (1,{n}).")
        else:
            st.subheader("Results")
            for i, lbl in enumerate(LABEL_COLS):
                p = float(scores[0, i])
                yhat = int(p >= th.get(lbl, 0.5))
                st.write(f"**{lbl}**: {yhat}  _(p={p:.2f}, thr={th.get(lbl,0.5):.2f})_")

with st.expander("Model & Training Info"):
    try:
        st.json(json.loads((ARTI/"metrics_report.json").read_text()))
    except Exception:
        st.info("No metrics file found.")
    try:
        st.json(json.loads((ARTI/"versions.json").read_text()))
    except Exception:
        pass
