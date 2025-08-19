# Group4updated_sess.py
# Toxic Comment Detection — stylish, interactive, leakage-free, with session_state + full evaluation
# Python 3.9 compatible (uses typing.Optional, no union '|')

import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from gensim.models import FastText
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# This downloads the stopwords file so your app can use it
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Create a set of English stopwords
stop_words = set(stopwords.words("english"))
# Optional wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# Page config (+ creative menu)

st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://streamlit.io/",
        "Report a bug": "mailto:group4-support@example.com?subject=Toxic%20Dashboard%20Bug",
        "About": "🧪 Toxic Comment Detection • Group 4 — University of Ghana\n\nFastText + Logistic Regression / Random Forest.\nLeakage-free with train/test separation."
    },
)


# Constants & paths

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
TEXT_COL = "comment_text"
ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH  = ROOT / "test.csv"


# CSS (scoped; no global layout overrides across pages)

st.markdown("""
<style>
/* Sidebar cosmetics */
.sb-header { font-weight:800; font-size:1.05rem; margin-bottom:6px; display:flex; align-items:center; gap:8px; }
.sb-subtle { color:#6b7280; font-size:.85rem; margin-top:-4px; margin-bottom:8px; }
.sb-chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.sb-chip {
  padding:3px 8px; border-radius:999px; font-size:.78rem; border:1px solid #e5e7eb;
  background:#f9fafb; color:#111827; display:inline-flex; align-items:center; gap:6px;
}
.sb-chip.ok { background:#dcfce7; border-color:#86efac; color:#065f46; }
.sb-chip.miss { background:#fee2e2; border-color:#fecaca; color:#7f1d1d; }
.sb-sep { margin:8px 0; height:1px; background:linear-gradient(90deg,#0000,#e5e7eb,#0000); }

/* Home-only scope */
.home-root { margin-top: 14px; }
.home-root .hero {
  border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
  background: radial-gradient(1100px 360px at 8% -20%, #eef2ff 6%, #fff 65%);
  border:1px solid #e5e7eb;
}
.home-root .hero-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.home-root .hero-title { display:flex; gap:10px; align-items:center; }
.home-root .hero-title .emoji { font-size:1.4rem }
.home-root .hero-title .heading { font-weight:800; font-size:1.05rem; }
.home-root .hero-title .subtitle { color:#4b5563; font-size:.95rem; white-space:nowrap; }
.home-root .chips { display:flex; gap:8px; overflow:auto; white-space:nowrap; padding-bottom:2px; }
.home-root .badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600;
  font-size:.83rem; border:1px solid #e5e7eb; background:#f9fafb; color:#111827;
}

.home-root .step-container { margin-top: 28px; }
.home-root .step-card {
  background: white; border-radius: 16px; padding: 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  transition: transform 0.2s ease; text-align: center; height: 100%;
  border:1px solid #e5e7eb;
}
.home-root .step-card:hover { transform: translateY(-4px); box-shadow: 0 6px 18px rgba(0,0,0,0.12); }
.home-root .step-number { font-size: 12px; letter-spacing: 2px; color: #6b7280; font-weight: 700; margin-bottom: 8px; }
.home-root .step-title { font-size: 18px; font-weight: 800; color: #2c3e50; margin-bottom: 6px; }
.home-root .step-desc { font-size: 14px; color: #555; }

/* Pills for predictions */
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:.85rem; border:1px solid #e5e7eb; background:#f9fafb; color:#111827; }
.badge.on { background:#dcfce7; border-color:#86efac; color:#065f46; }
.badge.off{ background:#fee2e2; border-color:#fecaca; color:#7f1d1d; }
.pills { display:flex; flex-wrap:wrap; gap:8px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* Hide default footer; use custom fixed one */
footer { visibility: hidden; }
.group4-footer {
  position: fixed; left:0; bottom:0; width:100%;
  background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
  color:#fff; text-align:center; padding:10px 8px; font-size:13.5px;
  border-top:1px solid rgba(255,255,255,0.15); z-index:9999;
}
.group4-footer a { color:#e5f2ff; text-decoration:none; font-weight:600; }
.group4-footer a:hover { text-decoration:underline; }

/* Divider */
hr { border:none; height:1px; background:linear-gradient(90deg,#0000,#e5e7eb,#0000); margin:.65rem 0; }
</style>
""", unsafe_allow_html=True)


# Initialize session state

def init_state():
    keys_defaults = {
        "train_df": None,
        "test_df": None,
        "clean_train": None,
        "clean_test": None,
        "ft_model": None,
        "X_train": None,
        "X_test": None,            # unlabeled test features
        "y_train": None,
        "lr_model": None,
        "rf_model": None,
        "metrics": {},             # overview metrics
        "reports": {},             # classification reports per model
        "confusions": {},          # confusion matrices per model per label
        "rocs": {},                # ROC data per model per label
    }
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# Data loading & preprocessing

stop_words = set(stopwords.words("english"))
stop_words.update({'article','wikipedia','page','edit','talk','user','please','thanks','thank'})
stemmer = PorterStemmer()

def clean_text_simple(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+','',t)
    t = re.sub(r'<.*?>','',t)
    t = re.sub(r'[^a-z\s]','',t)
    return t

def clean_text_pipeline(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+|https\S+","",t)
    t = re.sub(r"\d+","",t)
    t = t.translate(str.maketrans("","",string.punctuation))
    t = re.sub(r"\s+"," ",t).strip()
    t = re.sub(r"[^a-z\s]","",t)
    tokens = [w for w in t.split() if w not in stop_words and w.isalpha()]
    out=[]
    for w in tokens:
        try: out.append(stemmer.stem(w))
        except RecursionError: continue
    return " ".join(out)

def has_all_labels(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in LABEL_COLS)

@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_files():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        st.error("Missing `train.csv` or `test.csv` next to this script.")
        st.stop()

def ensure_data_in_state():
    """Load raw CSVs into session_state if missing."""
    if st.session_state.train_df is None:
        st.session_state.train_df = load_csv_cached(str(TRAIN_PATH))
    if st.session_state.test_df is None:
        st.session_state.test_df = load_csv_cached(str(TEST_PATH))

def preprocess_into_state():
    """Clean train & test (separately, no leakage) and store in state."""
    tr = st.session_state.train_df.copy()
    te = st.session_state.test_df.copy()

    # Guard rail for common column mismatch
    if TEXT_COL not in tr.columns or TEXT_COL not in te.columns:
        st.error(f"Both CSVs must include a `{TEXT_COL}` column.")
        st.stop()

    tr["cleaned_text"] = tr[TEXT_COL].astype(str).apply(clean_text_pipeline)
    te["cleaned_text"] = te[TEXT_COL].astype(str).apply(clean_text_pipeline)

    tr = tr[tr["cleaned_text"].str.strip()!=""].drop_duplicates(subset="cleaned_text")
    te = te[te["cleaned_text"].str.strip()!=""].drop_duplicates(subset="cleaned_text")

    tr["comment_length"] = tr["cleaned_text"].str.len()
    te["comment_length"] = te["cleaned_text"].str.len()
    if has_all_labels(tr):
        tr["num_labels"] = tr[LABEL_COLS].sum(axis=1)

    st.session_state.clean_train = tr
    st.session_state.clean_test  = te


# Embeddings & models

def train_or_load_fasttext(sentences: List[List[str]]) -> FastText:
    model_path = ROOT / "fasttext_model.bin"
    if model_path.exists():
        return FastText.load(str(model_path))
    ft = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)
    ft.save(str(model_path))
    return ft

def embed_comment(ft_model: FastText, comment: str) -> np.ndarray:
    tokens = str(comment).split()
    vecs = [ft_model.wv[w] for w in tokens if w in ft_model.wv.key_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(ft_model.vector_size)

def embed_series(ft_model: FastText, series: pd.Series) -> np.ndarray:
    return np.vstack(series.astype(str).apply(lambda x: embed_comment(ft_model, x)).values)

def train_or_load_models(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Any]:
    lr_path = ROOT / "logistic_model.pkl"
    rf_path = ROOT / "random_forest_model.pkl"

    lr = joblib.load(str(lr_path)) if lr_path.exists() else None
    rf = joblib.load(str(rf_path)) if rf_path.exists() else None
    if lr is not None and rf is not None:
        return lr, rf

    lr = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", penalty="l2", C=3.0, class_weight="balanced", max_iter=1000),
        n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    joblib.dump(lr, str(lr_path))
    joblib.dump(rf, str(rf_path))
    return lr, rf


# Evaluation utilities (multi-label)

def per_label_confusion(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> np.ndarray:
    """
    Return 2x2 confusion matrix for a single label: [[TN, FP],[FN, TP]]
    """
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])

def compute_reports_and_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    y_true, y_pred: shape (n_samples, n_labels) with 0/1 entries
    Returns: (classification_report_dict, {label: 2x2 confusion})
    """
    rep = classification_report(y_true, y_pred, target_names=LABEL_COLS, output_dict=True, zero_division=0)
    confs = {}
    for i, label in enumerate(LABEL_COLS):
        confs[label] = per_label_confusion(y_true[:, i], y_pred[:, i])
    return rep, confs

def compute_rocs(
    y_true: np.ndarray, y_proba: Optional[np.ndarray], model_name: str
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute ROC data per label if probabilities available.
    Returns dict: label -> {"fpr": array, "tpr": array, "auc": float}
    y_proba shape expected: (n_samples, n_labels) with positive class probability.
    """
    rocs = {}
    if y_proba is None:
        return rocs
    for i, label in enumerate(LABEL_COLS):
        # Skip if single-class ground truth (roc_curve would error)
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
        rocs[label] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return rocs


# Visualization helpers

def label_pill(name: str, on: int, prob: Optional[float]=None) -> str:
    cls = "on" if int(on)==1 else "off"
    conf = f' <span class="mono">({prob:.2f})</span>' if prob is not None else ""
    return f'<span class="badge {cls}">{name}{conf}</span>'

def chart_label_distribution(df: pd.DataFrame, title="Label Distribution"):
    counts = df[LABEL_COLS].sum(axis=0).reset_index()
    counts.columns = ["label","count"]
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("label:N", sort="-x", title=""),
        tooltip=["label","count"]
    ).properties(height=230, title=title)
    st.altair_chart(chart, use_container_width=True)

def chart_length_hist(df: pd.DataFrame, title="Length distribution (chars)"):
    data = pd.DataFrame({"length": df["cleaned_text"].str.len()})
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("length:Q", bin=alt.Bin(maxbins=60), title="Length (chars)"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    ).properties(height=230, title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def chart_label_correlation(df: pd.DataFrame, title="Label Correlation"):
    corr = df[LABEL_COLS].corr()
    long = corr.stack().reset_index()
    long.columns = ["x","y","value"]
    chart = alt.Chart(long).mark_rect().encode(
        x=alt.X("x:O", title="", sort=LABEL_COLS),
        y=alt.Y("y:O", title="", sort=LABEL_COLS),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=(-1,1))),
        tooltip=["x","y",alt.Tooltip("value:Q", format=".2f")]
    ).properties(height=260, title=title)
    st.altair_chart(chart, use_container_width=True)

def chart_label_cooccurrence(df: pd.DataFrame, title="Label Co-occurrence"):
    co = pd.DataFrame(0, index=LABEL_COLS, columns=LABEL_COLS, dtype=int)
    for i in LABEL_COLS:
        for j in LABEL_COLS:
            co.loc[i, j] = int((df[i].astype(int) & df[j].astype(int)).sum())
    long = co.stack().reset_index()
    long.columns = ["x","y","count"]
    chart = alt.Chart(long).mark_rect().encode(
        x=alt.X("x:O", title="", sort=LABEL_COLS),
        y=alt.Y("y:O", title="", sort=LABEL_COLS),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["x","y","count"]
    ).properties(height=260, title=title)
    st.altair_chart(chart, use_container_width=True)

def chart_ngram_top(texts: pd.Series, ngram=(1,1), topk=30, title="Top n-grams"):
    vec = CountVectorizer(ngram_range=ngram, min_df=2, max_df=0.9)
    X = vec.fit_transform(texts.values)
    if X.shape[1] == 0:
        st.info("No n-grams found with the current settings.")
        return
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(sorted(vec.vocabulary_.items(), key=lambda kv: kv[1]))
    terms = vocab[:,0]
    df = pd.DataFrame({"term": terms, "freq": freqs}).sort_values("freq", ascending=False).head(topk)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("freq:Q", title="Frequency"),
        y=alt.Y("term:N", sort="-x", title=""),
        tooltip=["term","freq"]
    ).properties(height=300, title=title)
    st.altair_chart(chart, use_container_width=True)

def chart_prob_bars(probs: pd.Series):
    df = pd.DataFrame({"label": LABEL_COLS, "prob": probs})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("prob:Q", title="Confidence", scale=alt.Scale(domain=(0,1))),
        y=alt.Y("label:N", sort="-x", title=""),
        tooltip=["label", alt.Tooltip("prob:Q", format=".2f")]
    ).properties(height=220, title="Per-label probability").interactive()
    st.altair_chart(chart, use_container_width=True)

def chart_pca_map(X: np.ndarray, labels_num: np.ndarray, sample_texts: pd.Series):
    n = min(2000, X.shape[0])
    idx = np.random.RandomState(42).choice(X.shape[0], n, replace=False) if X.shape[0] > n else np.arange(X.shape[0])
    Xs = X[idx]; ys = labels_num[idx]; texts = sample_texts.iloc[idx]
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(Xs)
    df = pd.DataFrame({"pc1": P[:,0], "pc2": P[:,1], "labels": ys, "snippet": texts.str.slice(0, 140)})
    chart = alt.Chart(df).mark_circle(size=50, opacity=0.7).encode(
        x=alt.X("pc1:Q", title="PC1"),
        y=alt.Y("pc2:Q", title="PC2"),
        color=alt.Color("labels:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=[alt.Tooltip("labels:Q", title="#labels"), alt.Tooltip("snippet:N", title="text")]
    ).properties(height=380, title="PCA map of FastText embeddings (sample)").interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_conf_matrix_matrix(cm: np.ndarray, title: str):
    """Render a 2x2 confusion matrix."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"], ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='both', labelsize=9)
    st.pyplot(fig, use_container_width=False)

def plot_roc_curves_per_label(roc_dict: Dict[str, Dict[str, np.ndarray]], model_name: str):
    """Plot ROC curves per label with Altair (interactive) if data available."""
    rows = []
    for label, data in roc_dict.items():
        fpr = data.get("fpr"); tpr = data.get("tpr"); auc_val = data.get("auc", None)
        if fpr is None or tpr is None:
            continue
        rows += [{"label": label, "fpr": float(f), "tpr": float(t), "auc": float(auc_val)} for f, t in zip(fpr, tpr)]
    if not rows:
        st.info(f"No ROC data available for {model_name} (likely missing probabilities or single-class labels).")
        return
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line().encode(
        x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=(0,1))),
        y=alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=(0,1))),
        color="label:N",
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f"),
                 alt.Tooltip("auc:Q", format=".2f")]
    ).properties(height=320, title=f"ROC Curves — {model_name}")
    diag = alt.Chart(pd.DataFrame({"x":[0,1], "y":[0,1]})).mark_line(strokeDash=[4,4]).encode(x="x", y="y")
    st.altair_chart(line + diag, use_container_width=True)


# Sidebar menu (selectbox + status chips)

def sidebar_nav():
    st.sidebar.markdown('<div class="sb-header">🧭 Menu</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sb-subtle">Jump to a section</div>', unsafe_allow_html=True)

    menu_labels = {
        "Home": "🏠 Home",
        "Dataset Preview": "📊 Dataset Preview",
        "Preprocessing": "🧼 Preprocessing",
        "Modeling": "🤖 Modeling",
        "Prediction": "🔍 Prediction",
    }
    page = st.sidebar.selectbox(
        "Go to",
        list(menu_labels.keys()),
        format_func=lambda k: menu_labels[k],
        index=0,
        label_visibility="collapsed",
    )

    # Quick artifact presence chips
    data_ok = TRAIN_PATH.exists() and TEST_PATH.exists()
    ft_ok   = (ROOT / "fasttext_model.bin").exists() or (st.session_state.ft_model is not None)
    lr_ok   = (ROOT / "logistic_model.pkl").exists() or (st.session_state.lr_model is not None)
    rf_ok   = (ROOT / "random_forest_model.pkl").exists() or (st.session_state.rf_model is not None)

    def chip(txt, ok):
        cls = "ok" if ok else "miss"
        icon = "✅" if ok else "⛔"
        return f'<span class="sb-chip {cls}">{icon} {txt}</span>'

    chips_html = " ".join([
        chip("Data", data_ok),
        chip("FastText", ft_ok),
        chip("LR", lr_ok),
        chip("RF", rf_ok),
    ])
    st.sidebar.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sb-chips">{chips_html}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.sidebar.caption("FastText ➜ LR / RF • cached • interactive visuals")
    return page


# Pages

def page_home():
    st.markdown('<div class="home-root">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div class="hero-row">
        <div class="hero-title">
          <div class="emoji">🧪</div>
          <div>
            <div class="heading">Toxic Comment Detection</div>
            <div class="subtitle">Leakage-free FastText ➜ Logistic Regression / Random Forest</div>
          </div>
        </div>
        <div class="chips">
          <span class="badge">Multi-label</span>
          <span class="badge">FastText embeddings</span>
          <span class="badge">Interactive charts</span>
          <span class="badge">Submission export</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 1</div>
            <div class="step-title">📊 Dataset Preview</div>
            <div class="step-desc">Explore head, lengths, n-grams, labels</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 2</div>
            <div class="step-title">🧼 Preprocessing</div>
            <div class="step-desc">Clean train & test separately (no leakage)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 3–4</div>
            <div class="step-title">🤖 Modeling & Prediction</div>
            <div class="step-desc">Train LR/RF • Single & batch prediction</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 👥 Group 4")
    st.markdown("""
    <div class="">
      <ul>
        <li><strong>Priscilla D. Gborbitey</strong> — 22253220</li>
        <li><strong>Philip Kwasi Adjartey</strong> — 22252449</li>
        <li><strong>Nicco-Annan Marilyn</strong> — 11410745</li>
        <li><strong>Naa Borteley Sebastian-Kumah</strong> — 22253153</li>
        <li><strong>Bernice Baadawo Abbe</strong> — 22253447</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_preview():
    ensure_files()
    ensure_data_in_state()

    train_df = st.session_state.train_df
    test_df  = st.session_state.test_df

    tab_overview, tab_text, tab_labels, tab_ngrams = st.tabs(["Overview", "Text Analysis", "Label Analysis", "N-gram Explorer"])

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows", f"{train_df.shape[0]:,}")
        c2.metric("Train cols", f"{train_df.shape[1]}")
        c3.metric("Test rows",  f"{test_df.shape[0]:,}")
        c4.metric("Test cols",  f"{test_df.shape[1]}")
        st.markdown("---")
        st.write("**Head (Train)**")
        st.dataframe(train_df.head(12), use_container_width=True, height=260)
        st.write("**Head (Test)**")
        st.dataframe(test_df.head(12), use_container_width=True, height=260)

    with tab_text:
        if TEXT_COL not in train_df.columns:
            st.error(f"Train missing `{TEXT_COL}`.")
        else:
            if st.session_state.clean_train is None or st.session_state.clean_test is None:
                preprocess_into_state()
            tr = st.session_state.clean_train
            te = st.session_state.clean_test
            c1, c2 = st.columns(2)
            with c1:
                chart_length_hist(tr, "Train: length distribution")
            with c2:
                chart_length_hist(te, "Test: length distribution")

            # Word frequencies / wordcloud
            all_words = ' '.join(tr[TEXT_COL].astype(str).apply(clean_text_simple))
            words = [w for w in re.findall(r'\b[a-z]+\b', all_words) if w not in stop_words and len(w) > 2]
            common = Counter(words).most_common(120)
            st.markdown("#### Top words (Train)")
            if WORDCLOUD_AVAILABLE:
                wc = WordCloud(width=1000, height=350, background_color='white', max_words=150)
                wc.generate_from_frequencies(dict(common))
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                st.pyplot(fig, use_container_width=True)
            else:
                dfw = pd.DataFrame(common[:40], columns=["word","freq"])
                chart = alt.Chart(dfw).mark_bar().encode(
                    x="freq:Q", y=alt.Y("word:N", sort="-x"), tooltip=["word","freq"]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

    with tab_labels:
        if has_all_labels(train_df):
            c1, c2 = st.columns(2)
            with c1:
                chart_label_distribution(train_df, "Train label distribution")
                # Any toxic vs clean
                any_toxic = (train_df[LABEL_COLS].sum(axis=1) > 0).value_counts().rename(index={True:"toxic", False:"clean"}).reset_index()
                any_toxic.columns = ["class","count"]
                pie = alt.Chart(any_toxic).mark_arc(outerRadius=100).encode(
                    theta="count:Q", color="class:N", tooltip=["class","count"]
                ).properties(title="Clean vs Toxic (any label)")
                st.altair_chart(pie, use_container_width=True)
            with c2:
                chart_label_correlation(train_df)
                chart_label_cooccurrence(train_df)
        else:
            st.warning("Label columns missing in train — label analysis skipped.")

    with tab_ngrams:
        if TEXT_COL not in train_df.columns:
            st.error(f"Train missing `{TEXT_COL}`.")
        else:
            if st.session_state.clean_train is None:
                preprocess_into_state()
            tr = st.session_state.clean_train
            st.markdown("#### N-gram Explorer (Train)")
            n = st.select_slider("N-gram size", options=["Unigrams (1)", "Bigrams (2)", "Trigrams (3)"], value="Unigrams (1)")
            topk = st.slider("Top K", 10, 60, 30, step=5)
            if n.startswith("Uni"):
                chart_ngram_top(tr["cleaned_text"], (1,1), topk, title="Top unigrams")
            elif n.startswith("Bi"):
                chart_ngram_top(tr["cleaned_text"], (2,2), topk, title="Top bigrams")
            else:
                chart_ngram_top(tr["cleaned_text"], (3,3), topk, title="Top trigrams")

def page_preprocess():
    ensure_files()
    ensure_data_in_state()

    if TEXT_COL not in st.session_state.train_df.columns or TEXT_COL not in st.session_state.test_df.columns:
        st.error(f"Both CSVs must include `{TEXT_COL}`.")
        return

    with st.spinner("Cleaning train & test…"):
        preprocess_into_state()

    tr = st.session_state.clean_train
    te = st.session_state.clean_test

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cleaned Train (sample)")
        cols = [TEXT_COL, "cleaned_text", "comment_length"] + (["num_labels"] if "num_labels" in tr.columns else [])
        st.dataframe(tr[cols].head(20), use_container_width=True, height=320)
    with c2:
        st.markdown("#### Cleaned Test (sample)")
        st.dataframe(te[[TEXT_COL, "cleaned_text", "comment_length"]].head(20), use_container_width=True, height=320)

    tr.to_csv("cleaned_train.csv", index=False)
    te.to_csv("cleaned_test.csv", index=False)
    st.success("Saved cleaned_train.csv & cleaned_test.csv")

def page_modeling():
    ensure_files()
    ensure_data_in_state()
    if st.session_state.clean_train is None or st.session_state.clean_test is None:
        preprocess_into_state()

    tr = st.session_state.clean_train
    te = st.session_state.clean_test

    if not has_all_labels(tr):
        st.error("Train lacks label columns required for evaluation.")
        return

    # Prepare FastText
    if st.session_state.ft_model is None:
        sentences = tr["cleaned_text"].astype(str).apply(lambda x: x.split()).tolist()
        with st.spinner("Preparing FastText model…"):
            st.session_state.ft_model = train_or_load_fasttext(sentences)
        st.success("FastText ready")

    ft = st.session_state.ft_model

    # Embeddings
    if st.session_state.X_train is None or st.session_state.y_train is None:
        with st.spinner("Embedding train…"):
            # Reset index so X and y align
            tr_reset = tr.reset_index(drop=True)

            # Drop rows where labels are missing
            tr_reset = tr_reset.dropna(subset=LABEL_COLS)

            # Embed only the clean text
            st.session_state.X_train = embed_series(ft, tr_reset["cleaned_text"])

            # Labels aligned with X
            st.session_state.y_train = tr_reset[LABEL_COLS].values

    if st.session_state.X_test is None:
        with st.spinner("Embedding test…"):
            st.session_state.X_test = embed_series(ft, te["cleaned_text"])
            joblib.dump(st.session_state.X_test, str(ROOT/"X_test_unlabeled.pkl"))

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    # Coverage metrics
    vocab_size = len(ft.wv)
    avg_tokens = int(np.mean([len(s) for s in tr["cleaned_text"].str.split()]) if len(tr) else 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", f"{tr.shape[0]:,}")
    c2.metric("FastText vocab", f"{vocab_size:,}")
    c3.metric("Avg tokens/row", f"{avg_tokens}")

    st.markdown("#### PCA embedding map (Train sample)")
    labels_num = tr["num_labels"].values if "num_labels" in tr.columns else tr[LABEL_COLS].sum(axis=1).values
    chart_pca_map(X_train, labels_num, tr[TEXT_COL])

    # Models
    if st.session_state.lr_model is None or st.session_state.rf_model is None:
        with st.spinner("Training / loading LR & RF…"):
            lr_model, rf_model = train_or_load_models(X_train, y_train)
            st.session_state.lr_model = lr_model
            st.session_state.rf_model = rf_model
        st.success("Models ready")

    lr_model = st.session_state.lr_model
    rf_model = st.session_state.rf_model

    # Training-set predictions (NOTE: test set has no labels, so evaluation is on train)
    lr_pred_train = lr_model.predict(X_train)
    rf_pred_train = rf_model.predict(X_train)

    # Probabilities for ROC (shape handling)
    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if isinstance(proba, list):  # OneVsRest on some sklearn versions
                return np.vstack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba]).T
            return proba  # expected (n_samples, n_labels)
        return None

    lr_proba_train = get_probs(lr_model, X_train)
    rf_proba_train = None
    try:
        rf_proba = rf_model.predict_proba(X_train)  # for multi-output RF, this is a list of arrays
        if isinstance(rf_proba, list):
            rf_proba_train = np.vstack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in rf_proba]).T
        else:
            rf_proba_train = rf_proba
    except Exception:
        rf_proba_train = None

    # Metrics overview (training set)
    def metrics_frame(y_true, y_pred, name):
        return pd.DataFrame({
            "Model": [name],
            "Macro F1": [f1_score(y_true, y_pred, average='macro')],
            "Micro F1": [f1_score(y_true, y_pred, average='micro')],
            "Weighted F1": [f1_score(y_true, y_pred, average='weighted')],
            "Hamming Loss": [hamming_loss(y_true, y_pred)],
        })

    mdf = pd.concat([
        metrics_frame(y_train, lr_pred_train, "Logistic Regression"),
        metrics_frame(y_train, rf_pred_train, "Random Forest")
    ], ignore_index=True)
    st.session_state.metrics = mdf.to_dict(orient="list")

    st.markdown("#### 📈 Training Metrics Overview *(evaluated on train; test has no labels)*")
    st.dataframe(
        mdf.style.format({"Macro F1": "{:.3f}","Micro F1":"{:.3f}","Weighted F1":"{:.3f}","Hamming Loss":"{:.3f}"}),
        use_container_width=True
    )

    # Classification reports + confusion matrices per label
    st.markdown("### 📝 Classification Reports & Confusion Matrices (Train)")

    # Logistic Regression
    lr_rep, lr_confs = compute_reports_and_confusions(y_train, lr_pred_train, "LR")
    st.session_state.reports["lr"] = lr_rep
    st.session_state.confusions["lr"] = lr_confs
    st.markdown("#### Logistic Regression — Classification Report")
    st.dataframe(pd.DataFrame(lr_rep).transpose().style.format("{:.3f}"), use_container_width=True)

    st.markdown("#### Logistic Regression — Confusion Matrices (per label)")
    for i in range(0, len(LABEL_COLS), 3):
        cols = st.columns(3)
        for j, label in enumerate(LABEL_COLS[i:i+3]):
            with cols[j]:
                plot_conf_matrix_matrix(lr_confs[label], f"{label}")

    # Random Forest
    rf_rep, rf_confs = compute_reports_and_confusions(y_train, rf_pred_train, "RF")
    st.session_state.reports["rf"] = rf_rep
    st.session_state.confusions["rf"] = rf_confs
    st.markdown("#### Random Forest — Classification Report")
    st.dataframe(pd.DataFrame(rf_rep).transpose().style.format("{:.3f}"), use_container_width=True)

    st.markdown("#### Random Forest — Confusion Matrices (per label)")
    for i in range(0, len(LABEL_COLS), 3):
        cols = st.columns(3)
        for j, label in enumerate(LABEL_COLS[i:i+3]):
            with cols[j]:
                plot_conf_matrix_matrix(rf_confs[label], f"{label}")

    # ROC Curves per label (if proba available)
    st.markdown("### 📉 ROC Curves per Label (Train)")
    lr_rocs = compute_rocs(y_train, lr_proba_train, "LR")
    st.session_state.rocs["lr"] = lr_rocs
    plot_roc_curves_per_label(lr_rocs, "Logistic Regression")

    rf_rocs = compute_rocs(y_train, rf_proba_train, "RF")
    st.session_state.rocs["rf"] = rf_rocs
    plot_roc_curves_per_label(rf_rocs, "Random Forest")

def page_prediction():
    st.markdown("### 🔍 Prediction")

    # Load from session or disk
    ft = st.session_state.ft_model or (FastText.load(str(ROOT/"fasttext_model.bin")) if (ROOT/"fasttext_model.bin").exists() else None)
    lr_model = st.session_state.lr_model or (joblib.load(str(ROOT/"logistic_model.pkl")) if (ROOT/"logistic_model.pkl").exists() else None)
    rf_model = st.session_state.rf_model or (joblib.load(str(ROOT/"random_forest_model.pkl")) if (ROOT/"random_forest_model.pkl").exists() else None)

    if ft is None or (lr_model is None and rf_model is None):
        st.error("Models not found. Train them first on the Modeling page.")
        return

    st.sidebar.header("⚙️ Options")
    model_choice = st.sidebar.radio("Choose a model", ["Logistic Regression","Random Forest"], index=0)

    tab_single, tab_batch = st.tabs(["Single Text", "Batch Prediction (test.csv)"])

    with tab_single:
        text = st.text_area("Enter a comment:", height=120, help="We’ll clean and embed it with FastText (same as training).")
        if st.button("Predict"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                cleaned = clean_text_pipeline(text)
                vec = embed_comment(ft, cleaned).reshape(1,-1)
                model = lr_model if model_choice=="Logistic Regression" else rf_model
                pred = model.predict(vec)[0]

                probs = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec)
                    if isinstance(proba, list):
                        probs = np.array([p[0,1] if p.shape[1]>1 else p[0,0] for p in proba])
                    else:
                        probs = proba[0]

                st.markdown("#### Result")
                pills=[]
                for i, label in enumerate(LABEL_COLS):
                    v = int(pred[i]) if isinstance(pred,(np.ndarray,list)) else int(pred)
                    p = None if probs is None else float(probs[i])
                    pills.append(label_pill(label, v, p))
                st.markdown(f'<div class="pills">{" ".join(pills)}</div>', unsafe_allow_html=True)

                if probs is not None:
                    chart_prob_bars(pd.Series(probs, index=LABEL_COLS))

    with tab_batch:
        ensure_files()
        # Prefer cached X_test in session, fallback to disk or recompute
        if st.session_state.X_test is not None:
            X_test = st.session_state.X_test
            test_df = st.session_state.clean_test if st.session_state.clean_test is not None else load_csv_cached(str(TEST_PATH))
        elif (ROOT/"X_test_unlabeled.pkl").exists():
            X_test = joblib.load(str(ROOT/"X_test_unlabeled.pkl"))
            test_df = load_csv_cached(str(TEST_PATH))
        else:
            ensure_data_in_state()
            if st.session_state.clean_test is None:
                preprocess_into_state()
            X_test = embed_series(ft, st.session_state.clean_test["cleaned_text"])
            test_df = st.session_state.clean_test
            st.session_state.X_test = X_test

        model = lr_model if model_choice=="Logistic Regression" else rf_model
        preds = model.predict(X_test)

        out = pd.DataFrame(preds, columns=LABEL_COLS)
        # try preserve id if present
        base_test = st.session_state.test_df if st.session_state.test_df is not None else load_csv_cached(str(TEST_PATH))
        if "id" in base_test.columns:
            out.insert(0, "id", base_test["id"].values[:len(out)])
        else:
            out.insert(0, "row_id", np.arange(len(out)))

        st.dataframe(out.head(25), use_container_width=True, height=330)
        st.download_button("⬇️ Download submission.csv", out.to_csv(index=False).encode("utf-8"),
                           file_name="submission.csv", mime="text/csv")

        st.markdown("#### Predicted label distribution")
        chart_label_distribution(out.rename(columns={"row_id":"id"}), "Predicted counts (test)")

# =========================
# Router
# =========================
page = sidebar_nav()

if page == "Home":
    page_home()
elif page == "Dataset Preview":
    page_preview()
elif page == "Preprocessing":
    page_preprocess()
elif page == "Modeling":
    page_modeling()
elif page == "Prediction":
    page_prediction()

# =========================
# Group 4 Footer
# =========================
st.markdown("""
<div class="group4-footer">
  📊 <strong>Toxic Comment Detection</strong> — Built by <strong>Group 4</strong> · University of Ghana
</div>
""", unsafe_allow_html=True)

