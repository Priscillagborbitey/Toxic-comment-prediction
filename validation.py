
import platform, importlib, json
import psutil
import streamlit as st

def _ok(msg): st.markdown(f"✅ {msg}")
def _warn(msg): st.warning(msg)
def _fail(msg): st.error(msg); st.stop()

def preflight_environment(min_versions=None):
    st.subheader("🔎 Environment Preflight")
    _ok(f"Python: **{platform.python_version()}**")
    try:
        mem = psutil.virtual_memory()
        _ok(f"System RAM: **{mem.total/1e9:.1f} GB** (avail: **{mem.available/1e9:.1f} GB**)")
    except Exception as e:
        _warn(f"Could not read system memory: {e}")
    pkgs = {
        "numpy": importlib.import_module("numpy").__version__,
        "pandas": importlib.import_module("pandas").__version__,
        "scikit-learn": importlib.import_module("sklearn").__version__,
        "gensim": importlib.import_module("gensim").__version__,
        "scipy": importlib.import_module("scipy").__version__,
        "nltk": importlib.import_module("nltk").__version__,
        "matplotlib": importlib.import_module("matplotlib").__version__,
        "seaborn": importlib.import_module("seaborn").__version__,
        "streamlit": importlib.import_module("streamlit").__version__,
    }
    st.code(json.dumps(pkgs, indent=2))
    if min_versions:
        try:
            from packaging import version
            for name, minv in min_versions.items():
                if version.parse(pkgs.get(name, "0")) < version.parse(minv):
                    _warn(f"{name} < {minv}. Consider upgrading.")
        except Exception:
            _warn("Install `packaging` to enforce version checks.")
