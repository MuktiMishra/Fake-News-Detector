"""
app.py
Streamlit interface for real-time Fake News Detection.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
import tensorflow as tf
from utils import load_tokenizer, predict_news, clean_text

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2.5rem; font-weight: 800; text-align: center; color: #1e3a5f; }
    .sub-title    { text-align: center; color: #555; margin-bottom: 1.5rem; }
    .result-fake  { background: #ffe4e4; border-left: 6px solid #e53e3e;
                    padding: 1rem; border-radius: 8px; }
    .result-real  { background: #e4ffe8; border-left: 6px solid #38a169;
                    padding: 1rem; border-radius: 8px; }
    .result-label { font-size: 1.8rem; font-weight: 800; }
    .confidence   { font-size: 1rem; color: #444; margin-top: 0.4rem; }
    .footer       { text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by LSTM Deep Learning · NLP-based Classification</div>',
            unsafe_allow_html=True)
st.divider()

# ─── Model / Tokenizer Loader ────────────────────────────────────────────────
MODEL_OPTIONS = {
    "LSTM (recommended)":   "model/lstm_model.keras",
    "Bi-LSTM (best accuracy)": "model/bilstm_model.keras",
    "Simple RNN":           "model/rnn_model.keras",
}
TOKENIZER_PATH = "model/tokenizer.pkl"


@st.cache_resource(show_spinner="Loading model …")
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner="Loading tokenizer …")
def load_tokenizer_cached(path: str):
    return load_tokenizer(path)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses **RNN / LSTM** deep learning models trained on news datasets "
        "to classify whether a news article is **Fake** or **Real**."
    )
    st.markdown("**Tech Stack:** Python · TensorFlow · Keras · Streamlit · NLP")
    st.markdown("---")

    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.success("Cache cleared. Model will reload.")

# ─── Model availability check ────────────────────────────────────────────────
if not os.path.exists(model_path) or not os.path.exists(TOKENIZER_PATH):
    st.warning("⚠️ Model files not found. Please train the model first.")
    st.code("python train.py --model lstm", language="bash")
    st.info("After training, restart this app.")
    st.stop()

model     = load_model_cached(model_path)
tokenizer = load_tokenizer_cached(TOKENIZER_PATH)

# ─── Input section ───────────────────────────────────────────────────────────
st.subheader("📝 Enter News Article")

col1, col2 = st.columns([3, 1])
with col1:
    title_input = st.text_input("Headline / Title (optional)", placeholder="e.g. Government announces new policy on taxation")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)

text_input = st.text_area(
    "Article Body",
    height=200,
    placeholder="Paste the full news article text here …",
)

analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ─── Sample news ─────────────────────────────────────────────────────────────
with st.expander("💡 Try Sample Articles"):
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📰 Sample: Real News"):
            st.session_state["sample_text"] = (
                "The Federal Reserve raised its benchmark interest rate by a quarter percentage point "
                "on Wednesday, the latest in a series of increases aimed at cooling inflation. "
                "Officials indicated they may slow the pace of future hikes as they assess the impact "
                "of previous increases on the economy. The decision was unanimous among voting members."
            )
            st.session_state["sample_title"] = "Federal Reserve raises interest rates by 0.25 percent"
    with col_b:
        if st.button("🚨 Sample: Fake News"):
            st.session_state["sample_text"] = (
                "SHOCKING leaked documents obtained by independent researchers have revealed that the "
                "deep state has been secretly adding mind-control chemicals to the water supply for "
                "over two decades. Multiple whistleblowers have come forward but mainstream media "
                "refuses to cover this story. Share this before it gets deleted forever."
            )
            st.session_state["sample_title"] = "Government secretly puts chemicals in water to control the population"

# ─── Auto-fill samples ───────────────────────────────────────────────────────
if "sample_text" in st.session_state and not text_input:
    text_input  = st.session_state.pop("sample_text", "")
    title_input = st.session_state.pop("sample_title", "")
    st.rerun()

# ─── Prediction ──────────────────────────────────────────────────────────────
if analyze_btn:
    combined = (title_input + " " + text_input).strip()
    if not combined:
        st.error("Please enter some news text to analyze.")
    else:
        with st.spinner("Analyzing …"):
            result = predict_news(combined, model, tokenizer)

        st.divider()
        label = result["label"]
        conf  = result["confidence"]

        if label == "FAKE":
            st.markdown(
                f'<div class="result-fake">'
                f'<div class="result-label">🚨 FAKE NEWS</div>'
                f'<div class="confidence">Confidence: <strong>{conf}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-real">'
                f'<div class="result-label">✅ REAL NEWS</div>'
                f'<div class="confidence">Confidence: <strong>{conf}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔬 Technical Details"):
            st.write(f"**Model used:** {model_name}")
            st.write(f"**Raw probability (FAKE):** {result['raw_probability']}")
            st.write(f"**Cleaned text preview:**")
            st.code(clean_text(combined)[:300] + " …", language=None)

# ─── Training plot ───────────────────────────────────────────────────────────
model_key = model_path.split("/")[-1].replace("_model.keras", "")
plot_path  = f"model/{model_key}_training_plot.png"
if os.path.exists(plot_path):
    with st.expander("📊 Training History"):
        st.image(plot_path, caption=f"{model_key.upper()} Training Accuracy & Loss")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with TensorFlow · Keras · Streamlit | Fake News Detection Project</div>',
    unsafe_allow_html=True,
)
