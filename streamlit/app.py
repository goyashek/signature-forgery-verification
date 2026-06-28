"""
Signature Forgery Verification — Streamlit demo.

Upload a reference (known-genuine) signature and a questioned signature.
The app embeds both with the trained Siamese tower, measures the distance
between the embeddings, and returns a GENUINE / FORGED verdict using the
threshold chosen at the Equal Error Rate during training.

Run:  streamlit run streamlit/app.py
"""
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# --- locate models dir (works whether run from repo root or streamlit/) ---
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "models")

# Prefer the transfer-learning model (best); fall back to the from-scratch CNN.
MODEL_CANDIDATES = [
    ("siamese_transfer_embedding.keras", "siamese_transfer_meta.json"),
    ("siamese_embedding.keras", "siamese_cnn_meta.json"),
]

st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model_and_meta():
    """Load the first available embedding tower + its metadata."""
    for model_file, meta_file in MODEL_CANDIDATES:
        mp = os.path.join(MODEL_DIR, model_file)
        jp = os.path.join(MODEL_DIR, meta_file)
        if os.path.exists(mp) and os.path.exists(jp):
            model = tf.keras.models.load_model(mp, compile=False)
            with open(jp) as f:
                meta = json.load(f)
            return model, meta, model_file
    return None, None, None


def preprocess(pil_img, meta):
    """Replicate the exact training preprocessing for the loaded model."""
    h, w = meta["img_h"], meta["img_w"]
    img = ImageOps.grayscale(pil_img).resize((w, h))   # PIL size is (w, h)
    arr = np.asarray(img, dtype="float32")

    if meta.get("preprocess") == "mobilenet_v2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        rgb = np.stack([arr, arr, arr], axis=-1)        # grayscale -> 3 channels
        return preprocess_input(rgb)[None, ...]         # [-1, 1]
    # from-scratch CNN tower: single channel scaled to [0, 1]
    return (arr / 255.0)[None, ..., None]


def embed(model, x):
    return model.predict(x, verbose=0)


# --------------------------- UI ---------------------------
st.title("✍️ Signature Forgery Verification")
st.caption(
    "Siamese metric-learning demo. Upload two signatures — the model compares "
    "their learned embeddings and decides whether they came from the same hand."
)

model, meta, model_file = load_model_and_meta()

if model is None:
    st.error(
        "No trained model found in `models/`. Run notebook 02 or 03 first to "
        "produce a `*_embedding.keras` file and its metadata JSON."
    )
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"**File:** `{model_file}`")
    st.write(f"**Type:** {meta.get('model', 'siamese')}")
    st.write(f"**Threshold (τ):** {meta['threshold']:.4f}")
    if "test_auc" in meta:
        st.write(f"**Test ROC-AUC:** {meta['test_auc']:.3f}")
    if "test_eer" in meta:
        st.write(f"**Test EER:** {meta['test_eer'] * 100:.2f}%")
    st.caption(
        "τ was selected at the Equal Error Rate on a writer-independent "
        "validation set. A pair is GENUINE when the embedding distance < τ."
    )

col1, col2 = st.columns(2)
with col1:
    f1 = st.file_uploader("Reference signature", type=["png", "jpg", "jpeg"], key="ref")
with col2:
    f2 = st.file_uploader("Questioned signature", type=["png", "jpg", "jpeg"], key="qst")

if f1 and f2:
    img1 = Image.open(f1).convert("RGB")
    img2 = Image.open(f2).convert("RGB")

    c1, c2 = st.columns(2)
    c1.image(img1, caption="Reference", use_container_width=True)
    c2.image(img2, caption="Questioned", use_container_width=True)

    if st.button("Verify signature", type="primary", use_container_width=True):
        e1 = embed(model, preprocess(img1, meta))
        e2 = embed(model, preprocess(img2, meta))
        distance = float(np.sqrt(np.sum((e1 - e2) ** 2)))
        tau = meta["threshold"]
        is_genuine = distance < tau

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Embedding distance", f"{distance:.4f}")
        m2.metric("Threshold (τ)", f"{tau:.4f}")

        # confidence = how far the distance sits from the threshold, normalised
        margin = meta.get("margin", 1.0)
        confidence = min(abs(distance - tau) / max(margin, 1e-6), 1.0)

        if is_genuine:
            st.success(f"✅ **GENUINE** — distance below threshold "
                       f"(confidence {confidence * 100:.0f}%)")
        else:
            st.error(f"🚫 **FORGED** — distance above threshold "
                     f"(confidence {confidence * 100:.0f}%)")

        st.progress(min(distance / (2 * tau), 1.0),
                    text="distance relative to threshold")
        st.caption(
            "This is an educational demo, not a production authentication system. "
            "Decisions should never be made on a single model score alone."
        )
else:
    st.info("Upload both a reference and a questioned signature to run verification.")
