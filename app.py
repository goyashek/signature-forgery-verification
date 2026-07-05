"""
Signature Forgery Verification — HuggingFace Spaces (Gradio) demo.

Loads the shipped batch-hard EfficientNet-B0 embedding tower (NB3c) and verifies a
questioned signature against one or more known-genuine references.

Two operating points, chosen automatically by how many references you provide:

  * 2+ references  -> per-writer ADAPTIVE threshold (the project's best operating point,
                      FAR ~3.5%). Enrols the writer from the references, measures their
                      natural genuine spread, and sets tau = mean + alpha*std of the
                      reference-to-reference distances (alpha tuned on validation = 1.5).
  * 1 reference    -> falls back to the GLOBAL EER threshold (FAR ~6.5%), the only option
                      when a writer's spread can't be estimated.

The decision is distance-based: a questioned signature is GENUINE when its mean distance
to the references falls below the threshold. Educational demo — never the sole check in a
real authentication system.
"""
import json
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "siamese_bh_embedding.keras")
META_PATH = os.path.join(HERE, "models", "siamese_bh_meta.json")


# --- model load (the §5a gotcha: the tower ends in a Lambda named "l2") -----------------
def l2(t):
    return tf.math.l2_normalize(t, axis=1)


def _load():
    tower = load_model(MODEL_PATH, custom_objects={"l2": l2}, safe_mode=False)
    with open(META_PATH) as f:
        meta = json.load(f)
    return tower, meta


TOWER, META = _load()
H, W = META["img_h"], META["img_w"]
GLOBAL_TAU = META["global_threshold"]   # distance below this = genuine (chosen at EER)
ALPHA = META["per_writer_alpha"]        # per-writer knob, tuned on validation


# --- preprocessing: replicate training exactly (invert -> 3-channel, bilinear resize) ---
def preprocess(pil_img):
    """Grayscale -> resize (W,H) bilinear -> invert (ink=signal, page=0) -> 3 channels."""
    g = ImageOps.grayscale(pil_img).resize((W, H), Image.BILINEAR)
    arr = 255.0 - np.asarray(g, dtype="float32")        # invert
    return np.repeat(arr[..., None], 3, axis=2)         # 1ch -> 3ch for EfficientNet


def embed_batch(pil_imgs):
    x = np.stack([preprocess(im) for im in pil_imgs])
    return TOWER.predict(x, verbose=0)


# --- verification -----------------------------------------------------------------------
def verify(reference_files, questioned_img):
    if not reference_files:
        return "### ⚠️ Add at least one reference signature.", None
    if questioned_img is None:
        return "### ⚠️ Add the questioned signature to verify.", None

    refs = [Image.open(f.name if hasattr(f, "name") else f) for f in reference_files]
    ref_emb = embed_batch(refs)                          # (n_ref, d)
    q_emb = embed_batch([questioned_img])[0]             # (d,)

    # score = mean distance from the query to every reference (NB3c protocol)
    score = float(np.sqrt(np.sum((q_emb - ref_emb) ** 2, axis=1) + 1e-12).mean())

    n_ref = len(refs)
    if n_ref >= 2:
        # per-writer adaptive: tau from the writer's own genuine spread
        rd = np.sqrt(np.maximum(np.sum((ref_emb[:, None] - ref_emb[None, :]) ** 2, -1), 0))
        iu = np.triu_indices(n_ref, 1)
        intra = rd[iu]
        tau = float(intra.mean() + ALPHA * intra.std())
        mode = f"per-writer adaptive (α={ALPHA}, {n_ref} refs)"
    else:
        tau = GLOBAL_TAU
        mode = "global EER threshold (1 ref)"

    genuine = score < tau
    margin = abs(score - tau)
    verdict = "✅ **GENUINE**" if genuine else "🚫 **FORGED**"
    color = "#137333" if genuine else "#c5221f"

    md = (
        f"## <span style='color:{color}'>{verdict}</span>\n\n"
        f"| | |\n|---|---|\n"
        f"| **Distance** | `{score:.4f}` |\n"
        f"| **Threshold (τ)** | `{tau:.4f}` |\n"
        f"| **Margin** | `{margin:.4f}` {'below' if genuine else 'above'} τ |\n"
        f"| **Operating point** | {mode} |\n\n"
        + ("> Distance is below the threshold — the questioned signature matches the "
           "references.\n" if genuine else
           "> Distance exceeds the threshold — the questioned signature does **not** match "
           "the references.\n")
        + "\n*Educational demo. A real system must never decide on a single model score.*"
    )
    # gallery of what was compared
    gallery = [(r, f"reference {i+1}") for i, r in enumerate(refs)]
    gallery.append((questioned_img, "questioned"))
    return md, gallery


# --- UI ---------------------------------------------------------------------------------
DESCRIPTION = """
# ✍️ Signature Forgery Verification

Deep **metric-learning** verifier (Siamese EfficientNet-B0 + batch-hard mining).
Upload **one or more genuine reference** signatures and a **questioned** one — the model
embeds each, measures the distance, and returns a verdict.

**Tip:** give **3–5 genuine references** to unlock the *per-writer adaptive* threshold
(the project's best operating point, FAR ≈ 3.5%). A single reference uses the global
threshold instead.
"""

with gr.Blocks(title="Signature Forgery Verification", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            refs_in = gr.File(
                label="Reference signatures (genuine — 1 to 5)",
                file_count="multiple",
                file_types=["image"],
            )
            q_in = gr.Image(label="Questioned signature", type="pil", sources=["upload"])
            btn = gr.Button("Verify signature", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_gallery = gr.Gallery(label="Compared signatures", columns=3, height="auto")

    btn.click(verify, inputs=[refs_in, q_in], outputs=[out_md, out_gallery])
    gr.Markdown(
        f"<sub>Model: `{META['model']}` · input {H}×{W} · "
        f"global τ={GLOBAL_TAU:.4f} · α={ALPHA} · test ROC-AUC 0.986 (writer-independent, "
        "leak-free). Not a production authentication system.</sub>"
    )

if __name__ == "__main__":
    demo.launch()
