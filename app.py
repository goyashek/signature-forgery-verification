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
import hashlib
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

# --- abstain band -----------------------------------------------------------------------
# Normalized margin m = (distance - tau) / tau:  m < 0 => accept side (genuine verdict),
# m > 0 => reject side (forgery verdict). When m lands in [-BAND_ACCEPT, +BAND_REJECT] the
# case is too close to call, so we return INCONCLUSIVE instead of guessing.
#
# The band is ASYMMETRIC and calibrated on the NB3c writer-independent validation split
# (see analyze_band.py) at alpha=1.25:
#   * BAND_ACCEPT (0.12) is wide -- genuine signatures cluster below tau, so skilled
#     forgeries that dip *just* below tau hide right under it. This is the dangerous zone
#     (false accepts). On the real user-tested forgeries this catches both (m=-0.03, -0.10).
#   * BAND_REJECT (0.02) is narrow -- the m>0 region is almost entirely clear forgeries we
#     should reject decisively, not soften. ~93% of validation forgeries stay confidently
#     FORGED; only ~2% of forgeries slip through as false accepts.
# Cost: ~22% of genuine queries land in the band and are asked for more references. Values
# live in meta so they can be recalibrated without a code change.
BAND_ACCEPT = META.get("abstain_band_accept", 0.12)
BAND_REJECT = META.get("abstain_band_reject", 0.02)


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
def _dedupe(pil_imgs):
    """Drop byte-identical duplicate references.

    The per-writer threshold is tau = mean + alpha*std of the reference-to-reference
    distances -- it measures the writer's *natural* genuine spread. A duplicated reference
    injects a distance of exactly 0 that never existed, which drags the mean down and
    distorts std, mis-calibrating tau. (Seen in real user testing: the same signature file
    uploaded twice.) Dedupe by content hash so only genuinely distinct references count.
    """
    seen, unique = set(), []
    for im in pil_imgs:
        h = hashlib.md5(im.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(im)
    return unique


def verify(reference_files, questioned_img):
    if not reference_files:
        return "### ⚠️ Add at least one reference signature.", None
    if questioned_img is None:
        return "### ⚠️ Add the questioned signature to verify.", None

    refs = [Image.open(f.name if hasattr(f, "name") else f) for f in reference_files]
    n_uploaded = len(refs)
    refs = _dedupe(refs)
    n_dropped = n_uploaded - len(refs)

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

    # Normalized margin: m<0 accept side, m>0 reject side. Abstain inside the asymmetric band.
    m = (score - tau) / tau

    # Three states: inside the band -> abstain rather than guess.
    if -BAND_ACCEPT <= m <= BAND_REJECT:
        verdict, color = "🟡 **INCONCLUSIVE**", "#b06000"
        note = (
            "> Distance is within the borderline band of the threshold — too close to call "
            "confidently. In verification a false accept is the costly error, so the demo "
            "abstains here rather than risk passing a skilled forgery. **Add more genuine "
            "references** (to sharpen the per-writer threshold) or route to manual review.\n"
        )
    elif score < tau:
        verdict, color = "✅ **GENUINE**", "#137333"
        note = ("> Distance is comfortably below the threshold — the questioned signature "
                "matches the references.\n")
    else:
        verdict, color = "🚫 **FORGED**", "#c5221f"
        note = ("> Distance is comfortably above the threshold — the questioned signature "
                "does **not** match the references.\n")

    side = "below" if score < tau else "above"
    dedupe_row = (
        f"| **References** | {len(refs)} distinct "
        f"({n_dropped} duplicate{'s' if n_dropped != 1 else ''} dropped) |\n"
        if n_dropped else f"| **References** | {len(refs)} distinct |\n"
    )

    md = (
        f"## <span style='color:{color}'>{verdict}</span>\n\n"
        f"| | |\n|---|---|\n"
        f"| **Distance** | `{score:.4f}` |\n"
        f"| **Threshold (τ)** | `{tau:.4f}` |\n"
        f"| **Margin** | `{abs(score - tau):.4f}` {side} τ (m=`{m:+.3f}`, abstain "
        f"`[-{BAND_ACCEPT:.2f}, +{BAND_REJECT:.2f}]`) |\n"
        f"{dedupe_row}"
        f"| **Operating point** | {mode} |\n\n"
        + note
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
