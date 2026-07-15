"""Hugging Face Gradio demo for the calibrated signature verifier."""

import os

import gradio as gr
from PIL import Image

from inference import (
    ALPHA, BAND_ACCEPT, BAND_REJECT, H, MAX_REFS, META, MIN_REFS,
    RECOMMENDED_REFS, W, verify_images,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_REFS = [
    os.path.join(HERE, "sign_data_nfi", "genuine", f"NFI-0010{i}001.png")
    for i in range(1, 6)
]
SAMPLE_QUERY = os.path.join(HERE, "sign_data_nfi", "forged", "NFI-00304001.png")


def verify(reference_files, questioned_img):
    if not reference_files:
        return f"### ⚠️ Add {MIN_REFS} to {MAX_REFS} distinct genuine references.", None
    if questioned_img is None:
        return "### ⚠️ Add the questioned signature to verify.", None

    refs = [Image.open(f.name if hasattr(f, "name") else f) for f in reference_files]
    try:
        result = verify_images(refs, questioned_img)
    except ValueError as error:
        for reference in refs:
            reference.close()
        return f"### ⚠️ {error}", None

    refs = result["references"]
    score = result["distance"]
    tau = result["threshold"]
    m = result["margin"]
    verdict_name = result["verdict"]
    reliability_note = (
        "\n\n> ⚠️ Three references give a less reliable threshold. Five are recommended."
        if len(refs) == MIN_REFS else ""
    )

    if verdict_name == "INCONCLUSIVE":
        verdict, color = "🟡 **INCONCLUSIVE**", "#b06000"
        note = "> The score is too close to the threshold. Try a clearer image or review it manually.\n"
    elif verdict_name == "GENUINE":
        verdict, color = "✅ **GENUINE**", "#137333"
        note = "> The questioned signature is below the writer-specific threshold.\n"
    else:
        verdict, color = "🚫 **FORGED**", "#c5221f"
        note = "> The questioned signature is above the writer-specific threshold.\n"

    side = "below" if score < tau else "above"

    md = (
        f"## <span style='color:{color}'>{verdict}</span>\n\n"
        f"| | |\n|---|---|\n"
        f"| **Distance** | `{score:.4f}` |\n"
        f"| **Threshold (τ)** | `{tau:.4f}` |\n"
        f"| **Margin** | `{abs(score - tau):.4f}` {side} τ (m=`{m:+.3f}`, abstain "
        f"`[-{BAND_ACCEPT:.2f}, +{BAND_REJECT:.2f}]`) |\n"
        f"| **References** | {len(refs)} distinct |\n"
        f"| **Operating point** | per-writer adaptive (α={ALPHA}) |\n\n"
        + note
        + reliability_note
        + "\n*Educational demo. A real system must never decide on a single model score.*"
    )
    gallery = [(r, f"reference {i+1}") for i, r in enumerate(refs)]
    gallery.append((questioned_img, "questioned"))
    return md, gallery


DESCRIPTION = """
# ✍️ Signature Forgery Verification

This demo uses the EfficientNet-B0 embedding model from notebook 3c. Upload **3-5 distinct genuine
references** and one **questioned signature**.

Five references are recommended. Three are accepted, but the writer threshold is less reliable.
"""

with gr.Blocks(title="Signature Forgery Verification", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            refs_in = gr.File(
                label=f"Reference signatures ({MIN_REFS}-{MAX_REFS} distinct; {RECOMMENDED_REFS} recommended)",
                file_count="multiple",
                file_types=["image"],
            )
            q_in = gr.Image(label="Questioned signature", type="pil", sources=["upload"])
            sample_btn = gr.Button("Load sample signatures")
            btn = gr.Button("Verify signature", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_gallery = gr.Gallery(label="Compared signatures", columns=3, height="auto")

    sample_btn.click(lambda: (SAMPLE_REFS, SAMPLE_QUERY), outputs=[refs_in, q_in])
    btn.click(verify, inputs=[refs_in, q_in], outputs=[out_md, out_gallery])
    gr.Markdown(
        f"<sub>Model: `{META['model']}` · input {H}×{W} · "
        f"{MIN_REFS}-{MAX_REFS} references ({RECOMMENDED_REFS} recommended) · α={ALPHA} · model ROC-AUC 0.986 (writer-independent, "
        "leak-free). Not a production authentication system.</sub>"
    )

if __name__ == "__main__":
    demo.launch()
