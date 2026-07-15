"""Shared inference path for the app and final demo notebook."""

import hashlib
import json
import os

import numpy as np
import tensorflow as tf
from PIL import ImageOps
from tensorflow.keras.models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "siamese_bh_embedding.keras")
META_PATH = os.path.join(HERE, "models", "siamese_bh_meta.json")


def l2(t):
    return tf.math.l2_normalize(t, axis=1)


TOWER = load_model(MODEL_PATH, custom_objects={"l2": l2}, safe_mode=False)
with open(META_PATH) as f:
    META = json.load(f)

H, W = META["img_h"], META["img_w"]
MIN_REFS = 3
RECOMMENDED_REFS = META["n_ref"]
MAX_REFS = RECOMMENDED_REFS
ALPHA = META["per_writer_alpha"]
BAND_ACCEPT = META["abstain_band_accept"]
BAND_REJECT = META["abstain_band_reject"]


def preprocess(pil_img):
    """Match OpenCV's training resize closely without adding OpenCV to the app."""
    gray = np.asarray(ImageOps.grayscale(pil_img), dtype="float32")
    resized = tf.image.resize(gray[..., None], (H, W), method="bilinear", antialias=False)
    inverted = 255.0 - np.rint(resized.numpy()[..., 0])
    return np.repeat(inverted[..., None], 3, axis=2).astype("float32")


def dedupe_images(images):
    seen, unique = set(), []
    for image in images:
        identity = f"{image.mode}:{image.size}".encode() + image.tobytes()
        digest = hashlib.sha256(identity).digest()
        if digest not in seen:
            seen.add(digest)
            unique.append(image)
    return unique


def decide_embeddings(reference_embeddings, questioned_embedding):
    """Apply the calibrated decision rule to 3-5 reference embeddings."""
    ref_emb = np.asarray(reference_embeddings)
    query_emb = np.asarray(questioned_embedding)
    n_refs = len(ref_emb)
    if not MIN_REFS <= n_refs <= MAX_REFS:
        raise ValueError(f"Expected {MIN_REFS} to {MAX_REFS} reference embeddings.")

    score = float(np.sqrt(np.sum((query_emb - ref_emb) ** 2, axis=1) + 1e-12).mean())
    ref_distances = np.sqrt(
        np.maximum(np.sum((ref_emb[:, None] - ref_emb[None, :]) ** 2, axis=-1), 0)
    )
    intra = ref_distances[np.triu_indices(n_refs, 1)]
    threshold = float(intra.mean() + ALPHA * intra.std())
    if threshold <= 0:
        raise ValueError("The reference set is too uniform to calculate a reliable threshold.")
    margin = (score - threshold) / threshold

    if -BAND_ACCEPT <= margin <= BAND_REJECT:
        verdict = "INCONCLUSIVE"
    elif score < threshold:
        verdict = "GENUINE"
    else:
        verdict = "FORGED"

    return {
        "verdict": verdict,
        "distance": score,
        "threshold": threshold,
        "margin": margin,
    }


def verify_images(reference_images, questioned_image):
    refs = dedupe_images(reference_images)
    if not MIN_REFS <= len(refs) <= MAX_REFS:
        raise ValueError(f"Add {MIN_REFS} to {MAX_REFS} distinct genuine references.")

    ref_emb = TOWER.predict(np.stack([preprocess(im) for im in refs]), verbose=0)
    query_emb = TOWER.predict(preprocess(questioned_image)[None], verbose=0)[0]
    result = decide_embeddings(ref_emb, query_emb)
    result["references"] = refs
    return result
