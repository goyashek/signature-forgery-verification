"""
Derive the abstain-band width and re-tune alpha for the shipped 3c model, using the
SAME writer-independent validation split, enrollment protocol, and preprocessing as
notebooks/03c_siamese_batchhard.ipynb (cells 5, 7, 9, 23).

Nothing here retrains. It only runs forward passes over the validation writers, so it
is safe on the 8 GB M1. Two outputs:

  1. BAND: for the per-writer adaptive protocol, collect every query's normalized margin
     m = (score - tau) / tau, split by genuine vs forgery, and find the band fraction f
     where the two distributions overlap. |m| <= f is the "too close to call" region.
  2. ALPHA: sweep alpha finely and report FAR/FRR on validation, so we can pick the knob
     at a target FAR (<=2%) instead of at EER (|FAR-FRR| min, which the notebook used).
"""
import os
import random
from collections import defaultdict

import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model

DATA_ROOT = "sign_data_combined"
MANIFEST = os.path.join(DATA_ROOT, "manifest.csv")
IMG_H, IMG_W = 155, 220
N_REF = 5


# --- model (same custom Lambda as app.py / notebook) ------------------------------------
def l2(t):
    return tf.math.l2_normalize(t, axis=1)


tower = load_model("models/siamese_bh_embedding.keras", custom_objects={"l2": l2}, safe_mode=False)


# --- manifest + writer-independent split (cell 5) ---------------------------------------
import csv

rows = list(csv.DictReader(open(MANIFEST)))
genuine, forg, script_of = defaultdict(list), defaultdict(list), {}
for r in rows:
    (genuine if r["kind"] == "genuine" else forg)[r["writer"]].append(r["relpath"])
    script_of[r["writer"]] = r["script"]

writers = sorted(set(genuine) & set(forg))
icdar = [w for w in writers if w.startswith("icdar_")]
bhh = [w for w in writers if w.startswith("bhh_")]
num = lambda w: int(w.split("_")[1])

val_w = [w for w in icdar if 41 <= num(w) <= 48] + [w for w in bhh if 111 <= num(w) <= 130]
print(f"val writers: {len(val_w)} (icdar {sum(w.startswith('icdar_') for w in val_w)} + "
      f"bhh {sum(w.startswith('bhh_') for w in val_w)})")


# --- preprocessing identical to cell 9 --------------------------------------------------
def to3(relpath):
    im = cv2.imread(os.path.join(DATA_ROOT, relpath), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (IMG_W, IMG_H))
    inv = 255.0 - im.astype("float32")
    return np.repeat(inv[..., None], 3, axis=2)


# --- embed each writer ONCE; cache alpha-independent quantities -------------------------
# A query's score (mean distance to its 5 refs) does NOT depend on alpha, and neither do
# the intra-ref mean/std. Only tau = intra_mean + alpha*intra_std does. So embed every
# writer a single time, then sweep alpha in pure numpy (avoids 9x redundant model passes).
def build_cache(wset, seed=11):
    rng = random.Random(seed)
    per_writer = []
    for w in sorted(wset):
        g = genuine[w]
        if len(g) < N_REF + 1:
            continue
        refs = rng.sample(g, N_REF)
        queries = [(p, 0) for p in g if p not in refs] + [(p, 1) for p in forg.get(w, [])]
        if not queries:
            continue
        all_paths = refs + [p for p, _ in queries]
        emb = tower.predict(np.stack([to3(p) for p in all_paths]), verbose=0)
        ref_emb, q_emb = emb[:N_REF], emb[N_REF:]
        rd = np.sqrt(np.maximum(np.sum((ref_emb[:, None] - ref_emb[None, :]) ** 2, -1), 0))
        intra = rd[np.triu_indices(N_REF, 1)]
        scores = [(float(np.sqrt(np.sum((e - ref_emb) ** 2, axis=1) + 1e-12).mean()), lab)
                  for e, (_, lab) in zip(q_emb, queries)]
        per_writer.append({"im": float(intra.mean()), "isd": float(intra.std()), "scores": scores})
        print(f"  embedded {w} ({len(queries)} queries)", flush=True)
    return per_writer


def recs_at_alpha(cache, alpha):
    out = []
    for wr in cache:
        tau = wr["im"] + alpha * wr["isd"]
        for s, lab in wr["scores"]:
            out.append((s, tau, lab))
    return np.array(out)


print("embedding validation writers once...", flush=True)
CACHE = build_cache(val_w)


def writer_eval_collect(wset, alpha, seed=11):
    return recs_at_alpha(CACHE, alpha)


# ---------------------------------------------------------------------------------------
# PART 1 — alpha sweep at the fixed alpha we use to define the band later
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 64)
print("ALPHA SWEEP on validation (FAR = forgeries accepted, the costly error)")
print("=" * 64)
sweep = {}
for a in [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    recs = writer_eval_collect(val_w, alpha=a)
    s, tau, lab = recs[:, 0], recs[:, 1], recs[:, 2].astype(int)
    accept = s < tau
    far = 100 * accept[lab == 1].mean()   # forgery accepted
    frr = 100 * (~accept[lab == 0]).mean()  # genuine rejected
    sweep[a] = (far, frr)
    print(f"  alpha={a:4.2f}  FAR {far:5.2f}%  FRR {frr:5.2f}%  |gap| {abs(far-frr):5.2f}")

print("\n  EER-style pick (min |FAR-FRR|):", min(sweep, key=lambda a: abs(sweep[a][0]-sweep[a][1])))
le2 = [a for a in sweep if sweep[a][0] <= 2.0]
print("  strictest alpha with FAR<=2%:", min(le2) if le2 else "none in sweep")


# ---------------------------------------------------------------------------------------
# PART 2 — ASYMMETRIC band at the candidate (lower) alphas.
# margin m = (score - tau)/tau.  m<0 => accepted (genuine verdict), m>0 => rejected.
# Abstain when  -F_ACC <= m <= F_REJ.
#   accept side (F_ACC): must be WIDE enough to catch forgeries that dip just below tau.
#   reject side (F_REJ): keep NARROW -- the m>0 region is mostly clear forgeries we should
#                        decisively reject, not soften to INCONCLUSIVE.
# ---------------------------------------------------------------------------------------
def margins(alpha):
    recs = recs_at_alpha(CACHE, alpha)
    s, tau, lab = recs[:, 0], recs[:, 1], recs[:, 2].astype(int)
    return (s - tau) / tau, lab


for A in [1.0, 1.25]:
    m, lab = margins(A)
    mg, mf = m[lab == 0], m[lab == 1]
    print("\n" + "=" * 64)
    print(f"ASYMMETRIC BAND at alpha={A}")
    print("=" * 64)
    print(f"  genuine n={len(mg)}: mean {mg.mean():+.3f} sd {mg.std():.3f} "
          f"[p5 {np.percentile(mg,5):+.3f}, p95 {np.percentile(mg,95):+.3f}]")
    print(f"  forgery n={len(mf)}: mean {mf.mean():+.3f} sd {mf.std():.3f} "
          f"[p5 {np.percentile(mf,5):+.3f}, p95 {np.percentile(mf,95):+.3f}]")
    print("\n  F_acc  F_rej | gen_abstain% | forg_caught% | forg_SLIP% | forg_decisiveREJECT%")
    print("  " + "-" * 74)
    for f_acc in [0.08, 0.10, 0.12, 0.15]:
        for f_rej in [0.02, 0.05]:
            in_band = (m >= -f_acc) & (m <= f_rej)
            gen_ab = 100 * in_band[lab == 0].mean()
            forg_caught = 100 * in_band[lab == 1].mean()
            slip = 100 * ((lab == 1) & (m < -f_acc)).sum() / max((lab == 1).sum(), 1)
            dec_rej = 100 * ((lab == 1) & (m > f_rej)).sum() / max((lab == 1).sum(), 1)
            print(f"   {f_acc:4.2f}  {f_rej:4.2f} |   {gen_ab:6.2f}     |   {forg_caught:6.2f}    "
                  f"|  {slip:6.2f}   |    {dec_rej:6.2f}")

print("\n  forg_SLIP% = forgeries STILL false-accepted (want ~0)")
print("  forg_decisiveREJECT% = forgeries confidently FORGED, not softened (want high)")


# ---------------------------------------------------------------------------------------
# PART 3 — overlay the REAL user-test cases at each candidate alpha
# ---------------------------------------------------------------------------------------
import glob
from PIL import Image, ImageOps


def to3_ext(path):  # external jpgs: same preprocessing as to3 but via PIL (matches app.py)
    g = ImageOps.grayscale(Image.open(path)).resize((IMG_W, IMG_H), Image.BILINEAR)
    inv = 255.0 - np.asarray(g, dtype="float32")
    return np.repeat(inv[..., None], 3, axis=2)


def real_case(folder, alphas=(1.0, 1.25, 1.5)):
    refs = sorted(glob.glob(folder + "/gen*.jpg"))
    forg = sorted(glob.glob(folder + "/forged*.jpg"))
    if not refs or not forg:
        return
    ref_emb = tower.predict(np.stack([to3_ext(p) for p in refs]), verbose=0)
    rd = np.sqrt(np.maximum(np.sum((ref_emb[:, None] - ref_emb[None, :]) ** 2, -1), 0))
    intra = rd[np.triu_indices(len(refs), 1)]
    im, isd = intra.mean(), intra.std()
    print(f"\n  {os.path.basename(folder)}: {len(refs)} refs, intra mean {im:.4f} sd {isd:.4f}")
    for p in forg:
        e = tower.predict(to3_ext(p)[None], verbose=0)[0]
        d = float(np.sqrt(np.sum((e - ref_emb) ** 2, axis=1) + 1e-12).mean())
        cells = []
        for a in alphas:
            tau = im + a * isd
            mm = (d - tau) / tau
            cells.append(f"a={a}: m={mm:+.3f}")
        print(f"    {os.path.basename(p):14s} dist={d:.4f}  " + "  ".join(cells))


# Optional: overlay real user-test folders (gen*.jpg refs + forged*.jpg queries). These are
# local hand-written phone photos, not part of the repo — pass their paths as CLI args to
# include them, e.g.  python3 analyze_band.py /path/to/test /path/to/test2
import sys
extra = sys.argv[1:]
if extra:
    print("\n" + "=" * 64)
    print("REAL USER-TEST FORGERIES — normalized margin at each alpha (want m > accept-band)")
    print("=" * 64)
    for folder in extra:
        real_case(folder)
