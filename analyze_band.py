"""Check alpha and abstention bands on the NB3c validation writers."""
import csv
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


def l2(t):
    return tf.math.l2_normalize(t, axis=1)


tower = load_model("models/siamese_bh_embedding.keras", custom_objects={"l2": l2}, safe_mode=False)


rows = list(csv.DictReader(open(MANIFEST)))
genuine, forg = defaultdict(list), defaultdict(list)
for r in rows:
    (genuine if r["kind"] == "genuine" else forg)[r["writer"]].append(r["relpath"])

writers = sorted(set(genuine) & set(forg))
icdar = [w for w in writers if w.startswith("icdar_")]
bhh = [w for w in writers if w.startswith("bhh_")]


def writer_number(writer):
    return int(writer.split("_")[1])


val_w = [w for w in icdar if 41 <= writer_number(w) <= 48]
val_w += [w for w in bhh if 111 <= writer_number(w) <= 130]
print(f"val writers: {len(val_w)} (icdar {sum(w.startswith('icdar_') for w in val_w)} + "
      f"bhh {sum(w.startswith('bhh_') for w in val_w)})")


def to3(relpath):
    im = cv2.imread(os.path.join(DATA_ROOT, relpath), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (IMG_W, IMG_H))
    inv = 255.0 - im.astype("float32")
    return np.repeat(inv[..., None], 3, axis=2)


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


# Alpha sweep
print("\n" + "=" * 64)
print("ALPHA SWEEP on validation (FAR = forgeries accepted, the costly error)")
print("=" * 64)
sweep = {}
for a in [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    recs = recs_at_alpha(CACHE, a)
    s, tau, lab = recs[:, 0], recs[:, 1], recs[:, 2].astype(int)
    accept = s < tau
    far = 100 * accept[lab == 1].mean()   # forgery accepted
    frr = 100 * (~accept[lab == 0]).mean()  # genuine rejected
    sweep[a] = (far, frr)
    print(f"  alpha={a:4.2f}  FAR {far:5.2f}%  FRR {frr:5.2f}%  |gap| {abs(far-frr):5.2f}")

print("\n  EER-style pick (min |FAR-FRR|):", min(sweep, key=lambda a: abs(sweep[a][0]-sweep[a][1])))
le2 = [a for a in sweep if sweep[a][0] <= 2.0]
print("  strictest alpha with FAR<=2%:", min(le2) if le2 else "none in sweep")


# Abstention-band sweep
# margin m = (score - tau)/tau.  m<0 => accepted (genuine verdict), m>0 => rejected.
# Abstain when  -F_ACC <= m <= F_REJ.
#   accept side (F_ACC): must be WIDE enough to catch forgeries that dip just below tau.
#   reject side (F_REJ): keep NARROW -- the m>0 region is mostly clear forgeries we should
#                        decisively reject, not soften to INCONCLUSIVE.
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

print("\n  forg_SLIP% = forgeries still false-accepted")
print("  forg_decisiveREJECT% = forgeries confidently rejected")
