#!/usr/bin/env python3
"""
Build a single, self-contained, leak-free-ready signature dataset by combining:
  1. ICDAR  (sign_data/train/ only — test/ is the known byte-identical leak)   -> Latin
  2. BHSig260-Hindi (archive/BHSig260-Hindi/...)                                -> Devanagari

Design notes (see CLAUDE.md §4/§4a for the leakage story this protects):
  * The writer-independent split lives in the NOTEBOOK, not here. This script only
    produces a clean, collision-proof per-writer folder layout + a manifest. The
    notebook builds leak-free pairs (match / forgery-neg / different-writer-neg) from
    the raw folders and filters by writer for the split.
  * Both source datasets number writers from 1, so writer ids WOULD collide. We
    namespace every writer:  icdar_NNN  /  bhh_NNN .
  * ICDAR keeps its ORIGINAL numeric id (icdar_049) so the existing split ranges
    (train <=40, val 41-48, test >=49) carry over unchanged for the ICDAR portion.
  * Everything is re-encoded to grayscale PNG (signatures are ink-on-paper; the color
    channels are redundant and every notebook already reads grayscale). This roughly
    halves ICDAR and shrinks the heavy BHSig .tif files ~100x.
  * sign_data/ is left UNTOUCHED so notebooks 01, 01b, 02 keep running as-is.

Output:
  sign_data_combined/
    manifest.csv                      relpath,writer,source,script,kind
    icdar_001/ icdar_001_forg/ ...    (Latin)
    bhh_001/   bhh_001_forg/   ...     (Devanagari)
"""
import os, re, glob, csv, sys
import cv2

# libtiff prints a harmless "Software tag null byte" warning on the BHSig .tif files;
# it goes to stderr and does not affect decoding.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

ROOT = os.path.dirname(os.path.abspath(__file__))
ICDAR_TRAIN = os.path.join(ROOT, "sign_data", "train")
BHH_ROOT    = os.path.join(ROOT, "archive", "BHSig260-Hindi", "BHSig260-Hindi")
OUT         = os.path.join(ROOT, "sign_data_combined")

PNG_OPTS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


def save_gray_png(src_path, dst_path):
    """Read any image, write it as grayscale PNG. Returns False if unreadable."""
    im = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return False
    cv2.imwrite(dst_path, im, PNG_OPTS)
    return True


def ensure(d):
    os.makedirs(d, exist_ok=True)


def is_img(fn):
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))


def build_icdar(rows):
    """ICDAR: folders like 049/ (genuine .png) and 049_forg/ (forged .PNG)."""
    genuine_dirs = sorted(
        d for d in os.listdir(ICDAR_TRAIN)
        if os.path.isdir(os.path.join(ICDAR_TRAIN, d)) and not d.endswith("_forg")
    )
    n_writers = 0
    for d in genuine_dirs:
        try:
            wid = int(d)
        except ValueError:
            continue
        writer = f"icdar_{wid:03d}"
        # ---- genuine ----
        g_src = os.path.join(ICDAR_TRAIN, d)
        g_files = sorted(f for f in os.listdir(g_src) if is_img(f))
        g_out = os.path.join(OUT, writer)
        ensure(g_out)
        for i, f in enumerate(g_files, 1):
            rel = f"{writer}/{writer}_g_{i:02d}.png"
            if save_gray_png(os.path.join(g_src, f), os.path.join(OUT, rel)):
                rows.append([rel, writer, "icdar", "latin", "genuine"])
        # ---- forged ----
        f_src = os.path.join(ICDAR_TRAIN, d + "_forg")
        if os.path.isdir(f_src):
            f_files = sorted(f for f in os.listdir(f_src) if is_img(f))
            f_out = os.path.join(OUT, writer + "_forg")
            ensure(f_out)
            for i, f in enumerate(f_files, 1):
                rel = f"{writer}_forg/{writer}_f_{i:02d}.png"
                if save_gray_png(os.path.join(f_src, f), os.path.join(OUT, rel)):
                    rows.append([rel, writer, "icdar", "latin", "forged"])
        n_writers += 1
    return n_writers


# BHSig filename: H-S-{person}-G-{nn}.tif (genuine) / H-S-{person}-F-{nn}.tif (forged)
BHH_RE = re.compile(r"^H-S-(\d+)-([GF])-(\d+)\.(?:tif|tiff|png|jpg|jpeg)$", re.I)


def build_bhh(rows):
    """BHSig260-Hindi: folders 1..160, files H-S-{p}-G/F-{nn}.tif."""
    if not os.path.isdir(BHH_ROOT):
        print(f"  !! BHSig Hindi not found at {BHH_ROOT} — skipping")
        return 0
    person_dirs = sorted(
        (d for d in os.listdir(BHH_ROOT) if os.path.isdir(os.path.join(BHH_ROOT, d))),
        key=lambda x: int(x) if x.isdigit() else 1 << 30,
    )
    n_writers = 0
    for d in person_dirs:
        if not d.isdigit():
            continue
        wid = int(d)
        writer = f"bhh_{wid:03d}"
        src = os.path.join(BHH_ROOT, d)
        gen, forg = [], []
        for f in os.listdir(src):
            m = BHH_RE.match(f)
            if not m:
                continue
            (gen if m.group(2).upper() == "G" else forg).append((int(m.group(3)), f))
        gen.sort(); forg.sort()
        if gen:
            ensure(os.path.join(OUT, writer))
            for i, (_, f) in enumerate(gen, 1):
                rel = f"{writer}/{writer}_g_{i:02d}.png"
                if save_gray_png(os.path.join(src, f), os.path.join(OUT, rel)):
                    rows.append([rel, writer, "bhsig260_hindi", "devanagari", "genuine"])
        if forg:
            ensure(os.path.join(OUT, writer + "_forg"))
            for i, (_, f) in enumerate(forg, 1):
                rel = f"{writer}_forg/{writer}_f_{i:02d}.png"
                if save_gray_png(os.path.join(src, f), os.path.join(OUT, rel)):
                    rows.append([rel, writer, "bhsig260_hindi", "devanagari", "forged"])
        n_writers += 1
    return n_writers


def main():
    if os.path.isdir(OUT):
        print(f"!! {OUT} already exists. Remove it first to rebuild cleanly.")
        sys.exit(1)
    ensure(OUT)
    rows = []
    print("Building ICDAR (Latin) ...")
    ni = build_icdar(rows)
    print(f"  {ni} ICDAR writers")
    print("Building BHSig260-Hindi (Devanagari) ...")
    nb = build_bhh(rows)
    print(f"  {nb} BHSig Hindi writers")

    rows.sort(key=lambda r: r[0])
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["relpath", "writer", "source", "script", "kind"])
        w.writerows(rows)

    # summary
    writers = sorted({r[1] for r in rows})
    g = sum(1 for r in rows if r[4] == "genuine")
    f = sum(1 for r in rows if r[4] == "forged")
    print(f"\nDONE -> {OUT}")
    print(f"  writers: {len(writers)}  (icdar={sum(w.startswith('icdar_') for w in writers)}, "
          f"bhh={sum(w.startswith('bhh_') for w in writers)})")
    print(f"  images : {len(rows)}  (genuine={g}, forged={f})")
    print(f"  manifest: {os.path.join(OUT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
