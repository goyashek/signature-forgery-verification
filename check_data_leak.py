#!/usr/bin/env python3
"""
check_data_leak.py — detect the two data-leakage traps in a paired signature dataset.

This is the reusable, automated version of the manual investigation that confirmed why
notebook 1 scored a suspiciously perfect ROC-AUC 0.999 (see README "Data integrity" and
notebooks/01b_data_leak_investigation.ipynb). Point it at any dataset laid out like the
shipped ICDAR `sign_data/` and it will flag both leaks before you train on it.

Two checks:

  LEAK 1 — Duplicate test set.
    Is the test/ folder a byte-identical duplicate of (part of) train/? If so, "test"
    metrics are really train metrics. Detected by md5-hashing every image in both folders
    and reporting the overlap.

  LEAK 2 — The pairing leak (label is a function of img2's folder).
    In the shipped CSVs the label is a deterministic function of which folder the SECOND
    image comes from: label=0 (match) -> img2 is always genuine, label=1 (forgery) ->
    img2 is always from a *_forg folder. A model can then ignore the reference signature
    (img1) entirely and just detect "is img2 forged?" — a single-image artifact detector
    that still generalizes to unseen writers, so a writer-independent split does NOT catch
    it. Detected by measuring how perfectly img2's folder type predicts the label.

Usage:
    python3 check_data_leak.py                      # defaults to ./sign_data
    python3 check_data_leak.py path/to/dataset
    python3 check_data_leak.py path/to/dataset --max-hash 4000

Exit code is non-zero if either leak is detected, so it can gate a pipeline.
"""
import os
import sys
import csv
import hashlib
import argparse
from collections import defaultdict


IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def iter_images(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                yield os.path.join(dirpath, f)


def check_duplicate_test(root, max_hash):
    """LEAK 1: is test/ a byte-identical duplicate subset of train/?"""
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print("  [skip] no train/ + test/ folders here — nothing to compare.")
        return False

    train_files = list(iter_images(train_dir))
    test_files = list(iter_images(test_dir))
    if max_hash and len(test_files) > max_hash:
        print(f"  (sampling {max_hash} of {len(test_files)} test images for the hash check)")
        test_files = test_files[:max_hash]

    print(f"  hashing {len(train_files)} train + {len(test_files)} test images ...")
    train_hashes = {md5_of(p) for p in train_files}
    dup = sum(1 for p in test_files if md5_of(p) in train_hashes)

    pct = 100.0 * dup / len(test_files) if test_files else 0.0
    print(f"  test images byte-identical to a train image: {dup}/{len(test_files)} ({pct:.1f}%)")
    if pct >= 99.0:
        print("  >> LEAK 1 DETECTED: test/ duplicates train/. 'Test' metrics = train metrics.")
        print("     Fix: re-split by WRITER ID so no person appears in both train and test.")
        return True
    if pct > 0:
        print(f"  >> WARNING: partial test/train overlap ({pct:.1f}%). Inspect before trusting the split.")
        return True
    print("  OK: no byte-identical test/train overlap.")
    return False


def folder_of(rel_path):
    """First path component, e.g. '068_forg/03_x.png' -> '068_forg'."""
    return rel_path.replace("\\", "/").split("/")[0]


def is_forg_folder(name):
    return name.lower().endswith("_forg")


def check_pairing_leak(root):
    """LEAK 2: is the label a deterministic function of img2's folder type?"""
    found_any = False
    for csv_name in ("train_data.csv", "test_data.csv"):
        path = os.path.join(root, csv_name)
        if not os.path.isfile(path):
            continue
        rows = []
        with open(path, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) < 3:
                    continue
                img1, img2, label = r[0].strip(), r[1].strip(), r[2].strip()
                # tolerate an optional header row
                if label not in ("0", "1"):
                    continue
                rows.append((img1, img2, int(label)))
        if not rows:
            continue

        # Cross-tabulate label vs. img2-folder-type.
        # table[label]["forg"|"genuine"] = count
        table = {0: defaultdict(int), 1: defaultdict(int)}
        for _, img2, label in rows:
            kind = "forg" if is_forg_folder(folder_of(img2)) else "genuine"
            table[label][kind] += 1

        n = len(rows)
        # How well does "img2 is forged" predict label==1?
        correct = table[1]["forg"] + table[0]["genuine"]
        acc = 100.0 * correct / n
        print(f"\n  {csv_name}: {n} pairs")
        print(f"    label=0 (match)  -> img2 genuine:{table[0]['genuine']:6d}  forg:{table[0]['forg']:6d}")
        print(f"    label=1 (forgery)-> img2 genuine:{table[1]['genuine']:6d}  forg:{table[1]['forg']:6d}")
        print(f"    'img2 folder type' alone predicts the label with {acc:.1f}% accuracy")

        if acc >= 99.0:
            print("    >> LEAK 2 DETECTED: label is (almost) a pure function of img2's folder.")
            print("       A model can ignore img1 and just detect 'is img2 forged?'.")
            found_any = True
        else:
            print("    OK: img2's folder does not determine the label.")

    if found_any:
        print("\n  Fix for LEAK 2: build pairs from the raw per-writer folders with a THIRD recipe —")
        print("    match            : genuine A vs genuine A          -> label 0")
        print("    hard negative    : genuine A vs forgery A (A_forg) -> label 1")
        print("    random negative  : genuine A vs genuine B (B != A) -> label 1   (NEW)")
        print("  Now a genuine img2 no longer implies a match, so the model must compare img1 vs img2.")
    return found_any


def main():
    ap = argparse.ArgumentParser(description="Detect duplicate-test and pairing data leaks.")
    ap.add_argument("root", nargs="?", default="sign_data", help="dataset root (default: sign_data)")
    ap.add_argument("--max-hash", type=int, default=0,
                    help="cap the number of test images hashed in leak 1 (0 = all)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"error: dataset root not found: {args.root}")
        sys.exit(2)

    print(f"Checking dataset: {args.root}\n")
    print("LEAK 1 — duplicate test set (md5):")
    leak1 = check_duplicate_test(args.root, args.max_hash)

    print("\nLEAK 2 — pairing leak (label vs. img2 folder):")
    leak2 = check_pairing_leak(args.root)

    print("\n" + "=" * 60)
    if leak1 or leak2:
        print("RESULT: data leakage detected — see fixes above before trusting any metric.")
        sys.exit(1)
    print("RESULT: no leakage detected by these checks.")
    sys.exit(0)


if __name__ == "__main__":
    main()
