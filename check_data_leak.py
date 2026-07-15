#!/usr/bin/env python3
"""Check an ICDAR-style dataset for duplicate test images and the pair-folder shortcut.

Usage: python3 check_data_leak.py [dataset] [--max-hash N]
The script exits with status 1 when either leak is found.
"""
import os
import sys
import csv
import hashlib
import argparse
from collections import defaultdict


IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def md5_of(path, chunk=1024 * 1024):
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
    """Return True when test contains images copied from train."""
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print("  [skip] train/ and test/ were not both found")
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
        print("  >> LEAK 1: test/ duplicates train/. Split again by writer id.")
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
    """Return True when img2's folder type predicts the pair label."""
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

        table = {0: defaultdict(int), 1: defaultdict(int)}
        for _, img2, label in rows:
            kind = "forg" if is_forg_folder(folder_of(img2)) else "genuine"
            table[label][kind] += 1

        n = len(rows)
        correct = table[1]["forg"] + table[0]["genuine"]
        acc = 100.0 * correct / n
        print(f"\n  {csv_name}: {n} pairs")
        print(f"    label=0 (match)  -> img2 genuine:{table[0]['genuine']:6d}  forg:{table[0]['forg']:6d}")
        print(f"    label=1 (forgery)-> img2 genuine:{table[1]['genuine']:6d}  forg:{table[1]['forg']:6d}")
        print(f"    'img2 folder type' alone predicts the label with {acc:.1f}% accuracy")

        if acc >= 99.0:
            print("    >> LEAK 2: img2's folder predicts the label, so img1 can be ignored.")
            found_any = True
        else:
            print("    OK: img2's folder does not determine the label.")

    if found_any:
        print("\n  Rebuild pairs with genuine A vs genuine B negatives so a genuine img2 can have either label.")
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
    print("LEAK 1: duplicate test images (md5)")
    leak1 = check_duplicate_test(args.root, args.max_hash)

    print("\nLEAK 2: label versus img2 folder")
    leak2 = check_pairing_leak(args.root)

    print("\n" + "=" * 60)
    if leak1 or leak2:
        print("RESULT: data leakage detected")
        sys.exit(1)
    print("RESULT: no leakage detected by these checks.")
    sys.exit(0)


if __name__ == "__main__":
    main()
