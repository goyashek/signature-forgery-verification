# sign_data_combined

A single, self-contained, **leak-free-ready** signature dataset combining two sources
under one collision-proof layout. Built by `../build_combined_dataset.py` (reproducible).

This folder is what **notebook 3 (rebuilt)** and the cross-dataset / cross-script work
read from. The original `sign_data/` is left untouched, so notebooks 01, 01b, and 02
keep running exactly as before.

## Sources

| source key | script | writers | per writer | from |
|---|---|---|---|---|
| `icdar` | Latin | 64 | ~12 genuine / ~12 forged | ICDAR `sign_data/train/` (the `test/` dup is the known leak — excluded) |
| `bhsig260_hindi` | Devanagari | 160 | 24 genuine / 30 forged | BHSig260-Hindi (Pal et al., ICDAR 2011) |

Totals: **224 writers, 10,289 images** (4,727 genuine + 5,562 forged).

## Layout

```
sign_data_combined/
├── manifest.csv                  # relpath, writer, source, script, kind
├── icdar_001/                    # genuine signatures of ICDAR writer 1
├── icdar_001_forg/               # forgeries of ICDAR writer 1
│   ...                           # icdar_049+ are the unseen-writer test range
├── bhh_001/                      # genuine signatures of BHSig Hindi person 1
├── bhh_001_forg/                 # forgeries of BHSig Hindi person 1
│   ...
└── bhh_160/  bhh_160_forg/
```

- **Writer keys are namespaced** (`icdar_NNN`, `bhh_NNN`). Both source datasets number
  writers from 1, so raw ids would collide — the prefix prevents any silent overlap.
- **ICDAR keeps its original numeric id** (`icdar_049`), so the existing ICDAR split
  ranges (train ≤40, val 41–48, test ≥49) carry over unchanged.
- **Filenames** are uniform and sortable: `icdar_049_g_03.png`, `bhh_012_f_27.png`
  (`_g_` = genuine, `_f_` = forged).
- **All images are grayscale PNG.** Signatures are ink-on-paper, so the color channels
  are redundant and every notebook already reads grayscale. Re-encoding roughly halves
  ICDAR and shrinks the heavy BHSig `.tif` files ~100×. Pixels the models use are
  identical; only the container/encoding changed (so these are no longer the byte-exact
  original files — use `sign_data/` if you need those).

## How to use it (notebook side)

The **writer-independent split lives in the notebook, not in this folder.** This dataset
deliberately stores every writer flat and tags them in `manifest.csv`; the notebook:

1. reads `manifest.csv` (or globs the folders),
2. picks which writers go to train / val / test **by writer key** (no person crosses
   splits — this is what keeps results honest, see `../CLAUDE.md` §4),
3. builds **leak-free pairs** from the raw folders with the 3-recipe `make_pairs`
   (match / forgery-negative / different-writer-negative), so a genuine `img2` never
   gives away the label (see `../CLAUDE.md` §4a).

Because every writer is tagged with `source` and `script`, the same manifest supports
both **mixed-script training** (train on icdar + bhh together) and **cross-script
held-out testing** (train on Latin, test on Devanagari) without rebuilding.

## Rebuild

```bash
python3 ../build_combined_dataset.py     # refuses to overwrite; rm -rf this folder first
```

Requires the raw `archive/BHSig260-Hindi/` download (gitignored) and `sign_data/train/`.
