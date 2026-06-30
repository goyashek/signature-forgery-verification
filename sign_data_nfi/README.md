# sign_data_nfi — clean NFI cross-dataset test set

A small, **clean** signature set derived from the NFI (Netherlands Forensic Institute)
`sample_Signature` collection. Used as an **independent cross-dataset test set** — a
second source, never trained on, to check whether a model trained on `sign_data`
generalizes beyond its own distribution.

## Why this exists (and why not the raw `sign_data2`)

The raw `sign_data2` download (gitignored, not in this repo) is 189 MB but only 486
unique images — `dataset3` is a byte-identical copy of `sample_Signature`, and
`dataset1`–`dataset4` have **inconsistent filename IDs**: 205 images share bytes with
another image but carry a *different* writer ID in the name. Deriving labels from those
folders is unsafe.

The `sample_Signature` subfolder, taken alone, is internally consistent: 300 unique
images, **zero label conflicts**, 30 owners. This folder is that subset only —
deduplicated, converted to grayscale, and capped at 600 px on the long side
(189 MB → 4.3 MB) since the notebooks downscale to ~160 px anyway.

## Layout

```
sign_data_nfi/
├── genuine/   # 150 imgs — signer == owner
└── forged/    # 150 imgs — signer != owner (targeted forgery of the owner's signature)
```

## Naming — `NFI-XXXYYZZZ.png`

- `XXX` — ID of the person who **physically signed**
- `YY`  — sample number
- `ZZZ` — ID of the person whose signature it is **meant to be**

So **genuine ⟺ `XXX == ZZZ`**, **forgery ⟺ `XXX != ZZZ`**. Examples:
- `NFI-02103021` — signer `021`, owner `021` → **genuine** (021 signing their own name)
- `NFI-00601023` — signer `006`, owner `023` → **forgery** (006 imitating 023's signature)

## Why it's the *honest* dataset

Forgeries here are **targeted**: someone imitating a specific person's signature. The
only way to flag one is to compare it against that person's genuine signature — exactly
the similarity task. Unlike `sign_data`, the label is **not** inferable from the
questioned image alone (verified: an `img2`-only probe scores ~0.55 AUC = chance here,
vs 0.91 on `sign_data`). See `notebooks/01b_data_leak_investigation.ipynb`.

## Caveats

- **Small:** 30 owners. Good for a held-out test, not for training a deep net from
  scratch — pair it with transfer learning (NB3) if used for fitting.
- Source: Kaggle "Handwritten Signatures" (NFI `sample_Signature`).
