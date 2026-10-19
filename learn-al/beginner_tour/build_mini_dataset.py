"""Build the small REAL dataset the beginner tour uses (run once, ~1 minute).

    python build_mini_dataset.py

Takes every 100th molecule of al-molecular's ENHITS pool (2,104,318 real
molecules with real AutoDock Vina docking scores against thymidylate kinase)
and copies their SMILES, scores and embeddings from 3 backbones into
learn-ai-research/data/enhits_mini/, in exactly the file format
run_experiment.py expects:

    scores.csv.gz              smiles,score    (score: lower = binds better)
    molformer_embeddings.npy   (21044, 768)    MoLFormer: reads the SMILES text
    unimol2_embeddings.npy     (21044, 768)    Uni-Mol2:  reads the 3D shape
    mhgged_embeddings.npy      (21044, 1024)   MHG-GED:   reads the molecular graph

The full pool file is sorted best score first, so taking every 100th row
keeps the whole range of scores. Rows are then shuffled (fixed seed), so the
order of the file says nothing about how good a molecule is.

This is real research data, so it lives OUTSIDE git (data/ is in .gitignore).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_EMB = Path("/N/project/SingleCell_Image/mengjing/enhits_large/embed")
SRC_SCORES = Path("/N/slate/mengjing/repos/al-molecular/data/EnamineHTS_scores.csv.gz")
OUT = Path(__file__).resolve().parents[2] / "data" / "enhits_mini"
BACKBONES = ["molformer", "unimol2", "mhgged"]
STEP = 100


def main():
    if (OUT / "scores.csv.gz").exists() and all((OUT / f"{b}_embeddings.npy").exists() for b in BACKBONES):
        print("already built:", OUT)
        return
    OUT.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SRC_SCORES)                      # 2,104,318 rows, row-aligned with the .npy files
    rows = np.arange(0, len(scores), STEP)
    order = np.random.default_rng(0).permutation(len(rows))   # shuffle so file order means nothing
    rows = rows[order]

    mini = scores.iloc[rows].reset_index(drop=True)
    assert mini["smiles"].is_unique, "duplicate SMILES would collapse in the oracle dict"
    mini.to_csv(OUT / "scores.csv.gz", index=False)
    print(f"scores: {len(mini):,} molecules, best {mini.score.min()}, worst {mini.score.max()}")

    sorted_rows = np.sort(rows)                           # read the big file in order (much faster) ...
    back = np.argsort(np.argsort(rows))                   # ... then put rows back in shuffled order
    for bb in BACKBONES:
        big = np.load(SRC_EMB / f"{bb}_embeddings.npy", mmap_mode="r")
        assert big.shape[0] == len(scores)
        emb = np.ascontiguousarray(big[sorted_rows])[back].astype(np.float32)
        np.save(OUT / f"{bb}_embeddings.npy", emb)
        print(f"{bb}: {emb.shape}")
    print("done:", OUT)


if __name__ == "__main__":
    sys.exit(main())
