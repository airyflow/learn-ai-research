"""Small helpers shared by the beginner-tour notebooks.

Every notebook starts with

    from tour import *

which gives you:
    AL_REPO     where the al-molecular code lives
    MINI_DIR    where the small real dataset lives (built by build_mini_dataset.py)
    BACKBONES   the three backbones in the mini dataset
    load_mini() -> (scores_table, embeddings)   the mini dataset
    register_mini(rx)   lets run_experiment.py use the mini dataset as --dataset MINI
    draw(smiles_list, labels)   pictures of molecules
    new_run_dir()   a fresh, empty folder for one experiment's output
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AL_REPO = Path("/N/slate/mengjing/repos/al-molecular-tutorial")
MINI_DIR = Path(__file__).resolve().parents[2] / "data" / "enhits_mini"
BACKBONES = ["molformer", "unimol2", "mhgged"]

if str(AL_REPO) not in sys.path:
    sys.path.insert(0, str(AL_REPO))          # so `import run_experiment`, `import surrogates` work

if not (MINI_DIR / "scores.csv.gz").exists():
    raise FileNotFoundError(
        f"The mini dataset isn't built yet. In a terminal, run:\n"
        f"    python {Path(__file__).parent / 'build_mini_dataset.py'}"
    )

_RUNS = Path(tempfile.mkdtemp(prefix="beginner_tour_"))


def load_mini():
    """The mini dataset: a table of (smiles, score) and a dict {backbone: (N, D) array}."""
    scores = pd.read_csv(MINI_DIR / "scores.csv.gz")
    embeddings = {bb: np.load(MINI_DIR / f"{bb}_embeddings.npy") for bb in BACKBONES}
    return scores, embeddings


def register_mini(rx):
    """Add the mini dataset to run_experiment's list of datasets, under the name MINI."""
    path = MINI_DIR / "scores.csv.gz"
    rx.DATASETS["MINI"] = {"library": path, "oracle": path, "embed_dir": MINI_DIR}


def draw(smiles_list, labels=None, per_row=4):
    """Draw molecules from their SMILES strings (uses RDKit)."""
    from rdkit import Chem
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=(220, 180),
                                legends=[str(x) for x in labels] if labels is not None else None)


def new_run_dir():
    """A fresh empty folder for one experiment's output (under the system temp folder)."""
    return Path(tempfile.mkdtemp(dir=_RUNS))
