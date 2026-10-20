"""Step 3: the repo's real surrogates, on fake "backbone" embeddings.

Run:   python 00_how_al_molecular_works/step3_real_surrogates.py      (CPU ~1 min; faster on a GPU node)

New idea in this step: BACKBONES.
    A backbone is a big pretrained model (MoLFormer, GROVER, Uni-Mol, ...) that
    turns a molecule into an embedding. al-molecular runs each backbone ONCE over
    the whole pool (embed/), saves the result as an (N, D) .npy file, and then
    never touches the backbone again during AL. (Only the "ft_*" surrogates
    fine-tune a backbone.) Different backbones see different things: one reads the
    SMILES text, one reads the molecular graph, one reads the 3D shape.

    The research question of the repo: is it better to use ONE backbone, or to
    FUSE several?

Here we fake three backbones: each sees the hidden truth through its own lens,
with its own blind spot. Then we plug in the classes from al-molecular/surrogates.py:
    SingleBackboneMVESurrogate  - one backbone, small net with MVE heads (mu + sigma)
    EnsembleFusionSurrogate     - one such net PER backbone, fused by ranks (method [3])
"""
import sys
from pathlib import Path

import numpy as np
import torch

AL_REPO = Path(__file__).resolve().parents[3] / "al-molecular-tutorial"
sys.path.insert(0, str(AL_REPO))
from surrogates import SingleBackboneMVESurrogate, EnsembleFusionSurrogate   # noqa: E402
from molpal.acquirer.metrics import greedy, ucb                              # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Part A. How rank fusion works, on 5 molecules.  (EnsembleFusionSurrogate.combine)
#   Each backbone's net predicts a score. Turn each column into RANKS (1 = best),
#   divide by n so ranks are in (0, 1]. Then:
#     mu    = -(mean rank)          -> molecules ALL backbones like come first
#     sigma ~ std of the ranks      -> molecules the backbones DISAGREE about
# ---------------------------------------------------------------------------
fusion_demo = EnsembleFusionSurrogate(dims={"grover": 1, "molformer": 1, "unimol": 1})
comps = np.array([   # per-backbone predicted (sign-flipped) scores; columns = grover, molformer, unimol
    [9.0, 8.8, 9.1],   # m0: everyone says great
    [5.0, 5.2, 4.9],   # m1: everyone says poor
    [9.5, 3.0, 4.0],   # m2: grover loves it, the others don't
    [7.0, 7.1, 6.9],   # m3: everyone says OK
    [2.0, 9.9, 2.5],   # m4: only molformer loves it
], dtype=np.float32)
for beta in (0.2, 1.0):
    fusion_demo._beta = beta                     # the exploration weight (default 0.2)
    mu, sigma = fusion_demo.combine(comps)
    u = ucb(mu, sigma ** 2)
    g_order = np.argsort(greedy(mu))[::-1].tolist()
    u_order = np.argsort(u)[::-1].tolist()
    print(f"Part A: rank fusion on 5 molecules, beta = {beta}")
    print("  mol  grover molformer unimol |    mu   sigma    ucb | greedy-rank  ucb-rank")
    for i, row in enumerate(comps):
        print(f"  m{i}   {row[0]:5.1f}  {row[1]:7.1f}  {row[2]:6.1f} | {mu[i]:6.2f}  {sigma[i]:5.3f}  {u[i]:5.2f} |"
              f"      {g_order.index(i) + 1}          {u_order.index(i) + 1}")
    print()
print("""  m2 and m4 (backbones disagree) get the biggest sigma. At beta = 0.2 that bonus
  is too small to change the order. At beta = 1.0 they jump past m3 ("everyone says OK").
  Whether that jump is a good idea is an experiment -- and in this repo, it mostly wasn't.
""")

# ---------------------------------------------------------------------------
# Part B. A small AL experiment with the real classes.
# ---------------------------------------------------------------------------
N_POOL, TOPK, INIT, BATCH, N_ROUNDS = 6_000, 60, 60, 60, 4
EPOCHS = 30
hidden = rng.normal(size=(N_POOL, 6)).astype(np.float32)        # the "real chemistry"
y_true = (1.2 * hidden[:, 0] - 0.8 * hidden[:, 1] + 0.7 * hidden[:, 2] * hidden[:, 3]
          + 0.5 * np.sin(2 * hidden[:, 4]) + 0.2 * rng.normal(size=N_POOL)).astype(np.float32)
# already sign-flipped: higher = better
true_top_k = set(np.argsort(y_true)[::-1][:TOPK].tolist())


def fake_backbone(dim, sees, seed):
    """An embedding that only sees some of the hidden features (its blind spot), mixed + noisy."""
    r = np.random.default_rng(seed)
    mix = r.normal(size=(len(sees), dim)).astype(np.float32)
    return hidden[:, sees] @ mix + 0.8 * r.normal(size=(N_POOL, dim)).astype(np.float32)


# Each fake backbone is blind to a different feature, like a 1D/2D/3D model missing different things.
emb_dict = {
    "grover":    fake_backbone(32, [0, 1, 2, 3], seed=1),      # misses feature 4
    "molformer": fake_backbone(24, [0, 1, 3, 4], seed=2),      # misses feature 2
    "unimol":    fake_backbone(16, [0, 2, 3, 4], seed=3),      # misses feature 1
}
dims = {k: v.shape[1] for k, v in emb_dict.items()}
# Fusion surrogates take ONE wide matrix: the backbones glued side by side (they split it back with _split()).
X_all = np.concatenate([emb_dict[k] for k in dims], axis=1)


def run(name, make_model, X, acq="greedy", seed=42):
    r = np.random.default_rng(seed)
    labeled = set(r.choice(N_POOL, INIT, replace=False).tolist())
    model = make_model()
    for _ in range(N_ROUNDS):
        idx = np.array(sorted(labeled))
        model.fit(X[idx], y_true[idx], epochs=EPOCHS)          # the repo's fit(X, y, epochs)
        pool_idx = np.setdiff1d(np.arange(N_POOL), idx)
        mu, sigma = model.predict(X[pool_idx])                  # the repo's predict(X) -> (mu, sigma)
        scores = ucb(mu, sigma ** 2) if acq == "ucb" else greedy(mu)
        labeled.update(pool_idx[np.argsort(scores)[::-1][:BATCH]].tolist())
    rec = len(labeled & true_top_k) / TOPK
    print(f"  {name:32s} top-{TOPK} recall after {N_ROUNDS} rounds = {rec:.0%}")
    return rec


print(f"Part B: {N_POOL:,} molecules, budget {INIT + N_ROUNDS * BATCH:,} "
      f"(random search would expect ~{(INIT + N_ROUNDS * BATCH) / N_POOL:.0%})")
for k in dims:
    run(f"single backbone: {k}", lambda k=k: SingleBackboneMVESurrogate(in_dim=dims[k]), emb_dict[k])
run("fusion of 3 (greedy)", lambda: EnsembleFusionSurrogate(dims=dims, epochs=EPOCHS), X_all)
run("fusion of 3 (ucb)", lambda: EnsembleFusionSurrogate(dims=dims, epochs=EPOCHS), X_all, acq="ucb")

print("""
This is one seed on toy data, so don't over-read small differences between lines.
What the real repo found on EnamineHTS (reports/figure4_and_fusion_comparison.html):
the frozen 3-backbone fusion reaches ~95% of the top-1000 at the largest batch,
close to MPN and fine-tuned MoLFormer (~95-98%), but 100-1000x faster, because
the backbones run once up front and only small nets are trained in the loop.

Look at where "fusion" lands next to the single backbones here. Rank fusion gives
every backbone an EQUAL vote, so a weak backbone can pull a strong one down.
That is why the repo went on to LEARN the weights (LearnedFusionSurrogate,
LTAllSurrogate, the current default), and why its biggest wins came from
better backbones, not cleverer acquisition (docs/adaptive_acq_results.md, sec. 1).""")

# EXERCISE 1: make one backbone blind to almost everything (sees=[0]). Does it drag the fusion down?
# EXERCISE 2: run each line with seed=43, 44. How much does recall move between seeds?
