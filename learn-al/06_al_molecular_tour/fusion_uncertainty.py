"""Case study: reading a surrogate's code against its docstring.

Run:   python 06_al_molecular_tour/fusion_uncertainty.py      (CPU, a few seconds)

NOTE (2026-10-20): this describes the code BEFORE the 2026-10-10 fix. EnsembleFusionSurrogate
now really returns rank disagreement as sigma, and ranks once over the whole pool. See
beginner_tour/8_tests_and_bugs.ipynb for the fixed code and the tests that guard it.

Before the fix, EnsembleFusionSurrogate (surrogates.py, class now at line 393) had a
docstring that said:
    UCB_score = -mean_rank_norm + beta * std_rank_norm
    (inter-backbone rank DISAGREEMENT used as uncertainty)
but its predict() returned (-borda, zeros): the beta and std terms
are never computed. This script rebuilds both versions on toy data so you can
see what difference that makes. Two lessons:
  1. With sigma == 0, UCB is exactly greedy (ucb = mu + beta * sigma).
  2. Rank-based scores computed per CHUNK are only comparable across chunks
     if the chunks are statistically alike. A pool stored sorted by score breaks that.
"""
import numpy as np

rng = np.random.default_rng(0)
N, BATCH, CHUNK = 50_000, 500, 5_000
true = rng.normal(size=N)                                   # higher = better here
top1 = set(np.argsort(true)[::-1][: N // 100])

# Three "backbones": each sees the truth through its own noise level.
backbone_preds = {name: true + rng.normal(scale=s, size=N)
                  for name, s in [("grover", 1.0), ("molformer", 0.8), ("unimol", 1.2)]}


def ranks_desc(mu):
    """rank 1 = best (highest mu), as in surrogates.py"""
    order = np.argsort(mu)[::-1]
    r = np.empty(len(mu))
    r[order] = np.arange(1, len(mu) + 1)
    return r


# --- version A: what the code does today -----------------------------------
def predict_as_implemented(preds):
    borda = sum(ranks_desc(p) for p in preds.values())
    return -borda, np.zeros(len(borda))                     # sigma is ALWAYS zero


# --- version B: what the docstring describes --------------------------------
def predict_as_documented(preds, beta=0.2):
    R = np.stack([ranks_desc(p) / len(p) for p in preds.values()])   # normalised ranks in (0, 1]
    return -R.mean(0), R.std(0), beta


def ucb(mu, var, beta=2.0):                                 # molpal/acquirer/metrics.py
    return mu + beta * np.sqrt(var)


def top_k(score, k=BATCH):
    return np.argsort(score)[::-1][:k]


def recall(sel):
    return len(top1 & set(sel.tolist())) / BATCH


# 1. sigma == 0  =>  UCB == greedy
mu, sigma = predict_as_implemented(backbone_preds)
greedy_pick = top_k(mu)
ucb_pick = top_k(ucb(mu, sigma ** 2))
print("1) as implemented: sigma all zero?", not sigma.any())
print("   UCB selection identical to greedy?", np.array_equal(np.sort(greedy_pick), np.sort(ucb_pick)))

m, s, beta = predict_as_documented(backbone_preds)
doc_pick = top_k(m + beta * s)
overlap = len(set(doc_pick.tolist()) & set(greedy_pick.tolist())) / BATCH
print(f"   documented rank-UCB differs from greedy: {1 - overlap:.0%} of the batch changes")
print(f"   precision@top-1%: greedy {recall(greedy_pick):.1%}  rank-UCB {recall(doc_pick):.1%}")


# 2. per-chunk ranking vs global ranking, on a shuffled vs a SORTED pool
def predict_chunked(preds, order):
    """Mimics run_experiment._chunked_get_means_and_vars: predict() is called
    on consecutive chunks of the pool in storage order, so ranks are per chunk."""
    out = np.empty(N)
    for start in range(0, N, CHUNK):
        idx = order[start:start + CHUNK]
        out[idx], _ = predict_as_implemented({k: v[idx] for k, v in preds.items()})
    return out


print("\n2) Borda computed per chunk of", CHUNK)
for name, order in [("shuffled pool", rng.permutation(N)),
                    ("pool sorted best-first (like a file sorted by docking score)", np.argsort(true)[::-1])]:
    pick = top_k(predict_chunked(backbone_preds, order))
    print(f"   {name:<62} precision@top-1% {recall(pick):6.1%}")
print(f"   {'global ranking (one call over the whole pool)':<62} precision@top-1% {recall(greedy_pick):6.1%}")

# EXERCISE 1: open surrogates.py and confirm by reading that EnsembleFusionSurrogate.predict
#   never uses self._beta. Then check in git history (`git log -L` on that class) whether
#   it ever did. What does that mean for "ensemble + ucb" vs "ensemble + greedy" runs?
# EXERCISE 2: explain why the sorted pool is so much worse in part 2. (Hint: every chunk
#   has a rank-1 molecule, and they all get the same score.) Then check the comments on
#   ENHITS / ENHITS_shuffled in run_experiment.py's DATASETS: is the pool there stored in
#   sorted order?
# EXERCISE 3: fix part 2 without loading the whole pool at once. One way: return raw per-backbone
#   mu from each chunk, concatenate, and rank once globally (the ranks need only N floats).
