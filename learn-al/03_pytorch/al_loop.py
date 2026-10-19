"""A real (small) active-learning loop in PyTorch: an ensemble surrogate on fingerprints.

Run:   python 03_pytorch/al_loop.py               (about 1-2 min on CPU)
       python 03_pytorch/al_loop.py --acq random  (baseline)

This is a miniature of what al-molecular / MolPAL do:
  pool of molecules (fingerprints) -> train surrogate on the labeled set
  -> predict mean and uncertainty over the pool -> acquire a batch -> query the oracle -> repeat.
Metric: the fraction of the true top-1% that has been found (the usual MolPAL metric).
"""
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

p = argparse.ArgumentParser()
p.add_argument("--acq", choices=["greedy", "ucb", "random"], default="ucb")
p.add_argument("--pool", type=int, default=50000)
p.add_argument("--init", type=int, default=100)
p.add_argument("--batch", type=int, default=100)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--ensemble", type=int, default=3)
p.add_argument("--beta", type=float, default=2.0)
args = p.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- 1. Synthetic "library": random 512-bit fingerprints + a hidden objective ---
def make_library(n, bits=512):
    fps = (torch.rand(n, bits) < 0.1).float()          # sparse binary, like Morgan FPs
    teacher = nn.Sequential(nn.Linear(bits, 64), nn.Tanh(), nn.Linear(64, 1))
    with torch.no_grad():
        y = teacher(fps).squeeze(1)                    # "docking score" (higher = better)
    return fps, y


X, Y = make_library(args.pool)
top_thresh = Y.quantile(0.99)
true_top = set(torch.nonzero(Y >= top_thresh).squeeze(1).tolist())


# --- 2. The surrogate: an nn.Module ---------------------------------------
class Surrogate(nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(x, y, epochs=60, seed=0):
    torch.manual_seed(seed)                            # each ensemble member starts differently
    model = Surrogate(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    mu, sd = y.mean(), y.std() + 1e-8                  # standardise targets
    dl = DataLoader(TensorDataset(x, (y - mu) / sd), batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()                            # grads accumulate - clear them
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


@torch.no_grad()                                       # inference: no graph, much less memory
def predict_pool(models, x, chunk=4096):
    """Mean and std over ensemble members, computed chunk by chunk."""
    outs = []
    for model, mu, sd in models:
        preds = [model(x[i:i + chunk].to(device)).cpu() for i in range(0, len(x), chunk)]
        outs.append(torch.cat(preds) * sd + mu)
    outs = torch.stack(outs)                           # [ensemble, n_pool]
    return outs.mean(0), outs.std(0)


def acquire(mean, std, k):
    if args.acq == "random":
        return torch.randperm(len(mean))[:k]
    score = mean if args.acq == "greedy" else mean + args.beta * std
    return torch.topk(score, k).indices                # indices INTO THE POOL, not the library


# --- 3. The AL loop -----------------------------------------------------------
perm = torch.randperm(args.pool)
labeled_idx = perm[:args.init]                         # library indices
pool_idx = perm[args.init:]

for it in range(args.iters + 1):
    found = len(true_top & set(labeled_idx.tolist())) / len(true_top)
    print(f"iter {it}  labeled {len(labeled_idx):5d}  found top-1%: {found:6.1%}  "
          f"best so far {Y[labeled_idx].max():.3f} (true best {Y.max():.3f})", flush=True)
    if it == args.iters:
        break

    models = [train_one(X[labeled_idx], Y[labeled_idx], seed=s) for s in range(args.ensemble)]
    mean, std = predict_pool(models, X[pool_idx])
    pick = acquire(mean, std, args.batch)              # positions in pool_idx

    keep = torch.ones(len(pool_idx), dtype=torch.bool)
    keep[pick] = False
    labeled_idx = torch.cat([labeled_idx, pool_idx[pick]])   # "query the oracle"
    pool_idx = pool_idx[keep]
    assert len(labeled_idx) + len(pool_idx) == args.pool

# EXERCISE 1: run with --acq greedy / ucb / random. Expect greedy and UCB to be close and
#   ~15x better than random (MolPAL found the same). When should UCB win? Try --init 20 or --beta 10.
# EXERCISE 2: put breakpoint() just after `mean, std = predict_pool(...)`. Check
#   mean.shape, std.min(), and whether high-std molecules are near labeled ones.
# EXERCISE 3: replace the ensemble with MC-dropout (add nn.Dropout and keep
#   model.train() during prediction). Compare the speed and the quality of the uncertainty.
# EXERCISE 4: map every step here onto al-molecular/run_experiment.py and
#   surrogates.py. Where do they keep pool_idx / labeled_idx? Where is acquire()?
