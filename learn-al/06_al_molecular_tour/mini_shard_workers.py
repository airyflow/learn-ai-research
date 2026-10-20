"""A miniature of al-molecular's ParallelMVEExplorer + predict_pool_shard_worker.py.

Run:   python 06_al_molecular_tour/mini_shard_workers.py                (CPU, ~20 s)
       python 06_al_molecular_tour/mini_shard_workers.py --kill-shard 2 (simulate a dead worker)

The real system (AmpC, 99.5M molecules):
  * ONE orchestrator job trains the small surrogate on the labeled set.
  * N worker jobs (a Slurm --array), each owning a fixed shard of the pool,
    predict their shard every round.
  * They don't talk over a network. They coordinate through FILES in a shared
    "coord" directory:
        orchestrator writes  round_{r}_surrogate.pt, then round_{r}_ready.marker
        worker i writes      round_{r}_shard_{i}_mu.npy, then round_{r}_shard_{i}.done
        orchestrator writes  STOP.marker after the last round
Here the "jobs" are local processes started with multiprocessing; everything else
works the same way.
"""
import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

AL_REPO = Path(__file__).resolve().parents[3] / "al-molecular-tutorial"
sys.path.insert(0, str(AL_REPO))           # lets us import al-molecular's own helper
from smiles_chunking import _chunk_bounds   # noqa: E402  (the exact function the real workers use)

N_POOL, DIM, N_SHARDS, N_ROUNDS, INIT, BATCH = 40_000, 64, 4, 3, 300, 300


def make_pool():
    g = torch.Generator().manual_seed(0)    # same seed -> every process builds the same pool
    X = (torch.rand(N_POOL, DIM, generator=g) < 0.2).float()
    y = torch.tanh(X @ torch.randn(DIM, generator=g) / 3)
    return X, y


def wait_for(path, stop_path, poll=0.2, timeout=None):
    """Same idea as predict_pool_shard_worker.wait_for: poll the filesystem."""
    t0 = time.time()
    while not path.exists():
        if stop_path is not None and stop_path.exists():
            return False
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError(f"gave up waiting for {path.name}")
        time.sleep(poll)
    return True


# ---------------------------------------------------------------- worker ----
def worker(shard_id, coord, die_after_round=None):
    torch.set_num_threads(1)                # 4 workers + orchestrator share this machine's CPUs
    X, _ = make_pool()
    start, end = _chunk_bounds(N_POOL, shard_id, N_SHARDS)
    shard = X[start:end]                    # loaded ONCE, reused every round
    stop = coord / "STOP.marker"
    for r in range(1, N_ROUNDS + 1):
        done = coord / f"round_{r}_shard_{shard_id}.done"
        if done.exists():                   # restart-safe: skip finished rounds
            continue
        if not wait_for(coord / f"round_{r}_ready.marker", stop):
            return
        model = torch.load(coord / f"round_{r}_surrogate.pt", weights_only=False)
        with torch.no_grad():
            mu = model(shard).squeeze(-1).numpy()
        # write the result, THEN the marker: whoever sees .done can trust the .npy is complete
        tmp = coord / f"round_{r}_shard_{shard_id}_mu.tmp.npy"
        np.save(tmp, mu)
        tmp.rename(coord / f"round_{r}_shard_{shard_id}_mu.npy")   # atomic on POSIX
        done.touch()
        print(f"    [worker {shard_id}] round {r}: predicted rows [{start}, {end})", flush=True)
        if die_after_round == r:
            print(f"    [worker {shard_id}] simulating a crash / Slurm timeout", flush=True)
            return


# ----------------------------------------------------------- orchestrator ----
def train_surrogate(X, y):
    model = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def orchestrator(coord, timeout):
    X, y = make_pool()
    true_top = set(torch.topk(y, N_POOL // 100).indices.tolist())
    rng = np.random.default_rng(42)
    labeled = set(rng.choice(N_POOL, INIT, replace=False).tolist())
    for r in range(1, N_ROUNDS + 1):
        idx = sorted(labeled)
        model = train_surrogate(X[idx], y[idx])
        torch.save(model, coord / f"round_{r}_surrogate.pt")   # whole object, like the real code
        (coord / f"round_{r}_ready.marker").touch()

        parts = []
        for s in range(N_SHARDS):
            wait_for(coord / f"round_{r}_shard_{s}.done", None, timeout=timeout)
            parts.append(np.load(coord / f"round_{r}_shard_{s}_mu.npy"))
        mu = np.concatenate(parts)
        assert len(mu) == N_POOL

        mask = np.ones(N_POOL, bool)
        mask[idx] = False                                       # never re-acquire labeled molecules
        pool_idx = np.where(mask)[0]
        picked = pool_idx[np.argsort(mu[pool_idx])[::-1][:BATCH]]
        labeled |= set(picked.tolist())
        found = len(true_top & labeled) / len(true_top)
        print(f"[orchestrator] round {r}: labeled {len(labeled):5d}  found top-1%: {found:.1%}", flush=True)
    (coord / "STOP.marker").touch()


if __name__ == "__main__":      # REQUIRED with multiprocessing: children re-import this file
    p = argparse.ArgumentParser()
    p.add_argument("--kill-shard", type=int, default=None)
    p.add_argument("--timeout", type=float, default=15.0)
    args = p.parse_args()

    coord = Path(tempfile.mkdtemp(prefix="coord_"))
    print("coord dir:", coord)
    ctx = mp.get_context("spawn")          # fresh interpreters, closer to separate Slurm jobs
    workers = [ctx.Process(target=worker, args=(s, coord, 1 if s == args.kill_shard else None))
               for s in range(N_SHARDS)]
    for w in workers:                      # like `sbatch --array=0-3` BEFORE the orchestrator
        w.start()
    try:
        orchestrator(coord, args.timeout)
    except TimeoutError as e:
        print(f"[orchestrator] {e} -> in the real run: `sbatch --array={args.kill_shard} ...` "
              f"to resubmit just that shard", flush=True)
        (coord / "STOP.marker").touch()
    for w in workers:
        w.join()
    print("files left in coord dir:", sorted(f.name for f in coord.iterdir())[:8], "...")
    shutil.rmtree(coord)                   # comment this out to inspect the marker files yourself

# EXERCISE 1: run with --kill-shard 2. Then add a "resubmit": after the TimeoutError, start a
#   NEW worker process for shard 2 and continue the round. Why does the `if done.exists(): continue`
#   check in worker() make this safe?
# EXERCISE 2: swap the two lines `tmp.rename(...)` and `done.touch()`. What could the orchestrator
#   read in the real system (on a slow shared filesystem) if the marker appears first?
# EXERCISE 3: compare with al-molecular/predict_pool_shard_worker.py. That worker predicts its
#   WHOLE shard every round, including labeled molecules. Find the comment that explains why.
# EXERCISE 4: compare with 05_distributed/ddp_al_iteration.py, which does the same job with
#   all_gather. List 3 advantages of each design on this cluster.
