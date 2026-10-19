"""One distributed AL iteration: DDP surrogate training + sharded pool scoring + global top-k.

CPU, 2 processes (login node, for learning the mechanics):
    torchrun --nproc_per_node=2 05_distributed/ddp_al_iteration.py
GPU, 4x H100:
    sbatch 05_distributed/run_ddp.sh

The mental model:
  * torchrun starts N copies of THIS SCRIPT. Each copy is a "rank" (0..N-1),
    and N is the "world size". It passes RANK / LOCAL_RANK / WORLD_SIZE /
    MASTER_ADDR / MASTER_PORT through environment variables.
  * TRAINING (DDP): every rank holds the same surrogate. DistributedSampler gives
    each rank a different 1/N of the labeled set, and loss.backward() averages
    the gradients across ranks, so the weights stay identical.
  * SCORING (data sharding): each rank predicts its own 1/N of the pool. No
    gradients, so no DDP is needed. Each rank takes its local top-k, then
    all_gather combines them and every rank agrees on the GLOBAL top-k.
  * Logging and checkpointing are done once, on rank 0.

al-molecular's predict_pool_shard_worker.py does the scoring part with separate
Slurm jobs instead of a process group. Same idea, different glue.
"""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

N_POOL, N_LABELED, BITS, BATCH_ACQ = 100_000, 2_000, 512, 500


def setup():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")   # nccl=GPU, gloo=CPU
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])                       # GPU index on THIS node
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, device


def log(rank, *msg):
    if rank == 0:
        print(*msg, flush=True)


def make_library():
    # Same seed on every rank -> every rank builds the SAME library (no need to send it).
    g = torch.Generator().manual_seed(0)
    fps = (torch.rand(N_POOL + N_LABELED, BITS, generator=g) < 0.1).float()
    w = torch.randn(BITS, generator=g)
    y = torch.tanh(fps @ w / 5) + 0.05 * torch.randn(len(fps), generator=g)
    return fps[:N_LABELED], y[:N_LABELED], fps[N_LABELED:], y[N_LABELED:]


def main():
    rank, world, device = setup()
    x_lab, y_lab, x_pool, y_pool = make_library()

    # ---- 0. Collective warm-up: every rank adds its value in ------------------
    t = torch.tensor([float(rank)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}/{world}] all_reduce sum of ranks = {t.item()}", flush=True)
    assert t.item() == sum(range(world))

    # ---- 1. DDP training of the surrogate on the labeled set ----------------
    ds = TensorDataset(x_lab, y_lab)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=64, sampler=sampler)   # NOT shuffle=True: the sampler shuffles
    log(rank, f"labeled {len(ds)} -> {len(sampler)} per rank; effective batch {64 * world}")

    torch.manual_seed(0)                                   # same init on every rank (DDP also broadcasts it)
    model = nn.Sequential(nn.Linear(BITS, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)                           # otherwise every epoch uses the same shuffle
        model.train()
        total = torch.zeros(2, device=device)              # [sum_loss, count]
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()                                # gradient all-reduce happens HERE
            opt.step()
            total += torch.stack([loss.detach() * len(xb), torch.tensor(float(len(xb)), device=device)])
        dist.all_reduce(total)                             # combine the metric across ranks
        if epoch % 5 == 4:
            log(rank, f"epoch {epoch:2d}  global train MSE {(total[0] / total[1]).item():.4f}")

    # ---- 2. Sharded pool scoring + global top-k ----------------------------
    net = model.module.eval()                              # unwrap: no DDP needed for inference
    shard = torch.arange(rank, N_POOL, world)              # rank r scores pool[r::world]
    with torch.no_grad():
        preds = torch.cat([net(x_pool[shard[i:i + 8192]].to(device)).squeeze(-1)
                           for i in range(0, len(shard), 8192)])
    local_val, local_pos = torch.topk(preds, BATCH_ACQ)    # local top-k is enough to find the global top-k
    local_idx = shard.to(device)[local_pos]                # convert to GLOBAL pool indices

    all_val = [torch.zeros_like(local_val) for _ in range(world)]
    all_idx = [torch.zeros_like(local_idx) for _ in range(world)]
    dist.all_gather(all_val, local_val)
    dist.all_gather(all_idx, local_idx)
    vals, idxs = torch.cat(all_val), torch.cat(all_idx)
    picked = idxs[torch.topk(vals, BATCH_ACQ).indices].cpu()

    # ---- 3. Rank 0 reports and "queries the oracle" -----------------------
    if rank == 0:
        true_top = set(torch.topk(y_pool, int(0.01 * N_POOL)).indices.tolist())
        hit = len(true_top & set(picked.tolist())) / len(picked)
        print(f"scored {N_POOL} pool molecules on {world} ranks; acquired {len(picked)}; "
              f"{hit:.1%} of them are true top-1% (random would give 1%)", flush=True)
        torch.save({"model": net.state_dict(), "picked": picked}, "/tmp/al_iter0.pt")
    dist.barrier()                                         # make the others wait until the file exists
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# EXERCISE 1: run with --nproc_per_node 1, 2 and 4. Is `picked` identical each time? Should it be?
# EXERCISE 2: why is taking each rank's LOCAL top-k (not just its local top-k/world) enough to get the
#   exact GLOBAL top-k? Convince yourself, then try BATCH_ACQ // world and see when it fails.
# EXERCISE 3: make rank 1 skip one training batch. Watch it hang: every rank must take part
#   in every collective. This is the #1 DDP bug. Set TORCH_DISTRIBUTED_DEBUG=DETAIL to diagnose.
# EXERCISE 4: wrap steps 1-3 in a loop over AL iterations (remove picked from the pool, add to labeled).
#   Then compare with al-molecular/predict_pool_shard_worker.py: when are separate Slurm jobs better
#   than one torchrun job? (Hint: queue time, fault tolerance, GPU count you can actually get.)
