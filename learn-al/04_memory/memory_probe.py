"""Memory in active learning: storing the pool, scoring the pool, training the surrogate.

Run on a GPU:   sbatch 04_memory/run_gpu.sh   (output goes to logs/)
Run on CPU:     python 04_memory/memory_probe.py   (parts A and B only)

Three places where AL runs out of memory:
  A. Pool STORAGE (CPU RAM): millions of molecules x fingerprint/embedding size.
  B. Pool SCORING: predicting all of them at once instead of in chunks.
  C. Surrogate TRAINING (GPU): weights + grads + optimizer state + activations.
     For Adam in fp32 the first three are 4 + 4 + 8 = 16 bytes per parameter;
     activations grow with batch size and are often the biggest part.
"""
import tracemalloc
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

MB = 2**20


# --- A. How you store the pool matters 10-100x ---------------------------------
def pool_storage_demo(n=20000, bits=2048):
    print("A. Pool storage for", n, "molecules x", bits, "bits")
    tracemalloc.start()
    as_lists = [[0] * bits for _ in range(n // 10)]          # only 1/10, it's that heavy
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del as_lists
    print(f"  python list of lists : {cur * 10 / MB:8.0f} MB  (estimated from 1/10)")
    for name, t in [("float32 tensor", torch.zeros(n, bits)),
                    ("uint8 tensor", torch.zeros(n, bits, dtype=torch.uint8)),
                    ("bit-packed uint8", torch.zeros(n, bits // 8, dtype=torch.uint8))]:
        print(f"  {name:<21}: {t.element_size() * t.nelement() / MB:8.0f} MB")
    print("  -> a 10M-molecule pool as float32 x 2048 = ~80 GB; packed bits = ~2.5 GB.")
    print("     Store compactly; convert chunk by chunk to float right before the model.")


# --- B/C helpers -------------------------------------------------------------
def make_surrogate(d_in=2048, width=4096, depth=6):
    layers = [nn.Linear(d_in, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(width, 1))


def peak_of(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MB


def pool_scoring_demo(model, n_pool=200_000, d_in=2048):
    print("\nB. Scoring a pool of", n_pool, "on GPU (pool itself kept on CPU as uint8)")
    pool = (torch.rand(n_pool, d_in) < 0.1).to(torch.uint8)

    def score(chunk, grad):
        with torch.set_grad_enabled(grad):
            for i in range(0, n_pool, chunk):
                model(pool[i:i + chunk].cuda().float())

    for chunk, grad in [(8192, True), (8192, False), (32768, False), (2048, False)]:
        try:
            pk = peak_of(lambda: score(chunk, grad))
            print(f"  chunk {chunk:6d}  grad={str(grad):<5}  peak {pk:8.0f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  chunk {chunk:6d}  grad={str(grad):<5}  OOM")
            torch.cuda.empty_cache()
    print("  -> always wrap pool prediction in torch.no_grad(); chunk size trades speed vs peak.")


def training_demo(d_in=2048, width=1024, depth=12):
    # narrower + deeper than part B, with big batches: here ACTIVATIONS dominate
    m = make_surrogate(d_in, width, depth)
    p = sum(q.numel() for q in m.parameters())
    print(f"\nC. Surrogate training: {p / 1e6:.0f}M params -> weights {p * 4 / MB:.0f} MB, "
          f"+grads {p * 4 / MB:.0f}, +Adam {p * 8 / MB:.0f} MB (fp32)")
    del m
    print(f"  {'setting':<30}{'PEAK MB':>10}")

    def step(batch, amp=False, ckpt=False):
        model = make_surrogate(d_in, width, depth).cuda()
        opt = torch.optim.Adam(model.parameters())
        x = (torch.rand(batch, d_in, device="cuda") < 0.1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if ckpt:
                # Keep only the inputs of 4 segments; recompute the activations inside each
                # segment during backward. (Checkpointing every single layer saves nothing:
                # each layer's input still gets stored. Segments are what save memory.)
                out = checkpoint_sequential(model, 4, x, use_reentrant=False)
            else:
                out = model(x)
            loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()                 # Adam state is allocated here
        opt.zero_grad(set_to_none=True)

    for name, kw in [("batch 512 fp32", dict(batch=512)),
                     ("batch 16384 fp32", dict(batch=16384)),
                     ("batch 16384 bf16 autocast", dict(batch=16384, amp=True)),
                     ("batch 16384 + grad checkpoint", dict(batch=16384, ckpt=True)),
                     ("batch 16384 bf16 + checkpoint", dict(batch=16384, amp=True, ckpt=True))]:
        print(f"  {name:<30}{peak_of(lambda: step(**kw)):>10.0f}")
    print("\nAllocator report (what to read when you hit OOM):")
    print(torch.cuda.memory_summary(abbreviated=True)[:1200])


if __name__ == "__main__":
    pool_storage_demo()
    if torch.cuda.is_available():
        surrogate = make_surrogate().cuda().eval()
        pool_scoring_demo(surrogate)
        del surrogate
        training_demo()
    else:
        print("\n(no GPU here - run `sbatch 04_memory/run_gpu.sh` for parts B and C)")

# EXERCISE 1: predict the PEAK for 'batch 512 fp32' from the 16-bytes/param rule before running.
# EXERCISE 2: in part B, why does grad=True cost so much more, even with no backward()?
# EXERCISE 3: how does al-molecular store its pool embeddings (see shared_embedding_store.py)?
#   Estimate its RAM for your largest pool with the arithmetic from part A.
# EXERCISE 4: record a memory timeline of one AL iteration:
#   torch.cuda.memory._record_memory_history(); ...run...;
#   torch.cuda.memory._dump_snapshot("snap.pickle")
#   then drag snap.pickle into https://pytorch.org/memory_viz
