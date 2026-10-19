"""NumPy patterns behind al-molecular's 99.5M-molecule embedding store.

Run:   python 06_al_molecular_tour/memmap_store.py      (CPU, a few seconds; writes to /tmp)

Covers, using al-molecular's own code where possible:
  1. _chunk_bounds: how a pool is split into shards (and its edge cases)
  2. views vs copies: why  emb[a:b]  is free but  emb[idx_list]  allocates
  3. np.memmap / open_memmap: arrays bigger than RAM, read lazily from disk
  4. shared_embedding_store.py: preallocate -> write_slice per shard -> one file
  5. the np.save ".npy" suffix trap, and atomic writes
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

AL_REPO = Path(__file__).resolve().parents[3] / "al-molecular"
sys.path.insert(0, str(AL_REPO))
from smiles_chunking import _chunk_bounds                       # noqa: E402
from shared_embedding_store import preallocate, write_slice     # noqa: E402

work = Path(tempfile.mkdtemp(prefix="memmap_demo_"))


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------
section("1. _chunk_bounds: ceil-divided contiguous shards")
for n, k in [(10, 3), (10, 4), (9, 4), (3, 5)]:
    print(f"n={n:2d}, {k} shards ->", [_chunk_bounds(n, i, k) for i in range(k)])
# Look closely at (9, 4) and (3, 5): some shards are EMPTY (start == end), or even start > n.
# A worker whose shard is empty must still write its .done marker, or the orchestrator waits forever.
assert sum(e - s for s, e in (_chunk_bounds(10, i, 4) for i in range(4))) == 10


# ---------------------------------------------------------------------------
section("2. Views vs copies")
emb = np.arange(20, dtype=np.float32).reshape(10, 2)
view = emb[2:5]                     # slice      -> VIEW: no new memory, shares data
fancy = emb[[2, 3, 4]]              # index list -> COPY: new memory
print("slice shares memory:", np.shares_memory(emb, view),
      "| fancy-index shares memory:", np.shares_memory(emb, fancy))
view[0, 0] = -1                     # modifies emb too!
print("emb[2] after editing the view:", emb[2])
# Why it matters: run_experiment._chunked_get_means_and_vars exists because fancy-indexing
# most of a 3-backbone pool built ~24 GB copies (read its docstring). Slicing would be free.


# ---------------------------------------------------------------------------
section("3. Memory-mapped arrays: bigger than RAM, loaded lazily")
N, D = 200_000, 256                                              # ~200 MB on disk
path = work / "pool_emb.npy"
mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(N, D))
mm[:] = 0.0
mm[123] = 7.0
mm.flush()
del mm                                                           # close it

ro = np.load(path, mmap_mode="r")                                # nothing read yet
print(type(ro).__name__, ro.shape, f"{ro.nbytes / 2**20:.0f} MB on disk; reading one row:", ro[123, :3])
chunk = ro[100_000:150_000]                                      # still a lazy view
print("a 50k-row slice is still lazy:", isinstance(chunk, np.memmap))
mean = sum(float(ro[s:s + 50_000].sum()) for s in range(0, N, 50_000)) / ro.size   # bounded memory
print(f"chunked mean computed without loading the whole array: {mean:.6f}")


# ---------------------------------------------------------------------------
section("4. shared_embedding_store: many shard jobs -> one stitched file")
store = work / "grover_embeddings.npy"
n_mol, dim, n_shards = 1_000, 8, 3
preallocate(store, n_molecules=n_mol, dim=dim)                   # header-only, near-zero cost
for shard_id in range(n_shards):                                 # in reality: one Slurm array task each
    s, e = _chunk_bounds(n_mol, shard_id, n_shards)
    fake_embeddings = np.full((e - s, dim), shard_id, dtype=np.float32)
    write_slice(store, s, e, fake_embeddings)
final = np.load(store, mmap_mode="r")
print("rows 0, 400, 999 came from shards:", final[0, 0], final[400, 0], final[999, 0])
try:
    write_slice(store, 0, 10, np.zeros((5, dim), np.float32))    # wrong size on purpose
except ValueError as err:
    print("defensive check caught a bug:", str(err)[:70], "...")


# ---------------------------------------------------------------------------
section("5. The np.save suffix trap + atomic rename")
np.save(work / "x.npy.tmp", np.zeros(3))
print("files after np.save('x.npy.tmp'):", sorted(p.name for p in work.glob("x*")))   # surprise!
tmp = work / "result.tmp.npy"                                    # the fix used in shared_embedding_store
np.save(tmp, np.ones(3))
tmp.rename(work / "result.npy")                                  # atomic: readers never see half a file
print("atomic write ok:", np.load(work / "result.npy"))

# EXERCISE 1: in part 1, what does _chunk_bounds(3, 4, 5) return? Is it safe to slice with it?
#   Find where predict_pool_shard_worker.py would hit this case.
# EXERCISE 2: in part 2, time emb[list(range(0, N, 2))] vs emb[::2] on the part-3 memmap.
#   Which one reads from disk sequentially? (ParallelMVEExplorer._get_X_cached is built around this.)
# EXERCISE 3: run preallocate() again on the same store with dim=16. Read the error, then find
#   the code in shared_embedding_store.py that raises it and say why refusing is the safe choice.
# EXERCISE 4: delete the demo files when done:  rm -rf /tmp/memmap_demo_*
print(f"\n(demo files are in {work})")
