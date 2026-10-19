"""Python idioms for people who already know another language.

Run:   python 01_python/idioms.py
Every section prints something. Read the code next to the output.
The asserts are self-checks: if one fails, your mental model is off.
"""
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from contextlib import contextmanager
import time


def section(title):
    print(f"\n=== {title} ===")
    
def section2(title):
    print(f"\n***=={title}==***")


# ---------------------------------------------------------------------------
section2("1. Everything is an object; variables are names, not boxes")
a = [1, 2, 3]
b = a            # b is the SAME list, not a copy
print(f"a: {a}, b: {b}")
print(f"a: {id(a)}, b: {id(b)}")         # decimal: a: 140234567890112, b: 140234567890112
print(f"a: {hex(id(a))}, b: {hex(id(b))}")  # hex, like C: a: 0x7f8a1c2b3d40, b: 0x7f8a1c2b3d40
print(f"a: {a} @ {hex(id(a))}")
print(f"b: {b} @ {hex(id(b))}")

b.append(4)
print(f"a after b.append(4): {a} @ {hex(id(a))}")
print(f"b after b.append(4): {b} @ {hex(id(b))}")
assert a == [1, 2, 3, 4]
c = a.copy()     # or list(a), or a[:]
print(f"c: {c} @ {hex(id(c))}")
c.append(5)
assert a == [1, 2, 3, 4]
print("a is b:", a is b, "| a is c:", a is c)   # `is` = identity, `==` = equality


# ---------------------------------------------------------------------------
section("2. Truthiness: empty things are False")
for v in [0, 0.0, "", [], {}, None, "x", [0]]:
    print(f"{v!r:>6} -> {bool(v)}")
  
  
# so idiomatic Python writes  `if items:`  not  `if len(items) > 0:`
# but for None checks always use  `if x is None:`


# ---------------------------------------------------------------------------
section("3. Indexing and slicing (0-based, end-exclusive)")
s = list(range(10))
print(s[0], s[-1], s[2:5], s[:3], s[-3:], s[::2], s[::-1])
assert s[2:5] == [2, 3, 4]


# ---------------------------------------------------------------------------
section("4. Loops: iterate over things, not indices")
names = ["aspirin", "caffeine", "ibuprofen"]
scores = [0.3, 0.9, 0.5]
for i, name in enumerate(names):          # instead of range(len(names))
    print(i, name)
for name, score in zip(names, scores):    # parallel iteration
    print(f"{name:<10} {score:.2f}")

    print(f"{name} {score:.2f}")



for score, name in zip(scores, names):
    print(f"{name:<15} {score:.3f}")

# ---------------------------------------------------------------------------
section("5. Comprehensions")
squares = [x * x for x in range(6)]
evens = [x for x in range(10) if x % 2 == 0]
by_name = {n: sc for n, sc in zip(names, scores)}   # dict comprehension
print(squares, evens, by_name)
best = max(by_name, key=by_name.get)
print("best:", best)
ranked = sorted(names, key=lambda n: by_name[n], reverse=True)
print("ranked:", ranked)


# ---------------------------------------------------------------------------
section("6. Unpacking")
first, *rest = [1, 2, 3, 4]
x, y = 10, 20
x, y = y, x                                          # swap, no temp
print(first, rest, x, y)


# ---------------------------------------------------------------------------
section("7. Dicts, defaultdict, Counter")
d = {"a": 1}
print(d.get("missing", -199999))       # no KeyError
counts = Counter("abracadabra")
print(counts.most_common(2))

groups = defaultdict(list)
for word in ["apple", "avocado", "banana"]:
    groups[word[0]].append(word)
print(dict(groups))


# ---------------------------------------------------------------------------
section("8. Functions: defaults, keyword args, *args, **kwargs")
def train(lr, epochs=10, *extra, verbose=False, **options):
    return dict(lr=lr, epochs=epochs, extra=extra, verbose=verbose, options=options)

print(train(0.1))
print(train(0.1, 5, "x", verbose=True, dropout=0.2))
cfg = {"lr": 0.01, "epochs": 3}
print(train(**cfg))              # unpack a dict into keyword arguments


# THE classic trap: default values are evaluated ONCE, at definition time.
def bad_append(item, bucket=[]):
    bucket.append(item)
    return bucket

def good_append(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


print("bad :", bad_append(1), bad_append(2))   # [1, 2] [1, 2]  <- shared list!
print("good:", good_append(1), good_append(2))


# ---------------------------------------------------------------------------
section("9. Classes and dataclasses")
class Molecule:
    def __init__(self, smiles, score=0.0):
        self.smiles = smiles        # `self` is explicit, like `this`
        self.score = score

    def __repr__(self):             # what print() / the debugger show
        return f"Molecule({self.smiles!r}, score={self.score})"

@dataclass                           # generates __init__, __repr__, __eq__
class RunConfig:
    lr: float = 1e-3
    batch_size: int = 32
    tags: list = field(default_factory=list)   # safe mutable default

print(Molecule("CCO", 0.7))
print(RunConfig(lr=0.01))
assert RunConfig() == RunConfig()


# ---------------------------------------------------------------------------
section("10. Exceptions")
try:
    int("not a number")
except ValueError as e:
    print("caught:", e)
finally:
    print("finally always runs")


# ---------------------------------------------------------------------------
section("11. Context managers (`with`) - guaranteed cleanup")
@contextmanager
def timer(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - t0:.4f}s")

with timer("sum of 1e6"):
    sum(range(1_000_000))
# `with open(path) as f:` closes the file even if an exception happens


# ---------------------------------------------------------------------------
section("12. Generators - lazy sequences (how DataLoaders/chunking work)")
def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

print(list(chunks(list(range(7)), 3)))
total = sum(x * x for x in range(1_000_000))   # no list is built in memory
print(total)


# ---------------------------------------------------------------------------
section("13. Putting it together: AL bookkeeping with sets and dicts")
library = {f"mol{i}": score for i, score in enumerate([0.2, 0.9, 0.4, 0.95, 0.1, 0.7])}
labeled = {"mol0": library["mol0"], "mol4": library["mol4"]}      # id -> observed score
pool = set(library) - set(labeled)                                 # set difference
predicted = {m: library[m] + 0.05 for m in pool}                   # pretend surrogate
batch = sorted(pool, key=predicted.get, reverse=True)[:2]          # greedy top-2
labeled.update({m: library[m] for m in batch})                     # "query the oracle"
pool -= set(batch)
print("acquired:", batch, "| pool left:", sorted(pool))
assert not pool & set(labeled) and len(pool) + len(labeled) == len(library)


# ---------------------------------------------------------------------------
# EXERCISE 1: write `normalize(scores: dict) -> dict` that min-max scales the
#   values to [0, 1] in one dict comprehension (useful for docking scores).
#   Handle the all-equal case.
# EXERCISE 2: wrap section 13 in a `for it in range(3):` loop and record the best
#   score found after each iteration in a list. Then make the batch size a
#   RunConfig field.
# EXERCISE 3: open al-molecular/smiles_chunking.py. Which idioms above does it
#   use? Rewrite one function without them, then put the idiom back - notice why
#   the idiom is preferred.
print("\nAll self-checks passed.")
