# Modules, packages and imports

This is the topic that confuses most people coming from other languages. It's
also the main thing you'll need when you reorganize `al-molecular`.

## Terms

- **Module**: any `.py` file. `import surrogates` runs `surrogates.py` once and
  caches the result in `sys.modules`.
- **Package**: a directory of modules, usually with an `__init__.py`. Then
  `import al.models.surrogates` means `al/models/surrogates.py`.
- **`sys.path`**: the list of directories Python searches. It contains the
  **directory of the script you ran** (not your current directory!), then
  `PYTHONPATH`, then site-packages.

```python
import sys; print(sys.path)          # always the first thing to check on ImportError
import surrogates; print(surrogates.__file__)   # which file did I actually get?
```

## `if __name__ == "__main__":`

A file is either run (`python foo.py` → `__name__ == "__main__"`) or imported
(`import foo` → `__name__ == "foo"`). Put "do the work" code under that guard so
that importing the file doesn't start a training run. This is **required** for
multiprocessing and DataLoader workers: each child process re-imports your file.

## `python foo.py` vs `python -m pkg.foo`

- `python scripts/foo.py` puts `scripts/` on `sys.path`. So `import pkg` fails
  unless `pkg` is inside `scripts/`.
- `python -m pkg.foo`, run from the project root, puts the project root on
  `sys.path`, so `import pkg.anything` works. This is usually what you want
  in a project with packages.
- Best long-term option: add a `pyproject.toml` and run `pip install -e .`.
  The package can then be imported from anywhere, including Slurm jobs.

## Relative vs absolute imports

Inside a package, `from .utils import x` means "the `utils` module next to me".
Relative imports only work when the file is imported as part of a package. If
you run that file directly with `python file.py`, you get
`ImportError: attempted relative import with no known parent package`. The fix
is `python -m`.

## Exercise

1. Create `mypkg/__init__.py` (empty), `mypkg/a.py` with `def hi(): return "hi"`,
   and `mypkg/b.py` with `from .a import hi; print(hi())`.
2. Run `python mypkg/b.py`. It fails; read the error.
3. Run `python -m mypkg.b`. It works. Explain why, using `sys.path`.
