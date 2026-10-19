"""calltrace: print the tree of al-molecular functions that a piece of code calls.

Used by classes_and_call_flow.ipynb. You don't need to read this to use it:

    from calltrace import trace
    result = trace(lambda: something_to_run())

How it works, if you're curious: sys.setprofile(f) makes Python call f on every
function call and return. We keep only frames whose file is inside al-molecular,
build a tree from the call/return events, name each method after the class that
DEFINES it (so inherited methods are visible), and collapse repeats.
"""
import sys
from pathlib import Path

REPO = "/N/slate/mengjing/repos/al-molecular"


def _is_repo(code):
    f = code.co_filename
    return f.startswith(REPO) and "site-packages" not in f and not code.co_name.startswith("<")


def _label(frame):
    code = frame.f_code
    name = code.co_name
    self = frame.f_locals.get("self")
    if self is not None:
        for klass in type(self).__mro__:                    # find the class that DEFINES this method
            fn = klass.__dict__.get(name)
            fn = getattr(fn, "fget", fn)                     # unwrap @property
            fn = getattr(fn, "__func__", fn)
            if getattr(fn, "__code__", None) is code:
                name = f"{klass.__name__}.{name}"
                if klass is not type(self):
                    name += f"   (on a {type(self).__name__})"
                break
    rel = Path(code.co_filename).relative_to(REPO)
    return f"{name}", f"{rel}:{code.co_firstlineno}"


class _Node:
    def __init__(self, label="", where=""):
        self.label, self.where, self.kids, self.count = label, where, [], 1

    def sig(self):
        return (self.label, tuple(k.sig() for k in self.kids))


def trace(fn, max_depth=6, max_lines=60):
    """Run fn(), then print the tree of al-molecular functions it called."""
    root = _Node()
    stack = [root]

    def prof(frame, event, arg):
        if not _is_repo(frame.f_code):
            return
        if event == "call":
            if not frame.f_code.co_flags & 0x1:    # not CO_OPTIMIZED: a class body running at import time
                prof_frames.append(None)
                stack.append(_Node())               # throwaway node, never attached to the tree
                return
            label, where = _label(frame)
            node = _Node(label, where)
            stack[-1].kids.append(node)
            stack.append(node)
            prof_frames.append(frame)
        elif event == "return" and prof_frames and (prof_frames[-1] is frame or prof_frames[-1] is None):
            prof_frames.pop()
            stack.pop()

    prof_frames = []
    sys.setprofile(prof)
    try:
        result = fn()
    finally:
        sys.setprofile(None)

    def collapse(node):
        for k in node.kids:
            collapse(k)
        kids, out, i = node.kids, [], 0
        while i < len(kids):
            best_k, best_n = 1, 1
            for k in range(1, 6):                     # look for a block of k calls that repeats
                block = [c.sig() for c in kids[i:i + k]]
                if len(block) < k:
                    break
                n = 1
                while [c.sig() for c in kids[i + n * k:i + (n + 1) * k]] == block:
                    n += 1
                if n > 1 and n * k > best_n * best_k:
                    best_k, best_n = k, n
            if best_k == 1:
                kids[i].count = best_n
                out.append(kids[i])
            else:
                out.extend(kids[i:i + best_k])
                note = _Node(f"^ the {best_k} calls above repeat {best_n} times in total")
                note.is_note = True
                out.append(note)
            i += best_k * best_n
        node.kids = out

    collapse(root)
    lines = []

    def show(node, depth):
        if depth > max_depth:
            return
        if getattr(node, "is_note", False):
            lines.append(f"{'    ' * depth}{node.label}")
            return
        times = f"  x{node.count}" if node.count > 1 else ""
        lines.append(f"{'    ' * depth}{node.label}{times}".ljust(70) + f"  {node.where}")
        for k in node.kids:
            show(k, depth + 1)

    for k in root.kids:
        show(k, 0)
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")
    return result
