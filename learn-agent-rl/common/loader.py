"""Import a lesson file by its path.

Lesson folders start with digits (`06_agents`), and Python can't `import 06_agents...`
because a module name can't start with a digit. This helper loads the file directly:

    from common.loader import load_lesson
    agent = load_lesson("06_agents/agent_controller.py")
    agent.MockLLM()
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_lesson(relpath):
    path = ROOT / relpath
    name = "lesson_" + path.stem
    if name in sys.modules:                 # load each file only once
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
