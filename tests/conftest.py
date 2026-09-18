"""
Shared pytest setup.

Two jobs:

1. Put the repository root on ``sys.path`` so ``import src...`` works without
   installing the project.
2. Stand in for the heavy optional dependencies (FastF1, scikit-learn, ...)
   when they are not installed.

The stubs only apply to packages that are genuinely *missing*. On a normal
development machine every real package is imported, so the tests exercise the
real code; in a lean CI job the stubs let the pure-logic tests run without
downloading hundreds of megabytes of wheels. No test in this suite depends on
what the stubbed packages actually do.
"""

import sys
import types
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Packages that some modules import at module scope but that none of these
# tests actually call into.
OPTIONAL_PACKAGES = (
    "fastf1",
    "fastf1.plotting",
    "fastf1.core",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "sklearn.model_selection",
    "scipy",
    "scipy.interpolate",
    "xgboost",
    "groq",
    "dotenv",
)


class _StubModule(types.ModuleType):
    """A module whose every attribute resolves to a permissive dummy class."""

    # Marking it as a package lets ``import parent.child`` resolve.
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {
            "__init__": lambda self, *args, **kwargs: None,
            "__call__": lambda self, *args, **kwargs: None,
        })


def _install_stubs():
    """Register a stub for each optional package that is not installed."""
    for name in OPTIONAL_PACKAGES:
        if name in sys.modules:
            continue

        top_level = name.split(".")[0]
        try:
            if find_spec(top_level) is not None:
                continue  # the real package is available - use it
        except (ImportError, ValueError):
            pass

        sys.modules[name] = _StubModule(name)

    if isinstance(sys.modules.get("dotenv"), _StubModule):
        sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None


_install_stubs()
