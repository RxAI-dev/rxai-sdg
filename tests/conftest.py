"""Pytest bootstrap: make the ``src`` layout importable without installation."""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
_SRC = os.path.abspath(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
