"""Import the official dfloat11 package on a GPU-less machine.

Puts the cupy stub on sys.path first. Import this instead of `dfloat11`
directly. See phase0/shim/cupy.py for why.
"""
import sys
from pathlib import Path

_SHIM = str(Path(__file__).resolve().parent / "shim")
if _SHIM not in sys.path:
    sys.path.insert(0, _SHIM)

import cupy as _cupy  # noqa: E402  (the stub)

assert "phase0/shim" in _cupy.__file__, f"real cupy got imported from {_cupy.__file__}"

from dfloat11 import compress_model  # noqa: E402,F401
from dfloat11 import dfloat11_utils  # noqa: E402,F401
