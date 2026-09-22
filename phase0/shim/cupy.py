"""A deliberately non-functional stand-in for cupy.

The official `dfloat11` package imports cupy at module scope and builds a
`RawModule` from decode.ptx on import, so it cannot be imported at all without
cupy present -- even to *compress*, which is pure CPU work.

Installing cupy-cuda12x costs ~2.5 GB of CUDA wheels on a machine with no GPU.
Instead this stub satisfies the import and raises on any real use. That is the
stronger option: if a compression run completes with this on sys.path, the
compression path provably never touched the GPU. If it ever does, the run fails
loudly rather than silently taking a different route.

Anything that actually needs cupy -- inference, check_correctness=True, and the
Phase 5 GPU verification path -- must run on a machine with the real package.
"""


class _GpuReached(RuntimeError):
    pass


def _refuse(what):
    raise _GpuReached(
        f"cupy stub: {what} was called, so this code path is NOT CPU-only. "
        "Run it on a machine with real cupy, or keep check_correctness=False."
    )


class _Function:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        _refuse(f"kernel {self._name!r}")


class RawModule:
    """Records the PTX path; never parses or loads it."""

    def __init__(self, *, path=None, code=None, **kw):
        self.path = path
        self.code = code

    def get_function(self, name):
        return _Function(name)


class _Device:
    def __init__(self, index=None):
        self.index = index

    def __enter__(self):
        _refuse("cuda.Device context")

    def __exit__(self, *a):
        return False


class cuda:
    Device = _Device


def __getattr__(name):
    # Any other cupy attribute reached at runtime is an error we want to see.
    def _missing(*a, **k):
        _refuse(f"cupy.{name}")
    return _missing
