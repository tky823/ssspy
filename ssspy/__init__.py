try:
    from .io import wavread, wavwrite
except ModuleNotFoundError:
    # to avoid module not found error during installation
    # e.g. numpy is not found in io.py
    pass

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0.1.7"

__all__ = ["__version__", "wavread", "wavwrite"]
