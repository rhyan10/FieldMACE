import warnings as _warnings

from .__version__ import __version__

# Known, harmless warnings from PyTorch/e3nn that would otherwise flood every run (dozens of lines per
# model build): e3nn's TorchScript-compiled layers, and PyTorch's deprecation notice for TorchScript,
# which FieldMACE still uses for compiled models and the OpenMM TorchForce. Other warnings are untouched.
_warnings.filterwarnings("ignore", message=r"The TorchScript type system doesn't support instance-level annotations")
_warnings.filterwarnings("ignore", message=r"`torch\.jit\.[a-z_]+` is deprecated", category=FutureWarning)
