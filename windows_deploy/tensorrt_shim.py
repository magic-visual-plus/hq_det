from tensorrt_bindings.tensorrt import *
import tensorrt_bindings.tensorrt as _t
__version__ = "10.13.3.9"
for _n in dir(_t):
    if not _n.startswith("_"):
        globals().setdefault(_n, getattr(_t, _n))
