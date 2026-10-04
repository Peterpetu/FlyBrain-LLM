"""Phase-3 principle modules: a configurable fast-weight associative memory in which every
fly-derived principle is a switch with a matched control, plus the stream tasks and the
module-level probe that measure it (docs/phase3/EXPERIMENT_PLAN.md)."""
from .memory import MemConfig, FastMemory
from .tasks import capacity_stream, rebind_stream, load_stream, consolidation_stream, reversal_stream, Vocab
from .probe import Probe, evaluate, train_probe

__all__ = ["MemConfig", "FastMemory", "capacity_stream", "rebind_stream", "load_stream", "consolidation_stream",
           "reversal_stream", "Vocab", "Probe", "evaluate", "train_probe"]
