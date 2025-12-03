"""
RAGDA - Roughly Approximated Gradient Descent Algorithm

A high-performance derivative-free optimization library for mixed variable types
with parallel worker exploration and ADAM-based gradient approximation.

This is a pure Cython/C implementation for maximum performance.

High-dimensional optimization is AUTOMATIC - when the problem has 100+ continuous
dimensions, RAGDA automatically uses dimensionality reduction (Kernel PCA, 
random projections) to optimize in a reduced space.
"""

__version__ = "2.1.0"

# Import main optimizer components
from .result import OptimizationResult, Trial
from .space import SearchSpace, Parameter

# Import Cython core - required, no fallback
from . import core

# Import optimizer after space (it depends on it)
from .optimizer import RAGDAOptimizer, ragda_optimize

# Try to import high-dim core (may not be built yet)
try:
    from . import highdim_core
    HIGHDIM_AVAILABLE = True
except ImportError:
    HIGHDIM_AVAILABLE = False

# Import high-dimensional utilities (for advanced users)
from .highdim import (
    HighDimRAGDAOptimizer,
    highdim_ragda_optimize,
    DimensionalityReducer
)

__all__ = [
    # Main classes - just use RAGDAOptimizer, it handles high-dim automatically
    "RAGDAOptimizer",
    "OptimizationResult",
    "Trial",
    "SearchSpace",
    "Parameter",
    # Convenience functions
    "ragda_optimize",
    # Core modules
    "core",
    # Version and availability
    "__version__",
    "HIGHDIM_AVAILABLE",
    # Advanced/explicit high-dim access (for users who want direct control)
    "HighDimRAGDAOptimizer",
    "highdim_ragda_optimize",
    "DimensionalityReducer",
]
