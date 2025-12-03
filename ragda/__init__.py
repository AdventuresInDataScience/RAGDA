"""
RAGDA - Roughly Approximated Gradient Descent Algorithm

A high-performance derivative-free optimization library for mixed variable types
with parallel worker exploration and ADAM-based gradient approximation.

This is a pure Cython/C implementation for maximum performance.
"""

__version__ = "2.0.0"

# Import main optimizer components
from .result import OptimizationResult, Trial
from .space import SearchSpace, Parameter

# Import Cython core - required, no fallback
from . import core

# Import optimizer after space (it depends on it)
from .optimizer import RAGDAOptimizer, ragda_optimize

__all__ = [
    # Main classes
    "RAGDAOptimizer",
    "OptimizationResult",
    "Trial",
    "SearchSpace",
    "Parameter",
    # Convenience function
    "ragda_optimize",
    # Core module
    "core",
    # Version
    "__version__",
]
