"""
RAGDA Quick Tests - Validates Cython build.

All objective functions defined at MODULE level for pickling/multiprocessing.
"""
import numpy as np
import sys


# =============================================================================
# Module-level objective functions (required for multiprocessing pickle)
# =============================================================================

def sphere_objective(x, y):
    """Simple sphere function: x^2 + y^2"""
    return x**2 + y**2


def categorical_objective(x, category):
    """Categorical test: minimize (x - offset[cat])^2"""
    offsets = {'A': 0, 'B': 1, 'C': 2}
    return (x - offsets[category])**2


# Global data for minibatch test (initialized lazily)
_X_data = None
_y_data = None


def _init_ml_data():
    """Initialize ML test data (global, for pickling)."""
    global _X_data, _y_data
    if _X_data is None:
        np.random.seed(42)
        _X_data = np.random.randn(1000, 2)
        _y_data = _X_data[:, 0] * 2 + _X_data[:, 1] * 3 + np.random.randn(1000) * 0.1


def ml_objective(w1, w2, batch_size=None):
    """Linear regression objective with optional mini-batch."""
    global _X_data, _y_data
    _init_ml_data()
    
    if batch_size and batch_size > 0:
        idx = np.random.choice(len(_X_data), min(batch_size, len(_X_data)), replace=False)
        X, y = _X_data[idx], _y_data[idx]
    else:
        X, y = _X_data, _y_data
    
    pred = X[:, 0] * w1 + X[:, 1] * w2
    return np.mean((pred - y)**2)


def rosenbrock_objective(x):
    """Rosenbrock function for scipy-style test."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


# =============================================================================
# Test Functions
# =============================================================================

def test_basic():
    """Test basic optimization."""
    from ragda import RAGDAOptimizer, core
    
    print(f"RAGDA Core Version: {core.get_version()}")
    print(f"Is Cython: {core.is_cython()}")
    
    space = {
        'x': {'type': 'continuous', 'bounds': [-5.0, 5.0]},
        'y': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
    }
    
    optimizer = RAGDAOptimizer(space, n_workers=2, random_state=42)
    result = optimizer.optimize(sphere_objective, n_trials=50, verbose=True)
    
    print(f"\nBest params: {result.best_params}")
    print(f"Best value: {result.best_value}")
    
    assert result.best_value < 0.5, f"Expected < 0.5, got {result.best_value}"
    print("\n[PASS] Basic test passed!")
    return True


def test_categorical():
    """Test categorical optimization."""
    from ragda import RAGDAOptimizer
    
    space = {
        'x': {'type': 'continuous', 'bounds': [0.0, 3.0]},
        'category': {'type': 'categorical', 'values': ['A', 'B', 'C']}
    }
    
    optimizer = RAGDAOptimizer(space, n_workers=2, random_state=123)
    result = optimizer.optimize(categorical_objective, n_trials=50, verbose=True)
    
    print(f"\nBest params: {result.best_params}")
    print(f"Best value: {result.best_value}")
    
    assert result.best_value < 0.5, f"Expected < 0.5, got {result.best_value}"
    print("\n[PASS] Categorical test passed!")
    return True


def test_minibatch():
    """Test mini-batch evaluation."""
    from ragda import RAGDAOptimizer
    
    # Initialize global data
    _init_ml_data()
    
    space = {
        'w1': {'type': 'continuous', 'bounds': [-5.0, 5.0]},
        'w2': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
    }
    
    optimizer = RAGDAOptimizer(space, n_workers=2, random_state=42)
    result = optimizer.optimize(
        ml_objective, n_trials=100,
        use_minibatch=True, data_size=1000,
        minibatch_schedule='inverse_decay', verbose=True
    )
    
    print(f"\nBest params: {result.best_params}")
    print(f"True weights: w1=2.0, w2=3.0")
    
    assert abs(result.best_params['w1'] - 2.0) < 1.0, "w1 not close to 2.0"
    assert abs(result.best_params['w2'] - 3.0) < 1.0, "w2 not close to 3.0"
    print("\n[PASS] Mini-batch test passed!")
    return True


def test_scipy_style():
    """Test scipy-style interface."""
    from ragda import ragda_optimize
    
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    
    x_best, f_best, info = ragda_optimize(
        rosenbrock_objective, bounds,
        n_trials=100, random_state=42, verbose=True
    )
    
    print(f"\nBest x: {x_best}")
    print(f"Best f: {f_best}")
    
    assert f_best < 5.0, f"Expected < 5.0, got {f_best}"
    print("\n[PASS] Scipy-style test passed!")
    return True


if __name__ == '__main__':
    print("="*70)
    print("RAGDA Cython Quick Tests")
    print("="*70)
    
    tests = [
        ("Basic", test_basic),
        ("Categorical", test_categorical),
        ("Mini-batch", test_minibatch),
        ("Scipy-style", test_scipy_style),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print("="*70)
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)
