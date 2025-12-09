"""
Quick smoke tests for API adapters.

Run with: pytest test_adapters_smoke.py -v
"""

import numpy as np
from ragda import create_study, Study, minimize, maximize, RAGDAOptimizer


def test_optuna_style_basic():
    """Test basic Optuna-style API."""
    
    def objective(trial):
        x = trial.suggest_float('x', -5, 5)
        y = trial.suggest_float('y', -5, 5)
        return x**2 + y**2
    
    study = create_study(direction='minimize', random_state=42)
    study.optimize(objective, n_trials=50)
    
    assert study.best_value < 0.5
    assert 'x' in study.best_params
    assert 'y' in study.best_params
    assert abs(study.best_params['x']) < 1.0
    assert abs(study.best_params['y']) < 1.0


def test_optuna_style_mixed_types():
    """Test Optuna-style API with mixed parameter types."""
    
    def objective(trial):
        x = trial.suggest_float('x', -5, 5)
        method = trial.suggest_categorical('method', ['A', 'B', 'C'])
        n = trial.suggest_int('n', 1, 10)
        
        base = x**2
        if method == 'A':
            base *= 1.0
        elif method == 'B':
            base *= 0.5
        else:
            base *= 2.0
        
        return base + n * 0.1
    
    study = create_study(direction='minimize', random_state=42)
    study.optimize(objective, n_trials=50)
    
    assert study.best_value < 1.5
    assert study.best_params['method'] in ['A', 'B', 'C']
    # Should prefer low n values (but don't require n=1)
    assert study.best_params['n'] <= 5


def test_scipy_minimize():
    """Test scipy-style minimize."""
    
    def sphere(x):
        return np.sum(x**2)
    
    result = minimize(
        sphere,
        bounds=[(-5, 5), (-5, 5)],
        options={'maxiter': 50, 'random_state': 42}
    )
    
    assert result.success
    assert result.fun < 0.5
    assert len(result.x) == 2
    assert abs(result.x[0]) < 1.0
    assert abs(result.x[1]) < 1.0


def test_scipy_maximize():
    """Test scipy-style maximize."""
    
    def neg_sphere(x):
        return -np.sum(x**2)
    
    result = maximize(
        neg_sphere,
        bounds=[(-5, 5), (-5, 5)],
        options={'maxiter': 50, 'random_state': 42}
    )
    
    assert result.success
    assert result.fun > -0.5  # Maximizing negative sphere should be close to 0
    assert len(result.x) == 2


def test_scipy_multidim():
    """Test scipy-style with higher dimensions."""
    
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    result = minimize(
        rosenbrock,
        bounds=[(-2, 2)] * 5,
        options={'maxiter': 100, 'random_state': 42}
    )
    
    assert result.success
    assert len(result.x) == 5
    # Rosenbrock is hard, just check it runs without error


def test_optuna_study_properties():
    """Test Study object properties and methods."""
    
    def objective(trial):
        x = trial.suggest_float('x', -5, 5)
        y = trial.suggest_float('y', -5, 5)
        return x**2 + y**2
    
    study = create_study(random_state=42)
    study.optimize(objective, n_trials=20)
    
    # Test properties
    assert isinstance(study.best_value, float)
    assert isinstance(study.best_params, dict)
    assert hasattr(study.best_trial, 'value')
    assert hasattr(study.best_trial, 'params')
    
    # Test trials list (may have fewer trials if converged early)
    assert len(study.trials) > 0
    assert len(study.trials) <= 20
    assert all(hasattr(t, 'value') for t in study.trials)
    assert all(hasattr(t, 'params') for t in study.trials)


def test_native_api_still_works():
    """Verify native RAGDA API still works."""
    
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x, y):
        return x**2 + y**2
    
    opt = RAGDAOptimizer(space, random_state=42)
    result = opt.optimize(objective, n_trials=50)
    
    assert result.best_value < 0.5
    assert 'x' in result.best_params
    assert 'y' in result.best_params


if __name__ == '__main__':
    print("Running adapter smoke tests...")
    
    test_optuna_style_basic()
    print("[PASS] Optuna basic")
    
    test_optuna_style_mixed_types()
    print("[PASS] Optuna mixed types")
    
    test_scipy_minimize()
    print("[PASS] Scipy minimize")
    
    test_scipy_maximize()
    print("[PASS] Scipy maximize")
    
    test_scipy_multidim()
    print("[PASS] Scipy multidim")
    
    test_optuna_study_properties()
    print("[PASS] Optuna Study properties")
    
    test_native_api_still_works()
    print("[PASS] Native API")
    
    print("\nAll adapter smoke tests passed!")
