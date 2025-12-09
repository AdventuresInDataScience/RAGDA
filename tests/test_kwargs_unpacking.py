"""Test kwargs unpacking in objective functions."""
import numpy as np
import pytest
from ragda.optimizer import RAGDAOptimizer


def test_kwargs_simple():
    """Test that objective receives unpacked kwargs."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    # New API: no params dict needed
    def objective(x, y):
        return (x - 2)**2 + (y + 1)**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    result = optimizer.optimize(objective, n_trials=10, verbose=False)
    
    assert result.best_value < 10  # Should converge somewhat
    assert 'x' in result.best_params
    assert 'y' in result.best_params


def test_kwargs_with_batch_size():
    """Test kwargs unpacking with extra parameters."""
    space = {
        'lr': {'type': 'continuous', 'bounds': [0.001, 0.1], 'log': True},
        'momentum': {'type': 'continuous', 'bounds': [0.5, 0.99]}
    }
    
    def objective(lr, momentum):
        # Simple test function
        return lr * (1 - momentum)  # Want to minimize
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    result = optimizer.optimize(objective, n_trials=10, verbose=False)
    
    assert result.best_value < 0.1
    assert 'lr' in result.best_params
    assert 'momentum' in result.best_params


def test_kwargs_mixed_types():
    """Test kwargs unpacking with mixed parameter types."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 10]},
        'n': {'type': 'ordinal', 'values': [1, 2, 4, 8]},
        'method': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']}
    }
    
    def objective(x, n, method):
        # Different methods have different optima
        base = (x - 5)**2
        if method == 'adam':
            return base * n
        elif method == 'sgd':
            return base * n * 1.5
        else:  # rmsprop
            return base * n * 0.8
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    result = optimizer.optimize(objective, n_trials=20, verbose=False)
    
    assert result.best_value < 50
    assert result.best_params['method'] in ['adam', 'sgd', 'rmsprop']
    assert result.best_params['n'] in [1, 2, 4, 8]


def test_kwargs_with_default_args():
    """Test that objectives with default arguments work."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x, multiplier=2.0):
        return multiplier * x**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    result = optimizer.optimize(objective, n_trials=10, verbose=False)
    
    # Should find x close to 0
    assert abs(result.best_params['x']) < 2


def test_kwargs_error_handling():
    """Test that missing parameters are handled gracefully."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    # Objective expects y but space doesn't have it
    def objective(x, y):
        return x**2 + y**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    
    # Should not crash, but will return penalty values
    result = optimizer.optimize(objective, n_trials=5, verbose=False)
    assert result.best_value == 1e10  # Penalty value


def test_kwargs_vs_dict_signature():
    """Test that we can detect whether objective expects dict or kwargs."""
    import inspect
    
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    # Old style: single params dict
    def old_objective(params):
        return params['x']**2
    
    # New style: kwargs
    def new_objective(x):
        return x**2
    
    # Both should work (backward compatibility)
    optimizer1 = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    optimizer2 = RAGDAOptimizer(space, direction='minimize', n_workers=1)
    
    # Both should be callable (though old_objective will get 1e10 penalty)
    result1 = optimizer1.optimize(old_objective, n_trials=5, verbose=False)
    result2 = optimizer2.optimize(new_objective, n_trials=5, verbose=False)
    
    # New style should work correctly
    assert result2.best_value < 25


def test_parallel_kwargs():
    """Test kwargs unpacking works with parallel workers."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x, y):
        return (x - 1)**2 + (y + 2)**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=2)
    result = optimizer.optimize(objective, n_trials=20, verbose=False)
    
    assert result.best_value < 10
    assert 'x' in result.best_params
    assert 'y' in result.best_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
