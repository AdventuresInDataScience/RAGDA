"""Test constraint integration with optimizer."""
import numpy as np
import pytest
from ragda.optimizer import RAGDAOptimizer


def test_simple_constraint():
    """Test optimization with a simple numeric constraint."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x, y):
        return (x - 2)**2 + (y - 3)**2
    
    # Constraint: x + y <= 0 (forces negative quadrant)
    constraints = ['x + y <= 0']
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=50, verbose=False, constraints=constraints)
    
    # Best solution should satisfy constraint
    assert result.best_params['x'] + result.best_params['y'] <= 0.01  # Small tolerance
    # Should converge to feasible region
    assert result.best_value < 50


def test_multiple_constraints():
    """Test optimization with multiple constraints."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 10]},
        'y': {'type': 'continuous', 'bounds': [0, 10]}
    }
    
    def objective(x, y):
        return -(x + y)  # Maximize x + y
    
    constraints = [
        'x + y <= 8',  # Linear constraint
        'x * y >= 4'   # Nonlinear constraint
    ]
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=100, verbose=False, constraints=constraints)
    
    # Check constraints are satisfied (with tolerance for optimization noise)
    x, y = result.best_params['x'], result.best_params['y']
    assert x + y <= 8.2, f"Constraint x + y <= 8 violated: {x} + {y} = {x+y}"
    assert x * y >= 3.8, f"Constraint x * y >= 4 violated: {x} * {y} = {x*y}"


def test_categorical_constraint():
    """Test constraints with categorical variables."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 10]},
        'method': {'type': 'categorical', 'values': ['A', 'B', 'C']}
    }
    
    def objective(x, method):
        # Method A is better
        if method == 'A':
            return x**2
        else:
            return x**2 + 10
    
    # Only allow method A
    constraints = ["method == 'A'"]
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=30, verbose=False, constraints=constraints)
    
    assert result.best_params['method'] == 'A'
    assert abs(result.best_params['x']) < 2  # Should find x near 0


def test_implication_constraint():
    """Test implication constraints (if-then)."""
    space = {
        'use_feature': {'type': 'categorical', 'values': [True, False]},
        'feature_param': {'type': 'continuous', 'bounds': [0, 1]}
    }
    
    def objective(use_feature, feature_param):
        if use_feature:
            return (feature_param - 0.7)**2
        else:
            return 0.1  # Small constant when not using feature
    
    # If use_feature is False, then feature_param must be 0
    constraints = ['not use_feature -> feature_param == 0']
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=30, verbose=False, constraints=constraints)
    
    use_feat = result.best_params['use_feature']
    feat_param = result.best_params['feature_param']
    
    # Check implication is satisfied
    if not use_feat:
        assert abs(feat_param) < 0.1


def test_constraint_with_math_functions():
    """Test constraints using math functions."""
    space = {
        'x': {'type': 'continuous', 'bounds': [1, 5]},
        'y': {'type': 'continuous', 'bounds': [1, 5]}
    }
    
    def objective(x, y):
        return (x - 3)**2 + (y - 3)**2
    
    # Constraint using sqrt
    constraints = ['sqrt(x) + sqrt(y) <= 4']
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=50, verbose=False, constraints=constraints)
    
    x, y = result.best_params['x'], result.best_params['y']
    # Verify constraint (with tolerance)
    assert np.sqrt(x) + np.sqrt(y) <= 4.1


def test_no_constraints():
    """Test that optimization works without constraints (backward compatibility)."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x, y):
        return x**2 + y**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=30, verbose=False)
    
    # Should find minimum near (0, 0)
    assert abs(result.best_params['x']) < 1
    assert abs(result.best_params['y']) < 1
    assert result.best_value < 1


def test_infeasible_constraint():
    """Test behavior with impossible constraints."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 1]}
    }
    
    def objective(x):
        return x**2
    
    # Impossible constraint (x always in [0,1])
    constraints = ['x > 5']
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=20, verbose=False, constraints=constraints)
    
    # All evaluations should return penalty
    assert result.best_value == 1e10


def test_constraint_penalty_custom():
    """Test custom constraint penalty value."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 10]}
    }
    
    def objective(x):
        return x**2
    
    constraints = ['x > 5']
    custom_penalty = 999.0
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(
        objective, 
        n_trials=20, 
        verbose=False, 
        constraints=constraints,
        constraint_penalty=custom_penalty
    )
    
    # Should try values > 5 and get penalty
    # Best might be near 5 or penalty value
    assert result.best_value <= custom_penalty + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
