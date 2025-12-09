"""Integration tests for the complete new API: dict space + kwargs + constraints."""
import numpy as np
import pytest
from ragda.optimizer import RAGDAOptimizer


def test_full_api_integration():
    """Test all new features together: dict space, kwargs, and constraints."""
    # Dict-based space definition
    space = {
        'learning_rate': {'type': 'continuous', 'bounds': [0.001, 0.1], 'log': True},
        'batch_size': {'type': 'ordinal', 'values': [16, 32, 64, 128]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']}
    }
    
    # Objective using kwargs (no params dict needed!)
    def objective(learning_rate, batch_size, optimizer):
        # Simulate ML training
        base_loss = learning_rate * batch_size / 100
        
        # Different optimizers have different performance
        if optimizer == 'adam':
            return base_loss * 0.8
        elif optimizer == 'sgd':
            return base_loss * 1.2
        else:  # rmsprop
            return base_loss
    
    # Constraints using string syntax
    constraints = [
        'learning_rate * batch_size <= 0.5',  # Numeric constraint
        "optimizer != 'sgd'"                   # Categorical constraint
    ]
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(
        objective, 
        n_trials=50, 
        verbose=False,
        constraints=constraints
    )
    
    # Check all features worked
    lr = result.best_params['learning_rate']
    bs = result.best_params['batch_size']
    opt = result.best_params['optimizer']
    
    # Verify constraints
    assert lr * bs <= 0.51, f"Constraint violated: {lr} * {bs} = {lr * bs}"
    assert opt != 'sgd', f"Constraint violated: optimizer is {opt}"
    
    # Should prefer adam
    assert opt == 'adam' or opt == 'rmsprop'
    
    # Should find low loss
    assert result.best_value < 0.5


def test_real_world_hyperparameter_tuning():
    """Realistic ML hyperparameter optimization example."""
    space = {
        'lr': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
        'n_layers': {'type': 'ordinal', 'values': [2, 4, 6, 8]},
        'hidden_dim': {'type': 'ordinal', 'values': [32, 64, 128, 256]},
        'dropout': {'type': 'continuous', 'bounds': [0.0, 0.5]},
        'activation': {'type': 'categorical', 'values': ['relu', 'tanh', 'gelu']}
    }
    
    def train_model(lr, n_layers, hidden_dim, dropout, activation):
        # Simulated training loss
        complexity = n_layers * hidden_dim / 1000
        regularization = dropout * 2
        
        # Activation function effects
        act_factor = {'relu': 1.0, 'tanh': 1.2, 'gelu': 0.9}[activation]
        
        base_loss = lr * 100 + complexity - regularization
        return base_loss * act_factor
    
    constraints = [
        'n_layers * hidden_dim <= 1500',  # Model capacity limit
        'dropout < 0.3',                   # Regularization limit
        "activation == 'gelu' -> hidden_dim >= 64"  # GELU needs more capacity
    ]
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=2, random_state=42)
    result = optimizer.optimize(
        train_model,
        n_trials=100,
        verbose=False,
        constraints=constraints
    )
    
    # Verify constraints
    assert result.best_params['n_layers'] * result.best_params['hidden_dim'] <= 1500
    assert result.best_params['dropout'] < 0.3
    if result.best_params['activation'] == 'gelu':
        assert result.best_params['hidden_dim'] >= 64
    
    # Should find good configuration
    assert result.best_value < 10


def test_constraint_penalty_effectiveness():
    """Test that constraint penalties guide search away from infeasible regions."""
    space = {
        'x': {'type': 'continuous', 'bounds': [0, 10]},
        'y': {'type': 'continuous', 'bounds': [0, 10]}
    }
    
    def objective(x, y):
        # Optimum is at (10, 10) without constraints
        return -(x + y)
    
    # Constrain to x + y <= 10
    constraints = ['x + y <= 10']
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(
        objective,
        n_trials=100,
        verbose=False,
        constraints=constraints
    )
    
    # Should find points near the constraint boundary
    assert result.best_params['x'] + result.best_params['y'] <= 10.1
    # Should be close to boundary (optimal constrained solution)
    assert result.best_params['x'] + result.best_params['y'] >= 9.0


def test_empty_constraints_list():
    """Test that empty constraints list is handled correctly."""
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]}
    }
    
    def objective(x):
        return x**2
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(objective, n_trials=20, verbose=False, constraints=[])
    
    # Should work normally
    assert abs(result.best_params['x']) < 1


def test_complex_constraint_expression():
    """Test complex constraint with multiple operations."""
    space = {
        'a': {'type': 'continuous', 'bounds': [0, 5]},
        'b': {'type': 'continuous', 'bounds': [0, 5]},
        'c': {'type': 'continuous', 'bounds': [0, 5]}
    }
    
    def objective(a, b, c):
        return (a - 2)**2 + (b - 2)**2 + (c - 2)**2
    
    # Complex constraint: a + b + c <= 8 AND a * b >= 2
    constraints = [
        'a + b + c <= 8',
        'a * b >= 2'
    ]
    
    optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=1, random_state=42)
    result = optimizer.optimize(
        objective,
        n_trials=100,
        verbose=False,
        constraints=constraints
    )
    
    a = result.best_params['a']
    b = result.best_params['b']
    c = result.best_params['c']
    
    assert a + b + c <= 8.1
    assert a * b >= 1.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
