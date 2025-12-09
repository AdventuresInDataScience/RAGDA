"""
Integration tests for Optuna-style API using real benchmark functions.

Tests the Optuna adapter with actual optimization problems from the benchmark suite.
"""

import pytest
import numpy as np
from ragda import create_study


class TestOptunaBenchmarks:
    """Test Optuna-style API with standard benchmark functions."""
    
    def test_ackley_2d(self):
        """Test Optuna API on 2D Ackley function."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            # Ackley function
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = x**2 + y**2
            sum2 = np.cos(c * x) + np.cos(c * y)
            return -a * np.exp(-b * np.sqrt(sum1/2)) - np.exp(sum2/2) + a + np.e
        
        study = create_study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=150)
        
        # Ackley global minimum is at (0, 0) with value 0
        assert study.best_value < 0.5
        assert abs(study.best_params['x']) < 2.0
        assert abs(study.best_params['y']) < 2.0
    
    def test_rosenbrock_2d(self):
        """Test Optuna API on 2D Rosenbrock function."""
        def objective(trial):
            x = trial.suggest_float('x', -2, 2)
            y = trial.suggest_float('y', -2, 2)
            # Rosenbrock function
            return 100.0 * (y - x**2)**2 + (1 - x)**2
        
        study = create_study(random_state=43)
        study.optimize(objective, n_trials=200)
        
        # Rosenbrock global minimum is at (1, 1) with value 0
        assert study.best_value < 10.0
        # Allow wide tolerance since Rosenbrock is hard
        assert abs(study.best_params['x'] - 1.0) < 1.0
        assert abs(study.best_params['y'] - 1.0) < 1.0
    
    def test_rastrigin_2d(self):
        """Test Optuna API on 2D Rastrigin function (multimodal)."""
        def objective(trial):
            x = trial.suggest_float('x', -5.12, 5.12)
            y = trial.suggest_float('y', -5.12, 5.12)
            # Rastrigin function
            A = 10
            return 2 * A + (x**2 - A * np.cos(2 * np.pi * x)) + \
                   (y**2 - A * np.cos(2 * np.pi * y))
        
        study = create_study(random_state=44)
        study.optimize(objective, n_trials=200)
        
        # Rastrigin global minimum is at (0, 0) with value 0
        # It's highly multimodal, so we allow more tolerance
        assert study.best_value < 5.0
        assert abs(study.best_params['x']) < 3.0
        assert abs(study.best_params['y']) < 3.0
    
    def test_sphere_5d(self):
        """Test Optuna API on 5D sphere function."""
        def objective(trial):
            x = [trial.suggest_float(f'x{i}', -10, 10) for i in range(5)]
            return sum(xi**2 for xi in x)
        
        study = create_study(random_state=45)
        study.optimize(objective, n_trials=150)
        
        # Sphere global minimum is at (0,...,0) with value 0
        assert study.best_value < 2.0
        for i in range(5):
            assert abs(study.best_params[f'x{i}']) < 2.0


class TestOptunaParameterTypes:
    """Test Optuna API with different parameter types."""
    
    def test_mixed_continuous_categorical(self):
        """Test Optuna API with mixed continuous and categorical parameters."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            method = trial.suggest_categorical('method', ['A', 'B', 'C'])
            
            # Method affects the objective function shape
            if method == 'A':
                return x**2
            elif method == 'B':
                return (x - 2)**2
            else:
                return (x + 2)**2
        
        study = create_study(random_state=46)
        study.optimize(objective, n_trials=100)
        
        # Should find near-optimal for whichever method was selected
        assert study.best_value < 1.0
        
        # Verify method is one of the allowed values
        assert study.best_params['method'] in ['A', 'B', 'C']
    
    def test_log_scale_parameters(self):
        """Test Optuna API with log-scale parameters."""
        def objective(trial):
            # Learning rate style log-scale parameter
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            x = trial.suggest_float('x', -5, 5)
            
            # Simulate objective where lr affects sensitivity
            return (x - 1)**2 + (np.log10(lr) + 3)**2
        
        study = create_study(random_state=47)
        study.optimize(objective, n_trials=150)
        
        # Should find x≈1 and lr≈1e-3
        assert study.best_value < 2.0
        assert abs(study.best_params['x'] - 1.0) < 1.0
        assert 1e-5 <= study.best_params['learning_rate'] <= 1e-1
    
    def test_integer_parameters(self):
        """Test Optuna API with integer parameters."""
        def objective(trial):
            # Integer parameters (e.g., batch size, num layers)
            n = trial.suggest_int('n', 1, 10)
            m = trial.suggest_int('m', 5, 20, step=5)  # 5, 10, 15, 20
            
            # Optimal at n=5, m=10
            return (n - 5)**2 + ((m - 10) / 5)**2
        
        study = create_study(random_state=48)
        study.optimize(objective, n_trials=100)
        
        # Should find near (5, 10)
        assert study.best_value < 5.0
        assert isinstance(study.best_params['n'], (int, np.integer))
        assert isinstance(study.best_params['m'], (int, np.integer))
        assert study.best_params['m'] in [5, 10, 15, 20]


class TestOptunaDirections:
    """Test Optuna API with different optimization directions."""
    
    def test_minimize_quadratic(self):
        """Test minimization (default)."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 3)**2
        
        study = create_study(direction='minimize', random_state=49)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value < 1.0
        assert abs(study.best_params['x'] - 3) < 1.0
    
    def test_maximize_negative_quadratic(self):
        """Test maximization."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            # Maximum at x=3 with value 0
            return -(x - 3)**2
        
        study = create_study(direction='maximize', random_state=50)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value > -1.0
        assert abs(study.best_params['x'] - 3) < 1.0


class TestOptunaRobustness:
    """Test Optuna API error handling and edge cases."""
    
    def test_noisy_objective(self):
        """Test with noisy objective function."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            # Add noise to sphere function
            noise = np.random.randn() * 0.1
            return x**2 + y**2 + noise
        
        study = create_study(random_state=51)
        study.optimize(objective, n_trials=100)
        
        # Should still converge to near (0, 0) despite noise
        assert study.best_value < 1.0
    
    def test_constrained_search_space(self):
        """Test with narrow search space."""
        def objective(trial):
            # Very narrow bounds
            x = trial.suggest_float('x', -0.1, 0.1)
            y = trial.suggest_float('y', -0.1, 0.1)
            return x**2 + y**2
        
        study = create_study(random_state=52)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value < 0.02
        assert -0.1 <= study.best_params['x'] <= 0.1
        assert -0.1 <= study.best_params['y'] <= 0.1
    
    def test_single_parameter(self):
        """Test with single parameter (1D)."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 4)**2
        
        study = create_study(random_state=53)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value < 1.0
        assert abs(study.best_params['x'] - 4) < 1.0
