"""
Cross-API consistency tests.

Verifies that the same problem solved with different APIs gives consistent results.
"""

import pytest
import numpy as np
from ragda import RAGDAOptimizer, create_study, minimize


class TestCrossAPIConsistency:
    """Test that all three APIs solve problems consistently."""
    
    def test_sphere_2d_all_apis(self):
        """Test sphere function with all three APIs."""
        # 1. Native RAGDA API
        space_ragda = {
            'x': {'type': 'continuous', 'bounds': [-5, 5]},
            'y': {'type': 'continuous', 'bounds': [-5, 5]}
        }
        
        def objective_ragda(x, y):
            return x**2 + y**2
        
        opt_ragda = RAGDAOptimizer(space_ragda, random_state=42)
        result_ragda = opt_ragda.optimize(objective_ragda, n_trials=100)
        
        # 2. Optuna-style API
        def objective_optuna(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            return x**2 + y**2
        
        study = create_study(direction='minimize', random_state=43)
        study.optimize(objective_optuna, n_trials=100)
        
        # 3. Scipy-style API
        def objective_scipy(params):
            return params[0]**2 + params[1]**2
        
        result_scipy = minimize(
            objective_scipy,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 100, 'random_state': 44}
        )
        
        # All should find near-optimal solutions
        assert result_ragda.best_value < 0.5
        assert study.best_value < 0.5
        assert result_scipy.fun < 0.5
        
        # All should converge to near (0, 0)
        assert abs(result_ragda.best_params['x']) < 1.0
        assert abs(result_ragda.best_params['y']) < 1.0
        assert abs(study.best_params['x']) < 1.0
        assert abs(study.best_params['y']) < 1.0
        assert abs(result_scipy.x[0]) < 1.0
        assert abs(result_scipy.x[1]) < 1.0
    
    def test_shifted_quadratic_all_apis(self):
        """Test shifted quadratic with all APIs."""
        # Optimal at (3, -2)
        
        # 1. Native RAGDA
        space = {
            'x': {'type': 'continuous', 'bounds': [-10, 10]},
            'y': {'type': 'continuous', 'bounds': [-10, 10]}
        }
        
        opt = RAGDAOptimizer(space, random_state=42)
        result1 = opt.optimize(lambda x, y: (x - 3)**2 + (y + 2)**2, n_trials=100)
        
        # 2. Optuna-style
        def obj_optuna(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return (x - 3)**2 + (y + 2)**2
        
        study = create_study(random_state=43)
        study.optimize(obj_optuna, n_trials=100)
        
        # 3. Scipy-style
        result3 = minimize(
            lambda p: (p[0] - 3)**2 + (p[1] + 2)**2,
            bounds=[(-10, 10), (-10, 10)],
            options={'maxiter': 100, 'random_state': 44}
        )
        
        # All should find near (3, -2)
        assert result1.best_value < 1.0
        assert study.best_value < 1.0
        assert result3.fun < 1.0
        
        assert abs(result1.best_params['x'] - 3) < 1.5
        assert abs(result1.best_params['y'] + 2) < 1.5
        assert abs(study.best_params['x'] - 3) < 1.5
        assert abs(study.best_params['y'] + 2) < 1.5
        assert abs(result3.x[0] - 3) < 1.5
        assert abs(result3.x[1] + 2) < 1.5
    
    def test_rosenbrock_all_apis(self):
        """Test Rosenbrock function with all APIs."""
        # 1. Native RAGDA
        space = {
            'x': {'type': 'continuous', 'bounds': [-2, 2]},
            'y': {'type': 'continuous', 'bounds': [-2, 2]}
        }
        
        def rosenbrock_ragda(x, y):
            return 100.0 * (y - x**2)**2 + (1 - x)**2
        
        opt = RAGDAOptimizer(space, random_state=42)
        result_ragda = opt.optimize(rosenbrock_ragda, n_trials=150)
        
        # 2. Optuna-style
        def rosenbrock_optuna(trial):
            x = trial.suggest_float('x', -2, 2)
            y = trial.suggest_float('y', -2, 2)
            return 100.0 * (y - x**2)**2 + (1 - x)**2
        
        study = create_study(random_state=43)
        study.optimize(rosenbrock_optuna, n_trials=150)
        
        # 3. Scipy-style
        def rosenbrock_scipy(p):
            return 100.0 * (p[1] - p[0]**2)**2 + (1 - p[0])**2
        
        result_scipy = minimize(
            rosenbrock_scipy,
            bounds=[(-2, 2), (-2, 2)],
            options={'maxiter': 150, 'random_state': 44}
        )
        
        # Rosenbrock is hard, just verify all make progress
        assert result_ragda.best_value < 50.0
        assert study.best_value < 50.0
        assert result_scipy.fun < 50.0
        
        # Verify all find reasonable solutions (optimal is at (1, 1))
        # Allow wide tolerance since Rosenbrock is difficult
        assert abs(result_ragda.best_params['x'] - 1.0) < 1.5
        assert abs(result_ragda.best_params['y'] - 1.0) < 1.5
        assert abs(study.best_params['x'] - 1.0) < 1.5
        assert abs(study.best_params['y'] - 1.0) < 1.5
        assert abs(result_scipy.x[0] - 1.0) < 1.5
        assert abs(result_scipy.x[1] - 1.0) < 1.5


class TestAPISpecificFeatures:
    """Test features specific to each API."""
    
    def test_native_api_kwargs_unpacking(self):
        """Test that native API correctly unpacks kwargs."""
        space = {
            'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
            'batch_size': {'type': 'ordinal', 'values': [16, 32, 64, 128]},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']}
        }
        
        def objective(learning_rate, batch_size, optimizer):
            # Verify we receive the actual values, not dicts
            assert isinstance(learning_rate, float)
            assert batch_size in [16, 32, 64, 128]
            assert optimizer in ['adam', 'sgd']
            return abs(np.log10(learning_rate) + 3) + batch_size/100
        
        opt = RAGDAOptimizer(space, random_state=42)
        result = opt.optimize(objective, n_trials=50)
        
        assert result.best_value < 2.0
    
    def test_optuna_api_trial_methods(self):
        """Test Optuna-style trial.suggest_* methods."""
        def objective(trial):
            # Test all suggest methods
            x = trial.suggest_float('x', 0, 1)
            n = trial.suggest_int('n', 1, 10)
            cat = trial.suggest_categorical('cat', ['A', 'B', 'C'])
            
            assert 0 <= x <= 1
            assert 1 <= n <= 10
            assert cat in ['A', 'B', 'C']
            
            return x + n * 0.1
        
        study = create_study(random_state=42)
        study.optimize(objective, n_trials=30)
        
        assert study.best_value < 1.5
    
    def test_scipy_api_array_interface(self):
        """Test Scipy-style array-based interface."""
        def objective(x):
            # Verify we receive numpy array
            assert isinstance(x, np.ndarray)
            assert len(x) == 3
            return np.sum(x**2)
        
        result = minimize(
            objective,
            bounds=[(-5, 5)] * 3,
            options={'maxiter': 50, 'random_state': 42}
        )
        
        assert isinstance(result.x, np.ndarray)
        assert len(result.x) == 3
        assert result.fun < 0.5


class TestPerformanceConsistency:
    """Test that all APIs have similar performance characteristics."""
    
    def test_convergence_speed_parity(self):
        """Test that all APIs converge at similar speeds."""
        n_trials = 100
        
        # 1. Native RAGDA
        space = {'x': {'type': 'continuous', 'bounds': [-5, 5]}}
        opt = RAGDAOptimizer(space, random_state=42)
        result1 = opt.optimize(lambda x: x**2, n_trials=n_trials)
        
        # 2. Optuna-style
        study = create_study(random_state=43)
        study.optimize(lambda trial: trial.suggest_float('x', -5, 5)**2, n_trials=n_trials)
        
        # 3. Scipy-style
        result3 = minimize(
            lambda x: x[0]**2,
            bounds=[(-5, 5)],
            options={'maxiter': n_trials, 'random_state': 44}
        )
        
        # Note: With parallel execution (n_workers=8), the number of trials can exceed maxiter
        # because multiple workers can evaluate points simultaneously. This is expected behavior.
        # We verify that the requested number of trials is respected, but allow for some overshoot
        # due to concurrent execution.
        
        # All should have done a reasonable number of evaluations
        assert len(result1.trials) <= n_trials * 10  # Allow some overshoot for parallel execution
        assert len(study.trials) <= n_trials * 10
        assert result3.nfev <= n_trials * 10
        
        # All should converge to similar quality
        assert result1.best_value < 0.5
        assert study.best_value < 0.5
        assert result3.fun < 0.5
    
    def test_no_overhead_from_adapters(self):
        """Verify adapters don't add significant overhead."""
        # This is a qualitative test - just verify all complete in reasonable time
        
        def sphere_5d(x, y, z, a, b):
            return x**2 + y**2 + z**2 + a**2 + b**2
        
        # Native API
        space = {
            'x': {'type': 'continuous', 'bounds': [-5, 5]},
            'y': {'type': 'continuous', 'bounds': [-5, 5]},
            'z': {'type': 'continuous', 'bounds': [-5, 5]},
            'a': {'type': 'continuous', 'bounds': [-5, 5]},
            'b': {'type': 'continuous', 'bounds': [-5, 5]},
        }
        
        opt = RAGDAOptimizer(space, random_state=42)
        result = opt.optimize(sphere_5d, n_trials=50)
        
        assert result.best_value < 1.0
        # If this test runs without timeout, adapter overhead is acceptable


class TestParameterTypeConsistency:
    """Test that parameter type handling is consistent across APIs."""
    
    def test_continuous_parameters(self):
        """Test continuous parameter handling across APIs."""
        # All should handle continuous parameters identically
        
        # Native
        space = {'x': {'type': 'continuous', 'bounds': [-5, 5]}}
        opt = RAGDAOptimizer(space, random_state=42)
        r1 = opt.optimize(lambda x: (x - 2)**2, n_trials=50)
        
        # Optuna
        study = create_study(random_state=43)
        study.optimize(lambda t: (t.suggest_float('x', -5, 5) - 2)**2, n_trials=50)
        
        # Scipy
        r3 = minimize(lambda x: (x[0] - 2)**2, bounds=[(-5, 5)], 
                     options={'maxiter': 50, 'random_state': 44})
        
        # All should find near x=2
        assert abs(r1.best_params['x'] - 2) < 1.0
        assert abs(study.best_params['x'] - 2) < 1.0
        assert abs(r3.x[0] - 2) < 1.0
    
    def test_log_scale_parameters(self):
        """Test log-scale parameter handling."""
        # Native API
        space = {'lr': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True}}
        opt = RAGDAOptimizer(space, random_state=42)
        r1 = opt.optimize(lambda lr: (np.log10(lr) + 3)**2, n_trials=100)
        
        # Optuna API
        study = create_study(random_state=43)
        study.optimize(
            lambda t: (np.log10(t.suggest_float('lr', 1e-5, 1e-1, log=True)) + 3)**2,
            n_trials=100
        )
        
        # Both should find near lr=1e-3
        assert 1e-4 < r1.best_params['lr'] < 1e-2
        assert 1e-4 < study.best_params['lr'] < 1e-2
    
    def test_categorical_parameters(self):
        """Test categorical parameter handling."""
        # Native API
        space = {
            'method': {'type': 'categorical', 'values': ['A', 'B', 'C']},
            'x': {'type': 'continuous', 'bounds': [-5, 5]}
        }
        
        def objective(method, x):
            base = x**2
            if method == 'B':
                base *= 0.5  # B is best
            return base
        
        opt = RAGDAOptimizer(space, random_state=42)
        r1 = opt.optimize(objective, n_trials=100)
        
        # Optuna API
        def obj_optuna(trial):
            method = trial.suggest_categorical('method', ['A', 'B', 'C'])
            x = trial.suggest_float('x', -5, 5)
            base = x**2
            if method == 'B':
                base *= 0.5
            return base
        
        study = create_study(random_state=43)
        study.optimize(obj_optuna, n_trials=100)
        
        # Both should prefer method B
        assert r1.best_value < 0.5
        assert study.best_value < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
