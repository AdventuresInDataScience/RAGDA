"""
Unit tests for RAGDA optimizer module.
"""
import numpy as np
import pytest
from ragda import RAGDAOptimizer, ragda_optimize
from ragda.result import OptimizationResult


class TestRAGDAOptimizer:
    """Test RAGDAOptimizer class."""
    
    @pytest.fixture
    def simple_space(self):
        """Simple 2D continuous space."""
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'y', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
    
    @pytest.fixture
    def mixed_space(self):
        """Mixed continuous and categorical space."""
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'cat', 'type': 'categorical', 'values': ['A', 'B', 'C']},
        ]
    
    def test_init_basic(self, simple_space):
        """Test basic initialization."""
        opt = RAGDAOptimizer(simple_space)
        assert opt.n_workers >= 1
        assert opt.direction == 'minimize'
    
    def test_init_with_workers(self, simple_space):
        """Test initialization with specific worker count."""
        opt = RAGDAOptimizer(simple_space, n_workers=4)
        assert opt.n_workers == 4
    
    def test_init_maximize(self, simple_space):
        """Test initialization for maximization."""
        opt = RAGDAOptimizer(simple_space, direction='maximize')
        assert opt.direction == 'maximize'
    
    def test_optimize_sphere(self, simple_space):
        """Test optimization on sphere function."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(simple_space, n_workers=2, random_state=42)
        result = opt.optimize(sphere, n_trials=50, verbose=False)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_value < 0.1
        assert 'x' in result.best_params
        assert 'y' in result.best_params
    
    def test_optimize_with_categorical(self, mixed_space):
        """Test optimization with categorical variables."""
        def objective(params):
            offsets = {'A': 0, 'B': 1, 'C': 2}
            return params['x']**2 + offsets[params['cat']]
        
        opt = RAGDAOptimizer(mixed_space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=50, verbose=False)
        
        assert result.best_params['cat'] == 'A'  # Best category
        assert result.best_value < 0.5
    
    def test_maximize(self, simple_space):
        """Test maximization direction."""
        def negative_sphere(params):
            return -(params['x']**2 + params['y']**2)  # Minimum at 0
        
        opt = RAGDAOptimizer(simple_space, direction='maximize', n_workers=2, random_state=42)
        result = opt.optimize(negative_sphere, n_trials=50, verbose=False)
        
        assert result.best_value > -0.1  # Should maximize to near 0
    
    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
    def test_different_worker_counts(self, simple_space, n_workers):
        """Test with different worker counts."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(simple_space, n_workers=n_workers, random_state=42)
        result = opt.optimize(sphere, n_trials=30, verbose=False)
        
        assert result.best_value < 1.0
    
    @pytest.mark.parametrize("n_trials", [10, 25, 50, 100])
    def test_different_trial_counts(self, simple_space, n_trials):
        """Test with different trial counts."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(simple_space, n_workers=2, random_state=42)
        result = opt.optimize(sphere, n_trials=n_trials, verbose=False)
        
        assert result.best_value is not None
    
    def test_custom_x0(self, simple_space):
        """Test with custom starting point."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        x0 = {'x': 1.0, 'y': 1.0}
        opt = RAGDAOptimizer(simple_space, n_workers=2, random_state=42)
        result = opt.optimize(sphere, n_trials=50, x0=x0, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_early_stopping(self, simple_space):
        """Test early stopping behavior."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(simple_space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=200, 
            early_stop_threshold=1e-6,
            early_stop_patience=20,
            verbose=False
        )
        
        # Should find a reasonably low value
        assert result.best_value < 0.01
    
    def test_shrinking(self, simple_space):
        """Test shrinking mechanism."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(simple_space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=100, 
            shrink_factor=0.9,
            shrink_patience=5,
            verbose=False
        )
        
        assert result.best_value < 0.01


class TestMinibatch:
    """Test minibatch/curriculum learning functionality."""
    
    @pytest.fixture
    def space(self):
        return [
            {'name': 'w1', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'w2', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
    
    def test_minibatch_optimization(self, space):
        """Test optimization with minibatch."""
        # Simulated data
        np.random.seed(42)
        X = np.random.randn(1000, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(1000) * 0.1
        
        def mse_objective(params, batch_size=-1):
            w = np.array([params['w1'], params['w2']])
            if batch_size > 0 and batch_size < len(X):
                idx = np.random.choice(len(X), batch_size, replace=False)
                pred = X[idx] @ w
                return np.mean((pred - y[idx])**2)
            else:
                pred = X @ w
                return np.mean((pred - y)**2)
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            mse_objective,
            n_trials=50,
            use_minibatch=True,
            minibatch_start=50,
            minibatch_end=500,
            verbose=False
        )
        
        # Should find weights close to [2, 3]
        assert abs(result.best_params['w1'] - 2.0) < 1.0
        assert abs(result.best_params['w2'] - 3.0) < 1.0


class TestRagdaOptimize:
    """Test scipy-style interface."""
    
    def test_basic_scipy_style(self):
        """Test basic scipy-style optimization."""
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)
        
        x_best, f_best, info = ragda_optimize(
            rosenbrock, 
            bounds, 
            n_trials=50,
            random_state=42
        )
        
        assert len(x_best) == 2
        assert f_best < 1.0
        assert 'n_trials' in info  # Check expected key in optimization_params
    
    def test_scipy_with_x0(self):
        """Test scipy-style with starting point."""
        def sphere(x):
            return np.sum(x**2)
        
        bounds = np.array([[-5, 5], [-5, 5], [-5, 5]], dtype=np.float64)
        x0 = np.array([1.0, 1.0, 1.0])
        
        x_best, f_best, info = ragda_optimize(
            sphere, 
            bounds, 
            x0=x0,
            n_trials=50,
            random_state=42
        )
        
        assert f_best < 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_dimension(self):
        """Test 1D optimization."""
        space = [{'name': 'x', 'type': 'continuous', 'bounds': [-10.0, 10.0]}]
        
        def objective(params):
            return (params['x'] - 3)**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=50, verbose=False)
        
        assert abs(result.best_params['x'] - 3.0) < 0.5
    
    def test_high_dimension(self):
        """Test high-dimensional optimization."""
        n_dims = 10
        space = [
            {'name': f'x{i}', 'type': 'continuous', 'bounds': [-5.0, 5.0]}
            for i in range(n_dims)
        ]
        
        def sphere(params):
            return sum(params[f'x{i}']**2 for i in range(n_dims))
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(objective=sphere, n_trials=100, verbose=False)
        
        assert result.best_value < 1.0
    
    def test_many_categories(self):
        """Test with many categorical values."""
        space = [
            {'name': 'cat', 'type': 'categorical', 'values': [f'val_{i}' for i in range(20)]},
        ]
        
        def objective(params):
            # Best is val_0
            return int(params['cat'].split('_')[1])
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=50, verbose=False)
        
        # Should find one of the best categories
        assert result.best_value < 5
    
    def test_exception_in_objective(self):
        """Test handling of exceptions in objective."""
        space = [{'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]}]
        
        call_count = [0]
        def flaky_objective(params):
            call_count[0] += 1
            if call_count[0] % 10 == 0:
                raise ValueError("Random failure")
            return params['x']**2
        
        opt = RAGDAOptimizer(space, n_workers=1, random_state=42)
        # Should not crash, should handle exceptions gracefully
        result = opt.optimize(flaky_objective, n_trials=30, verbose=False)
        
        assert result.best_value is not None
    
    def test_ordinal_variables(self):
        """Test ordinal variable handling."""
        space = [
            {'name': 'level', 'type': 'ordinal', 'values': [1, 2, 3, 4, 5]},
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
        
        def objective(params):
            # Best at level=3, x=0
            return (params['level'] - 3)**2 + params['x']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=50, verbose=False)
        
        assert result.best_params['level'] in [2, 3, 4]  # Should be near 3


class TestLambdaSchedules:
    """Test lambda (population size) schedule configurations."""
    
    @pytest.fixture
    def space(self):
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'y', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
    
    def test_default_lambda(self, space):
        """Test with default lambda settings."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(sphere, n_trials=50, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_custom_lambda_start_end(self, space):
        """Test with custom lambda_start and lambda_end."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_start=100,
            lambda_end=20,
            verbose=False
        )
        
        assert result.best_value < 0.5
    
    def test_small_lambda(self, space):
        """Test with small lambda values."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_start=10,
            lambda_end=5,
            verbose=False
        )
        
        assert result.best_value is not None
    
    def test_large_lambda(self, space):
        """Test with large lambda values."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=30,
            lambda_start=200,
            lambda_end=50,
            verbose=False
        )
        
        assert result.best_value < 0.5
    
    def test_constant_lambda(self, space):
        """Test with constant lambda (start = end)."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_start=30,
            lambda_end=30,
            verbose=False
        )
        
        assert result.best_value < 0.5
    
    @pytest.mark.parametrize("decay_rate", [1.0, 3.0, 5.0, 10.0])
    def test_different_lambda_decay_rates(self, space, decay_rate):
        """Test with different lambda decay rates."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_decay_rate=decay_rate,
            verbose=False
        )
        
        assert result.best_value < 1.0


class TestInitialGuesses:
    """Test initial guess (x0) configurations."""
    
    @pytest.fixture
    def space(self):
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-10.0, 10.0]},
            {'name': 'y', 'type': 'continuous', 'bounds': [-10.0, 10.0]},
            {'name': 'cat', 'type': 'categorical', 'values': ['A', 'B', 'C']},
        ]
    
    def test_no_initial_guess(self, space):
        """Test with no initial guess (x0=None)."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(objective, n_trials=50, x0=None, verbose=False)
        
        assert result.best_value < 1.0
    
    def test_full_initial_guess(self, space):
        """Test with complete initial guess for all params."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return (params['x'] - 5)**2 + (params['y'] + 3)**2 + offset
        
        x0 = {'x': 4.0, 'y': -2.0, 'cat': 'A'}
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=50, x0=x0, verbose=False)
        
        assert result.best_value < 5.0
    
    def test_partial_continuous_guess(self):
        """Test with partial initial guess (only some continuous params)."""
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [-10.0, 10.0]},
            {'name': 'y', 'type': 'continuous', 'bounds': [-10.0, 10.0]},
            {'name': 'z', 'type': 'continuous', 'bounds': [-10.0, 10.0]},
        ]
        
        def objective(params):
            return (params['x'] - 5)**2 + (params['y'] + 3)**2 + params['z']**2
        
        # Full guess required by current API, but we test near-optimal partial knowledge
        x0 = {'x': 5.0, 'y': -3.0, 'z': 0.0}  # Perfect starting point
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=30, x0=x0, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_multiple_initial_guesses(self, space):
        """Test with list of multiple initial guesses."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        x0_list = [
            {'x': 1.0, 'y': 1.0, 'cat': 'A'},
            {'x': -1.0, 'y': -1.0, 'cat': 'A'},
        ]
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(objective, n_trials=50, x0=x0_list, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_more_guesses_than_workers(self, space):
        """Test with more initial guesses than workers."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        x0_list = [
            {'x': 0.5, 'y': 0.5, 'cat': 'A'},
            {'x': -0.5, 'y': -0.5, 'cat': 'A'},
            {'x': 0.3, 'y': -0.3, 'cat': 'A'},
            {'x': -0.3, 'y': 0.3, 'cat': 'A'},
            {'x': 0.1, 'y': 0.1, 'cat': 'A'},
        ]
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)  # Only 2 workers
        result = opt.optimize(objective, n_trials=50, x0=x0_list, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_fewer_guesses_than_workers(self, space):
        """Test with fewer initial guesses than workers."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        x0_list = [
            {'x': 0.5, 'y': 0.5, 'cat': 'A'},
        ]
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)  # 4 workers, 1 guess
        result = opt.optimize(objective, n_trials=50, x0=x0_list, verbose=False)
        
        assert result.best_value < 0.5
    
    def test_initial_guess_at_optimum(self, space):
        """Test starting exactly at the optimum."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        x0 = {'x': 0.0, 'y': 0.0, 'cat': 'A'}  # Optimal point
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=30, x0=x0, verbose=False)
        
        assert result.best_value < 0.1
    
    def test_initial_guess_far_from_optimum(self, space):
        """Test starting far from the optimum."""
        def objective(params):
            offset = {'A': 0, 'B': 1, 'C': 2}[params['cat']]
            return params['x']**2 + params['y']**2 + offset
        
        x0 = {'x': 9.9, 'y': -9.9, 'cat': 'C'}  # Far from optimal (0, 0, A)
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(objective, n_trials=100, x0=x0, verbose=False)
        
        # Should still make progress
        assert result.best_value < 50.0


class TestOptimizerArgs:
    """Test various optimizer argument combinations."""
    
    @pytest.fixture
    def space(self):
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'y', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
    
    # Sigma (step size) configurations
    @pytest.mark.parametrize("sigma_init", [0.1, 0.3, 0.5, 0.8])
    def test_different_sigma_init(self, space, sigma_init):
        """Test with different initial sigma values."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            sigma_init=sigma_init,
            verbose=False
        )
        
        assert result.best_value is not None
    
    @pytest.mark.parametrize("sigma_final_fraction", [0.1, 0.2, 0.5, 0.8])
    def test_different_sigma_final_fraction(self, space, sigma_final_fraction):
        """Test with different sigma final fractions."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            sigma_final_fraction=sigma_final_fraction,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    @pytest.mark.parametrize("sigma_schedule", ['exponential', 'linear', 'cosine'])
    def test_different_sigma_schedules(self, space, sigma_schedule):
        """Test different sigma decay schedules."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            sigma_decay_schedule=sigma_schedule,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    # Top-N weighting configurations
    @pytest.mark.parametrize("top_n_min,top_n_max", [
        (0.1, 0.5),
        (0.2, 1.0),
        (0.5, 0.5),  # All workers same
        (0.1, 1.0),  # Wide range
    ])
    def test_different_top_n_ranges(self, space, top_n_min, top_n_max):
        """Test with different top_n fraction ranges."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            top_n_min=top_n_min,
            top_n_max=top_n_max,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    @pytest.mark.parametrize("weight_decay", [0.8, 0.9, 0.95, 0.99, 1.0])
    def test_different_weight_decay(self, space, weight_decay):
        """Test with different weight decay values."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            weight_decay=weight_decay,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    def test_no_improvement_weights(self, space):
        """Test with use_improvement_weights=False."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            use_improvement_weights=False,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    # ADAM configurations
    @pytest.mark.parametrize("lr", [0.0001, 0.001, 0.01, 0.1])
    def test_different_adam_learning_rates(self, space, lr):
        """Test with different ADAM learning rates."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            adam_learning_rate=lr,
            verbose=False
        )
        
        assert result.best_value is not None
    
    @pytest.mark.parametrize("beta1,beta2", [
        (0.9, 0.999),
        (0.8, 0.99),
        (0.95, 0.9999),
    ])
    def test_different_adam_betas(self, space, beta1, beta2):
        """Test with different ADAM beta values."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=2, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            adam_beta1=beta1,
            adam_beta2=beta2,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    # Sync frequency
    @pytest.mark.parametrize("sync_freq", [10, 50, 100, 500])
    def test_different_sync_frequencies(self, space, sync_freq):
        """Test with different worker sync frequencies."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=100,
            sync_frequency=sync_freq,
            verbose=False
        )
        
        assert result.best_value < 0.5
    
    # Combined configurations
    def test_aggressive_exploration(self, space):
        """Test configuration favoring exploration."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_start=100,
            lambda_end=50,
            sigma_init=0.5,
            sigma_final_fraction=0.3,
            top_n_min=0.5,
            top_n_max=1.0,
            shrink_factor=0.95,
            verbose=False
        )
        
        assert result.best_value < 1.0
    
    def test_aggressive_exploitation(self, space):
        """Test configuration favoring exploitation."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere, 
            n_trials=50,
            lambda_start=20,
            lambda_end=10,
            sigma_init=0.2,
            sigma_final_fraction=0.1,
            top_n_min=0.1,
            top_n_max=0.5,
            shrink_factor=0.8,
            shrink_patience=5,
            verbose=False
        )
        
        assert result.best_value < 0.5
    
    def test_all_default_args(self, space):
        """Test with all default arguments."""
        def sphere(params):
            return params['x']**2 + params['y']**2
        
        opt = RAGDAOptimizer(space)
        result = opt.optimize(sphere, verbose=False)
        
        assert result.best_value < 0.01  # Should converge well with 1000 trials


class TestConstructorArgs:
    """Test RAGDAOptimizer constructor arguments."""
    
    @pytest.fixture
    def space(self):
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
    
    def test_default_constructor(self, space):
        """Test with default constructor args."""
        opt = RAGDAOptimizer(space)
        assert opt.direction == 'minimize'
        assert opt.n_workers >= 1
    
    def test_minimize_direction(self, space):
        """Test with explicit minimize direction."""
        opt = RAGDAOptimizer(space, direction='minimize')
        assert opt.direction == 'minimize'
    
    def test_maximize_direction(self, space):
        """Test with maximize direction."""
        opt = RAGDAOptimizer(space, direction='maximize')
        assert opt.direction == 'maximize'
    
    @pytest.mark.parametrize("n_workers", [1, 2, 4, 8, 16])
    def test_different_n_workers(self, space, n_workers):
        """Test with different worker counts."""
        opt = RAGDAOptimizer(space, n_workers=n_workers)
        assert opt.n_workers == n_workers
    
    @pytest.mark.parametrize("seed", [None, 0, 42, 12345])
    def test_different_random_states(self, space, seed):
        """Test with different random states."""
        opt = RAGDAOptimizer(space, random_state=seed)
        assert opt.random_state == seed
    
    def test_highdim_threshold_parameter(self, space):
        """Test highdim_threshold constructor parameter."""
        opt = RAGDAOptimizer(space, highdim_threshold=50)
        assert opt.highdim_threshold == 50
        
        opt2 = RAGDAOptimizer(space, highdim_threshold=200)
        assert opt2.highdim_threshold == 200
    
    def test_variance_threshold_parameter(self, space):
        """Test variance_threshold constructor parameter."""
        opt = RAGDAOptimizer(space, variance_threshold=0.90)
        assert opt.variance_threshold == 0.90
        
        opt2 = RAGDAOptimizer(space, variance_threshold=0.80)
        assert opt2.variance_threshold == 0.80
    
    def test_reduction_method_parameter(self, space):
        """Test reduction_method constructor parameter."""
        for method in ['auto', 'kernel_pca', 'incremental_pca', 'random_projection']:
            opt = RAGDAOptimizer(space, reduction_method=method)
            assert opt.reduction_method == method
    
    def test_default_highdim_parameters(self, space):
        """Test default values for high-dim parameters."""
        opt = RAGDAOptimizer(space)
        assert opt.highdim_threshold == 100
        assert opt.variance_threshold == 0.95
        assert opt.reduction_method == 'auto'


class TestOutputValidation:
    """
    Comprehensive tests validating optimizer output correctness.
    
    Tests that best_params are:
    - Correctly descaled from unit [0,1] space to original bounds
    - Within declared bounds for continuous params
    - Valid values for categorical/ordinal params
    - Consistent with best_value
    """
    
    @pytest.fixture
    def continuous_space(self):
        """Continuous params with various bound ranges."""
        return [
            {'name': 'a', 'type': 'continuous', 'bounds': [-100.0, 100.0]},
            {'name': 'b', 'type': 'continuous', 'bounds': [0.0, 1.0]},
            {'name': 'c', 'type': 'continuous', 'bounds': [1e-6, 1e-1], 'log': True},
            {'name': 'd', 'type': 'continuous', 'bounds': [0.001, 1000.0]},
            {'name': 'e', 'type': 'continuous', 'bounds': [-1.5, 1.5]},
        ]
    
    @pytest.fixture
    def categorical_space(self):
        """Categorical params with various value types."""
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'cat_str', 'type': 'categorical', 'values': ['A', 'B', 'C', 'D']},
            {'name': 'cat_int', 'type': 'categorical', 'values': [1, 2, 3, 4, 5]},
            {'name': 'cat_float', 'type': 'categorical', 'values': [0.1, 0.5, 1.0, 2.0]},
        ]
    
    @pytest.fixture
    def ordinal_space(self):
        """Ordinal params with ordered values."""
        return [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'ord_int', 'type': 'ordinal', 'values': [1, 2, 4, 8, 16, 32]},
            {'name': 'ord_float', 'type': 'ordinal', 'values': [0.001, 0.01, 0.1, 1.0]},
        ]
    
    @pytest.fixture
    def mixed_space(self):
        """Mixed space with all param types."""
        return [
            {'name': 'lr', 'type': 'continuous', 'bounds': [1e-5, 1.0], 'log': True},
            {'name': 'momentum', 'type': 'continuous', 'bounds': [0.0, 0.99]},
            {'name': 'batch_size', 'type': 'ordinal', 'values': [16, 32, 64, 128, 256]},
            {'name': 'optimizer', 'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
        ]
    
    def _optimize(self, obj, space, n_trials=100, **kwargs):
        """Helper to run optimization with the class-based API."""
        opt = RAGDAOptimizer(space, **kwargs)
        return opt.optimize(obj, n_trials=n_trials, verbose=False)
    
    def test_continuous_params_within_bounds(self, continuous_space):
        """Test that all continuous params are within declared bounds."""
        def obj(params):
            return sum(v**2 if isinstance(v, (int, float)) else 0 for v in params.values())
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        for param_def in continuous_space:
            name = param_def['name']
            lower, upper = param_def['bounds']
            value = result.best_params[name]
            
            assert isinstance(value, (int, float, np.number)), \
                f"Param '{name}' should be numeric, got {type(value)}"
            assert lower <= value <= upper, \
                f"Param '{name}'={value} not in bounds [{lower}, {upper}]"
    
    def test_log_scale_params_positive(self, continuous_space):
        """Test that log-scale params are positive and within bounds."""
        def obj(params):
            return sum(v**2 if isinstance(v, (int, float)) else 0 for v in params.values())
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        # Find log-scale param (c)
        for param_def in continuous_space:
            if param_def.get('log', False):
                name = param_def['name']
                value = result.best_params[name]
                lower, upper = param_def['bounds']
                
                assert value > 0, f"Log-scale param '{name}'={value} should be positive"
                assert lower <= value <= upper, \
                    f"Log-scale param '{name}'={value} not in bounds [{lower}, {upper}]"
    
    def test_categorical_params_valid_values(self, categorical_space):
        """Test that categorical params have valid values from the list."""
        def obj(params):
            # Different values for different categories
            cat_penalty = 0
            if params['cat_str'] == 'A':
                cat_penalty = 0
            elif params['cat_str'] == 'B':
                cat_penalty = 1
            else:
                cat_penalty = 2
            return params['x']**2 + cat_penalty
        
        result = self._optimize(obj, categorical_space, n_trials=100)
        
        for param_def in categorical_space:
            if param_def['type'] == 'categorical':
                name = param_def['name']
                value = result.best_params[name]
                valid_values = param_def['values']
                
                assert value in valid_values, \
                    f"Categorical param '{name}'={value} not in valid values {valid_values}"
    
    def test_ordinal_params_valid_values(self, ordinal_space):
        """Test that ordinal params have valid values from the list."""
        def obj(params):
            return params['x']**2 + params['ord_int'] * 0.01
        
        result = self._optimize(obj, ordinal_space, n_trials=100)
        
        for param_def in ordinal_space:
            if param_def['type'] == 'ordinal':
                name = param_def['name']
                value = result.best_params[name]
                valid_values = param_def['values']
                
                assert value in valid_values, \
                    f"Ordinal param '{name}'={value} not in valid values {valid_values}"
    
    def test_mixed_space_all_valid(self, mixed_space):
        """Test that all params in mixed space are valid."""
        def obj(params):
            return params['lr'] * 1000 + (1 - params['momentum']) + params['batch_size'] / 256
        
        result = self._optimize(obj, mixed_space, n_trials=200)
        
        # Validate each param type
        for param_def in mixed_space:
            name = param_def['name']
            value = result.best_params[name]
            ptype = param_def['type']
            
            if ptype == 'continuous':
                lower, upper = param_def['bounds']
                # Allow small floating-point tolerance for log-scale params
                eps = 1e-15 * max(abs(lower), abs(upper), 1.0)
                assert lower - eps <= value <= upper + eps, \
                    f"Continuous param '{name}'={value} not in bounds [{lower}, {upper}]"
            elif ptype in ('categorical', 'ordinal'):
                assert value in param_def['values'], \
                    f"{ptype.capitalize()} param '{name}'={value} not in values"
    
    def test_best_params_consistent_with_best_value(self, continuous_space):
        """Test that best_value is the objective evaluated at best_params."""
        def obj(params):
            return sum((params[p['name']])**2 for p in continuous_space)
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        # Re-evaluate objective at best_params
        recalculated = obj(result.best_params)
        
        # Should be very close (might differ slightly due to full sample re-evaluation)
        assert abs(result.best_value - recalculated) < 1e-6, \
            f"best_value={result.best_value} != obj(best_params)={recalculated}"
    
    def test_result_object_has_all_fields(self, continuous_space):
        """Test that result object has all required fields populated."""
        def obj(params):
            return sum(v**2 if isinstance(v, (int, float)) else 0 for v in params.values())
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        # Check all expected fields
        assert hasattr(result, 'best_params'), "Missing best_params"
        assert hasattr(result, 'best_value'), "Missing best_value"
        assert hasattr(result, 'best_trial'), "Missing best_trial"
        assert hasattr(result, 'best_worker_id'), "Missing best_worker_id"
        assert hasattr(result, 'trials'), "Missing trials"
        assert hasattr(result, 'n_trials'), "Missing n_trials"
        assert hasattr(result, 'n_workers'), "Missing n_workers"
        assert hasattr(result, 'direction'), "Missing direction"
        
        # Check types
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, (int, float, np.number))
        assert isinstance(result.n_trials, int)
        assert isinstance(result.n_workers, int)
        assert result.direction in ('minimize', 'maximize')
    
    def test_result_best_params_has_all_param_names(self, mixed_space):
        """Test that best_params contains all declared parameters."""
        def obj(params):
            return 1.0
        
        result = self._optimize(obj, mixed_space, n_trials=50)
        
        expected_names = {p['name'] for p in mixed_space}
        actual_names = set(result.best_params.keys())
        
        assert expected_names == actual_names, \
            f"best_params keys {actual_names} != expected {expected_names}"
    
    def test_trials_all_have_valid_params(self, mixed_space):
        """Test that all trials in history have valid params."""
        def obj(params):
            return params['lr'] + params['momentum']
        
        result = self._optimize(obj, mixed_space, n_trials=100)
        
        for trial in result.trials:
            params = trial.params
            
            for param_def in mixed_space:
                name = param_def['name']
                ptype = param_def['type']
                value = params[name]
                
                if ptype == 'continuous':
                    lower, upper = param_def['bounds']
                    # Allow small floating-point tolerance for log-scale params
                    eps = 1e-15 * max(abs(lower), abs(upper), 1.0)
                    assert lower - eps <= value <= upper + eps, \
                        f"Trial {trial.trial_id}: '{name}'={value} not in [{lower}, {upper}]"
                elif ptype in ('categorical', 'ordinal'):
                    assert value in param_def['values'], \
                        f"Trial {trial.trial_id}: '{name}'={value} not in values"
    
    def test_descaling_asymmetric_bounds(self):
        """Test descaling with asymmetric bounds like [-100, 50]."""
        space = [
            {'name': 'a', 'type': 'continuous', 'bounds': [-100.0, 50.0]},
            {'name': 'b', 'type': 'continuous', 'bounds': [-0.5, 10.0]},
        ]
        
        # Objective with known optimum at bounds edge
        def obj(params):
            # Optimum at a=-100, b=-0.5
            return (params['a'] + 100)**2 + (params['b'] + 0.5)**2
        
        result = self._optimize(obj, space, n_trials=200)
        
        # Check bounds
        assert -100.0 <= result.best_params['a'] <= 50.0
        assert -0.5 <= result.best_params['b'] <= 10.0
    
    def test_maximize_direction_output(self):
        """Test that maximize direction returns correctly."""
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [0.0, 10.0]},
        ]
        
        # Objective: maximize x (optimal at x=10)
        def obj(params):
            return params['x']
        
        opt = RAGDAOptimizer(space, direction='maximize')
        result = opt.optimize(obj, n_trials=100, verbose=False)
        
        # best_value should be high (near 10)
        assert result.best_value > 5.0, f"Maximize should find high value, got {result.best_value}"
        assert result.direction == 'maximize'
        assert 0.0 <= result.best_params['x'] <= 10.0
    
    def test_single_param_output(self):
        """Test with single parameter space."""
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ]
        
        def obj(params):
            return params['x']**2
        
        result = self._optimize(obj, space, n_trials=100)
        
        assert 'x' in result.best_params
        assert -5.0 <= result.best_params['x'] <= 5.0
        assert result.best_value >= 0  # x^2 >= 0
    
    def test_categorical_only_space(self):
        """Test space with only categorical params."""
        space = [
            {'name': 'a', 'type': 'categorical', 'values': ['X', 'Y', 'Z']},
            {'name': 'b', 'type': 'categorical', 'values': [1, 2, 3]},
        ]
        
        def obj(params):
            # X+1 is best
            val = 0
            if params['a'] == 'X':
                val -= 1
            if params['b'] == 1:
                val -= 1
            return val
        
        result = self._optimize(obj, space, n_trials=100)
        
        assert result.best_params['a'] in ['X', 'Y', 'Z']
        assert result.best_params['b'] in [1, 2, 3]
        # Should find optimal
        assert result.best_params['a'] == 'X'
        assert result.best_params['b'] == 1
    
    def test_ordinal_only_space(self):
        """Test space with only ordinal params."""
        space = [
            {'name': 'n_layers', 'type': 'ordinal', 'values': [1, 2, 4, 8]},
            {'name': 'n_units', 'type': 'ordinal', 'values': [32, 64, 128, 256]},
        ]
        
        def obj(params):
            # Prefer smaller
            return params['n_layers'] + params['n_units'] / 256
        
        result = self._optimize(obj, space, n_trials=100)
        
        assert result.best_params['n_layers'] in [1, 2, 4, 8]
        assert result.best_params['n_units'] in [32, 64, 128, 256]
    
    def test_wide_bounds_descaling(self):
        """Test descaling with very wide bounds."""
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [-1e6, 1e6]},
            {'name': 'y', 'type': 'continuous', 'bounds': [1e-10, 1e10], 'log': True},
        ]
        
        def obj(params):
            return abs(params['x']) + np.log10(params['y']) / 10
        
        result = self._optimize(obj, space, n_trials=200)
        
        # Check bounds
        assert -1e6 <= result.best_params['x'] <= 1e6
        assert 1e-10 <= result.best_params['y'] <= 1e10
    
    def test_narrow_bounds_precision(self):
        """Test that narrow bounds maintain precision."""
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [0.5, 0.500001]},
        ]
        
        def obj(params):
            return (params['x'] - 0.5000005)**2
        
        result = self._optimize(obj, space, n_trials=100)
        
        # Must be within narrow bounds
        assert 0.5 <= result.best_params['x'] <= 0.500001
    
    def test_best_trial_consistency(self, continuous_space):
        """Test that best_trial matches best_params and best_value."""
        def obj(params):
            return sum(v**2 if isinstance(v, (int, float)) else 0 for v in params.values())
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        # best_trial should have matching values
        assert result.best_trial.value == result.best_value or \
               abs(result.best_trial.value - result.best_value) < 1e-6, \
               f"best_trial.value={result.best_trial.value} != best_value={result.best_value}"
        
        # best_trial.params should match best_params
        for name in result.best_params:
            trial_val = result.best_trial.params[name]
            best_val = result.best_params[name]
            
            if isinstance(trial_val, (int, float)):
                assert abs(trial_val - best_val) < 1e-10, \
                    f"best_trial.params['{name}']={trial_val} != best_params['{name}']={best_val}"
            else:
                assert trial_val == best_val
    
    def test_trials_df_structure(self, mixed_space):
        """Test that trials_df has correct structure."""
        def obj(params):
            return 1.0
        
        result = self._optimize(obj, mixed_space, n_trials=50)
        
        df = result.trials_df
        
        # Should have required columns
        assert 'trial_id' in df.columns
        assert 'worker_id' in df.columns
        assert 'iteration' in df.columns
        assert 'value' in df.columns
        
        # Should have all param columns
        for param_def in mixed_space:
            assert param_def['name'] in df.columns, \
                f"Missing column for param '{param_def['name']}'"
    
    def test_n_trials_matches_history(self, continuous_space):
        """Test that n_trials count matches trials list length."""
        def obj(params):
            return 1.0
        
        result = self._optimize(obj, continuous_space, n_trials=100)
        
        # n_trials should match length of trials list
        assert result.n_trials == len(result.trials), \
            f"n_trials={result.n_trials} != len(trials)={len(result.trials)}"
