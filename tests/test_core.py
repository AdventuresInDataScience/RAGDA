"""
Unit tests for RAGDA core Cython module.
"""
import numpy as np
import pytest
from ragda import core


class TestCoreBasics:
    """Test basic core functionality."""
    
    def test_version(self):
        """Test version string."""
        version = core.get_version()
        assert isinstance(version, str)
        assert "cython" in version.lower()
    
    def test_is_cython(self):
        """Test Cython detection."""
        assert core.is_cython() is True


class TestOptimizeWorkerCore:
    """Test optimize_worker_core function."""
    
    @pytest.fixture
    def simple_objective(self):
        """Simple sphere objective."""
        def objective(x_cont, x_cat, minibatch_size=-1):
            return np.sum(x_cont ** 2)
        return objective
    
    @pytest.fixture
    def basic_schedules(self):
        """Basic schedules for 50 iterations."""
        max_iter = 50
        t = np.arange(max_iter, dtype=np.float64)
        progress = t / max_iter
        
        lambda_sched = (10 + 40 * np.exp(-2 * progress)).astype(np.int32)
        mu_sched = np.maximum(2, (lambda_sched * 0.3).astype(np.int32))
        sigma_sched = (0.06 + 0.24 * np.exp(-3 * progress)).astype(np.float64)
        minibatch_sched = np.full(max_iter, -1, dtype=np.int32)
        
        return {
            'max_iter': max_iter,
            'lambda': lambda_sched,
            'mu': mu_sched,
            'sigma': sigma_sched,
            'minibatch': minibatch_sched
        }
    
    def test_continuous_only(self, simple_objective, basic_schedules):
        """Test with continuous variables only."""
        x0_cont = np.zeros(3, dtype=np.float64)
        x0_cat = np.array([], dtype=np.int32)
        cat_n_values = np.array([], dtype=np.int32)
        bounds = np.array([[-5, 5], [-5, 5], [-5, 5]], dtype=np.float64)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=simple_objective,
            max_iter=basic_schedules['max_iter'],
            lambda_schedule=basic_schedules['lambda'],
            mu_schedule=basic_schedules['mu'],
            sigma_schedule=basic_schedules['sigma'],
            minibatch_schedule=basic_schedules['minibatch'],
            use_minibatch=False,
            top_n_fraction=0.5,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=42,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        
        assert len(x_best_cont) == 3
        assert len(x_best_cat) == 0
        assert f_best < 1.0  # Should optimize close to 0
        assert 'fitness' in history
        assert 'sigma' in history
        assert len(history['fitness']) > 0
    
    def test_categorical_only(self, basic_schedules):
        """Test with categorical variables only."""
        def cat_objective(x_cont, x_cat, minibatch_size=-1):
            # Optimal is category 0
            return float(x_cat[0])
        
        x0_cont = np.array([], dtype=np.float64)
        x0_cat = np.array([2], dtype=np.int32)  # Start at worst
        cat_n_values = np.array([3], dtype=np.int32)
        bounds = np.array([], dtype=np.float64).reshape(0, 2)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=cat_objective,
            max_iter=basic_schedules['max_iter'],
            lambda_schedule=basic_schedules['lambda'],
            mu_schedule=basic_schedules['mu'],
            sigma_schedule=basic_schedules['sigma'],
            minibatch_schedule=basic_schedules['minibatch'],
            use_minibatch=False,
            top_n_fraction=0.5,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=42,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        
        assert len(x_best_cont) == 0
        assert len(x_best_cat) == 1
        assert f_best == 0.0  # Should find optimal category
    
    def test_mixed_variables(self, basic_schedules):
        """Test with both continuous and categorical variables."""
        def mixed_objective(x_cont, x_cat, minibatch_size=-1):
            # Minimize x^2 + category_offset
            offsets = [0, 1, 2]
            return x_cont[0]**2 + offsets[x_cat[0]]
        
        x0_cont = np.array([2.0], dtype=np.float64)
        x0_cat = np.array([1], dtype=np.int32)
        cat_n_values = np.array([3], dtype=np.int32)
        bounds = np.array([[-5, 5]], dtype=np.float64)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=mixed_objective,
            max_iter=basic_schedules['max_iter'],
            lambda_schedule=basic_schedules['lambda'],
            mu_schedule=basic_schedules['mu'],
            sigma_schedule=basic_schedules['sigma'],
            minibatch_schedule=basic_schedules['minibatch'],
            use_minibatch=False,
            top_n_fraction=0.5,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=42,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        
        assert len(x_best_cont) == 1
        assert len(x_best_cat) == 1
        assert x_best_cat[0] == 0  # Should find best category
        assert f_best < 1.0  # Should be reasonably close to 0
    
    def test_long_optimization(self, simple_objective):
        """Test with many iterations to verify stability."""
        max_iter = 300
        t = np.arange(max_iter, dtype=np.float64)
        progress = t / max_iter
        
        lambda_sched = (10 + 40 * np.exp(-2 * progress)).astype(np.int32)
        mu_sched = np.maximum(2, (lambda_sched * 0.3).astype(np.int32))
        sigma_sched = (0.06 + 0.24 * np.exp(-3 * progress)).astype(np.float64)
        minibatch_sched = np.full(max_iter, -1, dtype=np.int32)
        
        x0_cont = np.zeros(5, dtype=np.float64)
        x0_cat = np.array([], dtype=np.int32)
        cat_n_values = np.array([], dtype=np.int32)
        bounds = np.array([[-10, 10]] * 5, dtype=np.float64)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=simple_objective,
            max_iter=max_iter,
            lambda_schedule=lambda_sched,
            mu_schedule=mu_sched,
            sigma_schedule=sigma_sched,
            minibatch_schedule=minibatch_sched,
            use_minibatch=False,
            top_n_fraction=0.5,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=42,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        assert f_best < 0.01  # Should converge well
    
    @pytest.mark.parametrize("seed", [1, 42, 123, 999, 12345])
    def test_different_seeds(self, simple_objective, basic_schedules, seed):
        """Test reproducibility with different seeds."""
        x0_cont = np.zeros(2, dtype=np.float64)
        x0_cat = np.array([], dtype=np.int32)
        cat_n_values = np.array([], dtype=np.int32)
        bounds = np.array([[-5, 5], [-5, 5]], dtype=np.float64)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=simple_objective,
            max_iter=basic_schedules['max_iter'],
            lambda_schedule=basic_schedules['lambda'],
            mu_schedule=basic_schedules['mu'],
            sigma_schedule=basic_schedules['sigma'],
            minibatch_schedule=basic_schedules['minibatch'],
            use_minibatch=False,
            top_n_fraction=0.5,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=seed,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        assert f_best < 1.0  # Should still optimize
    
    @pytest.mark.parametrize("top_n", [0.2, 0.4, 0.6, 0.8, 1.0])
    def test_different_top_n(self, simple_objective, basic_schedules, top_n):
        """Test with different top_n fractions."""
        x0_cont = np.zeros(2, dtype=np.float64)
        x0_cat = np.array([], dtype=np.int32)
        cat_n_values = np.array([], dtype=np.int32)
        bounds = np.array([[-5, 5], [-5, 5]], dtype=np.float64)
        
        result = core.optimize_worker_core(
            x0_cont=x0_cont,
            x0_cat=x0_cat,
            cat_n_values=cat_n_values,
            bounds=bounds,
            evaluate_fitness_func=simple_objective,
            max_iter=basic_schedules['max_iter'],
            lambda_schedule=basic_schedules['lambda'],
            mu_schedule=basic_schedules['mu'],
            sigma_schedule=basic_schedules['sigma'],
            minibatch_schedule=basic_schedules['minibatch'],
            use_minibatch=False,
            top_n_fraction=top_n,
            alpha=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            shrink_factor=0.9,
            shrink_patience=10,
            shrink_threshold=1e-6,
            use_improvement_weights=True,
            random_seed=42,
            worker_id=0,
            sync_queue=None,
            sync_event=None,
            sync_frequency=0
        )
        
        x_best_cont, x_best_cat, f_best, history = result
        assert f_best < 1.0
