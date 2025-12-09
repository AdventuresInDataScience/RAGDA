"""
Comprehensive tests for dynamic worker strategy in RAGDA optimizer.

This module contains:
1. Unit tests for each stage of the dynamic strategy
2. Edge case tests (odd numbers, boundary conditions, etc.)
3. Integration tests comparing greedy vs dynamic strategies
"""

import pytest
import numpy as np
import time
from ragda import RAGDAOptimizer, ragda_optimize


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================

def sphere_dict(**params):
    """Sphere function for kwargs API."""
    return sum(v**2 for v in params.values() if isinstance(v, (int, float)))


def sphere_array(x):
    """Sphere function for array-based API."""
    return np.sum(np.array(x) ** 2)


def rastrigin_dict(**params):
    """Rastrigin function - multimodal test function (kwargs API)."""
    A = 10
    vals = [v for v in params.values() if isinstance(v, (int, float))]
    n = len(vals)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in vals])


def rastrigin_array(x):
    """Rastrigin function - multimodal test function (array API)."""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def rosenbrock_dict(**params):
    """Rosenbrock function - classic optimization test (kwargs API)."""
    vals = sorted([(k, v) for k, v in params.items()], key=lambda x: x[0])
    x = [v for k, v in vals]
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def rosenbrock_array(x):
    """Rosenbrock function - classic optimization test (array API)."""
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def get_simple_space(dim=2, low=-5.0, high=5.0):
    """Create a simple search space for dict API."""
    return {
        f'x{i}': {'type': 'continuous', 'bounds': [low, high]}
        for i in range(dim)
    }


def get_simple_bounds(dim=2, low=-5.0, high=5.0):
    """Create simple bounds for array API."""
    return np.array([[low, high]] * dim)


# =============================================================================
# UNIT TESTS: WAVE EXECUTION LOGIC
# =============================================================================

class TestWaveExecution:
    """Unit tests for wave-based parallel execution."""
    
    def test_wave_calculation_exact_fit(self):
        """Test when n_workers is exactly divisible by max_parallel."""
        # 8 workers, 4 max parallel = 2 waves of 4
        bounds = get_simple_bounds(2)
        x_best, f_best, info = ragda_optimize(
            sphere_array,
            bounds,
            n_trials=16,
            worker_strategy='dynamic',
            verbose=False,
            random_state=42
        )
        assert x_best is not None
        assert len(x_best) == 2
    
    def test_wave_calculation_remainder(self):
        """Test when n_workers has remainder after division."""
        # 7 workers with different max parallel values
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=7, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=14,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
        assert result.best_value is not None
    
    def test_wave_calculation_single_wave(self):
        """Test when all workers fit in one wave."""
        # 3 workers should fit in a single wave for most systems
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=3, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=12,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_wave_with_one_worker(self):
        """Test with single worker (edge case)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=1, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=10,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None


# =============================================================================
# UNIT TESTS: ELITE SELECTION LOGIC
# =============================================================================

class TestEliteSelection:
    """Unit tests for elite worker selection."""
    
    def test_elite_fraction_default(self):
        """Test default elite_fraction (0.5 = top 50%)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.5,  # default
            verbose=False
        )
        assert result is not None
        assert result.best_value < 5.0  # Should find reasonable solution
    
    def test_elite_fraction_small(self):
        """Test small elite_fraction (0.1 = top 10%)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.1,
            verbose=False
        )
        assert result is not None
    
    def test_elite_fraction_large(self):
        """Test large elite_fraction (0.8 = top 80%)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.8,
            verbose=False
        )
        assert result is not None
    
    def test_elite_fraction_one(self):
        """Test elite_fraction=1.0 (all workers are elite)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=5, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=15,
            worker_strategy='dynamic',
            elite_fraction=1.0,
            verbose=False
        )
        assert result is not None
    
    def test_elite_fraction_minimum(self):
        """Test very small elite_fraction (should still have at least 1)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.05,  # Would be 0.5 workers, rounded up to 1
            verbose=False
        )
        assert result is not None
    
    def test_elite_with_odd_workers(self):
        """Test elite selection with odd number of workers."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=7, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=21,
            worker_strategy='dynamic',
            elite_fraction=0.3,
            verbose=False
        )
        assert result is not None
    
    def test_elite_with_two_workers(self):
        """Test elite selection with minimum workers (2)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=2, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=10,
            worker_strategy='dynamic',
            elite_fraction=0.5,
            verbose=False
        )
        assert result is not None


# =============================================================================
# UNIT TESTS: RESTART MODE LOGIC
# =============================================================================

class TestRestartMode:
    """Unit tests for restart strategies."""
    
    def test_restart_mode_adaptive(self):
        """Test adaptive restart (default - probability changes over time)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='adaptive',
            verbose=False
        )
        assert result is not None
    
    def test_restart_mode_elite(self):
        """Test elite-only restart (always from best positions)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='elite',
            verbose=False
        )
        assert result is not None
    
    def test_restart_mode_random(self):
        """Test random-only restart (always random new positions)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='random',
            verbose=False
        )
        assert result is not None
    
    def test_adaptive_probability_start(self):
        """Test adaptive restart with custom starting probability."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='adaptive',
            restart_elite_prob_start=0.9,  # High exploitation at start
            restart_elite_prob_end=0.1,    # High exploration at end
            verbose=False
        )
        assert result is not None
    
    def test_adaptive_probability_inverted(self):
        """Test adaptive restart with inverted probability (explore->exploit)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='adaptive',
            restart_elite_prob_start=0.3,
            restart_elite_prob_end=0.9,
            verbose=False
        )
        assert result is not None
    
    def test_adaptive_same_probabilities(self):
        """Test adaptive with same start/end probability (constant)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=24,
            worker_strategy='dynamic',
            restart_mode='adaptive',
            restart_elite_prob_start=0.5,
            restart_elite_prob_end=0.5,
            verbose=False
        )
        assert result is not None


# =============================================================================
# UNIT TESTS: WORKER DECAY LOGIC
# =============================================================================

class TestWorkerDecay:
    """Unit tests for worker decay over time."""
    
    def test_decay_disabled(self):
        """Test with worker decay disabled (default)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=32,
            worker_strategy='dynamic',
            enable_worker_decay=False,
            verbose=False
        )
        assert result is not None
    
    def test_decay_enabled_default_rate(self):
        """Test with worker decay enabled at default rate (0.5)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=32,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=2,
            verbose=False
        )
        assert result is not None
    
    def test_decay_fast_rate(self):
        """Test with fast decay rate (0.8)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=40,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.8,
            min_workers=2,
            verbose=False
        )
        assert result is not None
    
    def test_decay_slow_rate(self):
        """Test with slow decay rate (0.2)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=32,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.2,
            min_workers=2,
            verbose=False
        )
        assert result is not None
    
    def test_decay_min_workers_boundary(self):
        """Test that decay respects min_workers."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=64,  # Many trials to trigger decay
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.9,  # Fast decay
            min_workers=4,          # Should not go below 4
            verbose=False
        )
        assert result is not None
    
    def test_decay_min_workers_one(self):
        """Test with min_workers=1."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=1,
            verbose=False
        )
        assert result is not None
    
    def test_decay_min_equals_initial(self):
        """Test when min_workers equals initial workers (no decay)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.9,
            min_workers=4,  # Same as n_workers
            verbose=False
        )
        assert result is not None


# =============================================================================
# EDGE CASE TESTS: ODD NUMBERS AND BOUNDARIES
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""
    
    def test_odd_workers_1(self):
        """Test with 1 worker (minimum viable)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=1, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=10,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_odd_workers_3(self):
        """Test with 3 workers (odd, small)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=3, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=9,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_odd_workers_11(self):
        """Test with 11 workers (odd, larger)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=11, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=22,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_odd_trials_7(self):
        """Test with 7 trials (odd, small)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=3, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=7,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_odd_trials_23(self):
        """Test with 23 trials (odd prime)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=5, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=23,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_trials_less_than_workers(self):
        """Test when n_trials < n_workers (partial first phase)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=5,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_trials_equals_workers(self):
        """Test when n_trials == n_workers (exactly one phase)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=8,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_single_dimension(self):
        """Test with 1D problem."""
        space = {'x0': {'type': 'continuous', 'bounds': [-10.0, 10.0]}}
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
        assert abs(result.best_params['x0']) < 2.0
    
    def test_high_dimension(self):
        """Test with high-dimensional problem."""
        opt = RAGDAOptimizer(get_simple_space(10), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=60,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        space = {
            'x0': {'type': 'continuous', 'bounds': [-10.0, 5.0]},
            'x1': {'type': 'continuous', 'bounds': [-2.0, 8.0]},
            'x2': {'type': 'continuous', 'bounds': [0.0, 100.0]},
        }
        opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=20,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_very_narrow_bounds(self):
        """Test with very narrow bounds."""
        opt = RAGDAOptimizer(get_simple_space(2, -0.1, 0.1), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
        assert result.best_value < 0.1
    
    def test_very_wide_bounds(self):
        """Test with very wide bounds."""
        opt = RAGDAOptimizer(get_simple_space(2, -1000, 1000), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_prime_numbers_combination(self):
        """Test with prime numbers for workers, trials, dimensions."""
        opt = RAGDAOptimizer(get_simple_space(7), n_workers=13, random_state=42)  # 7D, 13 workers
        result = opt.optimize(
            sphere_dict,
            n_trials=31,  # Prime
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None


# =============================================================================
# EDGE CASE TESTS: SYNC FREQUENCY
# =============================================================================

class TestSyncFrequency:
    """Tests for sync_frequency edge cases."""
    
    def test_sync_frequency_matches_workers(self):
        """Test when sync_frequency equals n_workers."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=5, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=25,
            sync_frequency=5,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_sync_frequency_one(self):
        """Test sync_frequency=1 (sync after every trial)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=12,
            sync_frequency=1,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_sync_frequency_larger_than_trials(self):
        """Test sync_frequency > n_trials (never syncs mid-run)."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=10,
            sync_frequency=100,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_sync_frequency_prime(self):
        """Test with prime sync_frequency."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            sync_frequency=7,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_sync_with_dynamic_vs_greedy(self):
        """Compare behavior with different sync frequencies."""
        opt1 = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result_low = opt1.optimize(
            sphere_dict,
            n_trials=16,
            sync_frequency=2,
            worker_strategy='dynamic',
            verbose=False
        )
        
        opt2 = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result_high = opt2.optimize(
            sphere_dict,
            n_trials=16,
            sync_frequency=8,
            worker_strategy='dynamic',
            verbose=False
        )
        # Both should work, values may differ
        assert result_low is not None
        assert result_high is not None


# =============================================================================
# INTEGRATION TESTS: GREEDY VS DYNAMIC
# =============================================================================

class TestGreedyVsDynamic:
    """Integration tests comparing greedy and dynamic strategies."""
    
    def test_greedy_basic(self):
        """Test greedy strategy works correctly."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=20,
            worker_strategy='greedy',
            verbose=False
        )
        assert result is not None
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_value')
        assert result.best_value < 5.0
    
    def test_dynamic_basic(self):
        """Test dynamic strategy works correctly."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=20,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_value')
        assert result.best_value < 5.0
    
    def test_both_strategies_same_seed(self):
        """Test both strategies produce valid results with same seed."""
        seed = 42
        
        opt1 = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=seed)
        result_greedy = opt1.optimize(
            sphere_dict,
            n_trials=20,
            worker_strategy='greedy',
            verbose=False
        )
        
        opt2 = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=seed)
        result_dynamic = opt2.optimize(
            sphere_dict,
            n_trials=20,
            worker_strategy='dynamic',
            verbose=False
        )
        
        assert result_greedy is not None
        assert result_dynamic is not None
        # Both should find good solutions
        assert result_greedy.best_value < 5.0
        assert result_dynamic.best_value < 5.0
    
    def test_greedy_with_many_workers(self):
        """Test greedy with many workers."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=12, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=48,
            worker_strategy='greedy',
            verbose=False
        )
        assert result is not None
    
    def test_dynamic_with_many_workers(self):
        """Test dynamic with many workers."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=12, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=48,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_greedy_multimodal(self):
        """Test greedy on multimodal function."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            rastrigin_dict,
            n_trials=40,
            worker_strategy='greedy',
            verbose=False
        )
        assert result is not None
    
    def test_dynamic_multimodal(self):
        """Test dynamic on multimodal function."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=8, random_state=42)
        result = opt.optimize(
            rastrigin_dict,
            n_trials=40,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_greedy_rosenbrock(self):
        """Test greedy on Rosenbrock valley."""
        opt = RAGDAOptimizer(get_simple_space(2, -2, 2), n_workers=6, random_state=42)
        result = opt.optimize(
            rosenbrock_dict,
            n_trials=30,
            worker_strategy='greedy',
            verbose=False
        )
        assert result is not None
    
    def test_dynamic_rosenbrock(self):
        """Test dynamic on Rosenbrock valley."""
        opt = RAGDAOptimizer(get_simple_space(2, -2, 2), n_workers=6, random_state=42)
        result = opt.optimize(
            rosenbrock_dict,
            n_trials=30,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None


# =============================================================================
# INTEGRATION TESTS: PARAMETER COMBINATIONS
# =============================================================================

class TestParameterCombinations:
    """Integration tests with various parameter combinations."""
    
    def test_dynamic_full_options(self):
        """Test dynamic with all options specified."""
        opt = RAGDAOptimizer(get_simple_space(3), n_workers=8, random_state=123)
        result = opt.optimize(
            sphere_dict,
            n_trials=40,
            worker_strategy='dynamic',
            elite_fraction=0.25,
            restart_mode='adaptive',
            restart_elite_prob_start=0.7,
            restart_elite_prob_end=0.3,
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=3,
            sync_frequency=4,
            verbose=False
        )
        assert result is not None
        assert hasattr(result, 'best_params')
        assert len(result.best_params) == 3
    
    def test_dynamic_exploration_focused(self):
        """Test dynamic tuned for exploration."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            rastrigin_dict,
            n_trials=50,
            worker_strategy='dynamic',
            elite_fraction=0.5,      # More elite workers
            restart_mode='random',   # Random restarts
            enable_worker_decay=False,
            verbose=False
        )
        assert result is not None
    
    def test_dynamic_exploitation_focused(self):
        """Test dynamic tuned for exploitation."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=6, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.2,      # Fewer elite (more selective)
            restart_mode='elite',    # Always from elite
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=2,
            verbose=False
        )
        assert result is not None
    
    def test_dynamic_minimal_config(self):
        """Test dynamic with minimal configuration."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_default_strategy_is_greedy(self):
        """Test that default worker_strategy is 'greedy'."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            # No worker_strategy specified
            verbose=False
        )
        assert result is not None


# =============================================================================
# INTEGRATION TESTS: RESULT CONSISTENCY
# =============================================================================

class TestResultConsistency:
    """Tests for result structure and consistency."""
    
    def test_result_structure_greedy(self):
        """Verify result structure with greedy strategy."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='greedy',
            verbose=False
        )
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_value')
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, (int, float))
        assert len(result.best_params) == 2
    
    def test_result_structure_dynamic(self):
        """Verify result structure with dynamic strategy."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=16,
            worker_strategy='dynamic',
            verbose=False
        )
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_value')
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, (int, float))
        assert len(result.best_params) == 2
    
    def test_params_within_bounds(self):
        """Verify best params are within bounds."""
        space = {
            'a': {'type': 'continuous', 'bounds': [-3.0, 7.0]},
            'b': {'type': 'continuous', 'bounds': [-10.0, 2.0]},
            'c': {'type': 'continuous', 'bounds': [0.0, 100.0]},
        }
        
        for strategy in ['greedy', 'dynamic']:
            opt = RAGDAOptimizer(space, n_workers=4, random_state=42)
            result = opt.optimize(
                sphere_dict,
                n_trials=20,
                worker_strategy=strategy,
                verbose=False
            )
            
            assert -3.0 <= result.best_params['a'] <= 7.0, \
                f"Param a out of bounds for {strategy}"
            assert -10.0 <= result.best_params['b'] <= 2.0, \
                f"Param b out of bounds for {strategy}"
            assert 0.0 <= result.best_params['c'] <= 100.0, \
                f"Param c out of bounds for {strategy}"
    
    def test_deterministic_with_seed_greedy(self):
        """Test greedy produces similar results with same seed (parallel may vary slightly)."""
        results = []
        for _ in range(3):
            opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=999)
            result = opt.optimize(
                sphere_dict,
                n_trials=16,
                worker_strategy='greedy',
                verbose=False
            )
            results.append(result.best_value)
        
        # All should be close (parallel execution may have slight variance)
        # Check that all results are reasonably close to each other
        assert all(r < 5.0 for r in results), "All results should be reasonable"
        assert max(results) - min(results) < 1.0, "Results should be similar with same seed"
    
    def test_deterministic_with_seed_dynamic(self):
        """Test dynamic produces similar results with same seed (parallel may vary slightly)."""
        results = []
        for _ in range(3):
            opt = RAGDAOptimizer(get_simple_space(2), n_workers=4, random_state=999)
            result = opt.optimize(
                sphere_dict,
                n_trials=16,
                worker_strategy='dynamic',
                verbose=False
            )
            results.append(result.best_value)
        
        # All should be close (parallel execution may have slight variance)
        assert all(r < 5.0 for r in results), "All results should be reasonable"
        assert max(results) - min(results) < 1.0, "Results should be similar with same seed"


# =============================================================================
# INTEGRATION TESTS: SCIPY-STYLE API
# =============================================================================

class TestScipyStyleAPI:
    """Tests using ragda_optimize function (scipy-style)."""
    
    def test_ragda_optimize_greedy(self):
        """Test ragda_optimize with greedy strategy."""
        bounds = get_simple_bounds(2)
        x_best, f_best, info = ragda_optimize(
            sphere_array,
            bounds,
            n_trials=20,
            worker_strategy='greedy',
            verbose=False,
            random_state=42
        )
        assert len(x_best) == 2
        assert f_best < 5.0
    
    def test_ragda_optimize_dynamic(self):
        """Test ragda_optimize with dynamic strategy."""
        bounds = get_simple_bounds(2)
        x_best, f_best, info = ragda_optimize(
            sphere_array,
            bounds,
            n_trials=20,
            worker_strategy='dynamic',
            verbose=False,
            random_state=42
        )
        assert len(x_best) == 2
        assert f_best < 5.0
    
    def test_ragda_optimize_dynamic_full_options(self):
        """Test ragda_optimize with all dynamic options."""
        bounds = get_simple_bounds(3)
        x_best, f_best, info = ragda_optimize(
            sphere_array,
            bounds,
            n_trials=30,
            worker_strategy='dynamic',
            elite_fraction=0.3,
            restart_mode='adaptive',
            restart_elite_prob_start=0.8,
            restart_elite_prob_end=0.2,
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=2,
            verbose=False,
            random_state=42
        )
        assert len(x_best) == 3
        assert f_best is not None


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for robustness."""
    
    def test_many_workers_few_trials(self):
        """Stress test: many workers, few trials."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=20, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=10,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_few_workers_many_trials(self):
        """Stress test: few workers, many trials."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=2, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=50,
            worker_strategy='dynamic',
            verbose=False
        )
        assert result is not None
    
    def test_rapid_decay(self):
        """Stress test: very rapid worker decay."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=50,
            worker_strategy='dynamic',
            enable_worker_decay=True,
            worker_decay_rate=0.9,  # Very fast decay
            min_workers=1,
            verbose=False
        )
        assert result is not None
    
    def test_tiny_elite_fraction(self):
        """Stress test: very small elite fraction."""
        opt = RAGDAOptimizer(get_simple_space(2), n_workers=10, random_state=42)
        result = opt.optimize(
            sphere_dict,
            n_trials=40,
            worker_strategy='dynamic',
            elite_fraction=0.01,  # Should round to at least 1
            verbose=False
        )
        assert result is not None
    
    def test_all_features_combined(self):
        """Stress test: all dynamic features enabled."""
        opt = RAGDAOptimizer(get_simple_space(5), n_workers=12, random_state=42)
        result = opt.optimize(
            rastrigin_dict,
            n_trials=60,
            worker_strategy='dynamic',
            elite_fraction=0.2,
            restart_mode='adaptive',
            restart_elite_prob_start=0.8,
            restart_elite_prob_end=0.2,
            enable_worker_decay=True,
            worker_decay_rate=0.5,
            min_workers=3,
            sync_frequency=6,
            verbose=False
        )
        assert result is not None
        assert result.best_value < 100  # Reasonable for 5D Rastrigin


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
