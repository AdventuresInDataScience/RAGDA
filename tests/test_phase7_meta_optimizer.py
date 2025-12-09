"""
Phase 7 Meta-Optimizer Tests

All tests for the meta-optimization system in one file.
Tests are organized by implementation step.
"""

import pytest
import numpy as np
from typing import List


# =============================================================================
# Section 1: AUC Metric Tests
# =============================================================================

class TestAUCMetric:
    """Tests for AUC (Area Under Curve) convergence metric."""
    
    def test_auc_perfect_convergence(self):
        """Immediate convergence should give near-zero AUC."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Best value found immediately and maintained
        convergence_history = [0.0, 0.0, 0.0, 0.0, 0.0]
        auc = calculate_auc(convergence_history)
        
        assert auc == 0.0, f"Perfect convergence should give AUC=0.0, got {auc}"
    
    def test_auc_no_convergence(self):
        """No improvement should give AUC near 1.0."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Best value found only on last iteration
        convergence_history = [100.0, 100.0, 100.0, 100.0, 0.0]
        auc = calculate_auc(convergence_history)
        
        # Should be close to 1.0 (very slow convergence)
        assert auc >= 0.85, f"No improvement until end should give AUC close to 1.0, got {auc}"
    
    def test_auc_linear_convergence(self):
        """Linear improvement should give AUC around 0.5."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Linear improvement from 100 to 0
        convergence_history = [100.0, 75.0, 50.0, 25.0, 0.0]
        auc = calculate_auc(convergence_history)
        
        # Linear convergence should be around 0.5
        assert 0.4 <= auc <= 0.6, f"Linear convergence should give AUC≈0.5, got {auc}"
    
    def test_auc_normalization(self):
        """Test normalization with different value ranges."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Two identical convergence patterns with different scales
        history1 = [10.0, 5.0, 2.5, 1.25, 0.625]
        history2 = [1000.0, 500.0, 250.0, 125.0, 62.5]
        
        auc1 = calculate_auc(history1)
        auc2 = calculate_auc(history2)
        
        # Should be very similar due to normalization (scale-invariant)
        assert abs(auc1 - auc2) < 0.01, \
            f"Normalized AUC should be scale-invariant: {auc1} vs {auc2}"
    
    def test_auc_maximize_direction(self):
        """Test AUC with maximization problems."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Maximization: starting low, improving to high
        convergence_history = [0.0, 25.0, 50.0, 75.0, 100.0]
        auc = calculate_auc(convergence_history, direction='maximize')
        
        # Linear improvement should be around 0.5
        assert 0.4 <= auc <= 0.6, f"Linear maximization should give AUC≈0.5, got {auc}"
    
    def test_auc_single_value(self):
        """Single value should give neutral AUC of 0.5."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        convergence_history = [42.0]
        auc = calculate_auc(convergence_history)
        
        assert auc == 0.5, f"Single value should give AUC=0.5, got {auc}"
    
    def test_auc_empty_history(self):
        """Empty history should give worst AUC of 1.0."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        convergence_history = []
        auc = calculate_auc(convergence_history)
        
        assert auc == 1.0, f"Empty history should give AUC=1.0, got {auc}"
    
    def test_interpret_auc(self):
        """Test AUC interpretation function."""
        from RAGDA_default_args.auc_metric import interpret_auc
        
        assert "excellent" in interpret_auc(0.02).lower()
        assert "very good" in interpret_auc(0.10).lower()
        assert "good" in interpret_auc(0.25).lower()
        assert "moderate" in interpret_auc(0.40).lower()
        assert "fair" in interpret_auc(0.60).lower()
        assert "poor" in interpret_auc(0.85).lower()
    
    def test_evaluate_optimizer_on_simple_problem(self):
        """Test wrapper that runs optimizer and computes AUC."""
        from RAGDA_default_args.auc_metric import evaluate_optimizer_on_problem
        
        # Create a simple optimizer factory (mock)
        def mock_optimizer_factory(direction='minimize', **kwargs):
            class MockTrial:
                def __init__(self, value):
                    self.value = value
            
            class MockStudy:
                def __init__(self):
                    self.trials = []
                
                def optimize(self, objective, n_trials):
                    # Simulate linear convergence
                    for i in range(n_trials):
                        value = 100.0 * (1 - i / n_trials)
                        self.trials.append(MockTrial(value))
            
            return MockStudy()
        
        # Simple problem (not actually called by mock)
        def simple_problem(trial):
            return 0.0
        
        auc, history = evaluate_optimizer_on_problem(
            mock_optimizer_factory,
            simple_problem,
            n_evaluations=10
        )
        
        assert 0.0 <= auc <= 1.0, f"AUC should be in [0, 1], got {auc}"
        assert len(history) == 10, f"Should have 10 evaluations, got {len(history)}"
        assert 0.4 <= auc <= 0.6, f"Linear convergence should give AUC≈0.5, got {auc}"
    
    def test_auc_monotonicity(self):
        """Later improvement should result in higher (worse) AUC."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Test that delaying improvement increases AUC
        aucs = []
        for delay in [0, 2, 4, 6, 8]:
            history = [100.0] * delay + [0.0] * (10 - delay)
            auc = calculate_auc(history)
            aucs.append(auc)
        
        # AUC should increase monotonically (worse convergence = higher AUC)
        for i in range(len(aucs) - 1):
            assert aucs[i] < aucs[i+1], \
                f"AUC should increase with delayed convergence: {aucs}"
    
    def test_auc_scale_invariance(self):
        """AUC should be identical across different scales."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        # Same convergence pattern at different scales
        patterns = [
            [10.0, 5.0, 2.5, 1.25, 0.625],
            [100.0, 50.0, 25.0, 12.5, 6.25],
            [10000.0, 5000.0, 2500.0, 1250.0, 625.0],
        ]
        
        aucs = [calculate_auc(p) for p in patterns]
        
        # All should be identical (scale-invariant)
        for i in range(len(aucs) - 1):
            assert abs(aucs[i] - aucs[i+1]) < 1e-10, \
                f"AUC should be scale-invariant: {aucs}"
    
    def test_auc_direction_invariance(self):
        """AUC should work equally for minimize and maximize."""
        from RAGDA_default_args.auc_metric import calculate_auc
        
        minimize_history = [100.0, 75.0, 50.0, 25.0, 0.0]
        maximize_history = [0.0, 25.0, 50.0, 75.0, 100.0]
        
        auc_min = calculate_auc(minimize_history, direction='minimize')
        auc_max = calculate_auc(maximize_history, direction='maximize')
        
        # Should be identical (direction-agnostic)
        assert abs(auc_min - auc_max) < 1e-10, \
            f"AUC should be direction-invariant: {auc_min} vs {auc_max}"


# =============================================================================
# Section 2: Benchmark Functions Tests
# =============================================================================

class TestBenchmarkFunctions:
    """Tests for benchmark functions (Optuna API)."""
    
    def test_benchmark_functions_count(self):
        """Verify current chunk progress (after Chunk 2.1.4: 52 functions)."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 52, \
            f"Expected 52 functions after Chunk 2.1.4, got {len(ALL_BENCHMARK_FUNCTIONS)}"
    
    def test_unimodal_count(self):
        """Verify we have 12 unimodal functions."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        unimodal = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'unimodal']
        assert len(unimodal) == 12
    
    def test_multimodal_count(self):
        """Verify we have 40 multimodal functions (after Chunk 2.1.4)."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        multimodal = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'multimodal']
        assert len(multimodal) == 40
    
    def test_sphere_2d_optuna(self):
        """Test simple sphere function works with Optuna trial mock."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.0  # Always suggest 0
        
        problem = get_benchmark_function('sphere_2d')
        result = problem.objective(MockTrial())
        
        assert abs(result - 0.0) < 1e-10, f"Sphere at origin should be 0, got {result}"
        assert problem.dimension == 2
        assert problem.known_optimum == 0.0
        assert problem.category == 'unimodal'
    
    def test_ackley_10d_optuna(self):
        """Test multimodal ackley function (Chunk 2.1.2)."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.0  # At optimum
        
        problem = get_benchmark_function('ackley_10d')
        result = problem.objective(MockTrial())
        
        assert abs(result - 0.0) < 1e-6, f"Ackley at origin should be ~0, got {result}"
        assert problem.dimension == 10
        assert problem.category == 'multimodal'
        assert problem.known_optimum == 0.0
    
    def test_rastrigin_5d_optuna(self):
        """Test rastrigin function (Chunk 2.1.2)."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('rastrigin_5d')
        assert problem.dimension == 5
        assert problem.category == 'multimodal'
        assert callable(problem.objective)
    
    def test_styblinski_tang_20d(self):
        """Test styblinski-tang function (Chunk 2.1.2)."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('styblinski_tang_20d')
        assert problem.dimension == 20
        assert problem.category == 'multimodal'
        assert abs(problem.known_optimum - (-39.16599 * 20)) < 1
    
    def test_all_functions_callable(self):
        """Verify all registered functions are callable."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        
        for name, problem in ALL_BENCHMARK_FUNCTIONS.items():
            assert callable(problem.objective), f"{name} objective not callable"
    
    def test_problem_metadata_complete(self):
        """Verify all problems have required metadata."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        
        for name, problem in ALL_BENCHMARK_FUNCTIONS.items():
            assert problem.name == name
            assert problem.dimension > 0
            assert len(problem.bounds) == problem.dimension
            assert problem.category is not None
            assert problem.description
    
    # =========================================================================
    # Chunk 2.1.3: Special 2D Functions
    # =========================================================================
    
    def test_special_2d_count(self):
        """Verify we have 52 total functions after Chunk 2.1.4."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 52
    
    def test_eggholder_2d_optuna(self):
        """Test eggholder function with Optuna interface."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                # Suggest near global optimum
                if 'x0' in name:
                    return 512.0
                return 404.2319
        
        problem = get_benchmark_function('eggholder_2d')
        result = problem.objective(MockTrial())
        
        # Should be near global minimum
        assert result < -900  # Close to -959.6407
        assert problem.dimension == 2
        assert problem.category == 'multimodal'
    
    def test_hartmann_3d_dimension(self):
        """Test hartmann_3d has correct dimension."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('hartmann_3d')
        assert problem.dimension == 3
        assert problem.category == 'multimodal'
        assert all(b == (0, 1) for b in problem.bounds)
    
    # Chunk 2.1.4: Fixed Dimension Functions
    def test_fixed_dim_count(self):
        """After Chunk 2.1.4, should have exactly 52 functions (50 + 2 fixed dim)."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 52
    
    def test_shekel_4d_fixed(self):
        """Test Shekel is exactly 4D with correct bounds."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('shekel_4d')
        assert problem.dimension == 4
        assert problem.category == 'multimodal'
        assert all(b == (0, 10) for b in problem.bounds)
        assert problem.known_optimum is not None
    
    def test_hartmann_6d_fixed(self):
        """Test Hartmann 6D is exactly 6D with (0,1) bounds."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('hartmann_6d')
        assert problem.dimension == 6
        assert problem.category == 'multimodal'
        assert all(b == (0, 1) for b in problem.bounds)
        assert problem.known_optimum is not None


# =============================================================================
# Section 3: Problem Classification Tests
# =============================================================================

# Placeholder for Step 3 tests


# =============================================================================
# Section 4: RAGDA Parameter Space Tests
# =============================================================================

# Placeholder for Step 4 tests


# =============================================================================
# Section 5: Meta-Optimizer Tests
# =============================================================================

# Placeholder for Step 5 tests


# =============================================================================
# Section 6: Integration Tests
# =============================================================================

# Placeholder for Step 6 tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
