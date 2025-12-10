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
        """Verify current chunk progress (after Chunk 2.1.6: 78 functions)."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78, \
            f"Expected 78 functions after Chunk 2.1.6, got {len(ALL_BENCHMARK_FUNCTIONS)}"
    
    def test_unimodal_count(self):
        """Verify we have 12 unimodal functions."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        unimodal = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'unimodal']
        assert len(unimodal) == 12
    
    def test_multimodal_count(self):
        """Verify we have 39 multimodal functions (archive-verified: 5 each of ackley/rastrigin/schwefel/griewank/levy, 6 styblinski_tang, 8 special/fixed)."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        multimodal = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'multimodal']
        assert len(multimodal) == 39
    
    def test_sphere_2d_optuna(self):
        """Test simple sphere function with Optuna trial mock."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
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
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
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
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('rastrigin_5d')
        assert problem.dimension == 5
        assert problem.category == 'multimodal'
        assert callable(problem.objective)
    
    def test_styblinski_tang_20d(self):
        """Test styblinski-tang function (Chunk 2.1.2)."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('styblinski_tang_20d')
        assert problem.dimension == 20
        assert problem.category == 'multimodal'
        assert abs(problem.known_optimum - (-39.16599 * 20)) < 1
    
    def test_all_functions_callable(self):
        """Verify all registered functions are callable."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        
        for name, problem in ALL_BENCHMARK_FUNCTIONS.items():
            assert callable(problem.objective), f"{name} objective not callable"
    
    def test_problem_metadata_complete(self):
        """Verify all problems have required metadata."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        
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
        """After all chunks through 2.1.6, should have 78 functions."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78
    
    def test_eggholder_2d_optuna(self):
        """Test eggholder function with Optuna interface."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
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
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('hartmann_3d')
        assert problem.dimension == 3
        assert problem.category == 'multimodal'
        assert all(b == (0, 1) for b in problem.bounds)
    
    # Chunk 2.1.4: Fixed Dimension Functions
    def test_fixed_dim_count(self):
        """After all chunks through 2.1.6, should have 78 functions."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78
    
    def test_shekel_4d_fixed(self):
        """Test Shekel is exactly 4D with correct bounds."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('shekel_4d')
        assert problem.dimension == 4
        assert problem.category == 'multimodal'
        assert all(b == (0, 10) for b in problem.bounds)
        assert problem.known_optimum is not None
    
    def test_hartmann_6d_fixed(self):
        """Test Hartmann 6D is exactly 6D with (0,1) bounds."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('hartmann_6d')
        assert problem.dimension == 6
        assert problem.category == 'multimodal'
        assert all(b == (0, 1) for b in problem.bounds)
        assert problem.known_optimum is not None
    
    # =========================================================================
    # Chunk 2.1.5: Valley Functions
    # =========================================================================
    
    def test_valley_count(self):
        """Verify we have 78 total functions after all chunks through 2.1.6."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78, \
            f"Expected 78 functions after Chunk 2.1.6, got {len(ALL_BENCHMARK_FUNCTIONS)}"
    
    def test_valley_category_count(self):
        """Verify we have 19 valley functions (includes colville_4d)."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        valley = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'valley']
        assert len(valley) == 19, f"Expected 19 valley functions, got {len(valley)}"
    
    def test_rosenbrock_10d_valley(self):
        """Test Rosenbrock 10D function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('rosenbrock_10d')
        assert problem.dimension == 10
        assert problem.category == 'valley'
        assert all(b == (-5.0, 10.0) for b in problem.bounds)
        assert problem.known_optimum == 0.0
    
    def test_dixon_price_50d(self):
        """Test Dixon-Price 50D function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('dixon_price_50d')
        assert problem.dimension == 50
        assert problem.category == 'valley'
        assert all(b == (-10.0, 10.0) for b in problem.bounds)
    
    def test_six_hump_camel_2d(self):
        """Test Six-Hump Camel 2D function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('six_hump_camel_2d')
        assert problem.dimension == 2
        assert problem.category == 'valley'
        assert problem.bounds == [(-3, 3), (-2, 2)]
        assert abs(problem.known_optimum - (-1.0316)) < 0.01
    
    def test_powell_20d(self):
        """Test Powell 20D function (multiple of 4)."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('powell_20d')
        assert problem.dimension == 20
        assert problem.category == 'valley'
        assert all(b == (-4.0, 5.0) for b in problem.bounds)
        assert problem.known_optimum == 0.0
    
    # =========================================================================
    # Chunk 2.1.6: Plate Functions (8 functions)
    # =========================================================================
    
    def test_plate_count(self):
        """Verify we have 78 total functions after Chunk 2.1.6 (70 + 7 plate + 1 colville moved to valley)."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78, \
            f"Expected 78 functions after Chunk 2.1.6, got {len(ALL_BENCHMARK_FUNCTIONS)}"
    
    def test_plate_category_count(self):
        """Verify we have 7 plate functions."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        plate = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'plate']
        assert len(plate) == 7, f"Expected 7 plate functions, got {len(plate)}"
    
    def test_zakharov_20d_plate(self):
        """Test zakharov 20D plate function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('zakharov_20d')
        assert problem.dimension == 20
        assert problem.category == 'plate'
        assert all(b == (-5.0, 10.0) for b in problem.bounds)
        assert problem.known_optimum == 0.0
    
    def test_booth_2d_plate(self):
        """Test Booth 2D function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('booth_2d')
        assert problem.dimension == 2
        assert problem.category == 'plate'
        assert all(b == (-10.0, 10.0) for b in problem.bounds)
        assert problem.known_optimum == 0.0
    
    def test_colville_4d_valley(self):
        """Test Colville 4D function (valley category per archive)."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('colville_4d')
        assert problem.dimension == 4
        assert problem.category == 'valley'
        assert all(b == (-10.0, 10.0) for b in problem.bounds)
        assert problem.known_optimum == 0.0
    
    # =========================================================================
    # Chunk 2.1.7: Steep Functions (1 function) - COMPLETES MATHEMATICAL FUNCTIONS
    # =========================================================================
    
    def test_steep_count(self):
        """Verify we have 78 total functions after corrections (Chunk 2.1.7: +easom, -ackley_100d, colville→valley)."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78, \
            f"Expected 78 functions after Chunk 2.1.7, got {len(ALL_BENCHMARK_FUNCTIONS)}"
    
    def test_steep_category_count(self):
        """Verify we have 1 steep function."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        steep = [p for p in ALL_BENCHMARK_FUNCTIONS.values() if p.category == 'steep']
        assert len(steep) == 1, f"Expected 1 steep function, got {len(steep)}"
    
    def test_easom_2d_steep(self):
        """Test Easom 2D function."""
        from RAGDA_default_args.benchmark_mathematical_problems import get_benchmark_function
        
        problem = get_benchmark_function('easom_2d')
        assert problem.dimension == 2
        assert problem.category == 'steep'
        assert all(b == (-100.0, 100.0) for b in problem.bounds)
        assert problem.known_optimum == -1.0
    
    def test_mathematical_functions_complete(self):
        """Verify all 78 mathematical functions match archive counts exactly."""
        from RAGDA_default_args.benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
        
        cats = {}
        for p in ALL_BENCHMARK_FUNCTIONS.values():
            cats[p.category] = cats.get(p.category, 0) + 1
        
        # Archive counts (verified):
        assert cats['unimodal'] == 12
        assert cats['multimodal'] == 39
        assert cats['valley'] == 19
        assert cats['plate'] == 7
        assert cats['steep'] == 1
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78


# =============================================================================
# Chunk 2.2.1: Chaotic Systems Tests (16 functions)
# =============================================================================

class TestChaoticSystems:
    """Tests for chaotic system problems (Chunk 2.2.1)."""
    
    def test_chaotic_count(self):
        """Verify we added 133 real-world problems (all batches + 16 remaining)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 133
    
    def test_chaotic_category_count(self):
        """Verify all 16 are categorized as 'chaotic'."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        chaotic = [p for p in ALL_REALWORLD_PROBLEMS.values() if p.category == 'chaotic']
        assert len(chaotic) == 16
    
    def test_mackey_glass_4d(self):
        """Test Mackey-Glass 4D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                # Known good parameters
                if 'x0' in name: return 0.2  # beta
                if 'x1' in name: return 0.1  # gamma
                if 'x2' in name: return 10   # n
                if 'x3' in name: return 1.0  # tau_scale
                return (low + high) / 2
        
        problem = get_problem('MackeyGlass-4D')
        result = problem.objective(MockTrial())
        
        assert problem.dimension == 4
        assert problem.category == 'chaotic'
        assert problem.known_optimum is None  # Real-world problem
        assert isinstance(result, (int, float))
        assert result < 1e10  # Should not hit penalty
    
    def test_lorenz_3d(self):
        """Test Lorenz attractor 3D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                # Classic chaotic parameters
                if 'x0' in name: return 10.0  # sigma
                if 'x1' in name: return 28.0  # rho
                if 'x2' in name: return 8/3   # beta
                return (low + high) / 2
        
        problem = get_problem('Lorenz-3D')
        result = problem.objective(MockTrial())
        
        assert problem.dimension == 3
        assert problem.category == 'chaotic'
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_henon_2d(self):
        """Test Hénon map 2D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('Henon-2D')
        assert problem.dimension == 2
        assert problem.category == 'chaotic'
        assert problem.bounds == [(1.0, 1.5), (0.1, 0.5)]
    
    def test_logistic_map_1d(self):
        """Test Logistic map 1D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('LogisticMap-1D')
        assert problem.dimension == 1
        assert problem.category == 'chaotic'
        assert problem.bounds == [(3.5, 4.0)]
    
    def test_high_dimensional_chaotic(self):
        """Test high-dimensional chaotic problems exist."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        high_dim = [p for p in ALL_REALWORLD_PROBLEMS.values() if p.dimension >= 50]
        assert len(high_dim) >= 3  # Lorenz96-60D, CoupledMapLattice-64D, etc.
        
        # Check specific high-dim problems
        assert 'Lorenz96Extended-60D' in ALL_REALWORLD_PROBLEMS
        assert 'CoupledMapLattice-64D' in ALL_REALWORLD_PROBLEMS
        assert 'CoupledLogisticMaps-100D' in ALL_REALWORLD_PROBLEMS
    
    def test_all_chaotic_callable(self):
        """Verify all chaotic functions are callable."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        for name, problem in ALL_REALWORLD_PROBLEMS.items():
            assert callable(problem.objective), f"{name} objective not callable"
            assert problem.dimension > 0, f"{name} has invalid dimension"
            assert len(problem.bounds) == problem.dimension, f"{name} bounds mismatch"


# =============================================================================
# Chunk 2.2.2: Dynamical Systems Tests (6 functions)
# =============================================================================

class TestDynamicalSystems:
    """Tests for dynamical systems problems (Chunk 2.2.2)."""
    
    def test_dynamical_count(self):
        """Verify we added 6 dynamical systems (133 total: 16 + 6 + 16 + 43 + 18 + 18 + 16)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 133  # 16 chaotic + 6 dynamical + 16 nn_weights + 43 ml_training + 18 pde + 18 meta_control + 16 remaining
    
    def test_dynamical_category_count(self):
        """Verify all 7 are categorized as 'dynamical'."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        dynamical = [p for p in ALL_REALWORLD_PROBLEMS.values() if p.category == 'dynamical']
        assert len(dynamical) == 7
    
    def test_lotka_volterra_4d(self):
        """Test Lotka-Volterra 4D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                # Classic parameter values
                if 'x0' in name: return 1.5  # alpha
                if 'x1' in name: return 1.0  # beta
                if 'x2' in name: return 1.0  # delta
                if 'x3' in name: return 3.0  # gamma
                return (low + high) / 2
        
        problem = get_problem('LotkaVolterra-4D')
        result = problem.objective(MockTrial())
        
        assert problem.dimension == 4
        assert problem.category == 'dynamical'
        assert problem.known_optimum is None
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_van_der_pol_1d(self):
        """Test Van der Pol 1D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('VanDerPol-1D')
        assert problem.dimension == 1
        assert problem.category == 'dynamical'
        assert problem.bounds == [(0.1, 5)]
    
    def test_kuramoto_20d(self):
        """Test Kuramoto oscillators 20D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('KuramotoOscillators-20D')
        assert problem.dimension == 20
        assert problem.category == 'dynamical'
    
    def test_neural_field_70d(self):
        """Test Neural Field 70D is high-dimensional."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('NeuralField-70D')
        assert problem.dimension == 70
        assert problem.category == 'dynamical'
    
    def test_all_dynamical_callable(self):
        """Verify all dynamical functions are callable."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problems_by_category
        
        dynamical_probs = get_problems_by_category('dynamical')
        assert len(dynamical_probs) == 7
        
        for name, problem in dynamical_probs.items():
            assert callable(problem.objective), f"{name} objective not callable"
            assert problem.dimension > 0, f"{name} has invalid dimension"
            assert len(problem.bounds) == problem.dimension, f"{name} bounds mismatch"


# =============================================================================
# Chunk 2.2.3: Neural Network Weight Optimization Problems
# =============================================================================

class TestNNWeightProblems:
    """Test neural network weight optimization problems."""
    
    def test_nn_weights_count(self):
        """Verify we added 16 NN weight problems (133 total: 16 + 6 + 16 + 43 + 18 + 18 + 16)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 133
    
    def test_nn_weights_category_count(self):
        """Verify all 16 are categorized as 'nn_weights'."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        nn_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'nn_weights'}
        assert len(nn_probs) == 16
    
    def test_nn_xor_17d(self):
        """Test NN-XOR-17D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('NN-XOR-17D')
        assert problem.dimension == 17
        assert problem.category == 'nn_weights'
        
        # Test with random parameters
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(17) * 0.5)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_nn_regression_89d(self):
        """Test NN-Regression-89D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('NN-Regression-89D')
        assert problem.dimension == 89
        assert problem.category == 'nn_weights'
        
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(89) * 0.1)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_nn_mnist_1074d(self):
        """Test high-dimensional NN-MNIST-1074D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('NN-MNIST-1074D')
        assert problem.dimension == 1074
        assert problem.category == 'nn_weights'
        
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(1074) * 0.01)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_nn_large_1377d(self):
        """Test very high-dimensional NN-Large-1377D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('NN-Large-1377D')
        assert problem.dimension == 1377
        assert problem.category == 'nn_weights'
        
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(1377) * 0.01)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_sparse_autoencoder_100d(self):
        """Test SparseAutoencoder-100D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('SparseAutoencoder-100D')
        assert problem.dimension == 100
        assert problem.category == 'nn_weights'
        
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(100) * 0.1)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_neural_hessian_80d(self):
        """Test NeuralHessian-80D (expensive problem)."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        problem = get_problem('NeuralHessian-80D')
        assert problem.dimension == 80
        assert problem.category == 'nn_weights'
        assert 'expensive' in problem.description.lower()
        
        import numpy as np
        trial = MockTrial({'x': list(np.random.randn(80) * 0.1)})
        loss = problem.objective(trial)
        assert isinstance(loss, float)
        assert loss < 1e10
    
    def test_all_nn_weights_callable(self):
        """Verify all 16 NN weight problems are callable with proper metadata."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        nn_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'nn_weights'}
        
        assert len(nn_probs) == 16
        assert all(callable(prob.objective) for prob in nn_probs.values())
        assert all(prob.dimension > 0 for prob in nn_probs.values())
        assert all(prob.bounds for prob in nn_probs.values())
        assert all(len(prob.bounds) == prob.dimension for prob in nn_probs.values())


class TestMLTrainingProblems:
    """Test ML training/CV benchmark problems (Chunk 2.2.4)."""
    
    def test_ml_training_count(self):
        """Verify we added 15 ML training problems."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 133  # 16 chaotic + 6 dynamical + 16 nn_weights + 43 ml_training + 18 pde + 18 meta_control + 16 remaining
    
    def test_ml_training_category_count(self):
        """Verify all 43 are categorized as 'ml_training'."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        ml_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'ml_training'}
        assert len(ml_probs) == 43
    
    def test_svm_cv_2d(self):
        """Test SVM-CV-2D problem (small, fast test)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['SVM-CV-2D']
        assert problem.dimension == 2
        assert len(problem.bounds) == 2
        
        # Create mock trial
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        # Test with reasonable parameters
        trial = MockTrial({'x': np.array([0.0, -2.0])})  # C=1, gamma=0.01
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10  # Should not error
        assert result < 0  # Negative accuracy (minimize)
        assert result > -1  # Reasonable accuracy bound
    
    def test_ridge_cv_1d(self):
        """Test Ridge-CV-1D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['Ridge-CV-1D']
        assert problem.dimension == 1
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.0])})  # alpha=1
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result > 0  # Positive MSE
    
    def test_elasticnet_cv_2d(self):
        """Test ElasticNet-CV-2D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['ElasticNet-CV-2D']
        assert problem.dimension == 2
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.0, 0.5])})  # alpha=1, l1_ratio=0.5
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result > 0
    
    def test_rf_cv_4d(self):
        """Test RF-CV-4D problem (moderate size)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['RF-CV-4D']
        assert problem.dimension == 4
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5, 0.5, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_gradientboost_cv_3d(self):
        """Test GradientBoost-CV-3D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['GradientBoost-CV-3D']
        assert problem.dimension == 3
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.3, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_nested_cv_5d(self):
        """Test NestedCV-5D problem (expensive)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['NestedCV-5D']
        assert problem.dimension == 5
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.0, 0.0, 0.0, 0.0, 0.0])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_bayesian_acquisition_6d(self):
        """Test BayesianAcquisition-6D problem (expensive)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['BayesianAcquisition-6D']
        assert problem.dimension == 6
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    # Chunk 2.2.5: Additional ML Training Problems Tests
    def test_ml_training_count_updated(self):
        """Verify we added 10 more ML training problems (43 total, 133 overall)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        assert len(ALL_REALWORLD_PROBLEMS) == 133  # 16+6+16+43+18+18+16
    
    def test_ml_training_category_count_updated(self):
        """Verify all 43 are categorized as 'ml_training'."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        ml_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'ml_training'}
        assert len(ml_probs) == 43
    
    def test_neuralnet_dropout_20d(self):
        """Test NeuralNet-Dropout-20D problem (medium)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['NeuralNet-Dropout-20D']
        assert problem.dimension == 20
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 20)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_lightgbm_cv_4d(self):
        """Test LightGBM-CV-4D problem (small)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['LightGBM-CV-4D']
        assert problem.dimension == 4
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5, 0.5, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_stacking_ensemble_15d(self):
        """Test StackingEnsemble-15D problem (large, expensive)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['StackingEnsemble-15D']
        assert problem.dimension == 15
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 15)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_feature_selection_rfe_20d(self):
        """Test FeatureSelection-RFE-20D problem (large)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['FeatureSelection-RFE-20D']
        assert problem.dimension == 20
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 20)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_pca_components_1d(self):
        """Test PCA-Components-1D problem (small, fast)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['PCA-Components-1D']
        assert problem.dimension == 1
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_tsne_hyperparams_3d(self):
        """Test TSNE-Hyperparams-3D problem (medium, expensive)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['TSNE-Hyperparams-3D']
        assert problem.dimension == 3
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_isolation_forest_cv_3d(self):
        """Test IsolationForest-CV-3D problem (medium)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['IsolationForest-CV-3D']
        assert problem.dimension == 3
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative inlier ratio
    
    def test_autoencoder_hyperparams_5d(self):
        """Test AutoEncoder-Hyperparams-5D problem (medium)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['AutoEncoder-Hyperparams-5D']
        assert problem.dimension == 5
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5, 0.5, 0.5, 0.5, 0.5])})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result > 0  # MSE (reconstruction error)
    
    def test_ml_training_chunk_2_2_6_count(self):
        """Verify Chunk 2.2.6 added 10 more ML problems (total 43)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        ml_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'ml_training'}
        assert len(ml_probs) == 43  # 15 + 18 + 10
    
    def test_sparse_coding_30d(self):
        """Test SparseCoding-30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['SparseCoding-30D']
        assert problem.dimension == 30
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result < 0  # Negative accuracy
    
    def test_nmf_factorization_25d(self):
        """Test NMF-Factorization-25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['NMF-Factorization-25D']
        assert problem.dimension == 25
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_multi_output_regression_40d(self):
        """Test MultiOutputRegression-40D problem (high-dimensional)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['MultiOutputRegression-40D']
        assert problem.dimension == 40
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 40)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
        assert result >= 0  # Positive MSE
    
    def test_semi_supervised_30d(self):
        """Test SemiSupervised-30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['SemiSupervised-30D']
        assert problem.dimension == 30
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_transfer_learning_35d(self):
        """Test TransferLearning-35D problem (expensive)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['TransferLearning-35D']
        assert problem.dimension == 35
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 35)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_cost_sensitive_learning_22d(self):
        """Test CostSensitiveLearning-22D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        import numpy as np
        
        problem = ALL_REALWORLD_PROBLEMS['CostSensitiveLearning-22D']
        assert problem.dimension == 22
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 22)})
        result = problem.objective(trial)
        
        assert isinstance(result, (int, float))
        assert result < 1e10
    
    def test_all_ml_training_callable(self):
        """Verify all ML training problems are callable and well-formed."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        ml_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'ml_training'}
        
        assert len(ml_probs) == 43  # 15 + 18 + 10
        assert all(callable(prob.objective) for prob in ml_probs.values())
        assert all(prob.dimension > 0 for prob in ml_probs.values())
        assert all(prob.bounds for prob in ml_probs.values())
        assert all(len(prob.bounds) == prob.dimension for prob in ml_probs.values())


class TestPDEProblems:
    """Tests for PDE problems (Chunk 2.2.7)."""
    
    def test_pde_count(self):
        """Test total count includes PDE problems."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        assert len(ALL_REALWORLD_PROBLEMS) == 133  # 81 + 18 PDE + 18 meta_control + 16 remaining
    
    def test_pde_category_count(self):
        """Test PDE category has 18 problems."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        pde_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'pde'}
        assert len(pde_probs) == 18
    
    def test_burgers_equation_9d(self):
        """Test Burgers equation (1D nonlinear PDE) - 9D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['Burgers-9D']
        assert problem.dimension == 9
        assert problem.category == 'pde'
        
        # Test with mid-range parameters
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 9)})
        result = problem.objective(trial)
        
        # Should return valid finite value (not error sentinel)
        assert result < 1e10
        assert result >= 0  # Squared error
    
    def test_heat_equation_50d(self):
        """Test 1D heat equation with Fourier modes - 50D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['PDE-HeatEq-50D']
        assert problem.dimension == 50
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 50)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_wave_equation_30d(self):
        """Test wave equation - 30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['WaveEquation-30D']
        assert problem.dimension == 30
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_reaction_diffusion_30d(self):
        """Test reaction-diffusion (pattern formation) - 30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['ReactionDiffusion-30D']
        assert problem.dimension == 30
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_poisson_60d(self):
        """Test Poisson equation - 60D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['Poisson-60D']
        assert problem.dimension == 60
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 60)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_helmholtz_60d(self):
        """Test Helmholtz equation - 60D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['Helmholtz-60D']
        assert problem.dimension == 60
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 60)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_wave_equation_120d(self):
        """Test large wave equation - 120D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['WaveEquation-120D']
        assert problem.dimension == 120
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 120)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_ginzburg_landau_56d(self):
        """Test complex Ginzburg-Landau equation - 56D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['GinzburgLandau-56D']
        assert problem.dimension == 56
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 56)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_all_pde_callable(self):
        """Verify all PDE problems are callable and well-formed."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        pde_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() if v.category == 'pde'}
        
        assert len(pde_probs) == 18
        assert all(callable(prob.objective) for prob in pde_probs.values())
        assert all(prob.dimension > 0 for prob in pde_probs.values())
        assert all(prob.bounds for prob in pde_probs.values())
        assert all(len(prob.bounds) == prob.dimension for prob in pde_probs.values())


# =============================================================================
# Chunk 2.2.8: Meta-Optimization & Control Tests
# =============================================================================

class TestMetaControlProblems:
    """Test meta-optimization and control problems."""
    
    def test_genetic_algorithm_25d(self):
        """Test genetic algorithm meta-optimization - 25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['GeneticAlgorithm-25D']
        assert problem.dimension == 25
        assert problem.category == 'meta_optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_particle_swarm_30d(self):
        """Test particle swarm optimization - 30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['ParticleSwarm-30D']
        assert problem.dimension == 30
        assert problem.category == 'meta_optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_cma_es_25d(self):
        """Test CMA-ES meta-optimization - 25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['CMA-ES-25D']
        assert problem.dimension == 25
        assert problem.category == 'meta_optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_bayesian_opt_60d(self):
        """Test Bayesian optimization - 60D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['BayesianOpt-60D']
        assert problem.dimension == 60
        assert problem.category == 'meta_optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 60)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_pid_tuning_6d(self):
        """Test PID controller tuning - 6D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['PIDTuning-6D']
        assert problem.dimension == 6
        assert problem.category == 'control'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 6)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_lqr_control_8d(self):
        """Test LQR control synthesis - 8D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['LQRControl-8D']
        assert problem.dimension == 8
        assert problem.category == 'control'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 8)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_trajectory_opt_100d(self):
        """Test trajectory optimization - 100D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['TrajectoryOpt-100D']
        assert problem.dimension == 100
        assert problem.category == 'control'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 100)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_inverse_kinematics_80d(self):
        """Test inverse kinematics - 80D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['InverseKinematics-80D']
        assert problem.dimension == 80
        assert problem.category == 'control'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 80)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_all_meta_optimization_callable(self):
        """Verify all meta-optimization problems are callable and well-formed."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        meta_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() 
                      if v.category == 'meta_optimization'}
        
        assert len(meta_probs) == 11
        assert all(callable(prob.objective) for prob in meta_probs.values())
        assert all(prob.dimension > 0 for prob in meta_probs.values())
        assert all(prob.bounds for prob in meta_probs.values())
        assert all(len(prob.bounds) == prob.dimension for prob in meta_probs.values())
    
    def test_all_control_callable(self):
        """Verify all control problems are callable and well-formed."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        control_probs = {k: v for k, v in ALL_REALWORLD_PROBLEMS.items() 
                         if v.category == 'control'}
        
        assert len(control_probs) == 9
        assert all(callable(prob.objective) for prob in control_probs.values())
        assert all(prob.dimension > 0 for prob in control_probs.values())
        assert all(prob.bounds for prob in control_probs.values())
        assert all(len(prob.bounds) == prob.dimension for prob in control_probs.values())


# =============================================================================
# Chunk 2.2.9: Remaining Problems Tests
# =============================================================================

class TestRemainingProblems:
    """Test remaining real-world problems."""
    
    def test_sa_schedule_3d(self):
        """Test SA schedule optimization - 3D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['SA-Schedule-3D']
        assert problem.dimension == 3
        assert problem.category == 'meta_optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 3)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_cellular_automata_25d(self):
        """Test cellular automata - 25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['CellularAutomata-25D']
        assert problem.dimension == 25
        assert problem.category == 'simulation'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_spin_glass_150d(self):
        """Test spin glass - 150D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['SpinGlass-150D']
        assert problem.dimension == 150
        assert problem.category == 'physics'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 150)})
        result = problem.objective(trial)
        
        assert result < 1e10
    
    def test_coupled_pendulums_100d(self):
        """Test coupled pendulums - 100D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['CoupledPendulums-100D']
        assert problem.dimension == 100
        assert problem.category == 'dynamical'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 100)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_epidemic_control_25d(self):
        """Test epidemic control - 25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['EpidemicControl-25D']
        assert problem.dimension == 25
        assert problem.category == 'simulation'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_supply_chain_35d(self):
        """Test supply chain - 35D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['SupplyChain-35D']
        assert problem.dimension == 35
        assert problem.category == 'optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 35)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_graph_partition_25d(self):
        """Test graph partitioning - 25D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['GraphPartition-25D']
        assert problem.dimension == 25
        assert problem.category == 'optimization'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 25)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_risk_parity_30d(self):
        """Test risk parity portfolio - 30D problem."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        
        problem = ALL_REALWORLD_PROBLEMS['RiskParity-30D']
        assert problem.dimension == 30
        assert problem.category == 'finance'
        
        class MockTrial:
            def __init__(self, params):
                self._params = params
            def suggest_float(self, name, low, high):
                idx = int(name[1:])
                return self._params['x'][idx]
        
        trial = MockTrial({'x': np.array([0.5] * 30)})
        result = problem.objective(trial)
        
        assert result < 1e10
        assert result >= 0
    
    def test_total_count_updated(self):
        """Verify total count is now 133 (117 + 16 remaining)."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 133
    
    def test_remaining_problems_callable(self):
        """Verify all remaining problems are callable and well-formed."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS, _REMAINING_REGISTRY
        
        assert len(_REMAINING_REGISTRY) == 16
        assert all(callable(prob.objective) for prob in _REMAINING_REGISTRY.values())
        assert all(prob.dimension > 0 for prob in _REMAINING_REGISTRY.values())
        assert all(prob.bounds for prob in _REMAINING_REGISTRY.values())
        assert all(len(prob.bounds) == prob.dimension for prob in _REMAINING_REGISTRY.values())


# =============================================================================
# Section 3: Problem Classification Tests
# =============================================================================

# Placeholder for Step 3 tests


# =============================================================================
# Chunk 2.3: ML Problems (19 functions)
# =============================================================================

class TestMLProblems:
    """Tests for ML hyperparameter tuning problems."""
    
    def test_ml_problems_count(self):
        """Verify we have 19 ML problems."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        assert len(ALL_ML_PROBLEMS) == 19, f"Expected 19 ML problems, got {len(ALL_ML_PROBLEMS)}"
    
    def test_ml_tuning_category_count(self):
        """Verify ML tuning category count."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        ml_tuning = [p for p in ALL_ML_PROBLEMS.values() if p.category == 'ml_tuning']
        assert len(ml_tuning) == 14, f"Expected 14 ml_tuning problems, got {len(ml_tuning)}"
    
    def test_finance_category_count(self):
        """Verify finance category count (portfolios)."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        finance = [p for p in ALL_ML_PROBLEMS.values() if p.category == 'finance']
        assert len(finance) == 5, f"Expected 5 finance problems, got {len(finance)}"
    
    def test_lightgbm_breast_cancer(self):
        """Test LightGBM on breast cancer."""
        from RAGDA_default_args.benchmark_ml_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                # Reasonable defaults
                return 0.5
        
        problem = get_problem('LightGBM-BreastCancer-6D')
        assert problem.dimension == 6
        assert problem.category == 'ml_tuning'
        
        # Test objective is callable
        result = problem.objective(MockTrial())
        assert isinstance(result, float)
        assert 0 <= result <= 1  # Error rate between 0-1
    
    def test_xgboost_digits(self):
        """Test XGBoost on digits."""
        from RAGDA_default_args.benchmark_ml_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.5
        
        problem = get_problem('XGBoost-Digits-6D')
        assert problem.dimension == 6
        result = problem.objective(MockTrial())
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_svm_breast_cancer(self):
        """Test SVM on breast cancer (2D)."""
        from RAGDA_default_args.benchmark_ml_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.5
        
        problem = get_problem('SVM-BreastCancer-2D')
        assert problem.dimension == 2
        result = problem.objective(MockTrial())
        assert isinstance(result, float)
    
    def test_mlp_digits(self):
        """Test MLP on digits."""
        from RAGDA_default_args.benchmark_ml_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.5
        
        problem = get_problem('MLP-Digits-4D')
        assert problem.dimension == 4
        result = problem.objective(MockTrial())
        assert isinstance(result, float)
    
    def test_portfolio_5d(self):
        """Test portfolio optimization (5D)."""
        from RAGDA_default_args.benchmark_ml_problems import get_problem
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.2  # Equal weights
        
        problem = get_problem('Portfolio-5D')
        assert problem.dimension == 5
        assert problem.category == 'finance'
        result = problem.objective(MockTrial())
        assert isinstance(result, float)
        assert result >= 0  # Variance should be non-negative
    
    def test_all_ml_problems_callable(self):
        """Verify all ML problems are callable and return valid results."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.5
        
        for name, problem in ALL_ML_PROBLEMS.items():
            assert callable(problem.objective), f"{name} objective not callable"
            assert problem.dimension > 0, f"{name} has invalid dimension"
            assert len(problem.bounds) == problem.dimension, f"{name} bounds mismatch"
            
            # Test execution
            result = problem.objective(MockTrial())
            assert isinstance(result, float), f"{name} didn't return float"
            assert not np.isnan(result), f"{name} returned NaN"
            assert not np.isinf(result), f"{name} returned Inf"


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

