"""
Integration tests for Scipy-style API using real benchmark functions.

Tests the Scipy adapter with actual optimization problems from the benchmark suite.
"""

import pytest
import numpy as np
from ragda import minimize, maximize


class TestScipyBenchmarks:
    """Test Scipy-style API with standard benchmark functions."""
    
    def test_ackley_2d(self):
        """Test Scipy API on 2D Ackley function."""
        def ackley(x):
            # Ackley function
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1/2)) - np.exp(sum2/2) + a + np.e
        
        result = minimize(
            ackley,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 150, 'random_state': 42}
        )
        
        # Ackley global minimum is at (0, 0) with value 0
        assert result.success
        assert result.fun < 0.5
        assert abs(result.x[0]) < 2.0
        assert abs(result.x[1]) < 2.0
    
    def test_rosenbrock_2d(self):
        """Test Scipy API on 2D Rosenbrock function."""
        def rosenbrock(x):
            return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        result = minimize(
            rosenbrock,
            bounds=[(-2, 2), (-2, 2)],
            options={'maxiter': 200, 'random_state': 43}
        )
        
        # Rosenbrock global minimum is at (1, 1) with value 0
        assert result.success
        assert result.fun < 10.0
        # Allow wide tolerance since Rosenbrock is hard
        assert abs(result.x[0] - 1.0) < 1.0
        assert abs(result.x[1] - 1.0) < 1.0
    
    def test_rastrigin_2d(self):
        """Test Scipy API on 2D Rastrigin function (multimodal)."""
        def rastrigin(x):
            A = 10
            return 2 * A + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        
        result = minimize(
            rastrigin,
            bounds=[(-5.12, 5.12), (-5.12, 5.12)],
            options={'maxiter': 200, 'random_state': 44}
        )
        
        # Rastrigin global minimum is at (0, 0) with value 0
        # It's highly multimodal, so we allow more tolerance
        assert result.success
        assert result.fun < 5.0
        assert abs(result.x[0]) < 3.0
        assert abs(result.x[1]) < 3.0
    
    def test_sphere_5d(self):
        """Test Scipy API on 5D sphere function."""
        def sphere(x):
            return np.sum(x**2)
        
        result = minimize(
            sphere,
            bounds=[(-10, 10)] * 5,
            options={'maxiter': 150, 'random_state': 45}
        )
        
        # Sphere global minimum is at (0,...,0) with value 0
        assert result.success
        assert result.fun < 2.0
        assert len(result.x) == 5
        assert all(abs(xi) < 2.0 for xi in result.x)
    
    def test_sphere_10d(self):
        """Test Scipy API on higher-dimensional sphere function."""
        def sphere(x):
            return np.sum(x**2)
        
        result = minimize(
            sphere,
            bounds=[(-10, 10)] * 10,
            options={'maxiter': 200, 'random_state': 46}
        )
        
        assert result.success
        assert result.fun < 5.0
        assert len(result.x) == 10
    
    def test_griewank_2d(self):
        """Test Scipy API on 2D Griewank function (multimodal)."""
        def griewank(x):
            sum_term = np.sum(x**2) / 4000
            prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
            return sum_term - prod_term + 1
        
        result = minimize(
            griewank,
            bounds=[(-100, 100), (-100, 100)],
            options={'maxiter': 150, 'random_state': 47}
        )
        
        # Griewank global minimum is at (0, 0) with value 0
        assert result.success
        assert result.fun < 1.0


class TestScipyMaximization:
    """Test Scipy-style maximize() function."""
    
    def test_maximize_negative_sphere(self):
        """Test maximizing negative sphere (should find (0, 0))."""
        def neg_sphere(x):
            return -np.sum(x**2)
        
        result = maximize(
            neg_sphere,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 100, 'random_state': 48}
        )
        
        assert result.success
        assert result.fun > -0.5  # Maximum is 0 at (0, 0)
        assert abs(result.x[0]) < 1.0
        assert abs(result.x[1]) < 1.0
    
    def test_maximize_inverted_quadratic(self):
        """Test maximizing inverted quadratic."""
        def inverted_quad(x):
            # Maximum at (2, 3) with value 0
            return -(x[0] - 2)**2 - (x[1] - 3)**2
        
        result = maximize(
            inverted_quad,
            bounds=[(-10, 10), (-10, 10)],
            options={'maxiter': 100, 'random_state': 49}
        )
        
        assert result.success
        assert result.fun > -1.0
        assert abs(result.x[0] - 2.0) < 1.5
        assert abs(result.x[1] - 3.0) < 1.5
    
    def test_maximize_with_offset(self):
        """Test maximization with offset objective."""
        def offset_func(x):
            # Maximum at (5, -3) with value 10
            return 10 - (x[0] - 5)**2 - (x[1] + 3)**2
        
        result = maximize(
            offset_func,
            bounds=[(0, 10), (-10, 0)],
            options={'maxiter': 100, 'random_state': 50}
        )
        
        assert result.success
        assert result.fun > 8.0
        assert 0 <= result.x[0] <= 10
        assert -10 <= result.x[1] <= 0


class TestScipyHighDimensional:
    """Test Scipy API on higher-dimensional problems."""
    
    def test_sphere_20d(self):
        """Test 20D sphere optimization."""
        def sphere(x):
            return np.sum(x**2)
        
        result = minimize(
            sphere,
            bounds=[(-5, 5)] * 20,
            options={'maxiter': 300, 'random_state': 51}
        )
        
        assert result.success
        assert result.fun < 10.0
        assert len(result.x) == 20
    
    def test_rosenbrock_5d(self):
        """Test 5D Rosenbrock (harder than 2D)."""
        def rosenbrock_nd(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        result = minimize(
            rosenbrock_nd,
            bounds=[(-2, 2)] * 5,
            options={'maxiter': 300, 'random_state': 52}
        )
        
        assert result.success
        assert result.fun < 50.0  # Rosenbrock is hard in higher dimensions
    
    def test_ackley_10d(self):
        """Test 10D Ackley function."""
        def ackley(x):
            a, b, c = 20, 0.2, 2 * np.pi
            d = len(x)
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e
        
        result = minimize(
            ackley,
            bounds=[(-5, 5)] * 10,
            options={'maxiter': 250, 'random_state': 53}
        )
        
        assert result.success
        assert result.fun < 2.0


class TestScipyComplexLandscapes:
    """Test Scipy API on complex optimization landscapes."""
    
    def test_levy_function(self):
        """Test Levy function (multimodal with plateaus)."""
        def levy(x):
            w = 1 + (x - 1) / 4
            term1 = np.sin(np.pi * w[0])**2
            term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
            term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
            return term1 + term2 + term3
        
        result = minimize(
            levy,
            bounds=[(-10, 10)] * 3,
            options={'maxiter': 200, 'random_state': 54}
        )
        
        # Levy global minimum is at (1, 1, 1) with value 0
        assert result.success
        assert result.fun < 2.0
    
    def test_schwefel_2d(self):
        """Test Schwefel function (deceptive multimodal)."""
        def schwefel(x):
            d = len(x)
            return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        
        result = minimize(
            schwefel,
            bounds=[(-500, 500), (-500, 500)],
            options={'maxiter': 200, 'random_state': 55}
        )
        
        # Schwefel is very difficult - just verify reasonable progress
        assert result.success
        assert result.fun < 800  # Should make significant progress from worst case
    
    def test_asymmetric_bounds(self):
        """Test with highly asymmetric bounds."""
        def shifted_sphere(x):
            # Optimal at (7, -5, 3)
            return (x[0] - 7)**2 + (x[1] + 5)**2 + (x[2] - 3)**2
        
        result = minimize(
            shifted_sphere,
            bounds=[(0, 10), (-10, 0), (0, 5)],
            options={'maxiter': 150, 'random_state': 56}
        )
        
        assert result.success
        assert result.fun < 2.0
        # Verify bounds are respected
        assert 0 <= result.x[0] <= 10
        assert -10 <= result.x[1] <= 0
        assert 0 <= result.x[2] <= 5


class TestScipyRobustness:
    """Test Scipy API error handling and edge cases."""
    
    def test_noisy_objective(self):
        """Test with noisy objective function."""
        np.random.seed(57)
        
        def noisy_sphere(x):
            noise = np.random.randn() * 0.1
            return np.sum(x**2) + noise
        
        result = minimize(
            noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 100, 'random_state': 57}
        )
        
        # Should still converge despite noise
        assert result.success
        assert result.fun < 1.0
    
    def test_flat_region(self):
        """Test with flat regions in objective."""
        def plateau_function(x):
            # Flat in middle, steep at edges
            r = np.sqrt(np.sum(x**2))
            if r < 1.0:
                return 0.0
            else:
                return (r - 1.0)**2
        
        result = minimize(
            plateau_function,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 100, 'random_state': 58}
        )
        
        assert result.success
        assert result.fun < 0.5
    
    def test_narrow_optimum(self):
        """Test with very narrow optimum."""
        def narrow_valley(x):
            # Very narrow valley along x[0] = 0
            return 100 * x[0]**2 + x[1]**2
        
        result = minimize(
            narrow_valley,
            bounds=[(-5, 5), (-5, 5)],
            options={'maxiter': 100, 'random_state': 59}
        )
        
        assert result.success
        assert result.fun < 1.0
    
    def test_single_dimension(self):
        """Test 1D optimization."""
        def quadratic_1d(x):
            return (x[0] - 3)**2
        
        result = minimize(
            quadratic_1d,
            bounds=[(-10, 10)],
            options={'maxiter': 50, 'random_state': 60}
        )
        
        assert result.success
        assert result.fun < 1.0
        assert len(result.x) == 1
        assert abs(result.x[0] - 3) < 1.0
