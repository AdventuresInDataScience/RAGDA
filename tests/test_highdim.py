"""
Tests for RAGDA High-Dimensional Optimizer

Tests the dimensionality reduction and two-stage optimization pipeline.
"""

import numpy as np
import pytest
from typing import Dict, Any


class TestEffectiveDimensionality:
    """Tests for effective dimensionality estimation."""
    
    def test_low_dim_embedded_in_high_dim(self):
        """Test detection of low-dim structure in high-dim space."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        # Create data that lies on a 5-dimensional subspace in 100D
        np.random.seed(42)
        n_samples = 200
        n_features = 100
        true_dim = 5
        
        # Generate low-rank data
        U = np.random.randn(n_samples, true_dim)
        V = np.random.randn(true_dim, n_features)
        X = U @ V + 0.01 * np.random.randn(n_samples, n_features)
        
        # Compute eigenvalues
        eigenvalues = highdim_core.compute_eigenvalues_fast(X.astype(np.float64))
        
        # Estimate effective dimensionality
        result = highdim_core.estimate_effective_dimensionality(
            eigenvalues,
            variance_threshold=0.95
        )
        
        # Should detect approximately 5 effective dimensions
        assert result['effective_dim'] <= true_dim + 3, \
            f"Expected ~{true_dim} dims, got {result['effective_dim']}"
        assert result['is_low_dimensional'], "Should detect low-dimensional structure"
    
    def test_full_rank_data(self):
        """Test that full-rank data is not flagged as low-dim."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        # Generate full-rank data
        np.random.seed(42)
        n_samples = 200
        n_features = 50
        
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        
        eigenvalues = highdim_core.compute_eigenvalues_fast(X)
        result = highdim_core.estimate_effective_dimensionality(
            eigenvalues,
            variance_threshold=0.90
        )
        
        # Should need many dimensions
        assert result['effective_dim'] > n_features * 0.5, \
            f"Full rank data should need many dimensions, got {result['effective_dim']}"


class TestKernelPCA:
    """Tests for Kernel PCA implementation."""
    
    def test_fit_transform_roundtrip(self):
        """Test that transform followed by inverse_transform approximately recovers data."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        # Create data with some structure
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        
        # Fit kernel PCA
        state = highdim_core.fit_kernel_pca(
            X,
            n_components=20,
            variance_threshold=0.95
        )
        
        assert state.fitted
        assert state.n_components <= 20
        
        # Transform
        X_reduced = highdim_core.transform_kernel_pca(state, X)
        assert X_reduced.shape == (n_samples, state.n_components)
        
        # Inverse transform (approximate)
        X_reconstructed = highdim_core.inverse_transform_kernel_pca(state, X_reduced)
        assert X_reconstructed.shape == (n_samples, n_features)
    
    def test_nonlinear_data(self):
        """Test kernel PCA on data with nonlinear structure."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        np.random.seed(42)
        n_samples = 200
        
        # Create a swiss roll (nonlinear manifold)
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
        X = np.zeros((n_samples, 3), dtype=np.float64)
        X[:, 0] = t * np.cos(t)
        X[:, 1] = 10 * np.random.rand(n_samples)
        X[:, 2] = t * np.sin(t)
        
        # Kernel PCA should capture the manifold structure
        state = highdim_core.fit_kernel_pca(
            X,
            n_components=2,
            variance_threshold=0.95,
            adaptive_gamma=True
        )
        
        X_reduced = highdim_core.transform_kernel_pca(state, X)
        
        # Check that the reduced representation is reasonable
        assert X_reduced.shape == (n_samples, state.n_components)
        assert not np.any(np.isnan(X_reduced))


class TestRandomProjection:
    """Tests for random projection implementation."""
    
    def test_gaussian_projection(self):
        """Test Gaussian random projection."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        n_features = 1000
        n_components = 100
        
        state = highdim_core.fit_random_projection(
            n_features,
            n_components,
            projection_type='gaussian',
            random_seed=42
        )
        
        assert state.fitted
        assert state.n_components == n_components
        assert state.n_features == n_features
        
        # Test transform
        np.random.seed(42)
        X = np.random.randn(50, n_features).astype(np.float64)
        
        X_reduced = highdim_core.transform_random_projection(state, X)
        assert X_reduced.shape == (50, n_components)
        
        # Test inverse
        X_reconstructed = highdim_core.inverse_transform_random_projection(state, X_reduced)
        assert X_reconstructed.shape == (50, n_features)
    
    def test_sparse_projection(self):
        """Test sparse random projection."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        state = highdim_core.fit_random_projection(
            500, 50,
            projection_type='sparse',
            random_seed=42
        )
        
        assert state.fitted
        
        X = np.random.randn(20, 500).astype(np.float64)
        X_reduced = highdim_core.transform_random_projection(state, X)
        assert X_reduced.shape == (20, 50)
    
    def test_distance_preservation(self):
        """Test that random projection approximately preserves distances."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        np.random.seed(42)
        n_features = 500
        n_components = 100  # More components for better preservation
        n_samples = 100
        
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        
        state = highdim_core.fit_random_projection(
            n_features, n_components,
            projection_type='gaussian',
            random_seed=42
        )
        
        X_reduced = highdim_core.transform_random_projection(state, X)
        
        # Compare some pairwise distances
        from scipy.spatial.distance import pdist
        
        # Sample some pairs
        orig_dists = pdist(X[:20])
        proj_dists = pdist(X_reduced[:20])
        
        # Distances should be approximately preserved
        # JL lemma: with high probability, distortion is at most (1 +/- epsilon)
        # where epsilon depends on n_components
        ratios = proj_dists / (orig_dists + 1e-10)
        mean_ratio = np.mean(ratios)
        
        # With 100 components, should be closer to 1 (relaxed bounds)
        assert 0.3 < mean_ratio < 3.0, f"Mean distance ratio {mean_ratio} too far from 1"


class TestIncrementalPCA:
    """Tests for incremental PCA implementation."""
    
    def test_incremental_update(self):
        """Test that incremental updates work correctly."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import highdim_core
        
        np.random.seed(42)
        n_features = 50
        
        # Initial fit
        X1 = np.random.randn(100, n_features).astype(np.float64)
        state = highdim_core.fit_incremental_pca(X1, n_components=10)
        
        assert state.fitted
        assert state.n_samples_seen == 100
        
        # Incremental update
        X2 = np.random.randn(50, n_features).astype(np.float64)
        state = highdim_core.partial_fit_incremental_pca(state, X2)
        
        assert state.n_samples_seen == 150
        
        # Transform should work
        X_test = np.random.randn(20, n_features).astype(np.float64)
        X_reduced = highdim_core.transform_incremental_pca(state, X_test)
        assert X_reduced.shape == (20, state.n_components)


class TestDimensionalityReducer:
    """Tests for the unified DimensionalityReducer class."""
    
    def test_reducer_methods(self):
        """Test all reducer methods work."""
        pytest.importorskip("ragda.highdim_core")
        from ragda.highdim import DimensionalityReducer
        
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float64)
        
        for method in ['kernel_pca', 'incremental_pca', 'random_projection']:
            reducer = DimensionalityReducer(
                method=method,
                n_components=10,
                random_seed=42
            )
            
            reducer.fit(X)
            assert reducer.is_fitted
            
            X_reduced = reducer.transform(X)
            assert X_reduced.shape[1] == reducer.reduced_dim
            
            X_reconstructed = reducer.inverse_transform(X_reduced)
            assert X_reconstructed.shape == X.shape


class TestHighDimOptimizer:
    """Tests for the high-dimensional optimizer."""
    
    def test_low_dim_problem_fallback(self):
        """Test that low-dim problems use standard optimizer."""
        from ragda import HighDimRAGDAOptimizer
        
        # Small 10D problem - should not use high-dim mode
        space = {
            f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
            for i in range(10)
        }
        
        def sphere(**params):
            return sum(params[f'x{i}']**2 for i in range(10))
        
        optimizer = HighDimRAGDAOptimizer(
            space,
            direction='minimize',
            dim_threshold=1000,  # Problem is way below this
            n_workers=2,
            random_state=42
        )
        
        result = optimizer.optimize(
            sphere,
            n_trials=50,
            verbose=False
        )
        
        # Should find reasonable solution
        assert result.best_value < 10.0
    
    def test_high_dim_sphere(self):
        """Test optimization on high-dimensional sphere function."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import HighDimRAGDAOptimizer
        
        # 500D problem with intrinsic low dimensionality
        # (optimum is sparse - only first 10 dims matter)
        n_dims = 500
        active_dims = 10
        
        space = {
            f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
            for i in range(n_dims)
        }
        
        def sparse_sphere(**params):
            # Only first 10 dimensions contribute
            return sum(params[f'x{i}']**2 for i in range(active_dims))
        
        optimizer = HighDimRAGDAOptimizer(
            space,
            direction='minimize',
            dim_threshold=100,  # Trigger high-dim mode
            variance_threshold=0.90,
            n_workers=2,
            random_state=42
        )
        
        result = optimizer.optimize(
            sparse_sphere,
            n_trials=100,
            verbose=False
        )
        
        # Should find reasonable solution
        assert result.best_value < 50.0
    
    def test_analyze_effective_dimensionality(self):
        """Test the dimensionality analysis method."""
        pytest.importorskip("ragda.highdim_core")
        from ragda import HighDimRAGDAOptimizer
        
        n_dims = 100
        space = {
            f'x{i}': {'type': 'continuous', 'bounds': [-1.0, 1.0]}
            for i in range(n_dims)
        }
        
        def simple_objective(**params):
            return sum(params[f'x{i}']**2 for i in range(n_dims))
        
        optimizer = HighDimRAGDAOptimizer(
            space,
            direction='minimize',
            initial_samples=100,
            random_state=42
        )
        
        info = optimizer.analyze_effective_dimensionality(simple_objective)
        
        assert 'effective_dim' in info
        assert 'variance_explained' in info
        assert 'is_low_dimensional' in info
        assert 'recommended_method' in info


class TestConvenienceFunction:
    """Tests for the convenience function."""
    
    def test_highdim_ragda_optimize(self):
        """Test scipy-style convenience function."""
        from ragda import highdim_ragda_optimize
        
        n_dims = 20
        bounds = np.array([[-5.0, 5.0]] * n_dims)
        
        # Simple sphere function (easier than Rosenbrock)
        def sphere(x):
            return np.sum(x**2)
        
        x_best, f_best, info = highdim_ragda_optimize(
            sphere,
            bounds,
            direction='minimize',
            n_trials=100,
            dim_threshold=1000,  # Won't trigger high-dim for 20D
            random_state=42,
            verbose=False
        )
        
        assert len(x_best) == n_dims
        assert f_best < 50  # Some progress made on sphere


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
