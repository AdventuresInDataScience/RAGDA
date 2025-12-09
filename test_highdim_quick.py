"""Quick test of high-dimensional optimizer."""
import numpy as np
from ragda import HighDimRAGDAOptimizer, highdim_core

print("=" * 70)
print("Test 1: Verify pure C dimensionality detection on synthetic low-rank data")
print("=" * 70)

# Create data that lies in a 5D subspace of 100D
np.random.seed(42)
n_samples = 200
true_dim = 5
n_features = 100

# Low-rank structure: X = U @ V where U is (200, 5) and V is (5, 100)
U = np.random.randn(n_samples, true_dim)
V = np.random.randn(true_dim, n_features)
X_lowrank = (U @ V).astype(np.float64)

eigenvalues = highdim_core.compute_eigenvalues_fast(X_lowrank)
result = highdim_core.estimate_effective_dimensionality(eigenvalues, 0.95)

print(f"True intrinsic dim: {true_dim}")
print(f"Detected effective dim: {result['effective_dim']}")
print(f"Is low-dimensional: {result['is_low_dimensional']}")
print(f"Variance explained by top 5: {result['variance_explained'][4]:.4f}")
print()

print("=" * 70)
print("Test 2: High-dimensional optimization (falls back to standard)")
print("=" * 70)

# 200D problem - the sampling is uniform so won't have intrinsic low-D structure
# but the OBJECTIVE has sparse structure (only 10 dims matter)
n_dims = 200
active_dims = 10

space = {
    f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
    for i in range(n_dims)
}

def sparse_quadratic(**params):
    # Only first 10 dimensions matter
    return sum((params[f'x{i}'] - 1.0)**2 for i in range(active_dims))

optimizer = HighDimRAGDAOptimizer(
    space,
    direction='minimize',
    dim_threshold=100,
    variance_threshold=0.90,
    n_workers=2,
    random_state=42,
    initial_samples=200
)

result = optimizer.optimize(
    sparse_quadratic,
    n_trials=100,
    verbose=False
)

print(f'Best value: {result.best_value:.6f}')
print(f'Active dims values (should be close to 1.0):')
for i in range(min(5, active_dims)):
    print(f'  x{i} = {result.best_params[f"x{i}"]:.4f}')

print()
print("=" * 70)
print("Test 3: Kernel PCA performance check")
print("=" * 70)

import time

# Benchmark pure C implementation
X_test = np.random.randn(200, 500).astype(np.float64)

start = time.perf_counter()
state = highdim_core.fit_kernel_pca(X_test, n_components=50)
fit_time = time.perf_counter() - start
print(f"Kernel PCA fit (200 samples, 500 features -> 50 components): {fit_time*1000:.1f}ms")

start = time.perf_counter()
X_reduced = highdim_core.transform_kernel_pca(state, X_test)
transform_time = time.perf_counter() - start
print(f"Transform: {transform_time*1000:.1f}ms")

start = time.perf_counter()
X_recon = highdim_core.inverse_transform_kernel_pca(state, X_reduced)
inverse_time = time.perf_counter() - start
print(f"Inverse transform: {inverse_time*1000:.1f}ms")

print()
print("All tests completed successfully!")
