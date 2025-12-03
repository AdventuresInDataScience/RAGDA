# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True
# cython: binding=False
# cython: linetrace=False
# cython: profile=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
RAGDA High-Dimensional Core - Cython Implementation

Fast dimensionality reduction operations:
- Effective dimensionality testing via eigenvalue analysis
- Kernel PCA with RBF kernel (captures nonlinearities)
- Fast random projections (Johnson-Lindenstrauss)
- Incremental PCA updates for adaptive refinement
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, fabs, pow as c_pow
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
cimport cython

# Initialize NumPy C-API
cnp.import_array()

# Type definitions
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t INT32_t
ctypedef cnp.int64_t INT64_t

# Constants
DEF MAX_COMPONENTS = 10000
DEF EPSILON = 1e-10


# =============================================================================
# Effective Dimensionality Analysis
# =============================================================================

def compute_eigenvalues_fast(
    cnp.ndarray[DTYPE_t, ndim=2] X,
    bint center = True
):
    """
    Compute eigenvalues of the covariance matrix efficiently.
    
    Uses SVD which is more numerically stable and faster for n_samples < n_features.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix
    center : bool
        Whether to center the data
    
    Returns
    -------
    eigenvalues : ndarray, shape (min(n_samples, n_features),)
        Sorted eigenvalues in descending order
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_centered
    cdef cnp.ndarray[DTYPE_t, ndim=1] mean
    cdef cnp.ndarray[DTYPE_t, ndim=1] s
    cdef int i, j
    
    if center:
        mean = np.mean(X, axis=0)
        X_centered = X - mean
    else:
        X_centered = X
    
    # Use SVD - more stable than eigendecomposition of covariance
    # eigenvalues of cov matrix = singular values^2 / (n-1)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    
    # Convert to eigenvalues
    cdef cnp.ndarray[DTYPE_t, ndim=1] eigenvalues = (s * s) / (n_samples - 1)
    
    return eigenvalues


def estimate_effective_dimensionality(
    cnp.ndarray[DTYPE_t, ndim=1] eigenvalues,
    double variance_threshold = 0.90,
    double eigenvalue_ratio_threshold = 0.01
):
    """
    Estimate effective dimensionality using multiple criteria.
    
    Uses:
    1. Cumulative variance explained threshold
    2. Eigenvalue ratio test (drop below threshold of max)
    3. Participation ratio (entropy-based)
    
    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues in descending order
    variance_threshold : float
        Fraction of variance to explain (e.g., 0.90 = 90%)
    eigenvalue_ratio_threshold : float
        Minimum eigenvalue as fraction of largest
    
    Returns
    -------
    result : dict
        effective_dim : int
            Recommended number of dimensions
        variance_explained : ndarray
            Cumulative variance explained per component
        participation_ratio : float
            Participation ratio (1 = all equal, D = one dominant)
        is_low_dimensional : bool
            Whether significant dimensionality reduction is possible
        confidence : float
            Confidence in the estimate (0-1)
    """
    cdef int n = len(eigenvalues)
    cdef int i
    cdef double total_var, cumsum, pr_sum, pr
    cdef double max_eigenvalue
    cdef int dim_by_variance = n
    cdef int dim_by_ratio = n
    
    if n == 0:
        return {
            'effective_dim': 0,
            'variance_explained': np.array([]),
            'participation_ratio': 0.0,
            'is_low_dimensional': False,
            'confidence': 0.0
        }
    
    # Ensure non-negative eigenvalues
    cdef cnp.ndarray[DTYPE_t, ndim=1] eigs = np.maximum(eigenvalues, 0.0)
    
    # Total variance
    total_var = np.sum(eigs)
    if total_var < EPSILON:
        return {
            'effective_dim': 1,
            'variance_explained': np.ones(1),
            'participation_ratio': 1.0,
            'is_low_dimensional': True,
            'confidence': 1.0
        }
    
    # Cumulative variance explained
    cdef cnp.ndarray[DTYPE_t, ndim=1] variance_explained = np.cumsum(eigs) / total_var
    
    # Find dimension by variance threshold
    for i in range(n):
        if variance_explained[i] >= variance_threshold:
            dim_by_variance = i + 1
            break
    
    # Find dimension by eigenvalue ratio
    max_eigenvalue = eigs[0]
    if max_eigenvalue > EPSILON:
        for i in range(n):
            if eigs[i] / max_eigenvalue < eigenvalue_ratio_threshold:
                dim_by_ratio = i
                break
    
    # Participation ratio (inverse of IPR - Inverse Participation Ratio)
    # PR = (sum(lambda))^2 / sum(lambda^2)
    # High PR means eigenvalues are spread out, low PR means concentrated
    cdef cnp.ndarray[DTYPE_t, ndim=1] normalized_eigs = eigs / total_var
    pr_sum = np.sum(normalized_eigs * normalized_eigs)
    if pr_sum > EPSILON:
        pr = 1.0 / (n * pr_sum)  # Normalized to [0, 1]
    else:
        pr = 1.0
    
    # Take minimum of the two criteria
    cdef int effective_dim = min(dim_by_variance, dim_by_ratio)
    effective_dim = max(1, effective_dim)  # At least 1 dimension
    
    # Determine if data is effectively low-dimensional
    cdef bint is_low_dim = (effective_dim < n * 0.5) and (pr < 0.5)
    
    # Confidence based on eigenvalue decay rate
    # Fast decay = high confidence, slow decay = low confidence
    cdef double decay_rate = 0.0
    cdef int k
    if n > 1 and eigs[0] > EPSILON:
        # Fit exponential decay: eigenvalue[i] ~ eigenvalue[0] * exp(-decay * i)
        # Use simple approximation: decay = -log(eigenvalue[k] / eigenvalue[0]) / k
        k = min(10, n - 1)
        if eigs[k] > EPSILON:
            decay_rate = -log(eigs[k] / eigs[0]) / k
    
    cdef double confidence = min(1.0, decay_rate / 2.0)  # Normalize
    
    return {
        'effective_dim': effective_dim,
        'variance_explained': variance_explained,
        'participation_ratio': pr,
        'is_low_dimensional': is_low_dim,
        'confidence': confidence,
        'dim_by_variance': dim_by_variance,
        'dim_by_ratio': dim_by_ratio
    }


# =============================================================================
# Kernel Functions (nogil)
# =============================================================================

cdef inline double rbf_kernel_element(
    const double* x1,
    const double* x2,
    int n_features,
    double gamma
) noexcept nogil:
    """Compute RBF kernel k(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
    cdef double diff, sq_dist = 0.0
    cdef int i
    
    for i in range(n_features):
        diff = x1[i] - x2[i]
        sq_dist += diff * diff
    
    return exp(-gamma * sq_dist)


cdef void compute_rbf_kernel_matrix(
    const double* X,
    int n_samples,
    int n_features,
    double gamma,
    double* K
) noexcept nogil:
    """
    Compute RBF kernel matrix K[i,j] = exp(-gamma * ||X[i] - X[j]||^2)
    K is stored in row-major order, shape (n_samples, n_samples)
    """
    cdef int i, j
    cdef double k_val
    
    for i in range(n_samples):
        K[i * n_samples + i] = 1.0  # Diagonal
        for j in range(i + 1, n_samples):
            k_val = rbf_kernel_element(
                &X[i * n_features],
                &X[j * n_features],
                n_features,
                gamma
            )
            K[i * n_samples + j] = k_val
            K[j * n_samples + i] = k_val


cdef void center_kernel_matrix(
    double* K,
    int n
) noexcept nogil:
    """
    Center kernel matrix in feature space.
    K_centered = K - 1_n K - K 1_n + 1_n K 1_n
    where 1_n is the n x n matrix of 1/n
    """
    cdef int i, j
    cdef double* row_means = <double*>malloc(n * sizeof(double))
    cdef double total_mean = 0.0
    
    if row_means == NULL:
        return
    
    # Compute row means
    for i in range(n):
        row_means[i] = 0.0
        for j in range(n):
            row_means[i] += K[i * n + j]
        row_means[i] /= n
        total_mean += row_means[i]
    total_mean /= n
    
    # Center: K_ij = K_ij - row_mean[i] - row_mean[j] + total_mean
    for i in range(n):
        for j in range(n):
            K[i * n + j] = K[i * n + j] - row_means[i] - row_means[j] + total_mean
    
    free(row_means)


# =============================================================================
# Kernel PCA Implementation
# =============================================================================

cdef class KernelPCAState:
    """
    Holds the state for Kernel PCA transformations.
    Allows fast forward and inverse transforms.
    """
    cdef public cnp.ndarray X_fit          # Training data (n_samples, n_features)
    cdef public cnp.ndarray alphas         # Eigenvectors scaled (n_samples, n_components)
    cdef public cnp.ndarray lambdas        # Eigenvalues (n_components,)
    cdef public cnp.ndarray K_fit_rows     # Row sums of training kernel
    cdef public double K_fit_mean          # Mean of training kernel
    cdef public double gamma               # RBF kernel parameter
    cdef public int n_components           # Number of components
    cdef public int n_samples              # Number of training samples
    cdef public int n_features             # Original feature dimension
    cdef public bint fitted
    
    def __init__(self):
        self.fitted = False
        self.gamma = 1.0
        self.n_components = 0
        self.n_samples = 0
        self.n_features = 0


def fit_kernel_pca(
    cnp.ndarray[DTYPE_t, ndim=2] X,
    int n_components = 0,
    double variance_threshold = 0.90,
    double gamma = 0.0,
    bint adaptive_gamma = True
):
    """
    Fit Kernel PCA with RBF kernel.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data
    n_components : int
        Number of components. If 0, determined by variance_threshold.
    variance_threshold : float
        Variance to explain if n_components=0
    gamma : float
        RBF kernel parameter. If 0, uses 1/n_features (median heuristic approx).
    adaptive_gamma : bool
        If True and gamma=0, use median distance heuristic
    
    Returns
    -------
    state : KernelPCAState
        Fitted state for transforms
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int i, j
    
    # Determine gamma
    cdef double gamma_val = gamma
    if gamma_val <= 0:
        if adaptive_gamma:
            # Median heuristic: gamma = 1 / (2 * median(||x_i - x_j||^2))
            # Approximate with mean for speed
            gamma_val = _estimate_gamma_fast(X)
        else:
            gamma_val = 1.0 / n_features
    
    # Compute kernel matrix
    cdef cnp.ndarray[DTYPE_t, ndim=2] K = np.empty((n_samples, n_samples), dtype=np.float64)
    cdef double* K_ptr = &K[0, 0]
    cdef double* X_ptr = &X[0, 0]
    
    with nogil:
        compute_rbf_kernel_matrix(X_ptr, n_samples, n_features, gamma_val, K_ptr)
    
    # Store row means and total mean before centering
    cdef cnp.ndarray[DTYPE_t, ndim=1] K_row_means = np.mean(K, axis=1)
    cdef double K_mean = np.mean(K)
    
    # Center kernel matrix
    with nogil:
        center_kernel_matrix(K_ptr, n_samples)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Determine number of components
    cdef int n_comp
    eigenvalues = np.maximum(eigenvalues, 0.0)  # Numerical stability
    total_var = np.sum(eigenvalues)
    
    if n_components > 0:
        n_comp = min(n_components, n_samples)
    else:
        # Use variance threshold
        if total_var > EPSILON:
            cum_var = np.cumsum(eigenvalues) / total_var
            n_comp = np.searchsorted(cum_var, variance_threshold) + 1
            n_comp = min(n_comp, n_samples)
        else:
            n_comp = 1
    
    # Scale eigenvectors: alphas = eigenvectors / sqrt(eigenvalues)
    # Handle zero eigenvalues
    cdef cnp.ndarray[DTYPE_t, ndim=2] alphas = np.zeros((n_samples, n_comp), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] lambdas = eigenvalues[:n_comp].copy()
    
    for i in range(n_comp):
        if lambdas[i] > EPSILON:
            alphas[:, i] = eigenvectors[:, i] / sqrt(lambdas[i])
        else:
            alphas[:, i] = eigenvectors[:, i]
    
    # Create state
    state = KernelPCAState()
    state.X_fit = X.copy()
    state.alphas = alphas
    state.lambdas = lambdas
    state.K_fit_rows = K_row_means
    state.K_fit_mean = K_mean
    state.gamma = gamma_val
    state.n_components = n_comp
    state.n_samples = n_samples
    state.n_features = n_features
    state.fitted = True
    
    return state


# =============================================================================
# Fast Median Computation (pure C, nogil)
# =============================================================================

cdef void _swap_double(double* a, double* b) noexcept nogil:
    """Swap two doubles."""
    cdef double temp = a[0]
    a[0] = b[0]
    b[0] = temp


cdef double _quickselect_median(double* arr, int n) noexcept nogil:
    """
    Find median using Quickselect algorithm - O(n) average.
    Modifies the array in place.
    """
    if n == 0:
        return 0.0
    if n == 1:
        return arr[0]
    
    cdef int k = n // 2  # Median index
    cdef int left = 0
    cdef int right = n - 1
    cdef int pivot_idx, store_idx, i
    cdef double pivot_val
    
    while left < right:
        # Choose pivot (middle element)
        pivot_idx = left + (right - left) // 2
        pivot_val = arr[pivot_idx]
        
        # Move pivot to end
        _swap_double(&arr[pivot_idx], &arr[right])
        
        # Partition
        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot_val:
                _swap_double(&arr[store_idx], &arr[i])
                store_idx += 1
        
        # Move pivot to final position
        _swap_double(&arr[store_idx], &arr[right])
        
        # Narrow search
        if store_idx == k:
            break
        elif store_idx < k:
            left = store_idx + 1
        else:
            right = store_idx - 1
    
    return arr[k]


cdef double _compute_median_sq_distance_nogil(
    const double* X,
    int n_samples,
    int n_features,
    int max_pairs
) noexcept nogil:
    """
    Compute median squared distance using pure C with Quickselect.
    Fully nogil - no Python overhead.
    """
    cdef int i, j, k, pair_idx
    cdef double diff, sq_dist
    cdef double* sq_dists
    cdef double median_val
    cdef int actual_pairs
    
    # Calculate actual number of pairs
    actual_pairs = (n_samples * (n_samples - 1)) // 2
    if actual_pairs > max_pairs:
        actual_pairs = max_pairs
    
    if actual_pairs == 0:
        return 1.0
    
    # Allocate buffer for distances
    sq_dists = <double*>malloc(actual_pairs * sizeof(double))
    if sq_dists == NULL:
        return 1.0  # Fallback on allocation failure
    
    # Compute pairwise squared distances
    pair_idx = 0
    for i in range(n_samples):
        if pair_idx >= actual_pairs:
            break
        for j in range(i + 1, n_samples):
            if pair_idx >= actual_pairs:
                break
            
            # Squared Euclidean distance
            sq_dist = 0.0
            for k in range(n_features):
                diff = X[i * n_features + k] - X[j * n_features + k]
                sq_dist += diff * diff
            
            sq_dists[pair_idx] = sq_dist
            pair_idx += 1
    
    if pair_idx == 0:
        free(sq_dists)
        return 1.0
    
    # Find median using Quickselect - O(n) average
    median_val = _quickselect_median(sq_dists, pair_idx)
    
    free(sq_dists)
    return median_val


def _estimate_gamma_fast(cnp.ndarray[DTYPE_t, ndim=2] X):
    """
    Estimate RBF gamma using median heuristic.
    
    gamma = 1 / (2 * median(||x_i - x_j||^2))
    
    Uses pure C implementation with Quickselect for O(n) median finding.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_sub
    cdef int max_pairs = 2000
    cdef double median_sq_dist
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_sub
    
    # Subsample for speed if large (100 samples gives 4950 pairs)
    if n_samples > 100:
        idx = np.random.choice(n_samples, 100, replace=False)
        X_sub = np.ascontiguousarray(X[idx], dtype=np.float64)
        n_sub = 100
    else:
        X_sub = np.ascontiguousarray(X, dtype=np.float64)
        n_sub = n_samples
    
    # Compute median using pure C (no GIL)
    with nogil:
        median_sq_dist = _compute_median_sq_distance_nogil(
            &X_sub[0, 0],
            n_sub,
            n_features,
            max_pairs
        )
    
    if median_sq_dist < EPSILON:
        median_sq_dist = 1.0
    
    return 1.0 / (2.0 * median_sq_dist)


def transform_kernel_pca(
    KernelPCAState state,
    cnp.ndarray[DTYPE_t, ndim=2] X
):
    """
    Transform data using fitted Kernel PCA.
    
    Parameters
    ----------
    state : KernelPCAState
        Fitted state
    X : ndarray, shape (n_samples, n_features)
        Data to transform
    
    Returns
    -------
    X_transformed : ndarray, shape (n_samples, n_components)
        Transformed data
    """
    if not state.fitted:
        raise ValueError("KernelPCAState not fitted")
    
    cdef int n_new = X.shape[0]
    cdef int n_train = state.n_samples
    cdef int n_features = state.n_features
    cdef int n_components = state.n_components
    cdef int i, j
    
    # Compute kernel between new and training points
    cdef cnp.ndarray[DTYPE_t, ndim=2] K_new = np.empty((n_new, n_train), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_fit = state.X_fit
    cdef double gamma = state.gamma
    
    # Compute kernel elements
    for i in range(n_new):
        for j in range(n_train):
            K_new[i, j] = rbf_kernel_element(
                &X[i, 0], &X_fit[j, 0], n_features, gamma
            )
    
    # Center kernel
    # K_centered = K_new - mean(K_new, axis=1).reshape(-1,1) - K_fit_rows + K_fit_mean
    cdef cnp.ndarray[DTYPE_t, ndim=1] K_new_row_means = np.mean(K_new, axis=1)
    cdef cnp.ndarray[DTYPE_t, ndim=1] K_fit_rows = state.K_fit_rows
    cdef double K_fit_mean = state.K_fit_mean
    
    for i in range(n_new):
        for j in range(n_train):
            K_new[i, j] = K_new[i, j] - K_new_row_means[i] - K_fit_rows[j] + K_fit_mean
    
    # Project: X_transformed = K_centered @ alphas
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_transformed = K_new @ state.alphas
    
    return X_transformed


def inverse_transform_kernel_pca(
    KernelPCAState state,
    cnp.ndarray[DTYPE_t, ndim=2] X_transformed,
    int n_iter = 100,
    double tol = 1e-6
):
    """
    Approximate inverse transform for Kernel PCA.
    
    Uses pre-image approximation via iterative optimization.
    This is an approximate inverse since exact inverse doesn't exist for Kernel PCA.
    
    Parameters
    ----------
    state : KernelPCAState
        Fitted state
    X_transformed : ndarray, shape (n_samples, n_components)
        Transformed data
    n_iter : int
        Maximum iterations for optimization
    tol : float
        Convergence tolerance
    
    Returns
    -------
    X_reconstructed : ndarray, shape (n_samples, n_features)
        Approximate reconstruction in original space
    """
    if not state.fitted:
        raise ValueError("KernelPCAState not fitted")
    
    cdef int n_samples = X_transformed.shape[0]
    cdef int n_features = state.n_features
    cdef int n_train = state.n_samples
    cdef int n_components = state.n_components
    cdef int i, j, it
    
    # Use weighted combination of training points as pre-image approximation
    # Find weights based on similarity in transformed space
    
    # Transform training data to get reference
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_fit = state.X_fit
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_fit_transformed = transform_kernel_pca(state, X_fit)
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_reconstructed = np.empty((n_samples, n_features), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] distances = np.empty(n_train, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] weights = np.empty(n_train, dtype=np.float64)
    
    cdef double gamma_weights, total_weight, dist_sq
    cdef double diff
    
    # Use RBF weights in transformed space
    gamma_weights = 1.0 / (2.0 * n_components)  # Heuristic
    
    for i in range(n_samples):
        # Compute distances to all training points in transformed space
        total_weight = 0.0
        for j in range(n_train):
            dist_sq = 0.0
            for k in range(n_components):
                diff = X_transformed[i, k] - X_fit_transformed[j, k]
                dist_sq += diff * diff
            weights[j] = exp(-gamma_weights * dist_sq)
            total_weight += weights[j]
        
        # Normalize weights
        if total_weight > EPSILON:
            for j in range(n_train):
                weights[j] /= total_weight
        else:
            # Uniform weights
            for j in range(n_train):
                weights[j] = 1.0 / n_train
        
        # Weighted combination of training points
        for k in range(n_features):
            X_reconstructed[i, k] = 0.0
            for j in range(n_train):
                X_reconstructed[i, k] += weights[j] * X_fit[j, k]
    
    return X_reconstructed


# =============================================================================
# Random Projections (Johnson-Lindenstrauss)
# =============================================================================

cdef class RandomProjectionState:
    """State for random projection transforms."""
    cdef public cnp.ndarray projection_matrix  # (n_components, n_features)
    cdef public cnp.ndarray inverse_matrix     # (n_features, n_components) - pseudoinverse
    cdef public int n_components
    cdef public int n_features
    cdef public bint fitted
    
    def __init__(self):
        self.fitted = False
        self.n_components = 0
        self.n_features = 0


def fit_random_projection(
    int n_features,
    int n_components,
    str projection_type = 'gaussian',
    int random_seed = 0
):
    """
    Create random projection matrix.
    
    Parameters
    ----------
    n_features : int
        Original dimensionality
    n_components : int
        Target dimensionality
    projection_type : str
        'gaussian' - Gaussian random matrix (slower, most accurate)
        'sparse' - Sparse random matrix (faster, good accuracy)
        'rademacher' - {-1, +1} entries (fastest)
    random_seed : int
        Random seed
    
    Returns
    -------
    state : RandomProjectionState
        State for transforms
    """
    if random_seed > 0:
        np.random.seed(random_seed)
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] R
    cdef double scale
    
    if projection_type == 'gaussian':
        R = np.random.randn(n_components, n_features) / sqrt(n_components)
    
    elif projection_type == 'sparse':
        # Sparse random projection: 1/sqrt(3) * {-1, 0, 0, 0, 0, +1}
        # Probability: {1/6, 2/3, 1/6} for {-1, 0, +1}
        probs = np.random.random((n_components, n_features))
        R = np.zeros((n_components, n_features), dtype=np.float64)
        R[probs < 1.0/6.0] = -1.0
        R[probs > 5.0/6.0] = 1.0
        R *= sqrt(3.0) / sqrt(n_components)
    
    elif projection_type == 'rademacher':
        # Rademacher: uniform {-1, +1}
        R = np.random.randint(0, 2, (n_components, n_features)).astype(np.float64)
        R = 2.0 * R - 1.0
        R /= sqrt(n_components)
    
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")
    
    # Compute pseudoinverse for approximate inverse transform
    cdef cnp.ndarray[DTYPE_t, ndim=2] R_pinv = np.linalg.pinv(R)
    
    state = RandomProjectionState()
    state.projection_matrix = R
    state.inverse_matrix = R_pinv
    state.n_components = n_components
    state.n_features = n_features
    state.fitted = True
    
    return state


def transform_random_projection(
    RandomProjectionState state,
    cnp.ndarray[DTYPE_t, ndim=2] X
):
    """
    Apply random projection to reduce dimensionality.
    
    Parameters
    ----------
    state : RandomProjectionState
        Fitted state
    X : ndarray, shape (n_samples, n_features)
        Data to transform
    
    Returns
    -------
    X_transformed : ndarray, shape (n_samples, n_components)
    """
    if not state.fitted:
        raise ValueError("RandomProjectionState not fitted")
    
    return X @ state.projection_matrix.T


def inverse_transform_random_projection(
    RandomProjectionState state,
    cnp.ndarray[DTYPE_t, ndim=2] X_transformed
):
    """
    Approximate inverse transform using pseudoinverse.
    
    Parameters
    ----------
    state : RandomProjectionState
        Fitted state
    X_transformed : ndarray, shape (n_samples, n_components)
        Transformed data
    
    Returns
    -------
    X_reconstructed : ndarray, shape (n_samples, n_features)
    """
    if not state.fitted:
        raise ValueError("RandomProjectionState not fitted")
    
    return X_transformed @ state.inverse_matrix.T


# =============================================================================
# Incremental/Online PCA for Adaptive Refinement
# =============================================================================

cdef class IncrementalPCAState:
    """State for incremental PCA that can be updated with new samples."""
    cdef public cnp.ndarray components      # (n_components, n_features)
    cdef public cnp.ndarray mean            # (n_features,)
    cdef public cnp.ndarray singular_values # (n_components,)
    cdef public cnp.ndarray explained_variance
    cdef public int n_components
    cdef public int n_features
    cdef public int n_samples_seen
    cdef public bint fitted
    
    def __init__(self):
        self.fitted = False
        self.n_components = 0
        self.n_features = 0
        self.n_samples_seen = 0


def fit_incremental_pca(
    cnp.ndarray[DTYPE_t, ndim=2] X,
    int n_components = 0,
    double variance_threshold = 0.90
):
    """
    Fit incremental PCA.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data
    n_components : int
        Number of components (0 = auto based on variance)
    variance_threshold : float
        Variance threshold if n_components=0
    
    Returns
    -------
    state : IncrementalPCAState
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    
    # Center data
    cdef cnp.ndarray[DTYPE_t, ndim=1] mean = np.mean(X, axis=0)
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_centered = X - mean
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Determine n_components
    explained_var = (s ** 2) / (n_samples - 1)
    total_var = np.sum(explained_var)
    
    cdef int n_comp
    if n_components > 0:
        n_comp = min(n_components, len(s))
    else:
        if total_var > EPSILON:
            cum_var = np.cumsum(explained_var) / total_var
            n_comp = np.searchsorted(cum_var, variance_threshold) + 1
            n_comp = min(n_comp, len(s))
        else:
            n_comp = 1
    
    state = IncrementalPCAState()
    state.components = Vt[:n_comp].copy()
    state.mean = mean
    state.singular_values = s[:n_comp].copy()
    state.explained_variance = explained_var[:n_comp].copy()
    state.n_components = n_comp
    state.n_features = n_features
    state.n_samples_seen = n_samples
    state.fitted = True
    
    return state


def partial_fit_incremental_pca(
    IncrementalPCAState state,
    cnp.ndarray[DTYPE_t, ndim=2] X
):
    """
    Update PCA with new samples (incremental/online update).
    
    Uses the incremental SVD approach.
    
    Parameters
    ----------
    state : IncrementalPCAState
        Current state (will be modified)
    X : ndarray, shape (n_samples, n_features)
        New samples
    
    Returns
    -------
    state : IncrementalPCAState
        Updated state
    """
    if not state.fitted:
        return fit_incremental_pca(X, state.n_components)
    
    cdef int n_new = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_old = state.n_samples_seen
    cdef int n_total = n_old + n_new
    cdef int n_components = state.n_components
    
    # Update mean
    cdef cnp.ndarray[DTYPE_t, ndim=1] new_mean = np.mean(X, axis=0)
    cdef cnp.ndarray[DTYPE_t, ndim=1] updated_mean = (
        (n_old * state.mean + n_new * new_mean) / n_total
    )
    
    # Correct old components for new mean
    mean_correction = sqrt(n_old * n_new / n_total) * (state.mean - new_mean)
    
    # Build matrix for incremental SVD
    # [S * V^T; X_new_centered; mean_correction]
    X_centered = X - updated_mean
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] combined = np.vstack([
        np.diag(state.singular_values) @ state.components,
        X_centered,
        mean_correction.reshape(1, -1)
    ])
    
    # SVD of combined
    U, s, Vt = np.linalg.svd(combined, full_matrices=False)
    
    # Update state
    state.components = Vt[:n_components].copy()
    state.mean = updated_mean
    state.singular_values = s[:n_components].copy()
    state.explained_variance = (s[:n_components] ** 2) / (n_total - 1)
    state.n_samples_seen = n_total
    
    return state


def transform_incremental_pca(
    IncrementalPCAState state,
    cnp.ndarray[DTYPE_t, ndim=2] X
):
    """Transform data using fitted incremental PCA."""
    if not state.fitted:
        raise ValueError("IncrementalPCAState not fitted")
    
    return (X - state.mean) @ state.components.T


def inverse_transform_incremental_pca(
    IncrementalPCAState state,
    cnp.ndarray[DTYPE_t, ndim=2] X_transformed
):
    """Inverse transform from PCA space to original space."""
    if not state.fitted:
        raise ValueError("IncrementalPCAState not fitted")
    
    return X_transformed @ state.components + state.mean


# =============================================================================
# Adaptive Dimensionality Selection
# =============================================================================

def compute_adaptive_components(
    cnp.ndarray[DTYPE_t, ndim=1] eigenvalues,
    double base_threshold = 0.90,
    double min_threshold = 0.70,
    double max_threshold = 0.99,
    double progress = 0.0
):
    """
    Adaptively select number of components based on optimization progress.
    
    Early in optimization: use fewer components (faster, more exploration)
    Later in optimization: use more components (finer control)
    
    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues
    base_threshold : float
        Base variance threshold
    min_threshold : float
        Minimum threshold (early optimization)
    max_threshold : float
        Maximum threshold (late optimization)
    progress : float
        Optimization progress [0, 1]
    
    Returns
    -------
    n_components : int
        Recommended number of components
    threshold_used : float
        Actual threshold used
    """
    # Interpolate threshold based on progress
    # Start with min, end with max
    cdef double threshold = min_threshold + (max_threshold - min_threshold) * progress
    threshold = max(min_threshold, min(max_threshold, threshold))
    
    # Find components
    cdef int n = len(eigenvalues)
    if n == 0:
        return 1, threshold
    
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_var = np.sum(eigenvalues)
    
    if total_var < EPSILON:
        return 1, threshold
    
    cum_var = np.cumsum(eigenvalues) / total_var
    n_components = np.searchsorted(cum_var, threshold) + 1
    n_components = max(1, min(n_components, n))
    
    return n_components, threshold


# =============================================================================
# Utility Functions
# =============================================================================

def get_highdim_core_version():
    """Return version string."""
    return "1.0.0-cython"


def is_high_dimensional(int n_dims, int threshold = 1000):
    """Check if problem is high-dimensional."""
    return n_dims >= threshold
