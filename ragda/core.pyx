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
RAGDA Core - Pure Cython/C Implementation

High-performance optimization kernel with:
- All numeric operations in pure C (nogil)
- Pre-allocated memory pools
- Vectorized batch evaluation
- Cache-friendly memory access patterns
- Zero Python object overhead in hot paths
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, fabs, pow as c_pow, cos, sin
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy, memset
cimport cython

# Initialize NumPy C-API
cnp.import_array()

# =============================================================================
# Type Definitions
# =============================================================================
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t INT32_t
ctypedef cnp.int64_t INT64_t
ctypedef cnp.uint32_t UINT32_t

# Constants
DEF MAX_DIMS = 1000
DEF MAX_CATEGORIES = 256
DEF MAX_BATCH_SIZE = 10000
DEF PI = 3.14159265358979323846
DEF TWO_PI = 6.28318530717958647692


# =============================================================================
# Fast Random Number Generator (Xorshift128+)
# =============================================================================
cdef struct XorshiftState:
    UINT32_t s0
    UINT32_t s1
    UINT32_t s2
    UINT32_t s3


cdef inline void xorshift_seed(XorshiftState* state, UINT32_t seed) noexcept nogil:
    """Initialize Xorshift state from seed."""
    # Use bit operations to create constants without Python int coercion
    state.s0 = seed
    state.s1 = seed ^ (<UINT32_t>305419896)   # 0x12345678
    state.s2 = seed ^ (<UINT32_t>3735928559)  # 0xDEADBEEF  
    state.s3 = seed ^ (<UINT32_t>3405691582)  # 0xCAFEBABE


cdef inline UINT32_t xorshift_next(XorshiftState* state) noexcept nogil:
    """Generate next random uint32."""
    cdef UINT32_t t = state.s3
    cdef UINT32_t s = state.s0
    
    state.s3 = state.s2
    state.s2 = state.s1
    state.s1 = s
    
    t ^= t << 11
    t ^= t >> 8
    state.s0 = t ^ s ^ (s >> 19)
    
    return state.s0


cdef inline double xorshift_uniform(XorshiftState* state) noexcept nogil:
    """Generate uniform random in [0, 1)."""
    return <double>xorshift_next(state) / 4294967295.0  # 0xFFFFFFFF as double


cdef inline int xorshift_randint(XorshiftState* state, int n) noexcept nogil:
    """Generate uniform random integer in [0, n)."""
    if n <= 0:
        return 0
    return <int>(xorshift_uniform(state) * n)


cdef inline double xorshift_randn(XorshiftState* state) noexcept nogil:
    """Generate standard normal random (Box-Muller transform)."""
    cdef double u1, u2, r
    
    u1 = xorshift_uniform(state)
    u2 = xorshift_uniform(state)
    
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    
    r = sqrt(-2.0 * log(u1))
    return r * cos(TWO_PI * u2)


# =============================================================================
# Memory Pool for Pre-allocated Buffers
# =============================================================================
cdef struct MemoryPool:
    # Sample buffers
    double* samples_cont
    double* fitness
    double* weights
    double* pseudo_grad
    double* temp_buffer
    
    # Categorical
    INT32_t* samples_cat
    INT64_t* sorted_indices
    INT64_t* improved_indices
    INT32_t* cat_counts
    
    # ADAM state
    double* m
    double* v
    
    # Current position
    double* x_cont
    double* x_best_cont
    INT32_t* x_cat
    INT32_t* x_best_cat
    
    # Bounds
    double* lower
    double* upper
    
    # Dimensions
    int max_batch
    int n_dims
    int n_cat
    int allocated


cdef inline int pool_init(MemoryPool* pool, int max_batch, int n_dims, int n_cat) noexcept nogil:
    """Initialize memory pool. Returns 0 on success, -1 on failure."""
    pool.max_batch = max_batch
    pool.n_dims = n_dims
    pool.n_cat = n_cat
    
    # temp_buffer is always needed for sorting (size = max_batch)
    pool.temp_buffer = <double*>malloc(max_batch * sizeof(double))
    if pool.temp_buffer == NULL:
        return -1
    
    if n_dims > 0:
        pool.samples_cont = <double*>malloc(max_batch * n_dims * sizeof(double))
        pool.pseudo_grad = <double*>malloc(n_dims * sizeof(double))
        pool.m = <double*>malloc(n_dims * sizeof(double))
        pool.v = <double*>malloc(n_dims * sizeof(double))
        pool.x_cont = <double*>malloc(n_dims * sizeof(double))
        pool.x_best_cont = <double*>malloc(n_dims * sizeof(double))
        pool.lower = <double*>malloc(n_dims * sizeof(double))
        pool.upper = <double*>malloc(n_dims * sizeof(double))
        
        if (pool.samples_cont == NULL or pool.pseudo_grad == NULL or 
            pool.m == NULL or pool.v == NULL or
            pool.x_cont == NULL or pool.x_best_cont == NULL or
            pool.lower == NULL or pool.upper == NULL):
            return -1
        
        memset(pool.m, 0, n_dims * sizeof(double))
        memset(pool.v, 0, n_dims * sizeof(double))
    else:
        pool.samples_cont = NULL
        pool.pseudo_grad = NULL
        pool.m = NULL
        pool.v = NULL
        pool.x_cont = NULL
        pool.x_best_cont = NULL
        pool.lower = NULL
        pool.upper = NULL
    
    if n_cat > 0:
        pool.samples_cat = <INT32_t*>malloc(max_batch * n_cat * sizeof(INT32_t))
        pool.x_cat = <INT32_t*>malloc(n_cat * sizeof(INT32_t))
        pool.x_best_cat = <INT32_t*>malloc(n_cat * sizeof(INT32_t))
        
        if pool.samples_cat == NULL or pool.x_cat == NULL or pool.x_best_cat == NULL:
            return -1
    else:
        pool.samples_cat = NULL
        pool.x_cat = NULL
        pool.x_best_cat = NULL
    
    pool.fitness = <double*>malloc(max_batch * sizeof(double))
    pool.weights = <double*>malloc(max_batch * sizeof(double))
    pool.sorted_indices = <INT64_t*>malloc(max_batch * sizeof(INT64_t))
    pool.improved_indices = <INT64_t*>malloc(max_batch * sizeof(INT64_t))
    pool.cat_counts = <INT32_t*>malloc(MAX_CATEGORIES * sizeof(INT32_t))
    
    if (pool.fitness == NULL or pool.weights == NULL or 
        pool.sorted_indices == NULL or pool.improved_indices == NULL or
        pool.cat_counts == NULL):
        return -1
    
    pool.allocated = 1
    return 0


cdef inline void pool_free(MemoryPool* pool) noexcept nogil:
    """Free all memory in pool."""
    if pool.allocated == 0:
        return
    
    if pool.samples_cont != NULL: free(pool.samples_cont)
    if pool.pseudo_grad != NULL: free(pool.pseudo_grad)
    if pool.temp_buffer != NULL: free(pool.temp_buffer)
    if pool.m != NULL: free(pool.m)
    if pool.v != NULL: free(pool.v)
    if pool.x_cont != NULL: free(pool.x_cont)
    if pool.x_best_cont != NULL: free(pool.x_best_cont)
    if pool.lower != NULL: free(pool.lower)
    if pool.upper != NULL: free(pool.upper)
    if pool.samples_cat != NULL: free(pool.samples_cat)
    if pool.x_cat != NULL: free(pool.x_cat)
    if pool.x_best_cat != NULL: free(pool.x_best_cat)
    if pool.fitness != NULL: free(pool.fitness)
    if pool.weights != NULL: free(pool.weights)
    if pool.sorted_indices != NULL: free(pool.sorted_indices)
    if pool.improved_indices != NULL: free(pool.improved_indices)
    if pool.cat_counts != NULL: free(pool.cat_counts)
    
    pool.allocated = 0


# =============================================================================
# Sorting (nogil)
# =============================================================================
cdef struct IndexedValue:
    double value
    INT64_t index


cdef int compare_indexed(const void* a, const void* b) noexcept nogil:
    """Comparison function for qsort."""
    cdef double diff = (<IndexedValue*>a).value - (<IndexedValue*>b).value
    if diff < 0:
        return -1
    elif diff > 0:
        return 1
    return 0


cdef inline void argsort_nogil(const double* values, INT64_t* indices, int n) noexcept nogil:
    """Argsort without GIL using C qsort."""
    cdef IndexedValue* indexed = <IndexedValue*>malloc(n * sizeof(IndexedValue))
    cdef int i
    
    if indexed == NULL:
        return
    
    for i in range(n):
        indexed[i].value = values[i]
        indexed[i].index = i
    
    qsort(indexed, n, sizeof(IndexedValue), compare_indexed)
    
    for i in range(n):
        indices[i] = indexed[i].index
    
    free(indexed)


# =============================================================================
# Core Numeric Operations (all nogil)
# =============================================================================

cdef inline void generate_samples_cont(
    MemoryPool* pool,
    XorshiftState* rng,
    double sigma,
    int n_samples
) noexcept nogil:
    """Generate continuous samples around current position."""
    cdef int i, j
    cdef int n_dims = pool.n_dims
    cdef double* samples = pool.samples_cont
    cdef double* x = pool.x_cont
    cdef double* lower = pool.lower
    cdef double* upper = pool.upper
    cdef double val
    
    if n_dims <= 0 or samples == NULL:
        return
    
    for i in range(n_samples):
        for j in range(n_dims):
            val = x[j] + sigma * xorshift_randn(rng)
            
            if val < lower[j]:
                val = lower[j]
            elif val > upper[j]:
                val = upper[j]
            
            samples[i * n_dims + j] = val


cdef inline void generate_samples_cat(
    MemoryPool* pool,
    XorshiftState* rng,
    const INT32_t* cat_n_values,
    int n_samples,
    double mutation_rate
) noexcept nogil:
    """Generate categorical samples with mutation."""
    cdef int i, j
    cdef int n_cat = pool.n_cat
    cdef INT32_t* samples = pool.samples_cat
    cdef INT32_t* x = pool.x_cat
    cdef int mutation_threshold = <int>(mutation_rate * 100)
    
    if n_cat <= 0 or samples == NULL:
        return
    
    for i in range(n_samples):
        for j in range(n_cat):
            if xorshift_randint(rng, 100) < mutation_threshold:
                samples[i * n_cat + j] = xorshift_randint(rng, cat_n_values[j])
            else:
                samples[i * n_cat + j] = x[j]


cdef inline int find_improving_samples(
    MemoryPool* pool,
    double f_current,
    double top_n_fraction,
    double weight_decay,
    int n_samples
) noexcept nogil:
    """Find improving samples, apply top-n selection and weight decay."""
    cdef int i, n_improved = 0, n_selected
    cdef double total_weight
    cdef double* fitness = pool.fitness
    cdef double* weights = pool.weights
    cdef INT64_t* improved = pool.improved_indices
    cdef INT64_t* sorted_idx = pool.sorted_indices
    cdef double* temp = pool.temp_buffer
    
    # Find improving samples
    for i in range(n_samples):
        if fitness[i] < f_current:
            improved[n_improved] = i
            n_improved += 1
    
    if n_improved == 0:
        return 0
    
    # Sort by fitness (best first)
    for i in range(n_improved):
        weights[i] = fitness[improved[i]]
    
    argsort_nogil(weights, sorted_idx, n_improved)
    
    # Reorder
    for i in range(n_improved):
        temp[i] = <double>improved[sorted_idx[i]]
    for i in range(n_improved):
        improved[i] = <INT64_t>temp[i]
    
    # Top-n selection
    n_selected = <int>(n_improved * top_n_fraction)
    if n_selected < 1:
        n_selected = 1
    if n_selected > n_improved:
        n_selected = n_improved
    
    # Weight decay
    total_weight = 0.0
    for i in range(n_selected):
        weights[i] = c_pow(weight_decay, <double>i) * (f_current - fitness[improved[i]])
        total_weight += weights[i]
    
    if total_weight > 1e-10:
        for i in range(n_selected):
            weights[i] /= total_weight
    else:
        for i in range(n_selected):
            weights[i] = 1.0 / n_selected
    
    return n_selected


cdef inline void compute_pseudo_gradient(MemoryPool* pool, int n_selected) noexcept nogil:
    """Compute weighted pseudo-gradient from improving samples."""
    cdef int i, j
    cdef int n_dims = pool.n_dims
    cdef double* grad = pool.pseudo_grad
    cdef double* samples = pool.samples_cont
    cdef double* x = pool.x_cont
    cdef double* weights = pool.weights
    cdef INT64_t* improved = pool.improved_indices
    cdef INT64_t idx
    cdef double weighted_mean
    
    for j in range(n_dims):
        weighted_mean = 0.0
        for i in range(n_selected):
            idx = improved[i]
            weighted_mean += weights[i] * samples[idx * n_dims + j]
        grad[j] = weighted_mean - x[j]


cdef inline void adam_update_nogil(
    MemoryPool* pool,
    double alpha,
    double beta1,
    double beta2,
    double epsilon,
    int t
) noexcept nogil:
    """ADAM update in pure C."""
    cdef int j
    cdef int n_dims = pool.n_dims
    cdef double* x = pool.x_cont
    cdef double* m = pool.m
    cdef double* v = pool.v
    cdef double* grad = pool.pseudo_grad
    cdef double* lower = pool.lower
    cdef double* upper = pool.upper
    cdef double m_hat, v_hat, grad_sq
    cdef double t_d = <double>(t + 1)
    cdef double bc1 = 1.0 - c_pow(beta1, t_d)
    cdef double bc2 = 1.0 - c_pow(beta2, t_d)
    cdef double new_val
    
    for j in range(n_dims):
        m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j]
        grad_sq = grad[j] * grad[j]
        v[j] = beta2 * v[j] + (1.0 - beta2) * grad_sq
        
        m_hat = m[j] / bc1
        v_hat = v[j] / bc2
        
        new_val = x[j] + alpha * m_hat / (sqrt(v_hat) + epsilon)
        
        if new_val < lower[j]:
            new_val = lower[j]
        elif new_val > upper[j]:
            new_val = upper[j]
        
        x[j] = new_val


cdef inline int update_categorical_weighted_nogil(
    MemoryPool* pool,
    const INT32_t* cat_n_values,
    int n_selected,
    int cat_idx
) noexcept nogil:
    """Update categorical variable using weighted mode."""
    cdef int i, cat_val, n_cats
    cdef INT64_t sample_idx
    cdef INT32_t* samples = pool.samples_cat
    cdef INT32_t* counts = pool.cat_counts
    cdef INT64_t* improved = pool.improved_indices
    cdef int n_cat = pool.n_cat
    cdef int max_count = 0
    cdef int best_cat = pool.x_cat[cat_idx]
    
    if n_selected == 0:
        return best_cat
    
    n_cats = cat_n_values[cat_idx]
    if n_cats > MAX_CATEGORIES:
        n_cats = MAX_CATEGORIES
    
    memset(counts, 0, n_cats * sizeof(INT32_t))
    
    for i in range(n_selected):
        sample_idx = improved[i]
        cat_val = samples[sample_idx * n_cat + cat_idx]
        if cat_val >= 0 and cat_val < n_cats:
            counts[cat_val] += n_selected - i
    
    for i in range(n_cats):
        if counts[i] > max_count:
            max_count = counts[i]
            best_cat = i
    
    return best_cat


cdef inline void reset_adam_momentum(MemoryPool* pool) noexcept nogil:
    """Reset ADAM momentum."""
    memset(pool.m, 0, pool.n_dims * sizeof(double))
    memset(pool.v, 0, pool.n_dims * sizeof(double))


cdef inline void copy_to_best(MemoryPool* pool) noexcept nogil:
    """Copy current to best."""
    if pool.n_dims > 0:
        memcpy(pool.x_best_cont, pool.x_cont, pool.n_dims * sizeof(double))
    if pool.n_cat > 0:
        memcpy(pool.x_best_cat, pool.x_cat, pool.n_cat * sizeof(INT32_t))


cdef inline void copy_from_best(MemoryPool* pool) noexcept nogil:
    """Copy best to current."""
    if pool.n_dims > 0:
        memcpy(pool.x_cont, pool.x_best_cont, pool.n_dims * sizeof(double))
    if pool.n_cat > 0:
        memcpy(pool.x_cat, pool.x_best_cat, pool.n_cat * sizeof(INT32_t))


# =============================================================================
# Batch Evaluation
# =============================================================================

cdef class BatchEvaluator:
    """Wrapper for objective with batch support."""
    cdef object objective
    cdef bint supports_batch
    cdef bint checked_batch
    
    def __init__(self, objective):
        self.objective = objective
        self.supports_batch = False
        self.checked_batch = False
    
    cpdef cnp.ndarray[DTYPE_t, ndim=1] evaluate_batch(
        self,
        cnp.ndarray[DTYPE_t, ndim=2] x_cont,
        cnp.ndarray[INT32_t, ndim=2] x_cat,
        int batch_size,
        int minibatch_size
    ):
        """Evaluate batch of points."""
        cdef cnp.ndarray[DTYPE_t, ndim=1] results = np.empty(batch_size, dtype=np.float64)
        cdef int i
        cdef double val
        
        # Try vectorized on first call
        if not self.checked_batch:
            self.checked_batch = True
            try:
                if x_cont is not None and x_cat is not None:
                    result = self.objective(x_cont, x_cat, minibatch_size)
                elif x_cont is not None:
                    result = self.objective(x_cont, None, minibatch_size)
                else:
                    result = self.objective(None, x_cat, minibatch_size)
                
                if hasattr(result, '__len__') and len(result) == batch_size:
                    self.supports_batch = True
                    return np.asarray(result, dtype=np.float64)
            except:
                self.supports_batch = False
        
        if self.supports_batch:
            try:
                if x_cont is not None and x_cat is not None:
                    return np.asarray(self.objective(x_cont, x_cat, minibatch_size), dtype=np.float64)
                elif x_cont is not None:
                    return np.asarray(self.objective(x_cont, None, minibatch_size), dtype=np.float64)
                else:
                    return np.asarray(self.objective(None, x_cat, minibatch_size), dtype=np.float64)
            except:
                self.supports_batch = False
        
        # Sequential fallback
        for i in range(batch_size):
            try:
                if x_cont is not None and x_cat is not None:
                    val = self.objective(x_cont[i], x_cat[i], minibatch_size)
                elif x_cont is not None:
                    val = self.objective(x_cont[i], None, minibatch_size)
                else:
                    val = self.objective(None, x_cat[i], minibatch_size)
                results[i] = val
            except:
                results[i] = 1e10
        
        return results


# =============================================================================
# Main Optimization Kernel
# =============================================================================

def optimize_worker_core(
    cnp.ndarray[DTYPE_t, ndim=1] x0_cont,
    cnp.ndarray[INT32_t, ndim=1] x0_cat,
    cnp.ndarray[INT32_t, ndim=1] cat_n_values,
    cnp.ndarray[DTYPE_t, ndim=2] bounds,
    object evaluate_fitness_func,
    int max_iter,
    cnp.ndarray[INT32_t, ndim=1] lambda_schedule,
    cnp.ndarray[INT32_t, ndim=1] mu_schedule,
    cnp.ndarray[DTYPE_t, ndim=1] sigma_schedule,
    cnp.ndarray[INT32_t, ndim=1] minibatch_schedule,
    bint use_minibatch,
    double top_n_fraction,
    double alpha,
    double beta1,
    double beta2,
    double epsilon,
    double shrink_factor,
    int shrink_patience,
    double shrink_threshold,
    bint use_improvement_weights,
    int random_seed,
    int worker_id,
    object sync_queue,
    object sync_event,
    int sync_frequency,
    double weight_decay = 0.95,
    double early_stop_threshold = 1e-12,
    int early_stop_patience = 50
):
    """
    High-performance optimization kernel.
    All numeric operations run in pure C without GIL.
    """
    
    # Validation
    cdef int n_cont = len(x0_cont)
    cdef int n_cat = len(x0_cat)
    
    if n_cont > 0 and len(bounds) != n_cont:
        raise ValueError(f"Bounds length {len(bounds)} != n_cont {n_cont}")
    
    if n_cat > 0 and len(cat_n_values) != n_cat:
        raise ValueError(f"cat_n_values length {len(cat_n_values)} != n_cat {n_cat}")
    
    if n_cont > MAX_DIMS:
        raise ValueError(f"n_cont {n_cont} exceeds MAX_DIMS {MAX_DIMS}")
    
    # Memory allocation
    cdef int max_lambda = lambda_schedule.max()
    if max_lambda > MAX_BATCH_SIZE:
        raise ValueError(f"max_lambda {max_lambda} exceeds MAX_BATCH_SIZE {MAX_BATCH_SIZE}")
    
    cdef MemoryPool pool
    cdef XorshiftState rng
    cdef int alloc_result
    
    pool.allocated = 0
    alloc_result = pool_init(&pool, max_lambda, n_cont, n_cat)
    if alloc_result != 0:
        raise MemoryError("Failed to allocate memory pool")
    
    xorshift_seed(&rng, <UINT32_t>random_seed)
    
    # Copy initial values
    cdef int i, j
    cdef DTYPE_t[:] x0_cont_view = x0_cont
    cdef INT32_t[:] x0_cat_view = x0_cat
    cdef DTYPE_t[:, :] bounds_view = bounds
    cdef INT32_t[:] cat_n_values_view = cat_n_values
    
    for i in range(n_cont):
        pool.x_cont[i] = x0_cont_view[i]
        pool.x_best_cont[i] = x0_cont_view[i]
        pool.lower[i] = bounds_view[i, 0]
        pool.upper[i] = bounds_view[i, 1]
    
    for i in range(n_cat):
        pool.x_cat[i] = x0_cat_view[i]
        pool.x_best_cat[i] = x0_cat_view[i]
    
    # Batch evaluator
    cdef BatchEvaluator batch_eval = BatchEvaluator(evaluate_fitness_func)
    
    # Local variables
    cdef int t, lambda_t, mu_t, n_selected, best_idx
    cdef double sigma_t, sigma_current = 1.0
    cdef double f_current, f_best, current_best, improvement
    cdef int no_improvement_count = 0
    cdef int early_stop_counter = 0
    cdef int minibatch_size
    cdef bint converged = False
    cdef bint did_shrink = False
    
    # Schedule views
    cdef INT32_t[:] lambda_sched_view = lambda_schedule
    cdef INT32_t[:] mu_sched_view = mu_schedule
    cdef DTYPE_t[:] sigma_sched_view = sigma_schedule
    cdef INT32_t[:] minibatch_sched_view = minibatch_schedule
    
    # Pre-allocate history arrays
    cdef cnp.ndarray[DTYPE_t, ndim=1] fitness_history_arr = np.empty(max_iter, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] sigma_history_arr = np.empty(max_iter, dtype=np.float64)
    cdef cnp.ndarray[INT32_t, ndim=1] batch_size_history_arr = np.empty(max_iter, dtype=np.int32)
    cdef int history_idx = 0
    
    # Python lists for variable-length data
    shrink_events = []
    sync_events = []
    params_history = []
    
    # NumPy arrays for batch evaluation
    cdef cnp.ndarray[DTYPE_t, ndim=2] samples_cont_np
    cdef cnp.ndarray[INT32_t, ndim=2] samples_cat_np
    cdef cnp.ndarray[DTYPE_t, ndim=1] fitness_np
    
    if n_cont > 0:
        samples_cont_np = np.empty((max_lambda, n_cont), dtype=np.float64)
    else:
        samples_cont_np = None
    
    if n_cat > 0:
        samples_cat_np = np.empty((max_lambda, n_cat), dtype=np.int32)
    else:
        samples_cat_np = None
    
    # Initial evaluation
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_cont_np = np.empty(n_cont, dtype=np.float64)
    cdef cnp.ndarray[INT32_t, ndim=1] x_cat_np = np.empty(n_cat, dtype=np.int32)
    
    for i in range(n_cont):
        x_cont_np[i] = pool.x_cont[i]
    for i in range(n_cat):
        x_cat_np[i] = pool.x_cat[i]
    
    if use_minibatch:
        minibatch_size = minibatch_sched_view[0]
    else:
        minibatch_size = -1
    
    try:
        f_current = evaluate_fitness_func(x_cont_np, x_cat_np, minibatch_size)
    except Exception:
        f_current = 1e10
    
    f_best = f_current
    
    # Main loop
    try:
        for t in range(max_iter):
            lambda_t = lambda_sched_view[t]
            mu_t = mu_sched_view[t]
            sigma_t = sigma_sched_view[t] * sigma_current
            
            if use_minibatch:
                minibatch_size = minibatch_sched_view[t]
            else:
                minibatch_size = -1
            
            # Generate samples (nogil)
            with nogil:
                generate_samples_cont(&pool, &rng, sigma_t, lambda_t)
                if n_cat > 0:
                    generate_samples_cat(&pool, &rng, &cat_n_values_view[0], lambda_t, 0.3)
            
            # Copy to NumPy
            if n_cont > 0:
                for i in range(lambda_t):
                    for j in range(n_cont):
                        samples_cont_np[i, j] = pool.samples_cont[i * n_cont + j]
            
            if n_cat > 0:
                for i in range(lambda_t):
                    for j in range(n_cat):
                        samples_cat_np[i, j] = pool.samples_cat[i * n_cat + j]
            
            # Evaluate (requires GIL)
            fitness_np = batch_eval.evaluate_batch(
                samples_cont_np[:lambda_t] if n_cont > 0 else None,
                samples_cat_np[:lambda_t] if n_cat > 0 else None,
                lambda_t,
                minibatch_size
            )
            
            # Copy fitness to pool
            for i in range(lambda_t):
                pool.fitness[i] = fitness_np[i]
            
            # Find best and update (nogil)
            did_shrink = False
            with nogil:
                argsort_nogil(pool.fitness, pool.sorted_indices, lambda_t)
                best_idx = <int>pool.sorted_indices[0]
                current_best = pool.fitness[best_idx]
                
                if current_best < f_best:
                    improvement = (f_best - current_best) / (fabs(f_best) + 1e-10)
                    
                    if improvement > shrink_threshold:
                        f_best = current_best
                        
                        for j in range(n_cont):
                            pool.x_best_cont[j] = pool.samples_cont[best_idx * n_cont + j]
                        for j in range(n_cat):
                            pool.x_best_cat[j] = pool.samples_cat[best_idx * n_cat + j]
                        
                        no_improvement_count = 0
                        early_stop_counter = 0
                    else:
                        no_improvement_count += 1
                        early_stop_counter += 1
                else:
                    no_improvement_count += 1
                    early_stop_counter += 1
                
                # Adaptive shrinking
                if no_improvement_count >= shrink_patience:
                    sigma_current = sigma_current * shrink_factor
                    no_improvement_count = 0
                    did_shrink = True
                    
                    copy_from_best(&pool)
                    f_current = f_best
                    
                    # Partial momentum reset
                    for j in range(n_cont):
                        pool.m[j] *= 0.5
                        pool.v[j] *= 0.5
            
            if did_shrink:
                shrink_events.append(t)
            
            # Gradient and update (nogil)
            with nogil:
                if use_improvement_weights:
                    n_selected = find_improving_samples(
                        &pool, f_current, top_n_fraction, weight_decay, lambda_t
                    )
                    
                    if n_selected > 0:
                        compute_pseudo_gradient(&pool, n_selected)
                    else:
                        for j in range(n_cont):
                            pool.pseudo_grad[j] = pool.samples_cont[best_idx * n_cont + j] - pool.x_cont[j]
                        n_selected = 1
                        pool.improved_indices[0] = best_idx
                else:
                    n_selected = mu_t
                    for i in range(mu_t):
                        pool.improved_indices[i] = pool.sorted_indices[i]
                    for i in range(mu_t):
                        pool.weights[i] = 1.0 / mu_t
                    compute_pseudo_gradient(&pool, mu_t)
                
                if n_cont > 0:
                    adam_update_nogil(&pool, alpha, beta1, beta2, epsilon, t)
                
                if n_cat > 0 and n_selected > 0:
                    for j in range(n_cat):
                        pool.x_cat[j] = update_categorical_weighted_nogil(
                            &pool, &cat_n_values_view[0], n_selected, j
                        )
            
            # Re-evaluate current
            for i in range(n_cont):
                x_cont_np[i] = pool.x_cont[i]
            for i in range(n_cat):
                x_cat_np[i] = pool.x_cat[i]
            
            try:
                f_current = evaluate_fitness_func(x_cont_np, x_cat_np, minibatch_size)
            except:
                f_current = 1e10
            
            if f_current < f_best:
                f_best = f_current
                with nogil:
                    copy_to_best(&pool)
            
            # Synchronization
            if sync_frequency > 0 and (t + 1) % sync_frequency == 0 and sync_queue is not None:
                x_best_cont_out = np.empty(n_cont, dtype=np.float64)
                x_best_cat_out = np.empty(n_cat, dtype=np.int32)
                
                for i in range(n_cont):
                    x_best_cont_out[i] = pool.x_best_cont[i]
                for i in range(n_cat):
                    x_best_cat_out[i] = pool.x_best_cat[i]
                
                sync_queue.put({
                    'worker_id': worker_id,
                    'x_cont': x_best_cont_out,
                    'x_cat': x_best_cat_out,
                    'f_best': f_best
                })
                
                if sync_event is not None:
                    sync_event.wait()
                    
                    try:
                        global_best = sync_queue.get_nowait()
                        if global_best is not None and global_best['f_best'] < f_best:
                            f_best = global_best['f_best']
                            f_current = f_best
                            
                            for i in range(n_cont):
                                pool.x_cont[i] = global_best['x_cont'][i]
                                pool.x_best_cont[i] = global_best['x_cont'][i]
                            for i in range(n_cat):
                                pool.x_cat[i] = global_best['x_cat'][i]
                                pool.x_best_cat[i] = global_best['x_cat'][i]
                            
                            with nogil:
                                reset_adam_momentum(&pool)
                            
                            sync_events.append(t)
                    except:
                        pass
            
            # History (use preallocated arrays)
            fitness_history_arr[history_idx] = f_best
            sigma_history_arr[history_idx] = sigma_t
            batch_size_history_arr[history_idx] = minibatch_size
            history_idx += 1
            
            # Store params as copies of current arrays
            params_history.append({
                'x_cont': x_cont_np.copy(),
                'x_cat': x_cat_np.copy(),
                'f': float(f_current)
            })
            
            # Early stopping
            if f_best < early_stop_threshold:
                converged = True
                break
            
            if early_stop_counter >= early_stop_patience:
                if history_idx > early_stop_patience:
                    # Get value from preallocated array
                    old_fitness = fitness_history_arr[history_idx - early_stop_patience]
                    recent_improvement = old_fitness - f_best
                    if recent_improvement < shrink_threshold * fabs(f_best):
                        converged = True
                        break
    
    finally:
        # Extract results
        x_best_cont_result = np.empty(n_cont, dtype=np.float64)
        x_best_cat_result = np.empty(n_cat, dtype=np.int32)
        
        for i in range(n_cont):
            x_best_cont_result[i] = pool.x_best_cont[i]
        for i in range(n_cat):
            x_best_cat_result[i] = pool.x_best_cat[i]
        
        with nogil:
            pool_free(&pool)
    
    # Trim history arrays to actual size and convert to lists
    history = {
        'fitness': fitness_history_arr[:history_idx].tolist(),
        'sigma': sigma_history_arr[:history_idx].tolist(),
        'shrink_events': shrink_events,
        'sync_events': sync_events,
        'batch_sizes': batch_size_history_arr[:history_idx].tolist(),
        'params_history': params_history,
        'worker_id': worker_id,
        'converged': converged,
        'final_iteration': t
    }
    
    return (
        x_best_cont_result,
        x_best_cat_result,
        float(f_best),
        history
    )


def get_version():
    """Return RAGDA core version."""
    return "2.0.0-cython"


def is_cython():
    """Check if running Cython version."""
    return True
