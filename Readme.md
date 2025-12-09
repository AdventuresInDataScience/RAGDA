# RAGDA - Robust ADAM Gradient Descent Approximator

A **pure Cython/C** derivative-free optimization library for hyperparameter tuning and black-box optimization. RAGDA uses ADAM-based gradient approximation with parallel workers for fast, robust optimization.

## Features

- 🚀 **Pure Cython/C** - No Python fallbacks, maximum performance
- ⚡ **Parallel optimization** - Multiple workers with different exploration strategies  
- 🎯 **Mixed variable support** - Continuous, categorical, and ordinal parameters
- 📊 **Mini-batch curriculum learning** - Start with small batches, scale up
- 🔄 **Adaptive sigma shrinking** - Automatic step size reduction on plateaus
- 🛑 **Early stopping** - Convergence detection per worker
- 🔗 **Worker synchronization** - Share best solutions across workers
- 📈 **ADAM optimizer** - Momentum-based gradient approximation
- 🔬 **Automatic high-dim optimization** - Seamlessly handles 100+ dimensions with Kernel PCA, random projections, and adaptive dimensionality reduction - just use `RAGDAOptimizer`!

## Installation

### Using pip
```bash
pip install ragda
```

### From source
```bash
git clone https://github.com/AdventuresInDataScience/ragda.git
cd ragda
pip install .
```

### Requirements

RAGDA automatically installs these dependencies:
- `numpy>=1.20.0` - Core numerical operations
- `pandas>=1.3.0` - Results DataFrames and analysis
- `scipy>=1.7.0` - Latin Hypercube Sampling for initialization
- `cython>=3.0.0` - Build-time dependency for compilation
- `loky>=3.0.0` - Robust parallel execution (Windows-safe)

**Platform Support:** Linux ✓ | macOS ✓ | Windows ✓

**Python Support:** Python 3.8+

## Quick Start

### Basic Optimization

```python
from ragda import RAGDAOptimizer
import numpy as np

# Define search space (dict-based)
space = {
    'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    'dropout': {'type': 'continuous', 'bounds': [0.0, 0.5]},
    'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
    'layers': {'type': 'ordinal', 'values': [1, 2, 3, 4, 5]},
}

# Define objective function (parameters passed as kwargs)
def objective(learning_rate, dropout, optimizer, layers):
    # Your model training/evaluation code here
    # Parameters are passed directly - no dict access needed!
    # Return the metric to minimize (e.g., validation loss)
    return some_loss

# Optimize - works for ANY number of dimensions!
# High-dimensional optimization (100+ dims) is AUTOMATIC
opt = RAGDAOptimizer(space=space, n_workers=4, random_state=42)
result = opt.optimize(objective, n_trials=100)

print(f"Best params: {result.best_params}")
print(f"Best value: {result.best_value}")
```

### Scipy-Style Interface

For simple continuous optimization (array-based):

```python
from ragda import minimize
import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result = minimize(
    rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    options={'maxiter': 100, 'random_state': 42}
)

print(f"Best x: {result.x}")
print(f"Best f: {result.fun}")
```

### High-Dimensional Optimization

**High-dimensional optimization is AUTOMATIC!** Just use `RAGDAOptimizer` - it detects when you have 100+ continuous dimensions and automatically applies dimensionality reduction techniques:

```python
from ragda import RAGDAOptimizer
import numpy as np

# Define a 500-dimensional search space
n_dims = 500
space = {
    f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
    for i in range(n_dims)
}

def high_dim_objective(**params):
    # Only first 10 dimensions matter (common in practice)
    return sum((params[f'x{i}'] - 1.0)**2 for i in range(10))

# Just use RAGDAOptimizer - high-dim handling is automatic!
optimizer = RAGDAOptimizer(
    space=space,
    direction='minimize',
    n_workers=4,
    random_state=42,
    # Optional: customize high-dim behavior
    highdim_threshold=100,      # Trigger high-dim mode at 100+ dims (default)
    variance_threshold=0.95,    # Capture 95% of variance (default)
    reduction_method='auto',    # Auto-select best method (default)
)

result = optimizer.optimize(high_dim_objective, n_trials=200)
print(f"Best value: {result.best_value}")
```

#### Advanced: Using HighDimRAGDAOptimizer Directly

For more control over high-dimensional optimization, you can use the specialized class:

```python
from ragda import HighDimRAGDAOptimizer
import numpy as np

# Define a 500-dimensional search space
n_dims = 500
space = {
    f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]}
    for i in range(n_dims)
}

def high_dim_objective(**params):
    return sum((params[f'x{i}'] - 1.0)**2 for i in range(10))

optimizer = HighDimRAGDAOptimizer(
    space=space,
    direction='minimize',
    dim_threshold=100,          # Use high-dim methods for 100+ dimensions
    variance_threshold=0.95,    # Capture 95% of variance
    reduction_method='auto',    # Auto-select: kernel_pca, incremental_pca, or random_projection
    n_workers=4,
    random_state=42
)

result = optimizer.optimize(
    high_dim_objective,
    n_trials=200,
    initial_samples=150,        # Samples for dimensionality analysis
    stage2_trials_fraction=0.2, # Fraction for trust-region refinement
)

print(f"Best value: {result.best_value}")
```

#### Scipy-Style High-Dim Interface

```python
from ragda import minimize
import numpy as np

def sphere(x):
    return np.sum(x**2)

result = minimize(
    sphere,
    bounds=[(-5, 5)] * 200,  # 200 dimensions
    options={
        'maxiter': 200,
        'random_state': 42,
        'verbose': True
    }
)
```

#### Using the DimensionalityReducer Directly

```python
from ragda import DimensionalityReducer
import numpy as np

# Fit a reducer on your data
X = np.random.randn(1000, 500).astype(np.float64)

reducer = DimensionalityReducer(
    method='kernel_pca',  # or 'incremental_pca', 'random_projection'
    n_components=50,
    kernel='rbf',
    random_seed=42
)

reducer.fit(X)
X_reduced = reducer.transform(X)           # 1000 x 50
X_reconstructed = reducer.inverse_transform(X_reduced)  # 1000 x 500
```

### Mini-Batch Optimization (ML Training)

For expensive objectives with dataset subsampling:

```python
from ragda import RAGDAOptimizer
import numpy as np

# Your training data
X_train, y_train = load_data()  # 10,000 samples

def ml_objective(learning_rate, weight_decay, *, batch_size=-1):
    """
    When batch_size > 0, evaluate on a random subset.
    When batch_size = -1, evaluate on full dataset.
    """
    if batch_size > 0 and batch_size < len(X_train):
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        X, y = X_train[idx], y_train[idx]
    else:
        X, y = X_train, y_train
    
    # Train and evaluate model
    model = train_model(X, y, learning_rate=learning_rate, weight_decay=weight_decay)
    return evaluate_model(model, X, y)

space = {
    'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    'weight_decay': {'type': 'continuous', 'bounds': [1e-6, 1e-2]},
}

optimizer = RAGDAOptimizer(space=space, n_workers=4, random_state=42)
result = optimizer.optimize(
    ml_objective,
    n_trials=100,
    use_minibatch=True,
    data_size=10000,           # Total dataset size
    minibatch_start=100,       # Start with 100 samples
    minibatch_end=5000,        # End with 5000 samples
    minibatch_schedule='inverse_decay',  # Curriculum schedule
)

# Final result is re-evaluated on full dataset
print(f"Best params: {result.best_params}")
print(f"Best value: {result.best_value}")
```

### Dynamic Worker Strategy (Multi-Modal Optimization)

For problems with many local minima, the **dynamic worker strategy** preserves exploration diversity by using elite selection instead of resetting all workers to the global best:

```python
from ragda import RAGDAOptimizer
import numpy as np

# Rastrigin function - highly multi-modal with many local minima
def rastrigin(**params):
    # Extract x0, x1, ..., x9
    x = np.array([params[f'x{i}'] for i in range(10)])
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

space = {
    f'x{i}': {'type': 'continuous', 'bounds': [-5.12, 5.12]}
    for i in range(10)
}

optimizer = RAGDAOptimizer(
    space=space, 
    n_workers=8,
    random_state=42
)

result = optimizer.optimize(
    rastrigin,
    n_trials=500,
    
    # Use dynamic worker strategy (default is 'greedy')
    worker_strategy='dynamic',
    
    # Elite selection: keep top 50% of workers based on their best values
    elite_fraction=0.5,
    
    # How non-elite workers restart after sync
    restart_mode='adaptive',           # 'elite', 'random', or 'adaptive'
    restart_elite_prob_start=0.3,      # Early: 30% restart from elite, 70% random
    restart_elite_prob_end=0.8,        # Late: 80% restart from elite, 20% random
    
    # Optional: gradually reduce active workers for faster convergence
    enable_worker_decay=True,
    min_workers=2,                     # Never go below 2 workers
    worker_decay_rate=0.5,             # Reduce to ~50% of initial workers
    
    sync_frequency=50,                 # Sync and apply elite selection every 50 iterations
)

print(f"Best value: {result.best_value}")
```

#### Worker Strategy Comparison

| Strategy | Behavior at Sync | Best For |
|----------|-----------------|----------|
| `'greedy'` (default) | All workers reset to global best | Unimodal functions, fast convergence |
| `'dynamic'` | Top workers survive, others restart | Multi-modal landscapes, avoiding local minima |

#### Restart Modes (for `worker_strategy='dynamic'`)

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `'elite'` | Non-elite restart from elite positions with perturbation | Exploitation-focused, rugged landscapes |
| `'random'` | Non-elite restart randomly in search space | Exploration-focused, many local minima |
| `'adaptive'` | Transitions from mostly random → mostly elite | Balanced exploration/exploitation (recommended) |

#### Worker Decay

When `enable_worker_decay=True`, the number of active workers gradually decreases over the optimization run:
- Focuses compute on the most promising search directions
- Reduces overhead as the search converges
- `worker_decay_rate=0.5` means ~50% of workers remain by the end
- `min_workers` ensures you never go below a minimum (default: 2)

#### When to Use Dynamic vs Greedy

**Use `worker_strategy='greedy'` when:**
- Your objective is unimodal (single optimum)
- Fast convergence is the priority
- The search space is relatively smooth

**Use `worker_strategy='dynamic'` when:**
- Your objective has multiple local minima (Rastrigin, Ackley, etc.)
- You want to avoid premature convergence
- Exploration diversity matters more than speed

## API Reference

### RAGDAOptimizer

The main optimizer class - handles both standard and high-dimensional problems automatically:

```python
RAGDAOptimizer(
    space: Dict[str, Dict],         # Search space definition (dict-based)
    direction: str = 'minimize',    # 'minimize' or 'maximize'
    n_workers: int = None,          # Number of parallel workers (default: CPU count // 2)
    random_state: int = None,       # Random seed for reproducibility
    
    # High-dimensional settings (automatic when dims >= threshold)
    highdim_threshold: int = 100,   # Trigger high-dim mode at this many continuous dims
    variance_threshold: float = 0.95,  # Variance to capture in reduced space
    reduction_method: str = 'auto',    # 'auto', 'kernel_pca', 'incremental_pca', 'random_projection'
)
```

### optimize()

```python
result = optimizer.optimize(
    objective: Callable,           # Function to optimize
    n_trials: int = 1000,          # Iterations per worker
    x0: Dict = None,               # Starting point
    verbose: bool = True,          # Print progress
    
    # Population & Sampling
    lambda_start: int = 50,        # Initial samples per iteration
    lambda_end: int = 10,          # Final samples per iteration
    lambda_decay_rate: float = 5.0,  # Decay rate for sample count
    
    # Sample Space (sigma controls exploration radius)
    sigma_init: float = 0.3,       # Initial sampling radius in [0,1] unit space
    sigma_final_fraction: float = 0.2,  # Final sigma as fraction of initial
    sigma_decay_schedule: str = 'exponential',  # 'exponential', 'linear', 'cosine'
    
    # Adaptive Shrinking (reduces sigma on stagnation)
    shrink_factor: float = 0.9,    # Multiply sigma by this when stuck
    shrink_patience: int = 10,     # Iterations without improvement before shrinking
    shrink_threshold: float = 1e-6,  # Minimum improvement to count as progress
    
    # Early stopping
    early_stop_threshold: float = 1e-12,
    early_stop_patience: int = 50,
    
    # Worker synchronization
    sync_frequency: int = 100,     # Share best solution every N iters
    
    # ===== Worker Strategy =====
    worker_strategy: str = 'greedy',  # 'greedy' or 'dynamic'
    
    # Dynamic strategy settings (only used when worker_strategy='dynamic')
    elite_fraction: float = 0.5,      # Fraction of top workers to keep (0.0-1.0)
    restart_mode: str = 'adaptive',   # How non-elite workers restart:
                                      #   'elite': restart from elite positions
                                      #   'random': restart randomly
                                      #   'adaptive': transition from random to elite
    restart_elite_prob_start: float = 0.3,  # Initial probability of elite restart
    restart_elite_prob_end: float = 0.8,    # Final probability of elite restart
    enable_worker_decay: bool = False,      # Reduce workers over time
    min_workers: int = 2,                   # Minimum workers to keep
    worker_decay_rate: float = 0.5,         # Decay rate (0.5 = reduce to ~50%)
    
    # Mini-batch settings (for expensive ML objectives)
    use_minibatch: bool = False,
    data_size: int = None,         # Total dataset size
    minibatch_start: int = None,   # Starting batch size
    minibatch_end: int = None,     # Ending batch size
    minibatch_schedule: str = 'inverse_decay',  # 'constant', 'linear', 'exponential', 'inverse_decay', 'step'
    
    # ADAM optimizer settings
    adam_learning_rate: float = 0.001,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
)
```

### OptimizationResult

```python
result.best_params    # Dict of best parameters
result.best_value     # Best objective value
result.n_trials       # Number of iterations
result.n_workers      # Number of workers used
result.direction      # 'minimize' or 'maximize'
```

### HighDimRAGDAOptimizer

For advanced control over high-dimensional optimization (typically you don't need this - use `RAGDAOptimizer` instead):

```python
HighDimRAGDAOptimizer(
    space: Dict[str, Dict],             # Search space definition (dict-based)
    direction: str = 'minimize',        # 'minimize' or 'maximize'
    dim_threshold: int = 100,           # Use high-dim methods above this
    variance_threshold: float = 0.95,   # Variance to capture (0.7-0.99)
    reduction_method: str = 'auto',     # 'auto', 'kernel_pca', 'incremental_pca', 'random_projection'
    trust_region_fraction: float = 0.1, # Trust region size for refinement
    stage2_trials_fraction: float = 0.2,# Fraction of trials for stage 2
    n_workers: int = None,              # Number of parallel workers
    random_state: int = None,           # Random seed
    initial_samples: int = 100,         # Samples for dimensionality analysis
)
```

**Reduction Methods:**
- `kernel_pca`: RBF Kernel PCA for nonlinear structure (best for smooth objectives)
- `incremental_pca`: Online PCA for streaming/large datasets
- `random_projection`: Johnson-Lindenstrauss random projections (fastest)
- `auto`: Automatically selects based on problem characteristics

## Search Space Definition

RAGDA uses a **dict-based** space definition where keys are parameter names:

### Continuous Parameters
```python
'param_name': {'type': 'continuous', 'bounds': [lower, upper]}
```

### Categorical Parameters
```python
'param_name': {'type': 'categorical', 'values': ['a', 'b', 'c']}
```

### Ordinal Parameters
```python
'param_name': {'type': 'ordinal', 'values': [1, 2, 3, 4, 5]}
```

### Log-Scale Parameters
```python
'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True}
```

### Complete Example
```python
space = {
    'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
    'dropout': {'type': 'continuous', 'bounds': [0.0, 0.5]},
    'batch_size': {'type': 'ordinal', 'values': [16, 32, 64, 128]},
    'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
}
```

## How It Works

RAGDA uses a **derivative-free gradient approximation** approach:

1. **Sampling**: Generate candidate solutions around the current best
2. **Evaluation**: Evaluate candidates (optionally in parallel batches)
3. **Gradient Estimation**: Compute approximate gradient from improvement directions
4. **ADAM Update**: Apply ADAM optimizer with momentum
5. **Shrinking**: Reduce step size when stuck on plateaus

### Multi-Worker Strategy

Each worker uses a different `top_n` fraction (exploration vs exploitation):
- Worker 0: top_n=100% (full exploration - uses all samples for gradient)
- Worker 1: top_n=89%
- ...
- Worker N: top_n=20% (greedy exploitation - uses only top 20% of samples)

This diversity helps escape local minima - some workers exploit while others explore.

**Worker Synchronization Strategies:**

1. **Greedy Strategy** (`worker_strategy='greedy'`, default):
   - At each sync point, all workers reset to the global best position
   - Fast convergence for unimodal problems
   - Can get stuck in local minima for multi-modal objectives

2. **Dynamic Strategy** (`worker_strategy='dynamic'`):
   - Uses **elite selection** to maintain search diversity
   - At each sync point:
     1. Workers are ranked by their best objective value
     2. Top `elite_fraction` workers (e.g., top 50%) continue from their current positions
     3. Non-elite workers restart based on `restart_mode`:
        - `'elite'`: Restart from an elite worker's position with small perturbation
        - `'random'`: Restart from a random position in the search space
        - `'adaptive'`: Probability of elite restart increases over time (explore early, exploit late)
   - Optional **worker decay** (`enable_worker_decay=True`) gradually reduces active workers
   - Better for multi-modal landscapes (Rastrigin, Ackley, neural network hyperparameters, etc.)

**Example: Dynamic Strategy for Multi-Modal Optimization**

```python
result = optimizer.optimize(
    objective,
    n_trials=500,
    worker_strategy='dynamic',   # Use elite selection
    elite_fraction=0.5,          # Keep top 50% of workers
    restart_mode='adaptive',     # Explore early, exploit late
    restart_elite_prob_start=0.3,  # 30% elite restarts initially
    restart_elite_prob_end=0.8,    # 80% elite restarts by end
    sync_frequency=50,           # Apply selection every 50 iterations
)
```

### High-Dimensional Optimization Strategy

For problems with 100+ dimensions, RAGDA uses a **two-stage approach**:

1. **Dimensionality Analysis**: Sample the search space and analyze eigenvalue spectrum
2. **Effective Dimension Detection**: Identify if low-dimensional structure exists
3. **Stage 1 - Reduced Space**: Optimize in the low-dimensional subspace found by Kernel PCA
4. **Stage 2 - Trust Region**: Refine the solution in the full space around the Stage 1 result

**Key algorithms (all pure C/Cython for speed):**
- Kernel PCA with RBF kernel and adaptive gamma (median heuristic)
- Quickselect-based O(n) median computation for gamma estimation
- Johnson-Lindenstrauss random projections (Gaussian, sparse, Rademacher)
- Incremental PCA for online/streaming updates

## Running Tests

```bash
# Quick validation tests
python test_quick.py

# Full test suite
python -m pytest tests/ -v

# Exclude slow tests
python -m pytest tests/ -v -m "not slow"
```

## Performance Tips

1. **Use more workers** for hard multi-modal problems
2. **Enable mini-batch** for expensive objectives with large datasets
3. **Tune shrink_patience** - lower for fast convergence, higher for exploration
4. **Set sync_frequency** - more frequent syncing helps on unimodal problems
5. **High-dim is automatic** - `RAGDAOptimizer` detects 100+ dimensions and uses reduction
6. **Customize reduction** - use `reduction_method='kernel_pca'` for smooth objectives, `'random_projection'` for speed
7. **Use dynamic strategy** for multi-modal landscapes with `worker_strategy='dynamic'`
8. **Increase workers** for noisy objectives - more workers means more diverse search directions
9. **Enable worker decay** with `enable_worker_decay=True` for faster late-stage convergence
10. **Tune elite_fraction** - higher (0.6-0.8) for rugged landscapes, lower (0.3-0.5) for smoother ones
11. **Use adaptive restart** (`restart_mode='adaptive'`) for best balance of exploration and exploitation

## Alternative APIs

RAGDA supports multiple API styles for compatibility with existing code:

### Optuna-Style API

```python
from ragda import create_study

def objective(trial):
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    return (x - 1)**2 + (y - 2)**2

study = create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### Scipy-Style API

```python
from ragda import minimize, maximize
import numpy as np

# Minimize
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result = minimize(rosenbrock, bounds=[(-2, 2), (-2, 2)])
print(f"Minimum: {result.fun} at {result.x}")

# Maximize
def neg_sphere(x):
    return -np.sum(x**2)

result = maximize(neg_sphere, bounds=[(-5, 5), (-5, 5)])
print(f"Maximum: {result.fun} at {result.x}")
```

### Native RAGDA API (Most Flexible)

```python
from ragda import RAGDAOptimizer

space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5]},
    'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
}

def objective(x, y, optimizer):
    return (x - 1)**2 + (y - 2)**2

opt = RAGDAOptimizer(space=space, direction='minimize')
result = opt.optimize(objective, n_trials=100)
```

---

## Verifying Installation

```python
import ragda
print(f"RAGDA version: {ragda.__version__}")

# Quick verification with Scipy-style API
from ragda import minimize
import numpy as np

def sphere(x):
    return np.sum(x**2)

result = minimize(sphere, bounds=[(-5, 5)] * 3, options={'maxiter': 50})
print(f"Test optimization: f = {result.fun:.6f} (should be near 0)")

# Check high-dim module
if ragda.HIGHDIM_AVAILABLE:
    print("High-dimensional optimization: Available ✓")
else:
    print("High-dimensional optimization: Not built (run pip install -e .)")
```

## Changelog

### v2.2.0
- **Dynamic worker strategy** - New `worker_strategy='dynamic'` for multi-modal optimization
  - Use `worker_strategy='greedy'` (default) for fast convergence on unimodal problems
  - Use `worker_strategy='dynamic'` for better exploration on multi-modal landscapes
- **Elite selection**: Top workers survive, others restart based on configurable modes
- **Three restart modes** for non-elite workers:
  - `'elite'`: Restart from elite positions with perturbation (exploitation)
  - `'random'`: Restart randomly in search space (exploration)
  - `'adaptive'`: Transitions from random → elite over time (balanced)
- **Worker decay**: Optionally reduce active workers for faster late-stage convergence
  - `enable_worker_decay=True` to enable
  - `worker_decay_rate` controls how fast workers reduce
  - `min_workers` sets the floor
- **New `optimize()` parameters**:
  - `worker_strategy`: `'greedy'` or `'dynamic'`
  - `elite_fraction`: Fraction of top workers to keep (0.0-1.0)
  - `restart_mode`: `'elite'`, `'random'`, or `'adaptive'`
  - `restart_elite_prob_start`: Initial probability of elite restart (for adaptive mode)
  - `restart_elite_prob_end`: Final probability of elite restart (for adaptive mode)
  - `enable_worker_decay`: Enable gradual worker reduction
  - `min_workers`: Minimum workers to keep when decay is enabled
  - `worker_decay_rate`: Rate of worker decay

### v2.1.0
- **Automatic high-dimensional optimization** - `RAGDAOptimizer` now automatically detects and handles 100+ dimension problems
- New constructor parameters: `highdim_threshold`, `variance_threshold`, `reduction_method`
- Kernel PCA, random projections, and incremental PCA for dimensionality reduction
- Two-stage optimization: reduced space exploration + trust region refinement
- Adaptive variance threshold that increases during optimization
- Pure C implementations for all distance/kernel computations (no scipy dependency in hot paths)
- Quickselect-based O(n) median for RBF kernel gamma estimation
- `HighDimRAGDAOptimizer` class for advanced control (optional)
- `highdim_ragda_optimize` convenience function
- `DimensionalityReducer` wrapper for standalone dimensionality reduction
- 31 new tests for high-dimensional functionality

### v2.0.0
- Pure Cython/C implementation (no Python fallbacks)
- Fixed memory corruption issues with preallocated NumPy arrays for history tracking
- Added final full-sample re-evaluation for mini-batch optimization
- Comprehensive test suite (65+ unit and integration tests)
- Windows support via loky for multiprocessing

### v1.0.0
- Initial release

## License

MIT License - see LICENSE file for details.

## Citation

If you use RAGDA in your research, please cite:

```bibtex
@software{ragda2024,
  title = {RAGDA: Robust ADAM Gradient Descent Approximator},
  author = {Adventures In Data Science},
  year = {2024},
  url = {https://github.com/AdventuresInDataScience/ragda}
}
```