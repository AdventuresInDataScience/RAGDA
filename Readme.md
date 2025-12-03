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

# Define search space
space = [
    {'name': 'learning_rate', 'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    {'name': 'dropout', 'type': 'continuous', 'bounds': [0.0, 0.5]},
    {'name': 'optimizer', 'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
    {'name': 'layers', 'type': 'ordinal', 'values': [1, 2, 3, 4, 5]},
]

# Define objective function
def objective(params):
    # Your model training/evaluation code here
    # Return the metric to minimize (e.g., validation loss)
    return some_loss

# Optimize
optimizer = RAGDAOptimizer(space, n_workers=4, random_state=42)
result = optimizer.optimize(objective, n_trials=100)

print(f"Best params: {result.best_params}")
print(f"Best value: {result.best_value}")
```

### Scipy-Style Interface

For simple continuous optimization:

```python
from ragda import ragda_optimize
import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float64)
x_best, f_best, info = ragda_optimize(rosenbrock, bounds, n_trials=100)

print(f"Best x: {x_best}")
print(f"Best f: {f_best}")
```

### Mini-Batch Optimization (ML Training)

For expensive objectives with dataset subsampling:

```python
from ragda import RAGDAOptimizer
import numpy as np

# Your training data
X_train, y_train = load_data()  # 10,000 samples

def ml_objective(params, batch_size=-1):
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
    model = train_model(X, y, **params)
    return evaluate_model(model, X, y)

space = [
    {'name': 'learning_rate', 'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    {'name': 'weight_decay', 'type': 'continuous', 'bounds': [1e-6, 1e-2]},
]

optimizer = RAGDAOptimizer(space, n_workers=4, random_state=42)
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

## API Reference

### RAGDAOptimizer

```python
RAGDAOptimizer(
    space: List[Dict],              # Search space definition
    direction: str = 'minimize',    # 'minimize' or 'maximize'
    n_workers: int = None,          # Number of parallel workers (default: CPU count)
    random_state: int = None,       # Random seed for reproducibility
)
```

### optimize()

```python
result = optimizer.optimize(
    objective: Callable,           # Function to optimize
    n_trials: int = 1000,          # Iterations per worker
    x0: Dict = None,               # Starting point
    verbose: bool = True,          # Print progress
    
    # Early stopping
    early_stop_threshold: float = 1e-12,
    early_stop_patience: int = 50,
    
    # Sigma shrinking (adaptive step size)
    shrink_factor: float = 0.9,
    shrink_patience: int = 10,
    shrink_threshold: float = 1e-6,
    
    # Mini-batch settings
    use_minibatch: bool = False,
    data_size: int = 1000,
    minibatch_start: int = 50,
    minibatch_end: int = 500,
    minibatch_schedule: str = 'linear',  # 'linear', 'exponential', 'inverse_decay'
    
    # Worker synchronization
    sync_frequency: int = 100,     # Share best solution every N iters
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

## Search Space Definition

### Continuous Parameters
```python
{'name': 'param_name', 'type': 'continuous', 'bounds': [lower, upper]}
```

### Categorical Parameters
```python
{'name': 'param_name', 'type': 'categorical', 'values': ['a', 'b', 'c']}
```

### Ordinal Parameters
```python
{'name': 'param_name', 'type': 'ordinal', 'values': [1, 2, 3, 4, 5]}
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
- Worker 0: top_n=100% (full exploration)
- Worker 1: top_n=89%
- ...
- Worker N: top_n=20% (greedy exploitation)

Workers periodically synchronize, sharing the global best solution.

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

## Verifying Installation

```python
import ragda
print(f"RAGDA version: {ragda.__version__}")

# Quick verification
from ragda import ragda_optimize
import numpy as np

def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-5, 5]] * 3, dtype=np.float64)
x_best, f_best, info = ragda_optimize(sphere, bounds, n_trials=50)
print(f"Test optimization: f_best = {f_best:.6f} (should be near 0)")
```

## Changelog

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