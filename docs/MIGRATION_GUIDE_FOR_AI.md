# RAGDA API Migration Guide (For AI Code Assistants)

**Purpose**: This document is for AI code assistants to understand the API changes when refactoring legacy code in `RAGDA_default_args/` and `RAGDA_research/` directories.

**Note**: End users never had access to the old API. This guide is purely for internal code migration.

---

## API Changes Summary

### Space Definition: List → Dict

**OLD (v1.x - Internal Only):**
```python
space = [
    {'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]},
    {'name': 'y', 'type': 'continuous', 'bounds': [-5, 5]},
    {'name': 'lr', 'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
]
```

**NEW (v2.0 - Current):**
```python
space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5]},
    'lr': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
}
```

**Migration Rule**: Convert list-of-dicts to dict-of-dicts where keys are parameter names.

---

### Objective Function: Dict Access → Kwargs

**OLD (v1.x - Internal Only):**
```python
def objective(params):
    x = params['x']
    y = params['y']
    lr = params['lr']
    return (x - 1)**2 + (y - 2)**2
```

**NEW (v2.0 - Current):**
```python
def objective(x, y, lr):
    return (x - 1)**2 + (y - 2)**2
```

**Migration Rules**:
1. Change function signature from `def objective(params):` to `def objective(**param_names):`
2. Remove all `param = params['param']` lines
3. Use parameters directly as function arguments
4. The optimizer calls `objective(**sampled_params)` internally

**Important**: If objective needs extra arguments (e.g., data), use:
```python
def objective(x, y, lr, *, data=None):  # Extra args after *
    # Use x, y, lr directly
    return loss
```

---

### RAGDAOptimizer Constructor

**OLD (v1.x - Internal Only):**
```python
optimizer = RAGDAOptimizer(space, direction='minimize', n_workers=4)
result = optimizer.optimize(objective, n_trials=100)
```

**NEW (v2.0 - Current):**
```python
optimizer = RAGDAOptimizer(
    space=space,              # Now a dict
    direction='minimize',
    n_workers=4
)
result = optimizer.optimize(objective, n_trials=100)
```

**Migration Rule**: No change needed except space format.

---

### SearchSpace Class (Internal)

**OLD (v1.x - Internal Only):**
```python
from ragda.space import SearchSpace
space_obj = SearchSpace(space_list)  # Takes list
```

**NEW (v2.0 - Current):**
```python
from ragda.space import SearchSpace
space_obj = SearchSpace(space_dict)  # Takes dict
```

---

## Common Migration Patterns

### Pattern 1: Simple Optimization

**OLD:**
```python
space = [
    {'name': 'x0', 'type': 'continuous', 'bounds': [-5, 5]},
    {'name': 'x1', 'type': 'continuous', 'bounds': [-5, 5]},
]

def objective(params):
    x = np.array([params['x0'], params['x1']])
    return np.sum(x**2)

optimizer = RAGDAOptimizer(space, direction='minimize')
result = optimizer.optimize(objective, n_trials=100)
```

**NEW:**
```python
space = {
    'x0': {'type': 'continuous', 'bounds': [-5, 5]},
    'x1': {'type': 'continuous', 'bounds': [-5, 5]},
}

def objective(x0, x1):
    x = np.array([x0, x1])
    return np.sum(x**2)

optimizer = RAGDAOptimizer(space=space, direction='minimize')
result = optimizer.optimize(objective, n_trials=100)
```

---

### Pattern 2: Categorical Parameters

**OLD:**
```python
space = [
    {'name': 'lr', 'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    {'name': 'optimizer', 'type': 'categorical', 'values': ['adam', 'sgd']},
]

def objective(params):
    lr = params['lr']
    opt = params['optimizer']
    return train_model(lr, opt)
```

**NEW:**
```python
space = {
    'lr': {'type': 'continuous', 'bounds': [1e-5, 1e-1]},
    'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
}

def objective(lr, optimizer):
    return train_model(lr, optimizer)
```

---

### Pattern 3: High-Dimensional Problems

**OLD:**
```python
n_dims = 200
space = [
    {'name': f'x{i}', 'type': 'continuous', 'bounds': [-5, 5]}
    for i in range(n_dims)
]

def objective(params):
    x = np.array([params[f'x{i}'] for i in range(n_dims)])
    return np.sum(x**2)
```

**NEW:**
```python
n_dims = 200
space = {
    f'x{i}': {'type': 'continuous', 'bounds': [-5, 5]}
    for i in range(n_dims)
}

def objective(**params):
    # Option 1: Unpack all params
    x = np.array([params[f'x{i}'] for i in range(n_dims)])
    return np.sum(x**2)

# Or Option 2: Use *args if parameters are x0, x1, x2, ...
def objective(*args):
    x = np.array(args)
    return np.sum(x**2)
```

---

### Pattern 4: Meta-Optimizer (Nested Optimization)

**OLD:**
```python
def meta_objective(params):
    # Unpack meta-parameters
    lambda_start = params['lambda_start']
    sigma_init = params['sigma_init']
    
    # Create inner optimizer with these params
    inner_opt = RAGDAOptimizer(problem_space, lambda_start=lambda_start, sigma_init=sigma_init)
    result = inner_opt.optimize(problem_objective, n_trials=100)
    
    return result.best_value

meta_space = [
    {'name': 'lambda_start', 'type': 'ordinal', 'values': [10, 20, 50, 100]},
    {'name': 'sigma_init', 'type': 'continuous', 'bounds': [0.1, 0.5]},
]
```

**NEW:**
```python
def meta_objective(lambda_start, sigma_init):
    # Use parameters directly
    inner_opt = RAGDAOptimizer(
        space=problem_space,
        lambda_start=lambda_start,
        sigma_init=sigma_init
    )
    result = inner_opt.optimize(problem_objective, n_trials=100)
    return result.best_value

meta_space = {
    'lambda_start': {'type': 'ordinal', 'values': [10, 20, 50, 100]},
    'sigma_init': {'type': 'continuous', 'bounds': [0.1, 0.5]},
}
```

---

### Pattern 5: Benchmark Functions

**OLD:**
```python
def ackley_ragda(params):
    x = np.array([params['x'], params['y']])
    return ackley(x)  # ackley() expects array

space = [
    {'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]},
    {'name': 'y', 'type': 'continuous', 'bounds': [-5, 5]},
]
```

**NEW (Option 1 - Use Scipy API):**
```python
from ragda import minimize
import numpy as np

result = minimize(
    ackley,  # No wrapper needed!
    bounds=[(-5, 5), (-5, 5)],
    options={'maxiter': 100}
)
```

**NEW (Option 2 - Use Native API with wrapper):**
```python
space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5]},
}

def objective(x, y):
    return ackley(np.array([x, y]))

optimizer = RAGDAOptimizer(space=space)
result = optimizer.optimize(objective, n_trials=100)
```

---

## Files That Need Migration

### High Priority (Used in meta-optimizer and benchmarks)

1. **`RAGDA_default_args/meta_optimizer.py`**
   - Meta-optimization space definition
   - Meta-objective function
   - All nested optimizer calls

2. **`RAGDA_default_args/benchmark_functions.py`**
   - Wrapper functions for standard benchmarks
   - Space definitions for each benchmark

3. **`RAGDA_default_args/benchmark_ml_problems.py`**
   - ML problem space definitions
   - Objective functions

4. **`RAGDA_default_args/benchmark_realworld_problems.py`**
   - Real-world problem spaces
   - Objective functions

5. **`RAGDA_default_args/benchmark_comprehensive.py`**
   - Main benchmark runner
   - Result processing

### Medium Priority (Testing and debugging)

6. **`RAGDA_default_args/debug_problems.py`**
7. **`RAGDA_default_args/audit_problems.py`**
8. **`RAGDA_default_args/test_marsopt_debug.py`**
9. **`RAGDA_default_args/test_meta_simple.py`**

### Low Priority (Documentation)

10. **`RAGDA_research/generate_paper.py`**
11. **`RAGDA_research/Readme_Benchmarking.md`**

---

## Validation Checklist

After migrating a file, verify:

1. ✅ All space definitions use dict format
2. ✅ All objective functions use kwargs unpacking
3. ✅ No `params['key']` dict access in objectives
4. ✅ `RAGDAOptimizer` constructor uses `space=` keyword arg
5. ✅ Tests pass after migration
6. ✅ No deprecation warnings

---

## Example Migration Script

```python
# Before running migration, check current API:
from ragda.space import SearchSpace

# Try creating with dict (should work in v2.0)
space_dict = {'x': {'type': 'continuous', 'bounds': [-5, 5]}}
space_obj = SearchSpace(space_dict)
print("Dict-based API available ✓")

# Try creating with list (should fail or warn in v2.0)
space_list = [{'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]}]
try:
    space_obj = SearchSpace(space_list)
    print("List-based API still works (migration incomplete)")
except:
    print("List-based API removed (migration complete)")
```

---

## API Compatibility Notes

### What Changed
- Space definition format (list → dict)
- Objective function signature (dict access → kwargs)
- Internal `SearchSpace` class constructor

### What Stayed the Same
- `RAGDAOptimizer` constructor (except space format)
- `optimize()` method signature
- `OptimizationResult` structure
- All optimizer parameters (lambda, sigma, etc.)
- High-dimensional optimization interface
- Mini-batch interface
- Worker strategy interface

### New Features in v2.0
- Optuna-style API (`create_study`, `Study` class)
- Scipy-style API (`minimize`, `maximize` functions)
- API adapters for compatibility with other libraries
- Kwargs unpacking for cleaner code

---

## Testing After Migration

Run these tests to verify migration:

```bash
# All core tests should pass
pytest tests/test_optimizer.py -v
pytest tests/test_space_new_api.py -v

# All adapter tests should pass
pytest tests/test_api_*.py -v
pytest tests/test_integration_*.py -v

# Verify no old API usage
grep -r "params\['" RAGDA_default_args/
grep -r "{'name':" RAGDA_default_args/
```

If grep returns results, those files still use the old API.

---

## Questions?

If you encounter edge cases not covered here:
1. Check `tests/test_space_new_api.py` for examples
2. Check `tests/test_api_consistency.py` for cross-API patterns
3. Check `ragda/api_adapters.py` for adapter implementation details
