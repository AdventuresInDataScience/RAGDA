# RAGDA Multi-API Architecture Implementation

## Core Principle
**All API modes funnel into the same canonical internal representation before hitting the core optimizer.**

## Architecture Flow

```
User API → Adapter → Canonical Format → Shared Core → Result Adapter → User API
```

---

## 1. Canonical Internal Format

This is what ALL adapters must produce. It's your existing internal format:

```python
# Canonical space definition (already exists in your SearchSpace class)
canonical_space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5], 'log': True},
    'method': {'type': 'categorical', 'values': ['adam', 'sgd']},
}

# Canonical objective: always expects **kwargs, returns float
def canonical_objective(**params):
    return some_value
```

---

## 2. API Adapters

Each adapter translates its API into canonical format.

### A. Optuna-Style Adapter

**User writes:**
```python
def user_objective(trial):
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', 1e-5, 1e-1, log=True)
    method = trial.suggest_categorical('method', ['adam', 'sgd'])
    return x**2 + y**2

study = ragda.create_study(direction='minimize', api_mode='optuna')
study.optimize(user_objective, n_trials=100)
```

**Adapter does:**
```python
class OptunaAdapter:
    def __init__(self):
        self.space_definition = {}  # Built dynamically
        self.user_objective = None
    
    def create_trial_wrapper(self, params_dict):
        """Create a Trial object that captures suggest_* calls"""
        trial = RAGDATrial(params_dict, self.space_definition)
        return trial
    
    def convert_to_canonical(self):
        """After first trial, we know the space definition"""
        # self.space_definition was populated by trial.suggest_* calls
        # Now convert to canonical format
        canonical_space = {}
        for name, spec in self.space_definition.items():
            if spec['type'] == 'float':
                canonical_space[name] = {
                    'type': 'continuous',
                    'bounds': [spec['low'], spec['high']],
                    'log': spec.get('log', False)
                }
            elif spec['type'] == 'categorical':
                canonical_space[name] = {
                    'type': 'categorical',
                    'values': spec['choices']
                }
        return canonical_space
    
    def wrap_objective(self, user_objective):
        """Wrap user's objective to work with canonical params"""
        def canonical_objective(**params):
            # Create trial object with these params
            trial = self.create_trial_wrapper(params)
            # User's objective calls trial.suggest_*
            result = user_objective(trial)
            return result
        return canonical_objective

class RAGDATrial:
    """Mock Optuna Trial object that reads from pre-sampled params"""
    def __init__(self, params_dict, space_definition):
        self.params = params_dict
        self.space_definition = space_definition
    
    def suggest_float(self, name, low, high, log=False):
        # Record space definition on first call
        if name not in self.space_definition:
            self.space_definition[name] = {
                'type': 'float', 'low': low, 'high': high, 'log': log
            }
        # Return pre-sampled value
        return self.params[name]
    
    def suggest_categorical(self, name, choices):
        if name not in self.space_definition:
            self.space_definition[name] = {
                'type': 'categorical', 'choices': choices
            }
        return self.params[name]
```

**Key insight:** On the **first trial**, the objective function calls `trial.suggest_*`, which:
1. Records the space definition
2. Returns the pre-sampled value from SearchSpace

After the first trial, you have the canonical space definition!

### B. Scipy-Style Adapter

**User writes:**
```python
def user_objective(x):  # x is numpy array
    return x[0]**2 + x[1]**2

bounds = [(-5, 5), (-5, 5)]
result = ragda.minimize(user_objective, bounds=bounds, method='ragda')
```

**Adapter does:**
```python
class ScipyAdapter:
    def to_canonical_space(self, bounds):
        """Convert scipy bounds to canonical space"""
        canonical_space = {}
        for i, (low, high) in enumerate(bounds):
            canonical_space[f'x{i}'] = {
                'type': 'continuous',
                'bounds': [low, high]
            }
        return canonical_space
    
    def wrap_objective(self, user_objective):
        """Convert dict params → array → user objective"""
        def canonical_objective(**params):
            # Convert dict to array in correct order
            x = np.array([params[f'x{i}'] for i in range(len(params))])
            return user_objective(x)
        return canonical_objective
    
    def wrap_result(self, ragda_result):
        """Convert RAGDA result to scipy result format"""
        # Extract array from dict
        x = np.array([ragda_result.best_params[f'x{i}'] 
                      for i in range(len(ragda_result.best_params))])
        
        return scipy.optimize.OptimizeResult(
            x=x,
            fun=ragda_result.best_value,
            success=True,
            nfev=len(ragda_result.trials),
            message='Optimization terminated successfully.'
        )
```

### C. RAGDA Native (Your New API)

**User writes:**
```python
space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5]},
}

def objective(x, y):
    return x**2 + y**2

opt = RAGDAOptimizer(space, api_mode='ragda')
result = opt.optimize(objective, n_trials=100)
```

**Adapter does:**
```python
class RAGDANativeAdapter:
    def to_canonical_space(self, space):
        """Already in canonical format!"""
        return space
    
    def wrap_objective(self, objective):
        """Already expects **kwargs!"""
        return objective
```

---

## 3. Unified Optimizer Core

**This is your existing optimizer with a pre-processing layer:**

```python
class RAGDAOptimizer:
    def __init__(self, space=None, direction='minimize', api_mode='ragda', **kwargs):
        self.api_mode = api_mode
        self.direction = direction
        
        # Select adapter
        if api_mode == 'ragda':
            self.adapter = RAGDANativeAdapter()
            # Space provided upfront
            canonical_space = self.adapter.to_canonical_space(space)
            self.space = SearchSpace(canonical_space)
        
        elif api_mode == 'optuna':
            self.adapter = OptunaAdapter()
            # Space discovered dynamically
            self.space = None  # Will be built on first trial
        
        elif api_mode == 'scipy':
            self.adapter = ScipyAdapter()
            # Space from bounds parameter (set later)
            self.space = None
        
        # Store other kwargs
        self.kwargs = kwargs
    
    def optimize(self, objective, n_trials=100, **opt_kwargs):
        # Wrap objective through adapter
        canonical_objective = self.adapter.wrap_objective(objective)
        
        # For Optuna mode, need to discover space on first call
        if self.api_mode == 'optuna' and self.space is None:
            # Sample dummy params to trigger space definition
            dummy_params = {}
            canonical_objective(**dummy_params)  # This populates adapter.space_definition
            canonical_space = self.adapter.convert_to_canonical()
            self.space = SearchSpace(canonical_space)
        
        # NOW everything is canonical - use existing optimizer logic
        result = self._optimize_canonical(
            canonical_objective, 
            n_trials=n_trials, 
            **opt_kwargs
        )
        
        # Optionally adapt result back to API-specific format
        return self.adapter.wrap_result(result)
    
    def _optimize_canonical(self, objective, n_trials, **kwargs):
        """
        THIS IS YOUR EXISTING OPTIMIZE METHOD - UNCHANGED!
        
        It expects:
        - self.space: SearchSpace object
        - objective: callable that accepts **kwargs
        - Everything else as normal
        """
        # Your existing 1800 lines of optimization logic here
        # Sample from space, call objective(**params), etc.
        # NOTHING CHANGES HERE
        pass
```

---

## 4. Key Implementation Points

### Space Discovery for Optuna

The clever trick: On the **first trial**, when the objective calls `trial.suggest_float('x', -5, 5)`:
1. We don't have 'x' sampled yet
2. So we sample it immediately: `x = np.random.uniform(-5, 5)`
3. Record the space: `space_definition['x'] = {'type': 'float', 'low': -5, 'high': 5}`
4. Return the sampled value

After the first trial, we convert `space_definition` to canonical format and use that for all subsequent trials.

### Objective Wrapping Layers

```python
User's objective
    ↓
[API Adapter wraps it]
    ↓
Canonical objective (**kwargs)
    ↓
[Constraint wrapper if constraints exist]
    ↓
[Sign flip if maximize]
    ↓
Actual evaluation
```

All these wrappers are composed in the adapter layer. Your core optimizer only sees the final canonical objective.

---

## 5. Testing Strategy

```python
def test_all_apis_same_result():
    """All APIs should produce identical results for same problem"""
    
    # Define problem in each API style
    
    # 1. RAGDA Native
    space_ragda = {'x': {'type': 'continuous', 'bounds': [-5, 5]}}
    opt_ragda = RAGDAOptimizer(space_ragda, api_mode='ragda', random_state=42)
    result_ragda = opt_ragda.optimize(lambda x: x**2, n_trials=100)
    
    # 2. Optuna style
    def obj_optuna(trial):
        x = trial.suggest_float('x', -5, 5)
        return x**2
    
    study = ragda.create_study(direction='minimize', api_mode='optuna', random_state=42)
    study.optimize(obj_optuna, n_trials=100)
    result_optuna = study.best_value
    
    # 3. Scipy style
    result_scipy = ragda.minimize(
        lambda x: x[0]**2, 
        bounds=[(-5, 5)], 
        method='ragda',
        options={'maxiter': 100, 'random_state': 42}
    )
    
    # All should find approximately the same optimum
    assert abs(result_ragda.best_value) < 0.1
    assert abs(result_optuna) < 0.1
    assert abs(result_scipy.fun) < 0.1
```

---

## 6. File Structure

```
ragda/
├── __init__.py                 # Public API exports
├── optimizer.py                # Core optimizer (mostly unchanged)
├── space.py                    # SearchSpace (unchanged)
├── api_adapters.py             # NEW: All adapter classes
│   ├── OptunaAdapter
│   ├── ScipyAdapter
│   ├── RAGDANativeAdapter
│   └── Helper classes (RAGDATrial, etc.)
├── api_compat.py               # NEW: Top-level API functions
│   ├── create_study()          # Optuna-compatible
│   ├── minimize()              # Scipy-compatible
│   └── maximize()
└── core.pyx                    # Cython core (unchanged)
```

---

## 7. Benefits of This Architecture

✅ **Single source of truth**: All optimization logic in one place
✅ **Easy to test**: Test canonical format thoroughly, adapters are simple
✅ **Easy to add APIs**: Just write a new adapter, core is untouched
✅ **No performance overhead**: Adapter runs once per trial (negligible)
✅ **Maintainable**: Adapters are ~100 lines each, core is isolated
✅ **Backwards compatible**: RAGDA native mode works as before

---

## 8. Implementation Order

### Phase 1: Foundation (2-3 days)
1. Create `api_adapters.py` with base adapter class
2. Implement `RAGDANativeAdapter` (simplest - pass-through)
3. Refactor existing `RAGDAOptimizer` to use adapter pattern
4. Ensure all existing tests still pass

### Phase 2: Optuna (3-4 days)
1. Implement `RAGDATrial` class
2. Implement `OptunaAdapter`
3. Implement `create_study()` top-level function
4. Test against real Optuna on same problems

### Phase 3: Scipy (2-3 days)
1. Implement `ScipyAdapter`
2. Implement `minimize()` and `maximize()` functions
3. Match scipy's OptimizeResult format
4. Test against scipy on same problems

### Phase 4: Documentation & Benchmarks (2-3 days)
1. Write API comparison guide
2. Create unified benchmarking harness
3. Run cross-optimizer benchmarks
4. Document when to use each API

**Total: ~2 weeks for full implementation**

---

## 9. Example: Complete Optuna Flow

```python
# USER CODE
def objective(trial):
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)
    return x**2 + y**2

study = ragda.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# WHAT HAPPENS INTERNALLY:

# 1. create_study() creates RAGDAOptimizer with OptunaAdapter
study = RAGDAStudy(
    optimizer=RAGDAOptimizer(api_mode='optuna', direction='minimize')
)

# 2. First trial - discover space
optimizer.space is None
params = {}  # Empty
trial = RAGDATrial(params, space_definition={})
objective(trial)  # Calls suggest_float, which populates space_definition

# Now space_definition = {
#     'x': {'type': 'float', 'low': -5, 'high': 5},
#     'y': {'type': 'float', 'low': -5, 'high': 5}
# }

# Convert to canonical
canonical_space = {
    'x': {'type': 'continuous', 'bounds': [-5, 5]},
    'y': {'type': 'continuous', 'bounds': [-5, 5]}
}
optimizer.space = SearchSpace(canonical_space)

# 3. All subsequent trials - use canonical flow
for trial_num in range(1, 100):
    # Sample from canonical space
    params = optimizer.space.sample()  # {'x': 1.23, 'y': -2.45}
    
    # Wrap in trial object
    trial = RAGDATrial(params, space_definition)
    
    # Evaluate (suggest_* just returns pre-sampled values)
    value = objective(trial)
    
    # Rest of optimization logic (workers, ADAM, etc.) - UNCHANGED
```

---

## 10. Critical Success Factor

**The adapter layer must be THIN and FAST.** It should:
- Add < 1ms overhead per trial
- Not copy data unnecessarily
- Use simple dictionary operations
- Fail fast with clear error messages

Since optimization trials typically take >>1ms (often seconds), this overhead is negligible.
