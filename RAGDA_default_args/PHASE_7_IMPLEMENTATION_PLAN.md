# Phase 7: Meta-Optimizer Implementation Plan

## Overview

**Goal**: Optimize RAGDA's default parameters using MARsOpt across 234 benchmark problems, classified into 28 categories (27 specific + 1 general).

**Environment**: UV virtual environment (`.venv\Scripts\activate`)

**Core Principle**: 
- All benchmark problems use **Optuna-style API** for cross-optimizer compatibility
- All tests in single file (`tests/test_phase7_meta_optimizer.py`)
- Each step validated before proceeding to next
- Archive old implementation, build clean from scratch

---

## Step 0: Archive Current Implementation

**STATUS**: ⏳ PENDING

### Actions

1. Create archive directory:
   ```powershell
   New-Item -Path "RAGDA_default_args\archive" -ItemType Directory -Force
   ```

2. Move all current files to archive:
   ```powershell
   Get-ChildItem -Path "RAGDA_default_args\*" -Exclude "archive","PHASE_7_IMPLEMENTATION_PLAN.md" | Move-Item -Destination "RAGDA_default_args\archive\"
   ```

3. Verify archive contents:
   - audit_problems.py
   - benchmark_comprehensive.py
   - benchmark_functions.py
   - benchmark_ml_problems.py
   - benchmark_realworld_problems.py
   - debug_marsopt.py
   - debug_problems.py
   - meta_optimizer.py
   - META_OPTIMIZER_DEBUG_SUMMARY.md
   - parameter_audit.py
   - problem_classifications_cache.json
   - problem_classifier.py
   - ragda_parameter_space.py
   - test_marsopt_debug.py
   - test_meta_simple.py

**Validation**: Confirm all files moved, only archive/ and this plan remain.

---

## Step 1: AUC Metric Implementation & Verification

**STATUS**: ⏳ PENDING

**Purpose**: Verify/extract the AUC (Area Under Curve) metric used to evaluate optimizer convergence speed.

### 1.1 Extract AUC Implementation from Archive

**Actions**:

1. Search archive for AUC calculation:
   - Search `archive/meta_optimizer.py` for "auc", "area_under", "convergence"
   - Search `archive/benchmark_*.py` files
   - Identify the exact metric calculation

2. Document the AUC metric:
   - What is being measured? (loss vs iterations, loss vs evaluations, etc.)
   - How is normalization done? (0-1 range)
   - Lower is better (faster convergence)
   - Does it handle cases where true minimum is unknown?

### 1.2 Create AUC Utilities Module

**File**: `RAGDA_default_args/auc_metric.py`

**Content**:
```python
"""
AUC Metric for Optimizer Convergence Evaluation

Measures the area under the convergence curve (normalized 0-1).
Lower AUC = faster convergence = better optimizer performance.
"""

def calculate_auc(
    convergence_history: List[float],
    normalize: bool = True,
    best_known_value: Optional[float] = None
) -> float:
    """
    Calculate AUC from convergence history.
    
    Args:
        convergence_history: List of best-so-far objective values
        normalize: If True, normalize to [0, 1] range
        best_known_value: Known optimal value (if available)
    
    Returns:
        AUC value (0-1, lower is better)
    """
    pass  # Implementation extracted from archive

def evaluate_optimizer_on_problem(
    optimizer_factory: Callable,
    problem: Callable,
    n_evaluations: int,
    **optimizer_kwargs
) -> Tuple[float, List[float]]:
    """
    Run optimizer on problem and return AUC + convergence history.
    """
    pass
```

### 1.3 Tests for AUC Metric

**File**: `tests/test_phase7_meta_optimizer.py` (Section 1)

**Tests**:
```python
def test_auc_perfect_convergence():
    """Immediate convergence should give near-zero AUC."""
    pass

def test_auc_no_convergence():
    """No improvement should give AUC near 1.0."""
    pass

def test_auc_linear_convergence():
    """Linear improvement should give AUC around 0.5."""
    pass

def test_auc_normalization():
    """Test normalization with known bounds."""
    pass

def test_evaluate_optimizer_on_simple_problem():
    """Test wrapper that runs optimizer and computes AUC."""
    pass
```

**Validation**: All AUC metric tests pass before proceeding.

---

## Step 2: Benchmark Functions - Optuna API

**STATUS**: ⏳ PENDING

**Purpose**: Create clean benchmark problem definitions using Optuna-style API.

### 2.1 Mathematical Benchmark Functions

**File**: `RAGDA_default_args/benchmark_functions.py`

**Structure**:
```python
"""
Mathematical Benchmark Functions (Optuna API)

~50 standard test functions from optimization literature.
All wrapped in Optuna-style objective(trial) format.
"""

from typing import Callable, Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class BenchmarkProblem:
    """Metadata for a benchmark problem."""
    name: str
    objective: Callable  # Optuna-style: objective(trial) -> float
    dimension: int
    bounds: List[Tuple[float, float]]
    known_optimum: Optional[float]
    category: str  # 'multimodal', 'unimodal', 'valley', etc.
    description: str

# Example structure:
def sphere_2d_optuna(trial):
    """2D Sphere function: f(x,y) = x² + y²"""
    x = trial.suggest_float('x', -5.0, 5.0)
    y = trial.suggest_float('y', -5.0, 5.0)
    return x**2 + y**2

SPHERE_2D = BenchmarkProblem(
    name='sphere_2d',
    objective=sphere_2d_optuna,
    dimension=2,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    known_optimum=0.0,
    category='unimodal',
    description='Simple convex quadratic function'
)

# ... repeat for all ~50 functions
```

**Functions to Include** (extract from archive):

**Multimodal** (~20 functions):
- Ackley (2D, 5D, 10D, 20D variants)
- Rastrigin (2D, 5D, 10D, 20D)
- Schwefel (2D, 5D, 10D)
- Griewank (2D, 5D, 10D)
- Levy (2D, 5D, 10D)
- Michalewicz (2D, 5D, 10D)
- Drop-Wave (2D)
- Shubert (2D)
- Eggholder (2D)
- Holder Table (2D)
- Langermann (2D)

**Unimodal** (~15 functions):
- Sphere (2D, 5D, 10D, 20D, 50D)
- Rosenbrock (2D, 5D, 10D, 20D)
- Sum of Squares (5D, 10D, 20D)
- Rotated Hyper-Ellipsoid (5D, 10D)

**Other Landscapes** (~15 functions):
- Beale (2D)
- Booth (2D)
- Matyas (2D)
- McCormick (2D)
- Six-Hump Camel (2D)
- Three-Hump Camel (2D)
- Dixon-Price (5D, 10D)
- Zakharov (5D, 10D)
- Powell (4D, 8D)
- Styblinski-Tang (2D, 5D, 10D)

**Registry**:
```python
# Dictionary of all benchmark functions
ALL_BENCHMARK_FUNCTIONS: Dict[str, BenchmarkProblem] = {
    'sphere_2d': SPHERE_2D,
    'sphere_5d': SPHERE_5D,
    # ... etc
}

def get_benchmark_function(name: str) -> BenchmarkProblem:
    """Get benchmark problem by name."""
    pass

def list_benchmark_functions() -> List[str]:
    """List all available benchmark function names."""
    pass
```

### 2.2 Real-World Benchmark Problems

**File**: `RAGDA_default_args/benchmark_realworld_problems.py`

**Structure**: Same as above, but with ~137 real-world problems:

**Categories**:
- Chaotic system parameter estimation (~30 problems)
  - Lorenz, Rössler, Hénon, Mackey-Glass, coupled maps
- Neural network training (~20 problems)
  - Small networks on MNIST/XOR/toy datasets
- Control systems (~25 problems)
  - PID tuning, LQR, trajectory optimization
- Physics simulations (~20 problems)
  - Pendulum, wave equations, spin glass
- System identification (~15 problems)
  - AR/ARMA parameter estimation
- Operations research (~15 problems)
  - Supply chain, routing, scheduling
- Acquisition functions (~12 problems)
  - Bayesian optimization test cases

**Registry**: Same structure as benchmark_functions.py

### 2.3 ML Benchmark Problems

**File**: `RAGDA_default_args/benchmark_ml_problems.py`

**Structure**: ~19 ML hyperparameter tuning problems:
- Scikit-learn models (SVM, RandomForest, GradientBoosting)
- Neural network architectures
- Preprocessing pipeline optimization

**Registry**: Same structure

### 2.4 Master Problem Registry

**File**: `RAGDA_default_args/problem_registry.py`

**Purpose**: Single point to access all 234 problems.

```python
"""
Master Registry of All Benchmark Problems

Aggregates all benchmark problems from:
- benchmark_functions.py (~50)
- benchmark_realworld_problems.py (~137)
- benchmark_ml_problems.py (~19)
Total: ~234 problems
"""

from typing import Dict, List
from benchmark_functions import ALL_BENCHMARK_FUNCTIONS
from benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
from benchmark_ml_problems import ALL_ML_PROBLEMS

# Combine all registries
ALL_PROBLEMS: Dict[str, BenchmarkProblem] = {
    **ALL_BENCHMARK_FUNCTIONS,
    **ALL_REALWORLD_PROBLEMS,
    **ALL_ML_PROBLEMS,
}

def get_problem(name: str) -> BenchmarkProblem:
    """Get any problem by name."""
    pass

def list_all_problems() -> List[str]:
    """List all 234 problem names."""
    pass

def get_problems_by_category(category: str) -> List[BenchmarkProblem]:
    """Get all problems in a category."""
    pass
```

### 2.5 Tests for Benchmark Problems

**File**: `tests/test_phase7_meta_optimizer.py` (Section 2)

**Tests**:
```python
class TestBenchmarkFunctions:
    def test_problem_count(self):
        """Verify we have ~234 total problems."""
        assert len(ALL_PROBLEMS) >= 230
        assert len(ALL_PROBLEMS) <= 240
    
    def test_sphere_2d_optuna(self):
        """Test simple sphere function works with Optuna trial mock."""
        pass
    
    def test_ackley_5d_optuna(self):
        """Test multimodal function."""
        pass
    
    def test_rosenbrock_10d_optuna(self):
        """Test valley function."""
        pass
    
    def test_all_functions_callable(self):
        """Verify all registered functions are callable."""
        pass
    
    def test_problem_metadata_complete(self):
        """Verify all problems have required metadata."""
        pass
    
    def test_sample_realworld_problem(self):
        """Test one chaotic system problem."""
        pass
    
    def test_sample_ml_problem(self):
        """Test one ML hyperparameter problem."""
        pass
```

**Validation**: All benchmark problem tests pass (8 tests).

---

## Step 3: Problem Classification System

**STATUS**: ⏳ PENDING

**Purpose**: Classify all 234 problems into 27 categories (3×3×3 dimensions).

### 3.1 Problem Classifier Module

**File**: `RAGDA_default_args/problem_classifier.py`

**Structure**:
```python
"""
Problem Classifier for RAGDA Benchmarking

Classifies problems by THREE dimensions (3 levels each = 27 total categories):
1. DIMENSIONALITY: low (1-10D), medium (11-50D), high (51D+)
2. COST: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
3. RUGGEDNESS: smooth, moderate, rugged

All functions are DETERMINISTIC. "Ruggedness" = landscape complexity, not noise.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Tuple, Dict
import time
import numpy as np

class DimensionLevel(Enum):
    LOW = 'low'       # 1-10D
    MEDIUM = 'medium' # 11-50D
    HIGH = 'high'     # 51D+

class CostLevel(Enum):
    CHEAP = 'cheap'         # <10ms
    MODERATE = 'moderate'   # 10-100ms
    EXPENSIVE = 'expensive' # 100ms+

class RuggednessLevel(Enum):
    SMOOTH = 'smooth'       # Low sensitivity, consistent gradients
    MODERATE = 'moderate'   # Some local structure
    RUGGED = 'rugged'       # Many local features / chaotic

@dataclass
class ProblemClassification:
    """Classification result for a problem."""
    problem_name: str
    dimension_level: DimensionLevel
    cost_level: CostLevel
    ruggedness_level: RuggednessLevel
    category_key: str  # e.g., "low_cheap_smooth"
    
    # Measurements
    actual_dimension: int
    avg_eval_time_ms: float
    ruggedness_score: float

def classify_dimension(dim: int) -> DimensionLevel:
    """Classify dimensionality (trivial)."""
    if dim <= 10:
        return DimensionLevel.LOW
    elif dim <= 50:
        return DimensionLevel.MEDIUM
    else:
        return DimensionLevel.HIGH

def measure_cost(problem: Callable, n_samples: int = 10) -> Tuple[float, CostLevel]:
    """
    Measure evaluation time by running problem multiple times.
    Returns (avg_time_ms, cost_level).
    """
    pass

def measure_ruggedness(problem: Callable, n_samples: int = 50) -> Tuple[float, RuggednessLevel]:
    """
    Measure landscape ruggedness via:
    - Local sensitivity: output change for small input perturbations
    - Gradient variability: how much improvement direction changes
    
    Returns (ruggedness_score, ruggedness_level).
    """
    pass

def classify_problem(problem: BenchmarkProblem) -> ProblemClassification:
    """
    Fully classify a problem into one of 27 categories.
    """
    pass

def get_category_key(
    dim_level: DimensionLevel,
    cost_level: CostLevel,
    ruggedness_level: RuggednessLevel
) -> str:
    """Generate category key like 'low_cheap_smooth'."""
    return f"{dim_level.value}_{cost_level.value}_{ruggedness_level.value}"
```

### 3.2 Classification Script

**File**: `RAGDA_default_args/classify_all_problems.py`

**Purpose**: Classify all 234 problems and save to JSON cache.

```python
"""
Classify All Benchmark Problems

Runs classification on all 234 problems and saves to cache file.
Supports resume (skips already-classified problems).
"""

import json
from pathlib import Path
from tqdm import tqdm
from problem_registry import ALL_PROBLEMS
from problem_classifier import classify_problem

CACHE_FILE = Path(__file__).parent / "problem_classifications.json"

def load_cache() -> Dict[str, Dict]:
    """Load existing classifications."""
    pass

def save_cache(classifications: Dict[str, Dict]):
    """Save classifications to cache."""
    pass

def classify_all_problems(resume: bool = True):
    """
    Classify all problems.
    
    Args:
        resume: If True, skip already-classified problems
    """
    cache = load_cache() if resume else {}
    
    for name, problem in tqdm(ALL_PROBLEMS.items(), desc="Classifying"):
        if resume and name in cache:
            continue
        
        try:
            classification = classify_problem(problem)
            cache[name] = {
                'dimension_level': classification.dimension_level.value,
                'cost_level': classification.cost_level.value,
                'ruggedness_level': classification.ruggedness_level.value,
                'category_key': classification.category_key,
                'actual_dimension': classification.actual_dimension,
                'avg_eval_time_ms': classification.avg_eval_time_ms,
                'ruggedness_score': classification.ruggedness_score,
            }
            
            # Save after each problem (crash-resistant)
            save_cache(cache)
            
        except Exception as e:
            print(f"Error classifying {name}: {e}")
            continue
    
    return cache

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore cache)')
    args = parser.parse_args()
    
    classifications = classify_all_problems(resume=not args.fresh)
    print(f"\nClassified {len(classifications)} problems")
    
    # Print category distribution
    category_counts = {}
    for data in classifications.values():
        cat = data['category_key']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory Distribution:")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]} problems")
```

### 3.3 Tests for Classification

**File**: `tests/test_phase7_meta_optimizer.py` (Section 3)

**Tests**:
```python
class TestProblemClassification:
    def test_dimension_classification(self):
        """Test dimension level assignment."""
        assert classify_dimension(5) == DimensionLevel.LOW
        assert classify_dimension(25) == DimensionLevel.MEDIUM
        assert classify_dimension(100) == DimensionLevel.HIGH
    
    def test_cost_measurement(self):
        """Test cost measurement on simple function."""
        pass
    
    def test_ruggedness_measurement(self):
        """Test ruggedness on known smooth vs rugged functions."""
        pass
    
    def test_classify_sphere_2d(self):
        """Sphere should be: low, cheap, smooth."""
        pass
    
    def test_classify_ackley_10d(self):
        """Ackley 10D should be: low/medium, cheap/moderate, rugged."""
        pass
    
    def test_category_key_generation(self):
        """Test category key string generation."""
        pass
    
    def test_all_27_categories_possible(self):
        """Verify we can generate all 27 category keys."""
        assert len(set([
            get_category_key(d, c, r)
            for d in DimensionLevel
            for c in CostLevel
            for r in RuggednessLevel
        ])) == 27
```

**Validation**: All classification tests pass (7 tests).

**Manual Validation**: Run `python classify_all_problems.py` and verify:
- All 234 problems classified
- All 27 categories have at least 1 problem
- Distribution looks reasonable (not all in one category)

---

## Step 4: RAGDA Parameter Space Definition

**STATUS**: ⏳ PENDING

**Purpose**: Define all 34 tunable RAGDA parameters for meta-optimization.

### 4.1 Parameter Space Module

**File**: `RAGDA_default_args/ragda_parameter_space.py`

**Content**: Extract from archive, verify compatibility with RAGDA v2.0 API.

**Structure**:
```python
"""
RAGDA Parameter Space for Meta-Optimization

Defines all 34 tunable parameters:
- 7 __init__ parameters (optimizer-level)
- 27 optimize() parameters (run-level)

Includes:
- Parameter bounds, types, defaults
- Constraint definitions
- Penalty functions for invalid configs
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

@dataclass
class ParameterDef:
    """Definition of a single RAGDA parameter."""
    name: str
    param_type: str  # 'int', 'float', 'bool', 'categorical'
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    choices: Optional[List[Any]] = None
    description: str = ""
    location: str = "optimize"  # 'init' or 'optimize'
    constraint_notes: str = ""

# All 34 parameters defined
RAGDA_PARAMETERS: Dict[str, ParameterDef] = {
    # __init__ parameters (7)
    'n_workers': ParameterDef(...),
    'random_state': ParameterDef(...),
    # ... etc
    
    # optimize() parameters (27)
    'maxiter': ParameterDef(...),
    'lambda_start': ParameterDef(...),
    'lambda_end': ParameterDef(...),
    # ... etc
}

def check_constraints(params: Dict[str, Any]) -> List[str]:
    """
    Check if parameter configuration violates constraints.
    Returns list of constraint violations (empty if valid).
    
    Constraints:
    - lambda_end <= lambda_start
    - min_workers <= n_workers
    - sigma_min < sigma_init
    - etc.
    """
    pass

def compute_constraint_penalty(params: Dict[str, Any]) -> float:
    """
    Compute penalty for constraint violations.
    Returns 0.0 if valid, large penalty (e.g., 1000.0 * num_violations) otherwise.
    """
    violations = check_constraints(params)
    return 1000.0 * len(violations)

def get_default_params() -> Dict[str, Any]:
    """Get RAGDA's current default parameters."""
    pass

def split_params_by_location(params: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Split parameters into init_params and optimize_params.
    Returns (init_params, optimize_params).
    """
    pass
```

### 4.2 Tests for Parameter Space

**File**: `tests/test_phase7_meta_optimizer.py` (Section 4)

**Tests**:
```python
class TestRAGDAParameterSpace:
    def test_parameter_count(self):
        """Verify we have 34 parameters defined."""
        assert len(RAGDA_PARAMETERS) == 34
    
    def test_parameter_locations(self):
        """Verify 7 init params, 27 optimize params."""
        init_params = [p for p in RAGDA_PARAMETERS.values() if p.location == 'init']
        opt_params = [p for p in RAGDA_PARAMETERS.values() if p.location == 'optimize']
        assert len(init_params) == 7
        assert len(opt_params) == 27
    
    def test_valid_config_no_penalty(self):
        """Valid config should have zero penalty."""
        valid_params = get_default_params()
        assert compute_constraint_penalty(valid_params) == 0.0
    
    def test_lambda_constraint_violation(self):
        """lambda_end > lambda_start should be penalized."""
        params = get_default_params()
        params['lambda_start'] = 10
        params['lambda_end'] = 20
        assert compute_constraint_penalty(params) > 0
    
    def test_split_params_by_location(self):
        """Test splitting params into init vs optimize."""
        pass
    
    def test_all_params_have_bounds(self):
        """Verify all numeric params have bounds."""
        pass
```

**Validation**: All parameter space tests pass (6 tests).

---

## Step 5: Meta-Optimizer Core Implementation

**STATUS**: ⏳ PENDING

**Purpose**: Implement the meta-optimizer that uses MARsOpt to optimize RAGDA parameters.

### 5.1 Meta-Optimizer Module

**File**: `RAGDA_default_args/meta_optimizer.py`

**Structure**:
```python
"""
RAGDA Meta-Optimizer

Uses MARsOpt to find optimal RAGDA parameters for each of 28 categories:
- 27 specific categories (3×3×3: dimension × cost × ruggedness)
- 1 general category (all problems)

For each category:
1. Select problems in that category
2. Create MARsOpt study
3. Optimize RAGDA params to minimize average AUC across problems
4. Save best parameters

Output: ragda_optimal_defaults.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time

# MARsOpt (external optimizer)
try:
    from marsopt import create_study
except ImportError:
    raise ImportError("MARsOpt required: uv pip install marsopt")

# RAGDA
from ragda import create_study as ragda_create_study

# Internal modules
from problem_registry import ALL_PROBLEMS, get_problem
from problem_classifier import ProblemClassification
from ragda_parameter_space import (
    RAGDA_PARAMETERS,
    compute_constraint_penalty,
    split_params_by_location,
    get_default_params,
)
from auc_metric import calculate_auc

@dataclass
class CategoryOptimizationResult:
    """Result of optimizing RAGDA params for one category."""
    category_key: str
    best_params: Dict[str, Any]
    best_auc: float
    n_problems: int
    n_trials: int
    optimization_time_seconds: float

def load_problem_classifications() -> Dict[str, Dict]:
    """Load problem classifications from cache."""
    cache_file = Path(__file__).parent / "problem_classifications.json"
    with open(cache_file, 'r') as f:
        return json.load(f)

def get_problems_for_category(
    category_key: str,
    classifications: Dict[str, Dict]
) -> List[str]:
    """
    Get list of problem names for a given category.
    
    Args:
        category_key: e.g., 'low_cheap_smooth' or 'general'
        classifications: Problem classification dict
    
    Returns:
        List of problem names in this category
    """
    if category_key == 'general':
        return list(classifications.keys())
    
    return [
        name for name, data in classifications.items()
        if data['category_key'] == category_key
    ]

def evaluate_ragda_on_problem(
    problem_name: str,
    ragda_params: Dict[str, Any],
    n_evaluations: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate RAGDA with given params on a problem.
    Returns AUC (lower is better).
    """
    problem = get_problem(problem_name)
    
    # Split params into init and optimize
    init_params, optimize_params = split_params_by_location(ragda_params)
    
    # Override seed if provided
    if seed is not None:
        init_params['random_state'] = seed
    
    # Override maxiter to use n_evaluations
    optimize_params['maxiter'] = n_evaluations
    
    try:
        # Run RAGDA on problem using Optuna API
        study = ragda_create_study(direction='minimize', **init_params)
        study.optimize(problem.objective, n_trials=n_evaluations)
        
        # Extract convergence history
        convergence_history = [trial.value for trial in study.trials]
        
        # Calculate AUC
        auc = calculate_auc(
            convergence_history,
            normalize=True,
            best_known_value=problem.known_optimum
        )
        
        return auc
        
    except Exception as e:
        print(f"Error evaluating {problem_name}: {e}")
        return 1.0  # Worst possible AUC

def create_marsopt_objective(
    problem_names: List[str],
    n_evaluations_per_problem: int = 100,
) -> Callable:
    """
    Create MARsOpt objective function for a category.
    
    The objective:
    - Takes MARsOpt trial
    - Extracts RAGDA params from trial
    - Evaluates RAGDA on all problems in category
    - Returns average AUC (lower is better)
    """
    def objective(trial):
        # Extract RAGDA parameters from MARsOpt trial
        ragda_params = {}
        
        for param_name, param_def in RAGDA_PARAMETERS.items():
            if param_def.param_type == 'float':
                if param_def.log_scale:
                    value = trial.suggest_float(
                        param_name,
                        param_def.low,
                        param_def.high,
                        log=True
                    )
                else:
                    value = trial.suggest_float(
                        param_name,
                        param_def.low,
                        param_def.high
                    )
            elif param_def.param_type == 'int':
                value = trial.suggest_int(
                    param_name,
                    int(param_def.low),
                    int(param_def.high)
                )
            elif param_def.param_type == 'bool':
                # MARsOpt doesn't handle bools, use categorical
                str_val = trial.suggest_categorical(
                    param_name,
                    ['True', 'False']
                )
                value = (str_val == 'True')
            elif param_def.param_type == 'categorical':
                value = trial.suggest_categorical(
                    param_name,
                    param_def.choices
                )
            else:
                raise ValueError(f"Unknown param type: {param_def.param_type}")
            
            ragda_params[param_name] = value
        
        # Check constraints
        penalty = compute_constraint_penalty(ragda_params)
        if penalty > 0:
            return 1000.0 + penalty  # Large penalty for invalid configs
        
        # Evaluate RAGDA on all problems in category
        aucs = []
        for problem_name in problem_names:
            auc = evaluate_ragda_on_problem(
                problem_name,
                ragda_params,
                n_evaluations=n_evaluations_per_problem
            )
            aucs.append(auc)
        
        # Return average AUC
        avg_auc = np.mean(aucs)
        return avg_auc
    
    return objective

def optimize_category(
    category_key: str,
    problem_names: List[str],
    n_trials: int = 50,
    n_evaluations_per_problem: int = 100,
    n_jobs: int = 1,
) -> CategoryOptimizationResult:
    """
    Optimize RAGDA parameters for one category.
    
    Args:
        category_key: Category name (e.g., 'low_cheap_smooth')
        problem_names: List of problems in this category
        n_trials: Number of MARsOpt trials
        n_evaluations_per_problem: RAGDA evaluations per problem
        n_jobs: Parallel workers for MARsOpt
    
    Returns:
        CategoryOptimizationResult with best params and AUC
    """
    print(f"\n{'='*80}")
    print(f"Optimizing category: {category_key}")
    print(f"Problems: {len(problem_names)}")
    print(f"Trials: {n_trials}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create MARsOpt objective
    objective = create_marsopt_objective(
        problem_names,
        n_evaluations_per_problem=n_evaluations_per_problem
    )
    
    # Create MARsOpt study
    study = create_study(direction='minimize')
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    
    # Extract best parameters
    best_params = study.best_params
    best_auc = study.best_value
    
    elapsed = time.time() - start_time
    
    print(f"\nBest AUC: {best_auc:.6f}")
    print(f"Time: {elapsed:.1f}s")
    
    return CategoryOptimizationResult(
        category_key=category_key,
        best_params=best_params,
        best_auc=best_auc,
        n_problems=len(problem_names),
        n_trials=n_trials,
        optimization_time_seconds=elapsed,
    )

def optimize_all_categories(
    n_trials: int = 50,
    n_evaluations_per_problem: int = 100,
    n_jobs: int = 1,
    categories: Optional[List[str]] = None,
    output_file: str = "ragda_optimal_defaults.json",
) -> Dict[str, CategoryOptimizationResult]:
    """
    Optimize RAGDA parameters for all 28 categories.
    
    Args:
        n_trials: MARsOpt trials per category
        n_evaluations_per_problem: RAGDA evaluations per problem
        n_jobs: Parallel workers
        categories: List of categories to optimize (None = all 28)
        output_file: Where to save results
    
    Returns:
        Dict mapping category_key -> CategoryOptimizationResult
    """
    # Load classifications
    classifications = load_problem_classifications()
    
    # Determine categories to optimize
    if categories is None:
        # All 27 specific + 1 general
        all_category_keys = set(
            data['category_key'] 
            for data in classifications.values()
        )
        categories = sorted(all_category_keys) + ['general']
    
    print(f"\n{'='*80}")
    print(f"RAGDA Meta-Optimizer")
    print(f"{'='*80}")
    print(f"Categories: {len(categories)}")
    print(f"Total problems: {len(classifications)}")
    print(f"Trials per category: {n_trials}")
    print(f"Evaluations per problem: {n_evaluations_per_problem}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"{'='*80}\n")
    
    # Optimize each category
    results = {}
    for i, category_key in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Category: {category_key}")
        
        problem_names = get_problems_for_category(category_key, classifications)
        
        if len(problem_names) == 0:
            print(f"WARNING: No problems for category {category_key}, skipping")
            continue
        
        result = optimize_category(
            category_key,
            problem_names,
            n_trials=n_trials,
            n_evaluations_per_problem=n_evaluations_per_problem,
            n_jobs=n_jobs,
        )
        
        results[category_key] = result
    
    # Save results
    output_path = Path(__file__).parent.parent / output_file
    with open(output_path, 'w') as f:
        json.dump(
            {
                key: asdict(result)
                for key, result in results.items()
            },
            f,
            indent=2
        )
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return results
```

### 5.2 Entry Point Script

**File**: `RAGDA_default_args/run_meta_optimizer.py`

**Purpose**: CLI interface to run meta-optimizer.

```python
"""
Run RAGDA Meta-Optimizer

Usage:
    python run_meta_optimizer.py                    # Full run (all 28 categories)
    python run_meta_optimizer.py --test_mode        # Quick test (1 category, 5 trials)
    python run_meta_optimizer.py --categories low_cheap_smooth medium_moderate_rugged
"""

import argparse
from meta_optimizer import optimize_all_categories

def main():
    parser = argparse.ArgumentParser(description="RAGDA Meta-Optimizer")
    
    parser.add_argument(
        '--categories',
        nargs='*',
        default=None,
        help='Specific categories to optimize (default: all 28)'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='MARsOpt trials per category (default: 50)'
    )
    
    parser.add_argument(
        '--n_evaluations_per_problem',
        type=int,
        default=100,
        help='RAGDA evaluations per problem (default: 100)'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='ragda_optimal_defaults.json',
        help='Output file (default: ragda_optimal_defaults.json)'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='Quick test: 1 category, 5 trials, 20 evaluations'
    )
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test_mode:
        print("\n*** TEST MODE ***\n")
        categories = ['low_cheap_smooth']  # Just one category
        n_trials = 5
        n_evaluations = 20
    else:
        categories = args.categories
        n_trials = args.n_trials
        n_evaluations = args.n_evaluations_per_problem
    
    # Run optimization
    results = optimize_all_categories(
        n_trials=n_trials,
        n_evaluations_per_problem=n_evaluations,
        n_jobs=args.n_jobs,
        categories=categories,
        output_file=args.output,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for category_key, result in results.items():
        print(f"{category_key:30s} | AUC: {result.best_auc:.6f} | Problems: {result.n_problems:3d}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
```

### 5.3 Tests for Meta-Optimizer

**File**: `tests/test_phase7_meta_optimizer.py` (Section 5)

**Tests**:
```python
class TestMetaOptimizer:
    def test_load_classifications(self):
        """Test loading problem classifications."""
        pass
    
    def test_get_problems_for_category(self):
        """Test filtering problems by category."""
        pass
    
    def test_evaluate_ragda_on_simple_problem(self):
        """Test RAGDA evaluation on sphere function."""
        pass
    
    def test_create_marsopt_objective(self):
        """Test MARsOpt objective creation."""
        pass
    
    def test_constraint_penalty_in_objective(self):
        """Test that invalid configs are penalized."""
        pass
    
    def test_optimize_single_category_minimal(self):
        """
        Minimal test: optimize 1 category with 2 problems, 3 trials.
        Should complete without errors.
        """
        pass
    
    def test_general_category_includes_all_problems(self):
        """Test that 'general' category includes all 234 problems."""
        pass
```

**Validation**: All meta-optimizer tests pass (7 tests).

---

## Step 6: Integration Testing & Validation

**STATUS**: ⏳ PENDING

**Purpose**: End-to-end validation before full run.

### 6.1 Integration Tests

**File**: `tests/test_phase7_meta_optimizer.py` (Section 6)

**Tests**:
```python
class TestMetaOptimizerIntegration:
    def test_end_to_end_single_category(self):
        """
        Full end-to-end test with single category:
        1. Load 2-3 problems in category
        2. Run MARsOpt with 3 trials
        3. Verify output format
        4. Check that AUC improves
        """
        pass
    
    def test_test_mode_execution(self):
        """
        Test that test mode runs successfully:
        python run_meta_optimizer.py --test_mode
        """
        pass
    
    def test_output_file_format(self):
        """
        Verify output JSON has correct structure:
        {
            "category_key": {
                "best_params": {...},
                "best_auc": 0.123,
                "n_problems": 10,
                ...
            }
        }
        """
        pass
    
    def test_all_28_categories_have_problems(self):
        """
        Verify that all 27 specific categories have at least 1 problem,
        and general category has all 234.
        """
        pass
```

**Validation**: All integration tests pass (4 tests).

### 6.2 Manual Validation

**Actions**:

1. **Test Mode Run**:
   ```powershell
   uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
   ```
   - Should complete in ~2-5 minutes
   - Verify no errors
   - Check output file created

2. **Single Category Run**:
   ```powershell
   uv run python RAGDA_default_args/run_meta_optimizer.py --categories low_cheap_smooth --n_trials 10
   ```
   - Should complete in ~10-20 minutes
   - Verify convergence
   - Check AUC values reasonable

3. **Classification Validation**:
   - Verify `problem_classifications.json` has 234 entries
   - Check distribution across 27 categories
   - Ensure no category is empty

---

## Step 7: Documentation

**STATUS**: ⏳ PENDING

**Purpose**: Document the meta-optimizer system.

### 7.1 README for RAGDA_default_args

**File**: `RAGDA_default_args/README.md`

**Content**:
```markdown
# RAGDA Meta-Optimizer

This directory contains the meta-optimization system for finding optimal RAGDA default parameters.

## Overview

- **Goal**: Find universally good default parameters for RAGDA
- **Method**: Use MARsOpt to optimize RAGDA params across 234 benchmark problems
- **Categories**: 28 total (27 specific + 1 general)
  - 3 dimensionality levels: low (1-10D), medium (11-50D), high (51D+)
  - 3 cost levels: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
  - 3 ruggedness levels: smooth, moderate, rugged
- **Metric**: AUC (area under convergence curve, 0-1, lower is better)

## File Structure

- `benchmark_functions.py` - 50 mathematical test functions (Optuna API)
- `benchmark_realworld_problems.py` - 137 real-world problems (Optuna API)
- `benchmark_ml_problems.py` - 19 ML hyperparameter problems (Optuna API)
- `problem_registry.py` - Master registry of all 234 problems
- `problem_classifier.py` - Problem classification logic
- `classify_all_problems.py` - Script to classify all problems
- `problem_classifications.json` - Classification cache
- `ragda_parameter_space.py` - RAGDA parameter definitions & constraints
- `auc_metric.py` - AUC calculation for optimizer evaluation
- `meta_optimizer.py` - Core meta-optimization logic
- `run_meta_optimizer.py` - CLI interface
- `archive/` - Old implementation (reference only)

## Usage

### 1. Classify Problems (one-time)

```powershell
uv run python RAGDA_default_args/classify_all_problems.py
```

This measures dimension, cost, and ruggedness for all 234 problems.
Results cached in `problem_classifications.json`.

### 2. Run Meta-Optimizer

**Test mode** (quick validation):
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
```

**Single category**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --categories low_cheap_smooth --n_trials 50
```

**Full run** (all 28 categories):
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --n_trials 50 --n_jobs 4
```

### 3. Results

Output saved to `ragda_optimal_defaults.json`:
```json
{
  "low_cheap_smooth": {
    "best_params": {
      "n_workers": 8,
      "lambda_start": 100,
      ...
    },
    "best_auc": 0.234,
    "n_problems": 15,
    "n_trials": 50,
    "optimization_time_seconds": 1234.5
  },
  ...
}
```

## Development

Tests located in `../tests/test_phase7_meta_optimizer.py`.

Run all tests:
```powershell
uv run pytest tests/test_phase7_meta_optimizer.py -v
```

## Future: Cross-Optimizer Benchmarking

All problems use Optuna API, enabling easy comparison with other optimizers:
- Optuna's TPE, NSGAII, etc.
- Scipy optimizers
- Hyperopt, Ax, etc.

Same problems, same metric (AUC), fair comparison.
```

### 7.2 Update Main Implementation Plan

**File**: `API_REDESIGN_IMPLEMENTATION_PLAN.md`

**Action**: Replace Phase 7 section with:
```markdown
## Phase 7: Meta-Optimizer & Default Parameter Optimization (Days 14-18)

**STATUS**: ✅ COMPLETE

Phase 7 involved building a comprehensive meta-optimization system to find optimal default parameters for RAGDA.

See `RAGDA_default_args/PHASE_7_IMPLEMENTATION_PLAN.md` for detailed plan and progress.

**Summary**:
- Archived old implementation
- Created 234 benchmark problems (Optuna API)
- Implemented problem classification (28 categories)
- Built meta-optimizer using MARsOpt
- Optimized RAGDA parameters for each category
- Generated `ragda_optimal_defaults.json`

**Test Results**: X/X tests passing in `tests/test_phase7_meta_optimizer.py`

**Key Deliverables**:
- Clean benchmark problem library (Optuna API)
- Problem classifier (3×3×3 = 27 categories + general)
- Meta-optimizer (MARsOpt-based)
- Optimal default parameters for 28 categories
```

---

## Step 8: Full Execution

**STATUS**: ⏳ PENDING

**Purpose**: Run the full meta-optimization.

### 8.1 Problem Classification

**Command**:
```powershell
uv run python RAGDA_default_args/classify_all_problems.py
```

**Expected**:
- Runtime: ~30-60 minutes (234 problems, measuring cost & ruggedness)
- Output: `problem_classifications.json` with 234 entries
- Distribution across 27 categories

**Validation**:
- Check all problems classified
- Verify no category is empty
- Review distribution manually

### 8.2 Meta-Optimizer Test Run

**Command**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
```

**Expected**:
- Runtime: ~2-5 minutes
- Tests single category with 5 trials
- Validates end-to-end flow

**Validation**:
- No errors
- Output file created
- AUC values reasonable (0-1 range)

### 8.3 Full Meta-Optimization

**Command**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --n_trials 50 --n_jobs 4
```

**Expected**:
- Runtime: Hours to days (28 categories × 50 trials × ~8 problems/category × 100 evals)
- Progress printed for each category
- Resumable if interrupted (can run specific categories)

**Alternative (sequential categories)**:
```powershell
# Run each category separately (resumable)
foreach ($cat in @('low_cheap_smooth', 'low_cheap_moderate', ...)) {
    uv run python RAGDA_default_args/run_meta_optimizer.py --categories $cat --n_trials 50
}
```

**Validation**:
- Check `ragda_optimal_defaults.json` has 28 entries
- Review AUC values (should be < 1.0, ideally < 0.5)
- Compare with default RAGDA params (should be improvements)

---

## Success Criteria

### Must Have:
- [ ] All files archived to `archive/` subdirectory
- [ ] AUC metric implemented and tested
- [ ] All 234 benchmark problems defined (Optuna API)
- [ ] Problem classifier working (27 categories + general)
- [ ] All problems classified and cached
- [ ] RAGDA parameter space defined (34 parameters)
- [ ] Meta-optimizer implemented and tested
- [ ] Test mode runs successfully
- [ ] All tests pass (`tests/test_phase7_meta_optimizer.py`)
- [ ] Documentation complete
- [ ] Full meta-optimization run completed
- [ ] `ragda_optimal_defaults.json` generated

### Testing Milestones:
- [ ] Step 1: 5 AUC metric tests passing
- [ ] Step 2: 8 benchmark problem tests passing
- [ ] Step 3: 7 classification tests passing
- [ ] Step 4: 6 parameter space tests passing
- [ ] Step 5: 7 meta-optimizer unit tests passing
- [ ] Step 6: 4 integration tests passing
- [ ] **Total: ~37 tests passing before full run**

### Quality Checks:
- [ ] No old API usage (all Optuna-style)
- [ ] All problems have metadata (bounds, dimension, category)
- [ ] All 27 specific categories have ≥1 problem
- [ ] General category has all 234 problems
- [ ] Constraint violations properly penalized
- [ ] AUC values in valid range (0-1)
- [ ] Results reproducible with fixed seed

---

## Timeline Estimate

| Step | Task | Estimated Time |
|------|------|----------------|
| 0 | Archive current files | 5 min |
| 1 | AUC metric implementation | 1-2 hours |
| 2 | Benchmark problems (234 × Optuna API) | 4-6 hours |
| 3 | Problem classification system | 2-3 hours |
| 4 | RAGDA parameter space | 1 hour |
| 5 | Meta-optimizer core | 3-4 hours |
| 6 | Integration testing | 1-2 hours |
| 7 | Documentation | 1 hour |
| 8 | Full execution | Hours-Days |
| **Total** | **Development: ~1-2 days** | **Execution: Hours-Days** |

---

## Notes

- **UV Environment**: Always activate with `.venv\Scripts\activate`
- **MARsOpt**: Already installed in UV env
- **Tests**: All in single file `tests/test_phase7_meta_optimizer.py`
- **Resumability**: Classification and meta-optimizer support resume
- **Parallel**: Use `--n_jobs` for parallel MARsOpt trials
- **Test Mode**: Use `--test_mode` for quick validation
- **Archive**: Reference only, do not modify

---

## Future Work (Post-Phase 7)

1. **Integration into RAGDA**: Load optimal defaults based on problem characteristics
2. **Cross-optimizer benchmarking**: Compare RAGDA vs Optuna, Scipy, etc.
3. **Research paper**: Publish benchmark suite and results
4. **Adaptive defaults**: Real-time problem classification and parameter selection
5. **Continuous optimization**: Re-run meta-optimizer as RAGDA evolves
