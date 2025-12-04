"""
Meta-Optimizer for RAGDA Default Parameters

Uses MARSOpt to find optimal RAGDA parameters for each of the 27 problem categories
(3 dimensions × 3 costs × 3 ruggedness levels) plus one general-purpose default.

The output is a config file that RAGDA can load to use smart defaults based on
problem characteristics.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# MARSOpt import
try:
    from marsopt import Study as MARSOptStudy
except ImportError:
    raise ImportError("MARSOpt is required. Install with: pip install marsopt")

# RAGDA import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ragda import RAGDAOptimizer

# Import the correct parameter space definition
from ragda_parameter_space import (
    RAGDA_PARAMETERS,
    compute_constraint_penalty,
    is_valid_config,
    split_params_by_location,
    get_default_params,
    ALL_TUNABLE_PARAMS,
    SIMPLE_PARAMS,
    CORE_PARAMS,
)


@dataclass
class CategoryResult:
    """Result of optimizing RAGDA params for one category."""
    category: str
    dim_class: str
    cost_class: str
    ruggedness_class: str
    best_params: Dict[str, Any]
    best_loss: float
    n_problems: int
    n_iterations: int
    optimization_time_seconds: float
    constraint_violations: List[str] = None


# =============================================================================
# HELPER FUNCTIONS FOR LOGGING
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def interpret_loss(loss: float) -> str:
    """
    Interpret a loss value and return a human-readable quality description.
    
    The loss is normalized so that:
    - 0.0 = always found the global optimum
    - 1.0 = equivalent to random sampling
    """
    if loss < 0.05:
        return "excellent - near-optimal solutions"
    elif loss < 0.15:
        return "very good - consistently strong"
    elif loss < 0.30:
        return "good - solid performance"
    elif loss < 0.50:
        return "moderate - room for improvement"
    elif loss < 0.75:
        return "fair - mediocre performance"
    elif loss < 1.0:
        return "poor - barely better than random"
    elif loss >= 1e9:
        return "FAILED - evaluation errors"
    else:
        return "very poor - worse than random or constraint violations"


# =============================================================================
# PROBLEM LOADING WITH CACHING
# =============================================================================

CLASSIFICATION_CACHE_FILE = "problem_classifications_cache.json"


def load_classification_cache() -> Dict[str, dict]:
    """Load cached problem classifications if available."""
    cache_path = Path(__file__).parent / CLASSIFICATION_CACHE_FILE
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            print(f"  Loaded classification cache with {len(cache)} problems")
            return cache
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
    return {}


def save_classification_cache(cache: Dict[str, dict], verbose: bool = False):
    """Save problem classifications to cache file."""
    cache_path = Path(__file__).parent / CLASSIFICATION_CACHE_FILE
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        if verbose:
            print(f"  Saved classification cache with {len(cache)} problems")
    except Exception as e:
        print(f"  Warning: Could not save cache: {e}")


def load_all_problems(use_cache: bool = True) -> Dict[str, List[dict]]:
    """
    Load all benchmark problems and organize by category.
    
    Args:
        use_cache: If True, use cached classifications when available
    
    Returns:
        Dict mapping category key (e.g., "low_cheap_smooth") to list of problem dicts
    """
    from benchmark_functions import get_all_functions
    from benchmark_ml_problems import get_all_ml_problems
    from benchmark_realworld_problems import get_all_genuine_problems
    from problem_classifier import classify_problem, DimClass, CostLevel, RuggednessLevel
    
    # Load classification cache
    cache = load_classification_cache() if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    
    all_problems = []
    
    # Load synthetic functions
    for key, tf in get_all_functions().items():
        # Create space in RAGDA format
        space = [
            {'name': f'x{i}', 'type': 'continuous', 'bounds': list(tf.bounds[i])}
            for i in range(tf.dim)
        ]
        
        # Create wrapper that takes dict params
        def make_dict_wrapper(orig_func, dim):
            def wrapper(params):
                x = np.array([params[f'x{i}'] for i in range(dim)])
                return orig_func(x)
            return wrapper
        
        all_problems.append({
            'name': tf.name,
            'func': make_dict_wrapper(tf.func, tf.dim),
            'space': space,
            'bounds': np.array(tf.bounds),
            'dim': tf.dim,
            'source': 'synthetic',
        })
    
    # Load ML problems (already use dict params)
    for key, mp in get_all_ml_problems().items():
        all_problems.append({
            'name': mp.name,
            'func': mp.func,
            'space': mp.space,
            'bounds': None,  # Mixed space, no simple bounds
            'dim': mp.dim,
            'source': 'ml',
        })
    
    # Load real-world problems
    for gp in get_all_genuine_problems():
        # Create space in RAGDA format
        space = [
            {'name': f'x{i}', 'type': 'continuous', 'bounds': list(gp.bounds[i])}
            for i in range(gp.dim)
        ]
        
        # Create wrapper that takes dict params
        def make_dict_wrapper2(orig_func, dim):
            def wrapper(params):
                x = np.array([params[f'x{i}'] for i in range(dim)])
                return orig_func(x)
            return wrapper
        
        all_problems.append({
            'name': gp.name,
            'func': make_dict_wrapper2(gp.func, gp.dim),
            'space': space,
            'bounds': np.array(gp.bounds),
            'dim': gp.dim,
            'source': 'realworld',
        })
    
    # Classify each problem (using cache when available)
    print(f"Classifying {len(all_problems)} problems...")
    sys.stdout.flush()
    categorized = {}
    current_problem = None  # Track for error reporting
    
    for i, p in enumerate(all_problems):
        current_problem = p['name']  # Track which problem we're on
        try:
            # Check cache first
            if p['name'] in cache:
                cached = cache[p['name']]
                key = cached['category_key']
                cache_hits += 1
            else:
                # Need to classify - Use bounds if available, else extract from space
                if p['bounds'] is not None:
                    bounds = p['bounds']
                else:
                    bounds = np.array([
                        s.get('bounds', [0, 1]) for s in p['space'] if s['type'] == 'continuous'
                    ])
                
                cp = classify_problem(
                    name=p['name'],
                    func=lambda x: p['func']({f'x{i}': x[i] for i in range(len(x))}),
                    bounds=bounds,
                    dim=p['dim'],
                    category=p['source'],
                    n_timing_samples=3,
                    n_ruggedness_samples=10,
                )
                
                c = cp.characteristics
                key = f"{c.dim_class.value}_{c.cost_level.value}_{c.ruggedness_level.value}"
                
                # Cache this classification and SAVE IMMEDIATELY
                cache[p['name']] = {
                    'category_key': key,
                    'dim_class': c.dim_class.value,
                    'cost_level': c.cost_level.value,
                    'ruggedness_level': c.ruggedness_level.value,
                    'dim': p['dim'],
                    'source': p['source'],
                }
                try:
                    save_classification_cache(cache)
                except Exception as e:
                    print(f"  Warning: Could not save cache after {p['name']}: {e}")
                
                cache_misses += 1
            
            if key not in categorized:
                categorized[key] = []
            
            categorized[key].append({
                'name': p['name'],
                'func': p['func'],
                'space': p['space'],
                'bounds': p['bounds'],
                'dim': p['dim'],
            })
        except Exception as e:
            import traceback
            print(f"\n  ERROR classifying problem #{i+1}: {p['name']}")
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            sys.stdout.flush()
        
        # Progress update every 20 problems (shows last problem name for debugging)
        if (i + 1) % 20 == 0:
            print(f"  Classified {i+1}/{len(all_problems)} (last: {p['name']})")
            sys.stdout.flush()
    
    # Final progress message
    print(f"  Classified {len(all_problems)}/{len(all_problems)} - DONE")
    print(f"  Classification loop complete. {len(categorized)} categories.")
    sys.stdout.flush()
    
    print(f"Organized into {len(categorized)} categories")
    print(f"  Cache: {cache_hits} hits, {cache_misses} misses")
    sys.stdout.flush()
    for key, problems in sorted(categorized.items()):
        print(f"  {key}: {len(problems)} problems")
    
    return categorized


# =============================================================================
# RAGDA EVALUATION
# =============================================================================

def run_ragda_single(
    func: Callable,
    space: List[Dict[str, Any]],
    init_params: Dict[str, Any],
    optimize_params: Dict[str, Any],
    seed: int,
    fixed_n_trials: Optional[int] = None,
) -> float:
    """
    Run RAGDA once with given parameters and return the best value found.
    
    Args:
        func: Objective function (takes dict params)
        space: RAGDA search space definition
        init_params: Parameters for RAGDAOptimizer.__init__()
        optimize_params: Parameters for RAGDAOptimizer.optimize()
        seed: Random seed for reproducibility
        fixed_n_trials: Override n_trials if provided (for budget control)
    
    Returns:
        Best objective value found
    """
    np.random.seed(seed)
    
    try:
        # Create optimizer with init params
        optimizer = RAGDAOptimizer(
            space=space,
            direction='minimize',
            random_state=seed,
            **init_params
        )
        
        # Override n_trials if fixed
        opt_params = optimize_params.copy()
        if fixed_n_trials is not None:
            opt_params['n_trials'] = fixed_n_trials
        
        # Run optimization
        result = optimizer.optimize(
            func,
            verbose=False,
            **opt_params
        )
        
        return result.best_value
    
    except Exception as e:
        # Return a large penalty value on failure
        print(f"    RAGDA failed: {e}")
        return 1e10


def evaluate_ragda_config(
    params: Dict[str, Any],
    problems: List[dict],
    n_runs: int = 3,
    fixed_n_trials: Optional[int] = None,
) -> Tuple[float, List[str]]:
    """
    Evaluate a RAGDA configuration across all problems in a category.
    
    Args:
        params: Full RAGDA configuration to evaluate
        problems: List of problem dicts
        n_runs: Number of runs per problem (for robustness)
        fixed_n_trials: Override n_trials for budget control
    
    Returns:
        (mean_loss, list_of_constraint_violations)
    """
    # Check constraints first
    penalty, violations = compute_constraint_penalty(params)
    
    if penalty > 0:
        # Return penalty without running (invalid config)
        return penalty, violations
    
    # Split params for init vs optimize
    init_params, optimize_params = split_params_by_location(params)
    
    all_results = []
    
    for problem in problems:
        problem_results = []
        
        for run_idx in range(n_runs):
            seed = hash((problem['name'], run_idx)) % (2**32)
            
            result = run_ragda_single(
                func=problem['func'],
                space=problem['space'],
                init_params=init_params,
                optimize_params=optimize_params,
                seed=seed,
                fixed_n_trials=fixed_n_trials,
            )
            
            problem_results.append(result)
        
        # Average across runs for this problem
        mean_result = np.mean(problem_results)
        all_results.append(mean_result)
    
    # Return mean across all problems
    return np.mean(all_results), []


# =============================================================================
# PARAMETER COUNT FOR DISPLAY
# =============================================================================

def count_params_for_subset(param_subset: str = 'all') -> int:
    """Count how many parameters are in the given subset."""
    if param_subset == 'simple':
        return len(SIMPLE_PARAMS)
    elif param_subset == 'core':
        return len(CORE_PARAMS)
    else:
        return len(ALL_TUNABLE_PARAMS)


def optimize_category(
    category_key: str,
    problems: List[dict],
    n_iterations: int = 100,
    n_runs_per_problem: int = 3,
    fixed_n_trials: Optional[int] = None,
    param_subset: str = 'all',
    verbose: bool = True,
) -> CategoryResult:
    """
    Find optimal RAGDA parameters for a problem category.
    
    Args:
        category_key: Category identifier (e.g., "low_cheap_smooth")
        problems: List of problems in this category
        n_iterations: Number of MARSOpt iterations
        n_runs_per_problem: Number of RAGDA runs per problem per iteration
        fixed_n_trials: Fixed n_trials budget for RAGDA (None = optimize it too)
        param_subset: 'all', 'core', or 'simple'
        verbose: Print progress
    
    Returns:
        CategoryResult with best params and metadata
    """
    parts = category_key.split("_")
    dim_class, cost_class, ruggedness_class = parts[0], parts[1], parts[2]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing category: {category_key}")
        print(f"  Dimension: {dim_class}, Cost: {cost_class}, Ruggedness: {ruggedness_class}")
        print(f"  Problems: {len(problems)}, Iterations: {n_iterations}")
        print(f"  Param subset: {param_subset}")
        if fixed_n_trials:
            print(f"  Fixed n_trials: {fixed_n_trials}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Get the space definition for the param subset
    if param_subset == 'simple':
        param_names = SIMPLE_PARAMS
    elif param_subset == 'core':
        param_names = CORE_PARAMS
    else:
        param_names = ALL_TUNABLE_PARAMS
    
    # Track best across all trials
    best_loss = float('inf')
    best_params = None
    best_violations = []
    
    # Create objective function for MARSOpt
    def objective(trial):
        nonlocal best_loss, best_params, best_violations
        
        # Suggest parameters using MARSOpt's trial interface
        params = {}
        for param_name in param_names:
            if param_name not in RAGDA_PARAMETERS:
                continue
            
            pdef = RAGDA_PARAMETERS[param_name]
            ptype = pdef.param_type
            
            if ptype == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    pdef.low,
                    pdef.high
                )
            elif ptype == 'float':
                if pdef.log_scale:
                    # MARSOpt doesn't have native log scale, sample in log space manually
                    import math
                    low_log = math.log(pdef.low)
                    high_log = math.log(pdef.high)
                    log_val = trial.suggest_float(f"{param_name}_log", low_log, high_log)
                    params[param_name] = math.exp(log_val)
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        pdef.low,
                        pdef.high
                    )
            elif ptype == 'bool':
                # MARSOpt requires strings for categorical - convert boolean
                str_val = trial.suggest_categorical(
                    param_name,
                    ["True", "False"]
                )
                params[param_name] = (str_val == "True")
            elif ptype == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    pdef.choices
                )
        
        # Evaluate this configuration
        loss, violations = evaluate_ragda_config(
            params=params,
            problems=problems,
            n_runs=n_runs_per_problem,
            fixed_n_trials=fixed_n_trials,
        )
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            best_violations = violations
        
        return loss
    
    # Create MARSOpt study and optimize
    study = MARSOptStudy(direction='minimize', random_state=42, verbose=verbose)
    study.optimize(objective, n_trials=n_iterations)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nCategory {category_key} complete!")
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Time: {elapsed:.1f}s")
        if best_violations:
            print(f"  WARNING: Best config has violations: {best_violations}")
    
    return CategoryResult(
        category=category_key,
        dim_class=dim_class,
        cost_class=cost_class,
        ruggedness_class=ruggedness_class,
        best_params=best_params,
        best_loss=best_loss,
        n_problems=len(problems),
        n_iterations=n_iterations,
        optimization_time_seconds=elapsed,
        constraint_violations=best_violations,
    )


def optimize_overall(
    categorized_problems: Dict[str, List[dict]],
    n_iterations: int = 100,
    n_runs_per_problem: int = 3,
    fixed_n_trials: Optional[int] = None,
    param_subset: str = 'all',
    verbose: bool = True,
) -> CategoryResult:
    """
    Find optimal RAGDA parameters across ALL problems.
    
    Each category is weighted equally (regardless of size) to avoid
    bias toward categories with more problems.
    
    Args:
        categorized_problems: Dict mapping category to problem list
        n_iterations: Number of MARSOpt iterations
        n_runs_per_problem: Number of RAGDA runs per problem
        fixed_n_trials: Fixed n_trials budget for RAGDA
        param_subset: 'all', 'core', or 'simple'
        verbose: Print progress
    
    Returns:
        CategoryResult with best general-purpose params
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Optimizing OVERALL (general-purpose defaults)")
        print(f"  Categories: {len(categorized_problems)}")
        print(f"  Total problems: {sum(len(p) for p in categorized_problems.values())}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Param subset: {param_subset}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Sample problems from each category for efficiency
    # (use all if few, sample if many)
    sampled_problems = {}
    max_per_category = 5
    
    for cat, probs in categorized_problems.items():
        if len(probs) <= max_per_category:
            sampled_problems[cat] = probs
        else:
            np.random.seed(42)
            indices = np.random.choice(len(probs), max_per_category, replace=False)
            sampled_problems[cat] = [probs[i] for i in indices]
    
    # Get the space definition for the param subset
    if param_subset == 'simple':
        param_names = SIMPLE_PARAMS
    elif param_subset == 'core':
        param_names = CORE_PARAMS
    else:
        param_names = ALL_TUNABLE_PARAMS
    
    # Track best
    best_loss = float('inf')
    best_params = None
    best_violations = []
    
    # Create objective function for MARSOpt
    def objective(trial):
        nonlocal best_loss, best_params, best_violations
        
        # Suggest parameters using MARSOpt's trial interface
        params = {}
        for param_name in param_names:
            if param_name not in RAGDA_PARAMETERS:
                continue
            
            pdef = RAGDA_PARAMETERS[param_name]
            ptype = pdef.param_type
            
            if ptype == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    pdef.low,
                    pdef.high
                )
            elif ptype == 'float':
                if pdef.log_scale:
                    import math
                    low_log = math.log(pdef.low)
                    high_log = math.log(pdef.high)
                    log_val = trial.suggest_float(f"{param_name}_log", low_log, high_log)
                    params[param_name] = math.exp(log_val)
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        pdef.low,
                        pdef.high
                    )
            elif ptype == 'bool':
                # MARSOpt requires strings for categorical - convert boolean
                str_val = trial.suggest_categorical(
                    param_name,
                    ["True", "False"]
                )
                params[param_name] = (str_val == "True")
            elif ptype == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    pdef.choices
                )
        
        # Check constraints first
        penalty, violations = compute_constraint_penalty(params)
        
        if penalty > 0:
            # Invalid config - return penalty without running
            return penalty
        
        # Evaluate across all categories with equal weight
        category_losses = []
        
        for cat, probs in sampled_problems.items():
            cat_loss, _ = evaluate_ragda_config(
                params=params,
                problems=probs,
                n_runs=n_runs_per_problem,
                fixed_n_trials=fixed_n_trials,
            )
            category_losses.append(cat_loss)
        
        # Mean across categories (equal weight)
        loss = np.mean(category_losses)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            best_violations = violations
        
        return loss
    
    # Create MARSOpt study and optimize
    study = MARSOptStudy(direction='minimize', random_state=42, verbose=verbose)
    study.optimize(objective, n_trials=n_iterations)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nOverall optimization complete!")
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Time: {elapsed:.1f}s")
    
    return CategoryResult(
        category="overall",
        dim_class="all",
        cost_class="all",
        ruggedness_class="all",
        best_params=best_params,
        best_loss=best_loss,
        n_problems=sum(len(p) for p in sampled_problems.values()),
        n_iterations=n_iterations,
        optimization_time_seconds=elapsed,
        constraint_violations=best_violations,
    )


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def get_marsopt_iterations_for_category(category_key: str) -> int:
    """
    Determine MARSOpt iterations based on category cost level.
    
    Cheap problems: 100 MARSOpt iterations
    Moderate problems: 40 MARSOpt iterations  
    Expensive problems: 30 MARSOpt iterations
    """
    if "expensive" in category_key:
        return 30
    elif "moderate" in category_key:
        return 40
    else:  # cheap
        return 100


def get_n_runs_for_category(category_key: str) -> int:
    """
    Determine number of RAGDA runs per problem based on category cost level.
    
    Cheap problems: 5 runs (for robustness)
    Moderate problems: 3 runs
    Expensive problems: 1 run (to save time)
    """
    if "expensive" in category_key:
        return 1
    elif "moderate" in category_key:
        return 3
    else:  # cheap
        return 5


def get_fixed_n_trials_for_category(category_key: str) -> int:
    """
    Determine fixed RAGDA n_trials based on category cost level.
    
    This is the budget for RAGDA optimization itself (not MARSOpt iterations).
    We fix this to control the evaluation cost per MARSOpt iteration.
    
    Cheap problems: 100 RAGDA trials (cheap to evaluate)
    Moderate problems: 75 RAGDA trials
    Expensive problems: 50 RAGDA trials
    """
    if "expensive" in category_key:
        return 50
    elif "moderate" in category_key:
        return 75
    else:  # cheap
        return 100


def load_checkpoint(output_path: str) -> Tuple[Dict[str, Any], set]:
    """
    Load existing checkpoint file to resume from.
    
    Returns:
        (output_dict, set_of_completed_categories)
    """
    path = Path(output_path)
    if not path.exists():
        return None, set()
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        completed = set()
        if "categories" in data:
            for cat_key, cat_data in data["categories"].items():
                # Only count as completed if it has valid results (not failed)
                if cat_data.get("status") != "failed" and cat_data.get("best_params") is not None:
                    completed.add(cat_key)
        
        return data, completed
    
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}")
        return None, set()


def save_checkpoint(
    output_path: str,
    results: Dict[str, Any],
    categorized: Dict[str, List[dict]],
    param_subset: str,
    failed_categories: Dict[str, str],
    cumulative_time: float = 0.0,
):
    """
    Save current progress to checkpoint file.
    
    Args:
        output_path: Path to save checkpoint
        results: Dict of CategoryResult objects (completed categories)
        categorized: All categories with their problems
        param_subset: Parameter subset being tuned
        failed_categories: Dict mapping failed category names to error messages
        cumulative_time: Total time spent so far in seconds
    """
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_categories": len(categorized),
            "total_problems": sum(len(p) for p in categorized.values()),
            "marsopt_iterations": {
                "cheap": 100,
                "moderate": 40,
                "expensive": 30,
            },
            "n_runs_per_problem": {
                "cheap": 5,
                "moderate": 3,
                "expensive": 1,
            },
            "fixed_ragda_trials": {
                "cheap": 100,
                "moderate": 75,
                "expensive": 50,
            },
            "param_subset": param_subset,
            "status": "in_progress",
            "completed_categories": len(results),
            "failed_categories": len(failed_categories),
            "cumulative_time_seconds": cumulative_time,
        },
        "categories": {},
        "failed": failed_categories,
    }
    
    for key, result in results.items():
        output["categories"][key] = {
            "status": "completed",
            "dim_class": result.dim_class,
            "cost_class": result.cost_class,
            "ruggedness_class": result.ruggedness_class,
            "best_params": result.best_params,
            "best_loss": result.best_loss,
            "n_problems": result.n_problems,
            "optimization_time_seconds": result.optimization_time_seconds,
            "constraint_violations": result.constraint_violations,
        }
    
    # Mark failed categories
    for cat_key, error_msg in failed_categories.items():
        if cat_key not in output["categories"]:
            output["categories"][cat_key] = {
                "status": "failed",
                "error": error_msg,
            }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def run_full_optimization(
    output_path: str = "ragda_optimal_defaults.json",
    param_subset: str = 'all',
    verbose: bool = True,
    resume: bool = True,
    skip_overall: bool = False,
) -> Dict[str, CategoryResult]:
    """
    Run the complete meta-optimization to find optimal RAGDA defaults.
    
    MARSOpt iterations and n_runs are determined per-category:
      - Cheap: 100 MARSOpt iters, 3 RAGDA runs per problem
      - Moderate: 40 MARSOpt iters, 1 RAGDA run per problem
      - Expensive: 30 MARSOpt iters, 1 RAGDA run per problem
    
    Features:
      - Checkpointing: Saves after each category completes
      - Resume: Skips already-completed categories if checkpoint exists
      - Error handling: Logs failures and continues with remaining categories
    
    Args:
        output_path: Where to save the results (also used for checkpointing)
        param_subset: 'all', 'core', or 'simple'
        verbose: Print progress
        resume: If True, resume from existing checkpoint file
        skip_overall: If True, skip the "overall" optimization phase
    
    Returns:
        Dict mapping category to CategoryResult
    """
    overall_start_time = time.time()
    
    print("="*70)
    print("RAGDA META-OPTIMIZER")
    print("Finding optimal default parameters for each problem category")
    print("="*70)
    print(f"\nParameter subset: {param_subset}")
    print(f"Parameters being tuned: {count_params_for_subset(param_subset)}")
    print(f"\nMARSOpt iterations: Cheap=100, Moderate=40, Expensive=30")
    print(f"RAGDA runs per problem: Cheap=5, Moderate=3, Expensive=1")
    print(f"Resume mode: {'ON' if resume else 'OFF'}")
    print(f"Output file: {output_path}")
    
    # Explain the loss metric
    print("\n" + "-"*70)
    print("LOSS METRIC INTERPRETATION")
    print("-"*70)
    print("The 'loss' is the mean normalized objective value across problems.")
    print("  - Range: 0.0 (perfect) to 1.0+ (poor)")
    print("  - 0.0 = Always found the global optimum")
    print("  - 0.5 = On average, found values halfway between best and worst")
    print("  - 1.0 = Performance equivalent to random sampling")
    print("  - >1.0 = Constraint violations or failures")
    print("-"*70)
    
    # Check for existing checkpoint
    completed_categories = set()
    if resume:
        print(f"\nChecking for existing checkpoint...")
        existing_data, completed_categories = load_checkpoint(output_path)
        if completed_categories:
            print(f"  Found checkpoint with {len(completed_categories)} completed categories:")
            for cat in sorted(completed_categories):
                print(f"    - {cat}")
            print(f"  Will skip these and continue from where we left off.")
        else:
            print(f"  No valid checkpoint found, starting fresh.")
    
    # Load and categorize problems
    print("\nStep 1: Loading and classifying problems...")
    try:
        categorized = load_all_problems()
    except Exception as e:
        print(f"FATAL ERROR: Could not load problems: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    results = {}
    failed_categories = {}
    total_categories = len(categorized)
    
    # If resuming, reload previous results
    if resume and existing_data and "categories" in existing_data:
        for cat_key in completed_categories:
            cat_data = existing_data["categories"][cat_key]
            parts = cat_key.split("_") if cat_key != "overall" else ["all", "all", "all"]
            if len(parts) >= 3:
                dim_class, cost_class, ruggedness_class = parts[0], parts[1], parts[2]
            else:
                dim_class = cost_class = ruggedness_class = "all"
            
            results[cat_key] = CategoryResult(
                category=cat_key,
                dim_class=dim_class,
                cost_class=cost_class,
                ruggedness_class=ruggedness_class,
                best_params=cat_data.get("best_params"),
                best_loss=cat_data.get("best_loss", float('inf')),
                n_problems=cat_data.get("n_problems", 0),
                n_iterations=0,  # Not stored in checkpoint
                optimization_time_seconds=cat_data.get("optimization_time_seconds", 0),
                constraint_violations=cat_data.get("constraint_violations"),
            )
    
    # Optimize each category
    print(f"\nStep 2: Optimizing {total_categories} categories...")
    categories_to_process = sorted(categorized.keys())
    
    # Track timing for ETA calculation
    category_times = []  # List of (category_key, elapsed_seconds)
    cumulative_time = 0.0
    
    for idx, category_key in enumerate(categories_to_process, 1):
        # Skip if already completed
        if category_key in completed_categories:
            print(f"\n[{idx}/{total_categories}] SKIPPING {category_key} (already completed)")
            continue
        
        problems = categorized[category_key]
        
        # Determine settings based on cost level
        n_iter = get_marsopt_iterations_for_category(category_key)
        n_runs = get_n_runs_for_category(category_key)
        fixed_n_trials = get_fixed_n_trials_for_category(category_key)
        
        # Calculate ETA based on previous category times
        remaining = total_categories - idx
        if category_times:
            avg_time = sum(t for _, t in category_times) / len(category_times)
            eta_seconds = avg_time * remaining
            eta_str = format_duration(eta_seconds)
        else:
            eta_str = "calculating..."
        
        print(f"\n[{idx}/{total_categories}] {category_key}")
        print(f"  Settings: {n_iter} MARSOpt iters, {n_runs} runs/problem, {fixed_n_trials} RAGDA trials")
        print(f"  Problems: {len(problems)}")
        print(f"  Cumulative time: {format_duration(cumulative_time)}, ETA for remaining: {eta_str}")
        
        try:
            start_time = time.time()
            
            result = optimize_category(
                category_key=category_key,
                problems=problems,
                n_iterations=n_iter,
                n_runs_per_problem=n_runs,
                fixed_n_trials=fixed_n_trials,
                param_subset=param_subset,
                verbose=verbose,
            )
            
            elapsed = time.time() - start_time
            results[category_key] = result
            
            # Track timing
            category_times.append((category_key, elapsed))
            cumulative_time += elapsed
            
            # Interpret loss value
            loss_quality = interpret_loss(result.best_loss)
            print(f"  ✓ COMPLETED in {format_duration(elapsed)}")
            print(f"    Loss: {result.best_loss:.6f} ({loss_quality})")
            
        except Exception as e:
            error_msg = str(e)
            failed_categories[category_key] = error_msg
            print(f"  ✗ FAILED: {error_msg}")
            import traceback
            if verbose:
                traceback.print_exc()
        
        # Save checkpoint after each category
        print(f"  Saving checkpoint...")
        try:
            save_checkpoint(output_path, results, categorized, param_subset, failed_categories, cumulative_time)
            print(f"  Checkpoint saved ({len(results)} completed, {len(failed_categories)} failed)")
        except Exception as e:
            print(f"  Warning: Could not save checkpoint: {e}")
    
    # Optimize overall (use moderate settings since it spans all categories)
    if not skip_overall:
        if "overall" in completed_categories:
            print(f"\nStep 3: SKIPPING overall optimization (already completed)")
        else:
            print(f"\nStep 3: Finding general-purpose defaults...")
            try:
                overall_result = optimize_overall(
                    categorized_problems=categorized,
                    n_iterations=50,  # Moderate budget for overall
                    n_runs_per_problem=1,  # Single run for efficiency
                    fixed_n_trials=75,  # Middle ground for RAGDA budget
                    param_subset=param_subset,
                    verbose=verbose,
                )
                results["overall"] = overall_result
                print(f"  ✓ Overall optimization completed, best_loss={overall_result.best_loss:.6f}")
            except Exception as e:
                error_msg = str(e)
                failed_categories["overall"] = error_msg
                print(f"  ✗ Overall optimization FAILED: {error_msg}")
                import traceback
                if verbose:
                    traceback.print_exc()
            
            # Save checkpoint after overall
            try:
                save_checkpoint(output_path, results, categorized, param_subset, failed_categories, cumulative_time)
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}")
    else:
        print(f"\nStep 3: SKIPPING overall optimization (--skip-overall flag)")
    
    # Final save with completed status
    print(f"\nStep 4: Saving final results to {output_path}...")
    
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_categories": len(categorized),
            "total_problems": sum(len(p) for p in categorized.values()),
            "marsopt_iterations": {
                "cheap": 100,
                "moderate": 40,
                "expensive": 30,
            },
            "n_runs_per_problem": {
                "cheap": 5,
                "moderate": 3,
                "expensive": 1,
            },
            "fixed_ragda_trials": {
                "cheap": 100,
                "moderate": 75,
                "expensive": 50,
            },
            "param_subset": param_subset,
            "status": "completed" if not failed_categories else "completed_with_errors",
            "completed_categories": len(results),
            "failed_categories": len(failed_categories),
            "total_optimization_time_seconds": time.time() - overall_start_time,
        },
        "categories": {},
    }
    
    if failed_categories:
        output["failed"] = failed_categories
    
    for key, result in results.items():
        output["categories"][key] = {
            "status": "completed",
            "dim_class": result.dim_class,
            "cost_class": result.cost_class,
            "ruggedness_class": result.ruggedness_class,
            "best_params": result.best_params,
            "best_loss": result.best_loss,
            "n_problems": result.n_problems,
            "optimization_time_seconds": result.optimization_time_seconds,
            "constraint_violations": result.constraint_violations,
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    total_elapsed = time.time() - overall_start_time
    print(f"\nDone! Results saved to {output_path}")
    
    # Print detailed summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"\nTotal optimization time: {format_duration(total_elapsed)}")
    print(f"Categories completed: {len(results)}")
    print(f"Categories failed: {len(failed_categories)}")
    
    if results:
        # Calculate aggregate statistics
        losses = [r.best_loss for r in results.values() if r.best_loss < 1e9]
        times = [r.optimization_time_seconds for r in results.values()]
        
        if losses:
            avg_loss = np.mean(losses)
            min_loss = np.min(losses)
            max_loss = np.max(losses)
            
            print(f"\n--- LOSS STATISTICS ---")
            print(f"Average loss: {avg_loss:.4f} ({interpret_loss(avg_loss)})")
            print(f"Best loss:    {min_loss:.4f} ({interpret_loss(min_loss)})")
            print(f"Worst loss:   {max_loss:.4f} ({interpret_loss(max_loss)})")
        
        if times:
            avg_time = np.mean(times)
            total_cat_time = np.sum(times)
            
            print(f"\n--- TIMING STATISTICS ---")
            print(f"Total category optimization time: {format_duration(total_cat_time)}")
            print(f"Average time per category: {format_duration(avg_time)}")
            print(f"Fastest category: {format_duration(np.min(times))}")
            print(f"Slowest category: {format_duration(np.max(times))}")
        
        # Group results by cost class for analysis
        print(f"\n--- RESULTS BY COST CLASS ---")
        for cost_class in ['cheap', 'moderate', 'expensive']:
            cat_results = [(k, r) for k, r in results.items() 
                           if r.cost_class == cost_class]
            if cat_results:
                cat_losses = [r.best_loss for _, r in cat_results if r.best_loss < 1e9]
                cat_times = [r.optimization_time_seconds for _, r in cat_results]
                if cat_losses:
                    print(f"\n{cost_class.upper()} categories ({len(cat_results)}):")
                    print(f"  Avg loss: {np.mean(cat_losses):.4f}, "
                          f"Avg time: {format_duration(np.mean(cat_times))}")
        
        # Detailed per-category results
        print(f"\n--- DETAILED CATEGORY RESULTS ---")
        print(f"{'Category':<30} {'Loss':>10} {'Quality':<25} {'Time':>10}")
        print("-" * 80)
        for key, result in sorted(results.items()):
            loss_str = f"{result.best_loss:.6f}" if result.best_loss < 1e9 else "FAILED"
            quality = interpret_loss(result.best_loss).split(' - ')[0]
            time_str = format_duration(result.optimization_time_seconds)
            violations = " *" if result.constraint_violations else ""
            print(f"{key:<30} {loss_str:>10} {quality:<25} {time_str:>10}{violations}")
        
        if any(r.constraint_violations for r in results.values()):
            print("\n* = has constraint violations")
    
    if failed_categories:
        print(f"\n--- FAILED CATEGORIES ---")
        for key, error in sorted(failed_categories.items()):
            print(f"  {key}: {error[:60]}...")
    
    print("\n" + "="*70)
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Meta-optimize RAGDA default parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh optimization
  python meta_optimizer.py --output results.json
  
  # Resume from previous run (default behavior)
  python meta_optimizer.py --output results.json
  
  # Start fresh, ignore checkpoint
  python meta_optimizer.py --output results.json --no-resume
  
  # Skip the overall optimization phase
  python meta_optimizer.py --output results.json --skip-overall
  
  # Only tune core parameters (faster)
  python meta_optimizer.py --param-subset core

MARSOpt iteration budgets:
  Cheap categories:    100 iterations, 5 runs/problem
  Moderate categories:  40 iterations, 3 runs/problem
  Expensive categories: 30 iterations, 1 run/problem
        """
    )
    parser.add_argument("--output", "-o", default="ragda_optimal_defaults.json",
                        help="Output file path (also used for checkpointing)")
    parser.add_argument("--param-subset", choices=['all', 'core', 'simple'], default='all',
                        help="Which parameter subset to tune")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore any existing checkpoint")
    parser.add_argument("--skip-overall", action="store_true",
                        help="Skip the 'overall' optimization phase")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("RAGDA META-OPTIMIZER")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Output file: {args.output}")
    print(f"  Parameter subset: {args.param_subset}")
    print(f"  Resume mode: {'OFF (--no-resume)' if args.no_resume else 'ON'}")
    print(f"  Skip overall: {'YES' if args.skip_overall else 'NO'}")
    print("\nMARSOpt iteration budgets:")
    print("  Cheap categories:    100 iterations, 5 runs/problem")
    print("  Moderate categories:  40 iterations, 3 runs/problem")
    print("  Expensive categories: 30 iterations, 1 run/problem")
    sys.stdout.flush()
    
    try:
        run_full_optimization(
            output_path=args.output,
            param_subset=args.param_subset,
            verbose=not args.quiet,
            resume=not args.no_resume,
            skip_overall=args.skip_overall,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        print("Progress has been saved to cache and checkpoint files.")
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*70)
        print("FATAL ERROR - META-OPTIMIZER CRASHED")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\nProgress has been saved to cache and checkpoint files.")
        print("You can resume from where you left off by running the script again.")
        sys.stdout.flush()
        sys.exit(1)
