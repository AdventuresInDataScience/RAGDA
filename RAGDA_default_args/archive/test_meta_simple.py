"""
Mini Meta-Optimizer Test

Runs the actual MARSOpt meta-optimizer on a small set of easy problems
to verify the optimization loop works correctly.

This follows the exact same patterns as meta_optimizer.py
"""

import numpy as np
import time
import math
import sys
from pathlib import Path

# MARSOpt import
from marsopt import Study as MARSOptStudy

# RAGDA import
sys.path.insert(0, str(Path(__file__).parent.parent))
from ragda import RAGDAOptimizer

# Import from the parameter space - same imports as meta_optimizer.py
from ragda_parameter_space import (
    RAGDA_PARAMETERS,
    compute_constraint_penalty,
    split_params_by_location,
    get_default_params,
    SIMPLE_PARAMS,
)


# =============================================================================
# Simple Test Problems
# =============================================================================

def sphere_5d(**params):
    """Simple 5D sphere - optimal at origin."""
    return sum(v**2 for v in params.values())

def rosenbrock_2d(x0, x1):
    """2D Rosenbrock - optimal at (1,1)."""
    return (1 - x0)**2 + 100 * (x1 - x0**2)**2

def rastrigin_3d(**params):
    """3D Rastrigin - multimodal, optimal at origin."""
    A = 10
    vals = [params[f'x{i}'] for i in range(3)]
    return A * 3 + sum(v**2 - A * np.cos(2 * np.pi * v) for v in vals)


TEST_PROBLEMS = [
    {
        'name': 'Sphere-5D',
        'func': sphere_5d,
        'space': {f'x{i}': {'type': 'continuous', 'bounds': [-5.0, 5.0]} for i in range(5)},
        'optimal': 0.0,
        'good_threshold': 0.1,
    },
    {
        'name': 'Rosenbrock-2D',
        'func': rosenbrock_2d,
        'space': {
            'x0': {'type': 'continuous', 'bounds': [-5.0, 5.0]},
            'x1': {'type': 'continuous', 'bounds': [-5.0, 5.0]},
        },
        'optimal': 0.0,
        'good_threshold': 0.1,
    },
    {
        'name': 'Rastrigin-3D',
        'func': rastrigin_3d,
        'space': {f'x{i}': {'type': 'continuous', 'bounds': [-5.12, 5.12]} for i in range(3)},
        'optimal': 0.0,
        'good_threshold': 1.0,
    },
]


# =============================================================================
# RAGDA Evaluation (copied from meta_optimizer.py patterns)
# =============================================================================

def run_ragda_single(func, space, init_params, optimize_params, seed, fixed_n_trials=None):
    """Run RAGDA once with given parameters and return the best value found."""
    # Ensure seed fits in 32 bits
    seed = seed % (2**31)
    np.random.seed(seed)
    
    try:
        optimizer = RAGDAOptimizer(
            space=space,
            direction='minimize',
            random_state=seed,
            **init_params
        )
        
        opt_params = optimize_params.copy()
        if fixed_n_trials is not None:
            opt_params['n_trials'] = fixed_n_trials
        
        result = optimizer.optimize(func, verbose=False, **opt_params)
        return result.best_value
        
    except Exception as e:
        print(f"    RAGDA failed: {e}")
        return 1e10


def evaluate_ragda_config(params, problems, fixed_n_trials=50):
    """Evaluate a RAGDA config across test problems."""
    # Check constraints first
    penalty, violations = compute_constraint_penalty(params)
    
    if penalty > 0:
        return 1000 + penalty, violations
    
    # Split params
    init_params, optimize_params = split_params_by_location(params)
    
    results = []
    for problem in problems:
        seed = hash(problem['name']) % (2**32)
        result = run_ragda_single(
            func=problem['func'],
            space=problem['space'],
            init_params=init_params,
            optimize_params=optimize_params,
            seed=seed,
            fixed_n_trials=fixed_n_trials,
        )
        # Normalize by threshold
        normalized = result / problem['good_threshold']
        results.append(normalized)
    
    return np.mean(results), []


# =============================================================================
# Mini Meta-Optimizer (follows meta_optimizer.py patterns exactly)
# =============================================================================

def run_mini_meta_optimizer(
    n_marsopt_trials: int = 20,
    n_ragda_trials: int = 50,
    verbose: bool = True
):
    """
    Run a mini version of the meta-optimizer.
    
    Args:
        n_marsopt_trials: Number of different RAGDA configs to try (MARSOpt iterations)
        n_ragda_trials: Budget for each RAGDA run
        verbose: Print progress
    """
    if verbose:
        print("=" * 70)
        print("MINI META-OPTIMIZER TEST")
        print("=" * 70)
        print(f"MARSOpt trials: {n_marsopt_trials}")
        print(f"RAGDA trials per problem: {n_ragda_trials}")
        print(f"Test problems: {[p['name'] for p in TEST_PROBLEMS]}")
        print(f"Parameters being tuned: {len(SIMPLE_PARAMS)} (SIMPLE_PARAMS subset)")
        print("=" * 70)
    
    # Track best
    best_loss = float('inf')
    best_params = None
    trial_count = 0
    history = []
    
    start_time = time.time()
    
    # Objective function for MARSOpt (same pattern as meta_optimizer.py)
    def objective(trial):
        nonlocal best_loss, best_params, trial_count
        trial_count += 1
        
        # Suggest parameters using MARSOpt's trial interface
        # (exactly as in meta_optimizer.py optimize_category function)
        params = {}
        for param_name in SIMPLE_PARAMS:
            if param_name not in RAGDA_PARAMETERS:
                continue
            
            pdef = RAGDA_PARAMETERS[param_name]
            ptype = pdef.param_type
            
            if ptype == 'int':
                params[param_name] = trial.suggest_int(param_name, pdef.low, pdef.high)
            elif ptype == 'float':
                if pdef.log_scale:
                    low_log = math.log(pdef.low)
                    high_log = math.log(pdef.high)
                    log_val = trial.suggest_float(f"{param_name}_log", low_log, high_log)
                    params[param_name] = math.exp(log_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, pdef.low, pdef.high)
            elif ptype == 'bool':
                str_val = trial.suggest_categorical(param_name, ["True", "False"])
                params[param_name] = (str_val == "True")
            elif ptype == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, pdef.choices)
        
        # Evaluate this configuration
        loss, violations = evaluate_ragda_config(
            params=params,
            problems=TEST_PROBLEMS,
            fixed_n_trials=n_ragda_trials,
        )
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            if verbose:
                print(f"  Trial {trial_count}: loss={loss:.4f} ✓ NEW BEST")
        else:
            if verbose:
                print(f"  Trial {trial_count}: loss={loss:.4f}")
        
        history.append({'trial': trial_count, 'loss': loss})
        return loss
    
    # Create and run MARSOpt
    if verbose:
        print("\nRunning MARSOpt optimization...")
        print("-" * 70)
    
    study = MARSOptStudy(direction='minimize', random_state=42, verbose=False)
    study.optimize(objective, n_trials=n_marsopt_trials)
    
    elapsed = time.time() - start_time
    
    # Report results
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Best loss: {best_loss:.4f}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Avg time per trial: {elapsed/n_marsopt_trials:.1f}s")
        
        valid_losses = [h['loss'] for h in history if h['loss'] < 1000]
        if valid_losses:
            print(f"\nValid trials: {len(valid_losses)}/{n_marsopt_trials}")
            print(f"Loss range: {min(valid_losses):.4f} - {max(valid_losses):.4f}")
        
        if best_params:
            print("\nBest parameters found:")
            for k, v in sorted(best_params.items()):
                print(f"  {k}: {v}")
    
    return {
        'best_loss': best_loss,
        'best_params': best_params,
        'history': history,
        'elapsed': elapsed,
    }


def compare_with_defaults():
    """Compare optimized params against defaults."""
    print("\n" + "=" * 70)
    print("STEP 1: Evaluate DEFAULT parameters")
    print("=" * 70)
    
    # Get defaults
    default_params = get_default_params(SIMPLE_PARAMS)
    print(f"Default params: {default_params}")
    
    default_loss, _ = evaluate_ragda_config(
        default_params, TEST_PROBLEMS, fixed_n_trials=50
    )
    print(f"Default loss: {default_loss:.4f}")
    
    # Run mini optimizer
    print("\n" + "=" * 70)
    print("STEP 2: Run MARSOpt to find better parameters")
    print("=" * 70)
    
    result = run_mini_meta_optimizer(
        n_marsopt_trials=20,
        n_ragda_trials=50,
        verbose=True
    )
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Default params loss:   {default_loss:.4f}")
    print(f"Optimized params loss: {result['best_loss']:.4f}")
    
    if result['best_loss'] < default_loss:
        improvement = (default_loss - result['best_loss']) / default_loss * 100
        print(f"Improvement: {improvement:.1f}% better than defaults")
    else:
        print("No improvement found (defaults are already good for these problems)")
    
    return result


if __name__ == '__main__':
    compare_with_defaults()
