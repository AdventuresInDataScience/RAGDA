"""
Mini Meta-Optimizer Test

Runs the actual MARSOpt meta-optimizer on a small set of easy problems
to verify the optimization loop works correctly.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '..')

from marsopt import Study as MARSOptStudy
from ragda import RAGDAOptimizer
from ragda_parameter_space import (
    RAGDA_PARAMETERS, 
    split_params_by_location,
    build_marsopt_space,
    sample_from_marsopt,
    check_constraints,
    compute_penalty,
)


# =============================================================================
# Simple Test Problems
# =============================================================================

def sphere_5d(params):
    """Simple 5D sphere - optimal at origin."""
    return sum(v**2 for v in params.values())

def rosenbrock_2d(params):
    """2D Rosenbrock - optimal at (1,1)."""
    x, y = params['x0'], params['x1']
    return (1 - x)**2 + 100 * (y - x**2)**2

def rastrigin_3d(params):
    """3D Rastrigin - multimodal, optimal at origin."""
    A = 10
    vals = [params[f'x{i}'] for i in range(3)]
    return A * 3 + sum(v**2 - A * np.cos(2 * np.pi * v) for v in vals)


TEST_PROBLEMS = [
    {
        'name': 'Sphere-5D',
        'func': sphere_5d,
        'space': [{'name': f'x{i}', 'type': 'continuous', 'bounds': [-5.0, 5.0]} for i in range(5)],
        'optimal': 0.0,
        'good_threshold': 0.1,  # Consider good if < this
    },
    {
        'name': 'Rosenbrock-2D',
        'func': rosenbrock_2d,
        'space': [
            {'name': 'x0', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
            {'name': 'x1', 'type': 'continuous', 'bounds': [-5.0, 5.0]},
        ],
        'optimal': 0.0,
        'good_threshold': 0.1,
    },
    {
        'name': 'Rastrigin-3D',
        'func': rastrigin_3d,
        'space': [{'name': f'x{i}', 'type': 'continuous', 'bounds': [-5.12, 5.12]} for i in range(3)],
        'optimal': 0.0,
        'good_threshold': 1.0,  # Rastrigin is harder
    },
]


# =============================================================================
# Mini Meta-Optimizer
# =============================================================================

def run_ragda_on_problem(problem, ragda_params, n_trials=50):
    """Run RAGDA on a single problem with given parameters."""
    init_params, optimize_params = split_params_by_location(ragda_params)
    
    # Override n_trials for speed
    optimize_params['n_trials'] = n_trials
    
    optimizer = RAGDAOptimizer(
        problem['space'],
        direction='minimize',
        **init_params
    )
    
    result = optimizer.optimize(
        problem['func'],
        verbose=False,
        **optimize_params
    )
    
    return result.best_value


def evaluate_config(ragda_params, problems, n_trials_per_problem=50):
    """
    Evaluate a RAGDA configuration across all test problems.
    Returns average normalized performance (lower is better).
    """
    scores = []
    
    for problem in problems:
        try:
            result = run_ragda_on_problem(problem, ragda_params, n_trials=n_trials_per_problem)
            # Normalize by good threshold
            normalized = result / problem['good_threshold']
            scores.append(normalized)
        except Exception as e:
            # Penalize failures heavily
            scores.append(100.0)
    
    return np.mean(scores)


def run_mini_meta_optimizer(
    n_marsopt_trials: int = 20,
    n_ragda_trials: int = 50,
    verbose: bool = True
):
    """
    Run a mini version of the meta-optimizer.
    
    Args:
        n_marsopt_trials: Number of different RAGDA configs to try
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
        print("=" * 70)
    
    # Build MARSOpt search space
    marsopt_space = build_marsopt_space()
    
    # Track best
    best_loss = float('inf')
    best_params = None
    best_trial = 0
    
    # History for analysis
    history = []
    
    start_time = time.time()
    
    def objective(trial):
        nonlocal best_loss, best_params, best_trial
        
        # Sample RAGDA parameters
        ragda_params = sample_from_marsopt(trial, marsopt_space)
        
        # Check constraints and compute penalty
        violations = check_constraints(ragda_params)
        penalty = compute_penalty(violations)
        
        if penalty > 0:
            loss = 1000 + penalty  # Large penalty for constraint violations
            if verbose:
                print(f"  Trial {trial.number}: CONSTRAINT VIOLATION (penalty={penalty:.1f})")
        else:
            # Actually evaluate the config
            try:
                loss = evaluate_config(ragda_params, TEST_PROBLEMS, n_ragda_trials)
                if verbose:
                    status = "✓" if loss < best_loss else " "
                    print(f"  Trial {trial.number}: loss={loss:.4f} {status}")
            except Exception as e:
                loss = 10000  # Failure penalty
                if verbose:
                    print(f"  Trial {trial.number}: FAILED - {e}")
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = ragda_params.copy()
            best_trial = trial.number
        
        history.append({
            'trial': trial.number,
            'loss': loss,
            'penalty': penalty,
        })
        
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
        print(f"Best loss: {best_loss:.4f} (trial {best_trial})")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Avg time per trial: {elapsed/n_marsopt_trials:.1f}s")
        
        # Show convergence
        valid_losses = [h['loss'] for h in history if h['loss'] < 1000]
        if valid_losses:
            print(f"\nValid trials: {len(valid_losses)}/{n_marsopt_trials}")
            print(f"Loss range: {min(valid_losses):.4f} - {max(valid_losses):.4f}")
        
        # Show best params (key ones only)
        if best_params:
            print("\nBest parameters (key ones):")
            key_params = ['n_workers', 'shrink_factor', 'sigma_init', 'worker_strategy', 
                         'lambda_start', 'lambda_end', 'use_minibatch']
            for k in key_params:
                if k in best_params:
                    print(f"  {k}: {best_params[k]}")
    
    return {
        'best_loss': best_loss,
        'best_params': best_params,
        'best_trial': best_trial,
        'history': history,
        'elapsed': elapsed,
    }


def test_best_vs_default():
    """Compare best found params against defaults."""
    print("\n" + "=" * 70)
    print("COMPARING BEST PARAMS VS DEFAULTS")
    print("=" * 70)
    
    # Get defaults
    default_params = {}
    for name, spec in RAGDA_PARAMETERS.items():
        if spec.default is not None:
            default_params[name] = spec.default
        elif spec.param_type == 'float':
            default_params[name] = (spec.low + spec.high) / 2
        elif spec.param_type == 'int':
            default_params[name] = (spec.low + spec.high) // 2
        elif spec.param_type == 'bool':
            default_params[name] = False
        elif spec.param_type == 'categorical':
            default_params[name] = spec.choices[0] if spec.choices else None
    
    # Run with defaults
    print("\nEvaluating DEFAULT parameters...")
    default_loss = evaluate_config(default_params, TEST_PROBLEMS, n_trials_per_problem=50)
    print(f"Default loss: {default_loss:.4f}")
    
    # Run mini optimizer
    result = run_mini_meta_optimizer(n_marsopt_trials=20, n_ragda_trials=50, verbose=True)
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Default params loss: {default_loss:.4f}")
    print(f"Optimized params loss: {result['best_loss']:.4f}")
    improvement = (default_loss - result['best_loss']) / default_loss * 100
    print(f"Improvement: {improvement:.1f}%")
    
    return result


if __name__ == '__main__':
    # Quick test first
    print("Testing that RAGDA runs with full params...")
    default_params = {}
    for name, spec in RAGDA_PARAMETERS.items():
        if spec.default is not None:
            default_params[name] = spec.default
        elif spec.param_type == 'float':
            default_params[name] = (spec.low + spec.high) / 2
        elif spec.param_type == 'int':
            default_params[name] = (spec.low + spec.high) // 2
        elif spec.param_type == 'bool':
            default_params[name] = False
        elif spec.param_type == 'categorical':
            default_params[name] = spec.choices[0] if spec.choices else None
    
    test_result = run_ragda_on_problem(TEST_PROBLEMS[0], default_params, n_trials=30)
    print(f"Quick test passed: Sphere-5D result = {test_result:.4f}\n")
    
    # Run the actual mini meta-optimizer test
    test_best_vs_default()
