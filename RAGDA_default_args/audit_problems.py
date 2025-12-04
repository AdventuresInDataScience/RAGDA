"""
Comprehensive Benchmark Problem Audit

Uses RUGGEDNESS instead of stochastic noise.
All functions are DETERMINISTIC - ruggedness measures landscape complexity.

Classification axes (3 levels each = 27 categories):
1. DIMENSIONALITY: low (≤10), medium (11-50), high (51+)
2. COST: cheap (<10ms), moderate (10-100ms), expensive (≥100ms)
3. RUGGEDNESS: smooth, moderate, rugged

Target: Good coverage across all 27 (3×3×3) categories.
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from problem_classifier import (
    classify_problem, ClassifiedProblem, 
    RuggednessLevel, CostLevel, DimClass,
    summarize_classified_problems
)


def create_wrapper(func, dim):
    """Create array-input wrapper for test functions."""
    def wrapper(x):
        return func(x)
    return wrapper


def load_synthetic_functions():
    """Load synthetic test functions from benchmark_functions.py"""
    from benchmark_functions import get_all_functions
    
    all_funcs = get_all_functions()
    problems = []
    
    for key, tf in all_funcs.items():
        problems.append({
            'name': tf.name,
            'func': tf.func,
            'bounds': np.array(tf.bounds),
            'dim': tf.dim,
            'category': f'synthetic-{tf.category}',
        })
    
    return problems


def load_ml_problems():
    """Load ML hyperparameter tuning problems."""
    from benchmark_ml_problems import get_all_ml_problems
    
    all_probs = get_all_ml_problems()
    problems = []
    
    for key, mp in all_probs.items():
        # Convert space to bounds
        bounds = []
        for s in mp.space:
            if s['type'] == 'continuous':
                bounds.append([s['bounds'][0], s['bounds'][1]])
            else:
                bounds.append([0, 1])  # Normalize categorical
        
        # Create array-input wrapper
        def make_wrapper(orig_func, space):
            def wrapper(x):
                params = {}
                for i, s in enumerate(space):
                    if s['type'] == 'continuous':
                        params[s['name']] = float(x[i])
                    else:
                        idx = int(x[i] * (len(s.get('values', [0, 1])) - 1))
                        params[s['name']] = s.get('values', [0, 1])[idx]
                return orig_func(params)
            return wrapper
        
        problems.append({
            'name': mp.name,
            'func': make_wrapper(mp.func, mp.space),
            'bounds': np.array(bounds),
            'dim': mp.dim,
            'category': f'ml-{mp.category}',
        })
    
    return problems


def load_genuine_problems():
    """Load real-world benchmark problems (chaotic systems, PDEs, ML training, etc.)."""
    from benchmark_realworld_problems import get_all_genuine_problems
    
    all_probs = get_all_genuine_problems()
    problems = []
    
    for gp in all_probs:
        problems.append({
            'name': gp.name,
            'func': gp.func,
            'bounds': np.array(gp.bounds),
            'dim': gp.dim,
            'category': gp.category,
        })
    
    return problems


def classify_all_problems(problems, verbose=True) -> list:
    """Classify all problems."""
    classified = []
    
    for i, p in enumerate(problems):
        if verbose:
            print(f"[{i+1}/{len(problems)}] {p['name'][:30]:<30}...", end=' ', flush=True)
        
        cp = classify_problem(
            name=p['name'],
            func=p['func'],
            bounds=p['bounds'],
            dim=p['dim'],
            category=p['category'],
            n_timing_samples=5,
            n_ruggedness_samples=15,
        )
        
        c = cp.characteristics
        if verbose:
            print(f"{c.dim_class.value:>10} / {c.cost_level.value:>14} / {c.ruggedness_level.value:<15} "
                  f"({c.mean_eval_time_ms:>7.1f}ms, CV={c.local_cv_score:.4f})")
        
        classified.append(cp)
    
    return classified


def generate_coverage_report(problems: list):
    """Generate coverage matrix."""
    
    dim_classes = [d.value for d in DimClass]
    cost_levels = [c.value for c in CostLevel]
    ruggedness_levels = [r.value for r in RuggednessLevel]
    
    coverage = {}
    for d in dim_classes:
        for c in cost_levels:
            for r in ruggedness_levels:
                coverage[(d, c, r)] = []
    
    for p in problems:
        key = p.get_class_key()
        if key in coverage:
            coverage[key].append(p)
    
    return coverage, dim_classes, cost_levels, ruggedness_levels


def print_coverage_table(coverage, dim_classes, cost_levels, ruggedness_levels):
    """Print formatted coverage table."""
    
    print("\n" + "="*100)
    print("COVERAGE MATRIX: Problems per (Dimension × Cost × Ruggedness) Category")
    print("="*100)
    print(f"Target: 3-5 problems per category for good coverage")
    print()
    
    header = f"{'Dim':<12} {'Cost':<14} {'Ruggedness':<15} {'Count':>6}  Status   Problems"
    print(header)
    print("-" * 100)
    
    total = 0
    empty = 0
    low = 0
    ok = 0
    
    for d in dim_classes:
        for c in cost_levels:
            for r in ruggedness_levels:
                key = (d, c, r)
                probs = coverage.get(key, [])
                count = len(probs)
                total += count
                
                if count == 0:
                    status = "❌ EMPTY"
                    empty += 1
                elif count < 3:
                    status = "⚠️ LOW"
                    low += 1
                else:
                    status = "✓ OK"
                    ok += 1
                
                if probs:
                    names = ", ".join([p.name[:15] for p in probs[:3]])
                    if len(probs) > 3:
                        names += f", +{len(probs)-3}"
                else:
                    names = "-"
                
                print(f"{d:<12} {c:<14} {r:<15} {count:>6}  {status:<8} {names[:50]}")
    
    print("-" * 100)
    print(f"TOTAL PROBLEMS: {total}")
    print(f"CATEGORIES: {empty} empty, {low} low (<3), {ok} OK (≥3)")
    
    return coverage


def main():
    print("="*100)
    print("COMPREHENSIVE BENCHMARK PROBLEM AUDIT v2")
    print("Classification: Dimension × Cost × RUGGEDNESS (not stochastic noise)")
    print("="*100)
    print()
    
    # Load all problem sources
    print("Loading problems...")
    
    synthetic = load_synthetic_functions()
    print(f"  Synthetic functions: {len(synthetic)}")
    
    ml_probs = load_ml_problems()
    print(f"  ML problems: {len(ml_probs)}")
    
    genuine = load_genuine_problems()
    print(f"  Genuine problems: {len(genuine)}")
    
    all_problems = synthetic + ml_probs + genuine
    print(f"\nTOTAL: {len(all_problems)} problems\n")
    
    print("-"*100)
    print("Classifying all problems...")
    print("-"*100)
    
    classified = classify_all_problems(all_problems, verbose=True)
    
    # Generate coverage report
    coverage, dims, costs, ruggednesses = generate_coverage_report(classified)
    print_coverage_table(coverage, dims, costs, ruggednesses)
    
    # Summary by category
    summary = summarize_classified_problems(classified)
    
    print("\n" + "="*100)
    print("SUMMARY BY AXIS")
    print("="*100)
    
    print("\nBy Dimensionality:")
    for k in ['low', 'medium', 'high', 'very_high']:
        v = summary['by_dim'].get(k, 0)
        print(f"  {k:>12}: {v:>4} problems")
    
    print("\nBy Cost:")
    for k in ['cheap', 'moderate', 'expensive', 'very_expensive']:
        v = summary['by_cost'].get(k, 0)
        print(f"  {k:>14}: {v:>4} problems")
    
    print("\nBy Ruggedness:")
    for k in ['smooth', 'moderate', 'rugged', 'highly_rugged']:
        v = summary['by_ruggedness'].get(k, 0)
        print(f"  {k:>14}: {v:>4} problems")
    
    # Identify gaps
    print("\n" + "="*100)
    print("GAPS TO FILL")
    print("="*100)
    
    gaps = []
    for key, probs in coverage.items():
        if len(probs) < 3:
            gaps.append((key, len(probs)))
    
    gaps.sort(key=lambda x: x[1])
    
    print(f"\nCategories needing more problems ({len(gaps)} total):\n")
    
    # Group by major category
    by_dim = {}
    for (d, c, r), count in gaps:
        if d not in by_dim:
            by_dim[d] = []
        by_dim[d].append(((d, c, r), count))
    
    for dim in ['low', 'medium', 'high', 'very_high']:
        if dim in by_dim:
            print(f"\n  {dim.upper()} DIMENSION:")
            for (d, c, r), count in by_dim[dim]:
                needed = 3 - count
                print(f"    {c:>14} + {r:<15}: {count} (need {needed} more)")


if __name__ == '__main__':
    main()
