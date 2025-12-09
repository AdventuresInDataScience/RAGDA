"""
Quick diagnostic to test AUC calculation and MARSOpt.
"""
import sys
import time
import numpy as np

print("=== Test 1: AUC Calculation ===")
sys.stdout.flush()

# Import the AUC function
sys.path.insert(0, '.')
from meta_optimizer import compute_normalized_auc

# Test cases
print("\nTest: Perfect convergence (best found immediately)")
values = [0.1, 0.1, 0.1, 0.1, 0.1]  # Already at best from start
auc = compute_normalized_auc(values, 'minimize')
print(f"  Values: {values} -> AUC: {auc:.4f} (expected: 0.0)")

print("\nTest: Linear improvement")
values = [1.0, 0.75, 0.5, 0.25, 0.0]  # Steady improvement
auc = compute_normalized_auc(values, 'minimize')
print(f"  Values: {values} -> AUC: {auc:.4f} (expected: ~0.5)")

print("\nTest: Worst case (improvement only at end)")
values = [1.0, 1.0, 1.0, 1.0, 0.0]  # Best only at very end
auc = compute_normalized_auc(values, 'minimize')
print(f"  Values: {values} -> AUC: {auc:.4f} (expected: ~0.8)")

print("\nTest: Early improvement then plateau")
values = [1.0, 0.2, 0.1, 0.1, 0.1]  # Early improvement
auc = compute_normalized_auc(values, 'minimize')
print(f"  Values: {values} -> AUC: {auc:.4f} (expected: low, ~0.1)")

print("\nTest: Realistic noisy optimization")
values = [10.5, 8.2, 9.1, 5.3, 4.8, 5.1, 3.2, 2.9, 2.5, 2.1]
auc = compute_normalized_auc(values, 'minimize')
print(f"  Values: {[f'{v:.1f}' for v in values]}")
print(f"  AUC: {auc:.4f}")

print("\n=== Test 2: Quick RAGDA + AUC test ===")
sys.stdout.flush()

# Quick test with a simple problem
from ragda import RAGDAOptimizer

space = {
    'x0': {'type': 'continuous', 'bounds': [-5, 5]},
    'x1': {'type': 'continuous', 'bounds': [-5, 5]},
}

def sphere(x0, x1):
    return x0**2 + x1**2

optimizer = RAGDAOptimizer(space=space, direction='minimize', random_state=42)
result = optimizer.optimize(sphere, n_trials=30, verbose=False)

trials_df = result.trials_df.sort_values('trial_id')
values = trials_df['value'].tolist()
auc = compute_normalized_auc(values, 'minimize')

print(f"  Sphere function: {len(values)} trials")
print(f"  Best value: {result.best_value:.6f}")
print(f"  AUC: {auc:.4f}")

print("\n=== Test 3: MARSOpt with 34 params (5 trials) ===")
sys.stdout.flush()

from marsopt import Study as MARSOptStudy

call_count = 0

def simple_objective(trial):
    global call_count
    call_count += 1
    print(f"  Objective call #{call_count}")
    sys.stdout.flush()
    
    # Simulate 34 parameters like meta_optimizer
    params = {}
    for i in range(34):
        params[f'param_{i}'] = trial.suggest_float(f'param_{i}', 0.0, 1.0)
    
    # Simple loss in valid AUC range
    loss = sum(params.values()) / 34  # Normalized to ~[0, 1]
    print(f"    -> loss={loss:.4f}")
    sys.stdout.flush()
    return loss

study = MARSOptStudy(direction='minimize', random_state=42, verbose=True)
start = time.time()
study.optimize(simple_objective, n_trials=5)
elapsed = time.time() - start
print(f"\nCompleted {call_count} calls in {elapsed:.1f}s")

print("\nAll tests complete!")
