"""Quick test for dynamic worker strategy."""

from ragda import RAGDAOptimizer
import numpy as np

# Test 1: Basic greedy strategy (default)
print('Test 1: Greedy strategy (default)')
space = {'x': {'type': 'continuous', 'bounds': [-5, 5]}}

def objective(x):
    return x**2

optimizer = RAGDAOptimizer(space, n_workers=2, random_state=42)
result = optimizer.optimize(objective, n_trials=50, verbose=False)
print(f'  Best value: {result.best_value:.6f}')
print(f'  Strategy: {result.optimization_params.get("worker_strategy")}')

# Test 2: Dynamic strategy
print('\nTest 2: Dynamic strategy')
optimizer = RAGDAOptimizer(space, n_workers=4, random_state=42)
result = optimizer.optimize(
    objective,
    n_trials=100,
    worker_strategy='dynamic',
    elite_fraction=0.5,
    sync_frequency=25,
    verbose=False
)
print(f'  Best value: {result.best_value:.6f}')
print(f'  Strategy: {result.optimization_params.get("worker_strategy")}')
print(f'  Elite fraction: {result.optimization_params.get("elite_fraction")}')

# Test 3: Wave execution (more workers than max_parallel)
print('\nTest 3: Wave execution (6 workers, 2 parallel)')
optimizer = RAGDAOptimizer(space, n_workers=6, max_parallel_workers=2, random_state=42)
print(f'  n_workers: {optimizer.n_workers}')
print(f'  max_parallel_workers: {optimizer.max_parallel_workers}')
result = optimizer.optimize(objective, n_trials=50, verbose=False)
print(f'  Best value: {result.best_value:.6f}')

# Test 4: Dynamic with worker decay
print('\nTest 4: Dynamic with worker decay')
optimizer = RAGDAOptimizer(space, n_workers=4, random_state=42)
result = optimizer.optimize(
    objective,
    n_trials=100,
    worker_strategy='dynamic',
    enable_worker_decay=True,
    min_workers=2,
    sync_frequency=25,
    verbose=False
)
print(f'  Best value: {result.best_value:.6f}')
print(f'  Worker decay enabled: {result.optimization_params.get("enable_worker_decay")}')

# Test 5: Verbose output with dynamic strategy
print('\nTest 5: Verbose output with dynamic strategy')
optimizer = RAGDAOptimizer(space, n_workers=4, random_state=42)
result = optimizer.optimize(
    objective,
    n_trials=100,
    worker_strategy='dynamic',
    elite_fraction=0.5,
    restart_mode='adaptive',
    sync_frequency=25,
    verbose=True
)

print('\nAll tests passed!')
