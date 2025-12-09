"""
Example: Basic RAGDA optimization with the v2.0 API.

Shows how to use dict-based space definition and kwargs unpacking.
"""

from ragda import RAGDAOptimizer
import numpy as np


def main():
    print("=" * 70)
    print("RAGDA Example: Basic Optimization")
    print("=" * 70)
    
    # Define search space (dict-based)
    space = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]},
    }
    
    # Define objective function (kwargs unpacking)
    def sphere(x, y):
        """Simple 2D sphere function - global minimum at (0, 0) with value 0."""
        return x**2 + y**2
    
    print("\nOptimizing sphere function: f(x, y) = x² + y²")
    print(f"Search space: x ∈ [-5, 5], y ∈ [-5, 5]")
    print(f"Global minimum: f(0, 0) = 0")
    
    # Create optimizer
    optimizer = RAGDAOptimizer(
        space=space,
        direction='minimize',
        n_workers=2,
        random_state=42
    )
    
    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize(
        sphere,
        n_trials=50,
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value:.6f}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Number of workers: {result.n_workers}")
    
    # Check convergence
    if result.best_value < 0.01:
        print("\n✓ Successfully converged to global minimum!")
    else:
        print("\n⚠ Not fully converged, try increasing n_trials")


if __name__ == "__main__":
    main()
