"""
Example: Using Optuna-style API with RAGDA.

Shows trial.suggest_* methods for parameter sampling.
"""

from ragda import create_study
import numpy as np


def main():
    print("=" * 70)
    print("RAGDA Example: Optuna-Style API")
    print("=" * 70)
    
    def objective(trial):
        """
        Objective function using Optuna-style Trial object.
        
        Trial methods:
        - trial.suggest_float(name, low, high, log=False)
        - trial.suggest_int(name, low, high, log=False, step=1)
        - trial.suggest_categorical(name, choices)
        """
        # Sample parameters
        x = trial.suggest_float('x', -5, 5)
        y = trial.suggest_float('y', -5, 5)
        
        # Rosenbrock function
        return 100.0 * (y - x**2)**2 + (1 - x)**2
    
    print("\nOptimizing Rosenbrock function")
    print("Global minimum: f(1, 1) = 0")
    
    # Create study
    study = create_study(direction='minimize')
    
    # Run optimization
    print("\nRunning optimization...")
    study.optimize(objective, n_trials=100)
    
    # Display results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Check convergence
    expected_x, expected_y = 1.0, 1.0
    x_error = abs(study.best_params['x'] - expected_x)
    y_error = abs(study.best_params['y'] - expected_y)
    
    if study.best_value < 10.0 and x_error < 1.0 and y_error < 1.0:
        print("\n✓ Successfully found region near global minimum!")
    else:
        print("\n⚠ Not fully converged, Rosenbrock is a hard problem")


def example_with_mixed_types():
    """Example with continuous, integer, and categorical parameters."""
    print("\n" + "=" * 70)
    print("Example with Mixed Parameter Types")
    print("=" * 70)
    
    def objective(trial):
        x = trial.suggest_float('x', -5, 5)
        n = trial.suggest_int('n', 1, 10)
        opt = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
        
        # Dummy objective that uses all parameters
        penalty = 0.1 if opt == 'sgd' else 0.0
        return x**2 + (n - 5)**2 + penalty
    
    study = create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"\nBest value: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    print(f"  Note: 'n' is returned as int type: {type(study.best_params['n'])}")


if __name__ == "__main__":
    main()
    example_with_mixed_types()
