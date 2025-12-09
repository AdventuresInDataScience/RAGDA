"""
Example: Mixed parameter types (continuous, categorical, ordinal).

Demonstrates handling different parameter types in optimization.
"""

from ragda import RAGDAOptimizer
import numpy as np


def main():
    print("=" * 70)
    print("RAGDA Example: Mixed Parameter Types")
    print("=" * 70)
    
    # Define search space with mixed types
    space = {
        'learning_rate': {'type': 'continuous', 'bounds': [1e-4, 1e-1], 'log': True},
        'dropout': {'type': 'continuous', 'bounds': [0.0, 0.5]},
        'batch_size': {'type': 'ordinal', 'values': [16, 32, 64, 128, 256]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
        'n_layers': {'type': 'ordinal', 'values': [1, 2, 3, 4, 5]},
    }
    
    print("\nSearch Space:")
    print(f"  - learning_rate: continuous, log-scale, [{space['learning_rate']['bounds'][0]}, {space['learning_rate']['bounds'][1]}]")
    print(f"  - dropout: continuous [{space['dropout']['bounds'][0]}, {space['dropout']['bounds'][1]}]")
    print(f"  - batch_size: ordinal {space['batch_size']['values']}")
    print(f"  - optimizer: categorical {space['optimizer']['values']}")
    print(f"  - n_layers: ordinal {space['n_layers']['values']}")
    
    # Define objective that uses all parameter types
    def ml_objective(learning_rate, dropout, batch_size, optimizer, n_layers):
        """
        Simulated ML training objective.
        
        Parameters are passed as kwargs - no dict access needed!
        """
        # Simulate model performance based on hyperparameters
        # In practice, this would train and evaluate a real model
        
        # Penalty for extreme learning rates
        lr_penalty = abs(np.log10(learning_rate) + 2.5)  # Best around 3e-3
        
        # Dropout should be moderate
        dropout_penalty = (dropout - 0.2)**2
        
        # Prefer larger batches
        batch_penalty = (256 - batch_size) / 256
        
        # SGD is harder to tune
        opt_penalty = 0.5 if optimizer == 'sgd' else 0.0
        
        # 2-3 layers is optimal for this problem
        layer_penalty = abs(n_layers - 2.5)
        
        # Combined loss
        loss = lr_penalty + dropout_penalty + batch_penalty + opt_penalty + layer_penalty * 0.5
        
        return loss
    
    print("\nOptimizing ML hyperparameters...")
    print("(This is a simulated objective - in practice, you'd train a real model)")
    
    # Create optimizer
    optimizer = RAGDAOptimizer(
        space=space,
        direction='minimize',
        n_workers=2,
        random_state=42
    )
    
    # Run optimization
    result = optimizer.optimize(
        ml_objective,
        n_trials=100,
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Best loss: {result.best_value:.6f}")
    print(f"\nBest hyperparameters:")
    for param, value in result.best_params.items():
        if param == 'learning_rate':
            print(f"  {param}: {value:.6e} (log-scale)")
        elif param in ['batch_size', 'n_layers']:
            print(f"  {param}: {value} (ordinal)")
        elif param == 'optimizer':
            print(f"  {param}: {value} (categorical)")
        else:
            print(f"  {param}: {value:.4f} (continuous)")
    
    print(f"\nParameter types:")
    print(f"  learning_rate: {type(result.best_params['learning_rate']).__name__}")
    print(f"  dropout: {type(result.best_params['dropout']).__name__}")
    print(f"  batch_size: {type(result.best_params['batch_size']).__name__}")
    print(f"  optimizer: {type(result.best_params['optimizer']).__name__}")
    print(f"  n_layers: {type(result.best_params['n_layers']).__name__}")


if __name__ == "__main__":
    main()
