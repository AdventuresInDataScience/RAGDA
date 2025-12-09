"""
AUC Metric for Optimizer Convergence Evaluation

Measures the area under the convergence curve (normalized 0-1).
Lower AUC = faster convergence = better optimizer performance.

The AUC metric is scale-invariant and doesn't require knowing the true optimum,
making it ideal for comparing optimizer performance across different problems.
"""

from typing import List, Optional, Callable, Tuple, Any
import numpy as np


def calculate_auc(
    convergence_history: List[float],
    normalize: bool = True,
    best_known_value: Optional[float] = None,
    direction: str = 'minimize'
) -> float:
    """
    Calculate AUC from convergence history.
    
    This metric measures how quickly an optimizer converges by computing the
    area under the best-so-far curve, normalized to [0, 1]:
    - 0.0 = Perfect (found best value immediately)
    - 0.5 = Linear improvement
    - 1.0 = Worst (best value found only on last iteration)
    
    Args:
        convergence_history: List of objective values from optimization (in order)
        normalize: If True, normalize to [0, 1] range (always True in practice)
        best_known_value: Known optimal value (if available, currently unused)
        direction: 'minimize' or 'maximize'
    
    Returns:
        AUC value (0-1, lower is better)
    """
    if len(convergence_history) == 0:
        return 1.0  # No values = worst possible
    
    if len(convergence_history) == 1:
        return 0.5  # Single value = neutral
    
    values = np.array(convergence_history, dtype=float)
    
    # Compute best-so-far curve
    if direction == 'minimize':
        best_so_far = np.minimum.accumulate(values)
    else:
        best_so_far = np.maximum.accumulate(values)
    
    # Normalize to [0, 1] based on observed range
    v_min = best_so_far[-1]  # Best found (final value)
    v_max = best_so_far[0]   # Worst (first value = starting point)
    
    if v_max == v_min:
        # All values same = perfect convergence from start
        return 0.0
    
    # Normalize curve: 0 = at best, 1 = at worst
    if direction == 'minimize':
        normalized = (best_so_far - v_min) / (v_max - v_min)
    else:
        normalized = (v_max - best_so_far) / (v_max - v_min)
    
    # AUC using trapezoidal rule, normalized by number of points
    # This gives area as fraction of total possible area
    auc = np.trapz(normalized) / (len(normalized) - 1)
    
    return float(auc)


def interpret_auc(auc: float) -> str:
    """
    Interpret an AUC value and return a human-readable quality description.
    
    AUC (Area Under Curve) measures convergence quality:
    - Lower is better (faster convergence)
    - 0.0 = perfect, 1.0 = worst
    """
    if auc < 0.05:
        return "excellent - near-optimal convergence"
    elif auc < 0.15:
        return "very good - fast convergence"
    elif auc < 0.30:
        return "good - steady convergence"
    elif auc < 0.50:
        return "moderate - gradual convergence"
    elif auc < 0.75:
        return "fair - mediocre convergence"
    elif auc < 1.0:
        return "poor - slow convergence"
    elif auc >= 1e9:
        return "failed - optimization crashed"
    else:
        return "very poor - minimal improvement"


def evaluate_optimizer_on_problem(
    optimizer_factory: Callable,
    problem: Callable,
    n_evaluations: int,
    direction: str = 'minimize',
    **optimizer_kwargs
) -> Tuple[float, List[float]]:
    """
    Run optimizer on problem and return AUC + convergence history.
    
    Args:
        optimizer_factory: Function that creates an optimizer study
                          (e.g., ragda.create_study, optuna.create_study)
        problem: Optuna-style objective function (takes trial object)
        n_evaluations: Number of evaluations to run
        direction: 'minimize' or 'maximize'
        **optimizer_kwargs: Additional kwargs for optimizer_factory
    
    Returns:
        Tuple of (auc, convergence_history)
        - auc: Normalized AUC value (0-1, lower is better)
        - convergence_history: List of objective values in evaluation order
    """
    try:
        # Create optimizer study
        study = optimizer_factory(direction=direction, **optimizer_kwargs)
        
        # Run optimization
        study.optimize(problem, n_trials=n_evaluations)
        
        # Extract convergence history
        convergence_history = [trial.value for trial in study.trials]
        
        if len(convergence_history) == 0:
            return 1.0, []
        
        # Calculate AUC
        auc = calculate_auc(convergence_history, direction=direction)
        
        return auc, convergence_history
        
    except Exception as e:
        print(f"Error evaluating optimizer: {e}")
        return 1.0, []  # Worst possible AUC on failure
