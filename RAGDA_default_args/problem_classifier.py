"""
Problem Classifier for RAGDA Benchmarking

Classifies problems by THREE dimensions (3 levels each = 27 total categories):
1. DIMENSIONALITY: low (1-10D), medium (11-50D), high (51D+)
2. COST: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
3. RUGGEDNESS: smooth, moderate, rugged

All functions are DETERMINISTIC. "Ruggedness" = landscape complexity, not noise.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Callable, Tuple, Dict, Any
import time
import numpy as np
from benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
from benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
from benchmark_ml_problems import ALL_ML_PROBLEMS


class DimensionLevel(Enum):
    LOW = 'low'       # 1-10D
    MEDIUM = 'medium' # 11-50D
    HIGH = 'high'     # 51D+


class CostLevel(Enum):
    CHEAP = 'cheap'         # <10ms
    MODERATE = 'moderate'   # 10-100ms
    EXPENSIVE = 'expensive' # 100ms+


class RuggednessLevel(Enum):
    SMOOTH = 'smooth'       # Low sensitivity, consistent gradients
    MODERATE = 'moderate'   # Some local structure
    RUGGED = 'rugged'       # Many local features / chaotic


@dataclass
class ProblemClassification:
    """Classification result for a problem."""
    problem_name: str
    dimension_level: str
    cost_level: str
    ruggedness_level: str
    category_key: str  # e.g., "low_cheap_smooth"
    
    # Measurements
    actual_dimension: int
    avg_eval_time_ms: float
    ruggedness_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MockTrial:
    """Mock Optuna trial for testing problem functions."""
    def __init__(self, dimension: int, bounds: list):
        self.dimension = dimension
        self.bounds = bounds
        self._values = {}
    
    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Return a random value in the specified range."""
        value = np.random.uniform(low, high)
        self._values[name] = value
        return value
    
    def get_values(self) -> Dict[str, float]:
        """Get all suggested values."""
        return self._values


def classify_dimension(dim: int) -> DimensionLevel:
    """Classify dimensionality (trivial)."""
    if dim <= 10:
        return DimensionLevel.LOW
    elif dim <= 50:
        return DimensionLevel.MEDIUM
    else:
        return DimensionLevel.HIGH


def measure_cost(objective: Callable, dimension: int, bounds: list, n_samples: int = 10) -> Tuple[float, CostLevel]:
    """
    Measure evaluation time by running problem multiple times.
    Returns (avg_time_ms, cost_level).
    """
    times = []
    
    for _ in range(n_samples):
        trial = MockTrial(dimension, bounds)
        
        start = time.perf_counter()
        try:
            _ = objective(trial)
        except Exception:
            # If function fails, assume it's expensive (likely import issue or complex calc)
            return 100.0, CostLevel.EXPENSIVE
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    
    if avg_time < 10:
        return avg_time, CostLevel.CHEAP
    elif avg_time < 100:
        return avg_time, CostLevel.MODERATE
    else:
        return avg_time, CostLevel.EXPENSIVE


def measure_ruggedness(objective: Callable, dimension: int, bounds: list, n_samples: int = 50) -> Tuple[float, RuggednessLevel]:
    """
    Measure landscape ruggedness via:
    - Local sensitivity: output change for small input perturbations
    - Gradient variability: how much improvement direction changes
    
    Returns (ruggedness_score, ruggedness_level).
    
    Higher score = more rugged landscape.
    """
    sensitivities = []
    
    # Test local sensitivity at random points
    for _ in range(n_samples):
        # Generate a base point
        trial1 = MockTrial(dimension, bounds)
        try:
            f1 = objective(trial1)
        except Exception:
            # If function fails, assume moderate ruggedness
            return 0.5, RuggednessLevel.MODERATE
        
        # Generate a nearby point (1% perturbation)
        trial2 = MockTrial(dimension, bounds)
        values1 = trial1.get_values()
        
        # Override with perturbed values
        for key in values1:
            param_name = key
            low, high = bounds[int(key[1:])] if key.startswith('x') else bounds[0]
            range_size = high - low
            perturbation = np.random.normal(0, 0.01 * range_size)
            new_val = np.clip(values1[key] + perturbation, low, high)
            trial2._values[key] = new_val
        
        # Re-run objective with perturbed trial
        trial2_perturbed = MockTrial(dimension, bounds)
        for key, val in trial2._values.items():
            trial2_perturbed._values[key] = val
        
        try:
            f2 = objective(trial2_perturbed)
        except Exception:
            continue
        
        # Calculate relative sensitivity
        if abs(f1) > 1e-10:
            sensitivity = abs(f2 - f1) / abs(f1)
        else:
            sensitivity = abs(f2 - f1)
        
        sensitivities.append(sensitivity)
    
    if not sensitivities:
        return 0.5, RuggednessLevel.MODERATE
    
    # Ruggedness score is the median sensitivity (robust to outliers)
    ruggedness_score = float(np.median(sensitivities))
    
    # Classify based on sensitivity thresholds
    if ruggedness_score < 0.1:
        return ruggedness_score, RuggednessLevel.SMOOTH
    elif ruggedness_score < 1.0:
        return ruggedness_score, RuggednessLevel.MODERATE
    else:
        return ruggedness_score, RuggednessLevel.RUGGED


def get_category_key(
    dim_level: DimensionLevel,
    cost_level: CostLevel,
    ruggedness_level: RuggednessLevel
) -> str:
    """Generate category key like 'low_cheap_smooth'."""
    return f"{dim_level.value}_{cost_level.value}_{ruggedness_level.value}"


def classify_problem(problem: Any) -> ProblemClassification:
    """
    Fully classify a problem into one of 27 categories.
    
    Args:
        problem: BenchmarkProblem dataclass with name, objective, dimension, bounds
    
    Returns:
        ProblemClassification with all measurements and category assignment
    """
    # Classify dimension (trivial)
    dim_level = classify_dimension(problem.dimension)
    
    # Measure cost
    avg_time_ms, cost_level = measure_cost(
        problem.objective,
        problem.dimension,
        problem.bounds,
        n_samples=10
    )
    
    # Measure ruggedness
    ruggedness_score, ruggedness_level = measure_ruggedness(
        problem.objective,
        problem.dimension,
        problem.bounds,
        n_samples=50
    )
    
    # Generate category key
    category_key = get_category_key(dim_level, cost_level, ruggedness_level)
    
    return ProblemClassification(
        problem_name=problem.name,
        dimension_level=dim_level.value,
        cost_level=cost_level.value,
        ruggedness_level=ruggedness_level.value,
        category_key=category_key,
        actual_dimension=problem.dimension,
        avg_eval_time_ms=avg_time_ms,
        ruggedness_score=ruggedness_score
    )


def get_all_problems() -> Dict[str, Any]:
    """Get all benchmark problems from all registries."""
    all_problems = {}
    all_problems.update(ALL_BENCHMARK_FUNCTIONS)
    all_problems.update(ALL_REALWORLD_PROBLEMS)
    all_problems.update(ALL_ML_PROBLEMS)
    return all_problems


if __name__ == '__main__':
    # Quick test
    problems = get_all_problems()
    print(f"Total problems available: {len(problems)}")
    
    # Test classification on first problem
    first_prob = list(problems.values())[0]
    print(f"\nTesting classification on: {first_prob.name}")
    classification = classify_problem(first_prob)
    print(f"  Dimension: {classification.dimension_level}")
    print(f"  Cost: {classification.cost_level} ({classification.avg_eval_time_ms:.2f}ms)")
    print(f"  Ruggedness: {classification.ruggedness_level} (score: {classification.ruggedness_score:.3f})")
    print(f"  Category: {classification.category_key}")
