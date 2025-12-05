"""
Updated Problem Classifier for RAGDA Benchmarking

KEY INSIGHT: We classify by THREE axes:
1. DIMENSIONALITY - low/medium/high/very_high 
2. COST - cheap/moderate/expensive/very_expensive (wall-clock time)
3. LANDSCAPE RUGGEDNESS - smooth/moderate/rugged/highly_rugged

ALL functions are DETERMINISTIC (same input = same output).
"Ruggedness" measures how chaotic/complex the landscape is, NOT stochasticity.

Ruggedness is measured by:
- Local sensitivity: How much output changes for small input perturbations
- Gradient variability: How much the "direction" of improvement changes
- These capture "many local minima", "chaotic dynamics", etc.
"""

import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics


class RuggednessLevel(Enum):
    """Landscape ruggedness classification (3 levels)."""
    SMOOTH = 'smooth'           # Low sensitivity, consistent gradients
    MODERATE = 'moderate'       # Some local structure
    RUGGED = 'rugged'           # Many local features / chaotic


class CostLevel(Enum):
    """Cost classification based on evaluation time (3 levels)."""
    CHEAP = 'cheap'                   # < 10ms
    MODERATE = 'moderate'             # 10-100ms
    EXPENSIVE = 'expensive'           # >= 100ms


class DimClass(Enum):
    """Dimensionality classification (3 levels)."""
    LOW = 'low'             # 1-10
    MEDIUM = 'medium'       # 11-50
    HIGH = 'high'           # 51+


@dataclass
class ProblemCharacteristics:
    """Measured characteristics of a benchmark problem."""
    # Core measurements
    dimension: int
    mean_eval_time_ms: float
    sensitivity_score: float  # How sensitive is output to input perturbations
    sign_change_score: float  # How often near local extrema
    local_cv_score: float     # Local coefficient of variation
    
    # Classifications
    dim_class: DimClass
    cost_level: CostLevel
    ruggedness_level: RuggednessLevel
    
    # Additional stats
    std_eval_time_ms: float = 0.0
    mean_output: float = 0.0
    std_output: float = 0.0
    
    # Raw data for debugging
    sample_outputs: List[float] = field(default_factory=list)


def classify_dimension(dim: int) -> DimClass:
    """Classify dimensionality (3 levels)."""
    if dim <= 10:
        return DimClass.LOW
    elif dim <= 50:
        return DimClass.MEDIUM
    else:
        return DimClass.HIGH


def classify_cost(mean_time_ms: float) -> CostLevel:
    """Classify cost based on mean evaluation time (3 levels)."""
    if mean_time_ms < 10:
        return CostLevel.CHEAP
    elif mean_time_ms < 100:
        return CostLevel.MODERATE
    else:
        return CostLevel.EXPENSIVE


def classify_ruggedness(sensitivity: float, sign_changes: float, local_cv: float) -> RuggednessLevel:
    """
    Classify landscape ruggedness based on multiple metrics (3 levels).
    
    sensitivity: How much output changes for small input changes
    sign_changes: How often we're near local extrema (0-1)  
    local_cv: Local coefficient of variation - KEY DISCRIMINATOR
    
    Higher values → more rugged landscape
    """
    # Local CV is the best discriminator:
    # - Sphere: ~0.006 (smooth, consistent gradients)
    # - Rastrigin: ~0.023 (many small bumps)
    # - Lorenz: ~0.09+ (chaotic, highly variable)
    
    # Also factor in sensitivity for chaotic systems
    ruggedness_score = 50 * local_cv + 0.1 * sensitivity
    
    if ruggedness_score < 0.5:
        return RuggednessLevel.SMOOTH
    elif ruggedness_score < 1.5:
        return RuggednessLevel.MODERATE
    else:
        return RuggednessLevel.RUGGED


def measure_ruggedness(
    func: Callable,
    bounds: np.ndarray,
    dim: int,
    n_base_points: int = 20,
    perturbation_scale: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Measure landscape ruggedness by sampling.
    
    Returns:
        sensitivity: Normalized output change per unit input change
        sign_changes: Fraction of perturbations that change gradient sign
        local_variation: Coefficient of variation of local function values
    """
    rng = np.random.default_rng(42)
    
    # Generate base points
    base_points = []
    base_values = []
    
    for _ in range(n_base_points):
        x = rng.uniform(bounds[:, 0], bounds[:, 1])
        try:
            val = func(x)
            if np.isfinite(val):
                base_points.append(x)
                base_values.append(val)
        except:
            pass
    
    if len(base_points) < 5:
        return 0.0, 0.0, 0.0
    
    base_points = np.array(base_points)
    base_values = np.array(base_values)
    
    sensitivities = []
    sign_change_counts = []
    local_cvs = []
    
    for i, (x, f_x) in enumerate(zip(base_points, base_values)):
        range_scale = bounds[:, 1] - bounds[:, 0]
        eps = perturbation_scale * range_scale
        
        # Sample in a local neighborhood
        local_values = [f_x]
        deltas = []
        
        for _ in range(8):
            direction = rng.standard_normal(dim)
            direction = direction / np.linalg.norm(direction)
            
            x_perturbed = x + eps * direction
            x_perturbed = np.clip(x_perturbed, bounds[:, 0], bounds[:, 1])
            
            try:
                f_perturbed = func(x_perturbed)
                if np.isfinite(f_perturbed):
                    local_values.append(f_perturbed)
                    delta_f = f_perturbed - f_x
                    delta_x = np.linalg.norm(x_perturbed - x)
                    
                    if delta_x > 1e-10:
                        f_scale = max(abs(f_x), abs(f_perturbed), 1.0)
                        sensitivity = abs(delta_f) / f_scale / perturbation_scale
                        sensitivities.append(sensitivity)
                        deltas.append(delta_f)
            except:
                pass
        
        # Count sign changes (indicates we're near a local extremum)
        if len(deltas) >= 4:
            signs = np.sign(deltas)
            n_positive = np.sum(signs > 0)
            n_negative = np.sum(signs < 0)
            # If roughly equal positive and negative, we're near an extremum
            balance = min(n_positive, n_negative) / max(len(deltas) / 2, 1)
            sign_change_counts.append(balance)
        
        # Local coefficient of variation
        if len(local_values) >= 4:
            mean_local = np.mean(local_values)
            std_local = np.std(local_values)
            if abs(mean_local) > 1e-10:
                local_cvs.append(std_local / abs(mean_local))
    
    sensitivity = np.mean(sensitivities) if sensitivities else 0.0
    sign_changes = np.mean(sign_change_counts) if sign_change_counts else 0.0
    local_variation = np.mean(local_cvs) if local_cvs else 0.0
    
    return sensitivity, sign_changes, local_variation


def measure_problem(
    func: Callable,
    bounds: np.ndarray,
    dim: int,
    n_timing_samples: int = 10,
    n_ruggedness_samples: int = 20,
) -> ProblemCharacteristics:
    """
    Measure all characteristics of a problem.
    """
    rng = np.random.default_rng(42)
    
    # Measure timing
    times = []
    outputs = []
    
    for _ in range(n_timing_samples):
        x = rng.uniform(bounds[:, 0], bounds[:, 1])
        
        start = time.perf_counter()
        try:
            val = func(x)
        except:
            val = float('inf')
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        times.append(elapsed_ms)
        if np.isfinite(val):
            outputs.append(val)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_output = np.mean(outputs) if outputs else 0.0
    std_output = np.std(outputs) if outputs else 0.0
    
    # Measure ruggedness
    sensitivity, sign_changes, local_cv = measure_ruggedness(
        func, bounds, dim, n_ruggedness_samples
    )
    
    return ProblemCharacteristics(
        dimension=dim,
        mean_eval_time_ms=mean_time,
        sensitivity_score=sensitivity,
        sign_change_score=sign_changes,
        local_cv_score=local_cv,
        dim_class=classify_dimension(dim),
        cost_level=classify_cost(mean_time),
        ruggedness_level=classify_ruggedness(sensitivity, sign_changes, local_cv),
        std_eval_time_ms=std_time,
        mean_output=mean_output,
        std_output=std_output,
        sample_outputs=outputs[:10],
    )


@dataclass
class ClassifiedProblem:
    """A benchmark problem with measured characteristics."""
    name: str
    func: Callable
    bounds: np.ndarray
    dim: int
    characteristics: ProblemCharacteristics
    category: str = 'unknown'
    description: str = ''
    
    def get_class_key(self) -> Tuple[str, str, str]:
        """Return (dim_class, cost_level, ruggedness_level) for grouping."""
        c = self.characteristics
        return (c.dim_class.value, c.cost_level.value, c.ruggedness_level.value)


def classify_problem(
    name: str,
    func: Callable,
    bounds: np.ndarray,
    dim: int,
    category: str = 'unknown',
    description: str = '',
    n_timing_samples: int = 10,
    n_ruggedness_samples: int = 20,
) -> ClassifiedProblem:
    """Classify a single problem."""
    
    chars = measure_problem(func, bounds, dim, n_timing_samples, n_ruggedness_samples)
    
    return ClassifiedProblem(
        name=name,
        func=func,
        bounds=bounds,
        dim=dim,
        characteristics=chars,
        category=category,
        description=description,
    )


def summarize_classified_problems(problems: List[ClassifiedProblem]) -> Dict:
    """Get summary statistics of classified problems."""
    summary = {
        'total': len(problems),
        'by_dim': {},
        'by_cost': {},
        'by_ruggedness': {},
        'by_combo': {},
    }
    
    for p in problems:
        c = p.characteristics
        
        d = c.dim_class.value
        summary['by_dim'][d] = summary['by_dim'].get(d, 0) + 1
        
        cost = c.cost_level.value
        summary['by_cost'][cost] = summary['by_cost'].get(cost, 0) + 1
        
        rug = c.ruggedness_level.value
        summary['by_ruggedness'][rug] = summary['by_ruggedness'].get(rug, 0) + 1
        
        key = p.get_class_key()
        summary['by_combo'][key] = summary['by_combo'].get(key, 0) + 1
    
    return summary


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("UPDATED PROBLEM CLASSIFIER - Testing Ruggedness Measurement")
    print("="*80)
    
    # Test 1: Sphere (should be SMOOTH)
    def sphere(x):
        return np.sum(x**2)
    
    print("\n1. Sphere (convex, should be SMOOTH):")
    p = classify_problem("Sphere-10D", sphere, np.array([[-5, 5]]*10), 10)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    
    # Test 2: Rastrigin (should be RUGGED - many local minima)
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    print("\n2. Rastrigin (many local minima, should be RUGGED/HIGHLY_RUGGED):")
    p = classify_problem("Rastrigin-10D", rastrigin, np.array([[-5.12, 5.12]]*10), 10)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    
    # Test 3: Rosenbrock (should be MODERATE - valley structure)
    def rosenbrock(x):
        return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
    
    print("\n3. Rosenbrock (valley, should be MODERATE):")
    p = classify_problem("Rosenbrock-10D", rosenbrock, np.array([[-5, 10]]*10), 10)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    
    # Test 4: Schwefel (should be HIGHLY_RUGGED - chaotic)
    def schwefel(x):
        return 418.9829*len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    print("\n4. Schwefel (chaotic, should be RUGGED/HIGHLY_RUGGED):")
    p = classify_problem("Schwefel-10D", schwefel, np.array([[-500, 500]]*10), 10)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    
    # Test 5: Ackley (should be RUGGED)
    def ackley(x):
        n = len(x)
        sum1 = -0.2 * np.sqrt(np.sum(x**2) / n)
        sum2 = np.sum(np.cos(2 * np.pi * x)) / n
        return -20 * np.exp(sum1) - np.exp(sum2) + 20 + np.e
    
    print("\n5. Ackley (multimodal, should be RUGGED):")
    p = classify_problem("Ackley-10D", ackley, np.array([[-5, 5]]*10), 10)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    
    print("\n" + "="*80)
    print("Testing chaotic systems:")
    print("="*80)
    
    def lorenz_objective(params):
        sigma, rho, beta = params
        if sigma <= 0 or rho <= 0 or beta <= 0:
            return 1e10
        
        T, dt = 1000, 0.01
        x, y, z = 1.0, 1.0, 1.0
        trajectory = []
        
        for _ in range(T):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dt * dx
            y += dt * dy
            z += dt * dz
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            trajectory.append([x, y, z])
        
        trajectory = np.array(trajectory)
        target_mean = np.array([0.0, 0.0, 23.5])
        target_std = np.array([7.9, 9.0, 8.5])
        return np.mean((trajectory.mean(axis=0) - target_mean)**2) + \
               np.mean((trajectory.std(axis=0) - target_std)**2)
    
    print("\n6. Lorenz Parameter Estimation (chaotic system):")
    p = classify_problem("Lorenz-3D", lorenz_objective, 
                        np.array([[1, 20], [10, 50], [0.5, 5]]), 3,
                        n_ruggedness_samples=30)
    c = p.characteristics
    print(f"   Sensitivity: {c.sensitivity_score:.4f}")
    print(f"   Sign Changes: {c.sign_change_score:.4f}")
    print(f"   Local CV: {c.local_cv_score:.4f}")
    print(f"   Ruggedness: {c.ruggedness_level.value}")
    print(f"   Cost: {c.cost_level.value} ({c.mean_eval_time_ms:.1f}ms)")
