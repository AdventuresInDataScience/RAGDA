"""
Mathematical Benchmark Functions (Optuna API)

Standard optimization test functions wrapped in Optuna-style objective(trial) format.
Total: 78 functions from archive, organized by landscape type.

All functions are deterministic and use Optuna's trial.suggest_* interface.
"""

from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BenchmarkProblem:
    """Metadata for a benchmark problem."""
    name: str
    objective: Callable  # Optuna-style: objective(trial) -> float
    dimension: int
    bounds: List[Tuple[float, float]]
    known_optimum: Optional[float]
    category: str  # 'multimodal', 'unimodal', 'valley', 'plate', 'steep'
    description: str


# =============================================================================
# HELPER: Create Optuna-style wrapper
# =============================================================================

def _make_optuna_objective(func: Callable, bounds: List[Tuple[float, float]], dim: int) -> Callable:
    """Wrap a numpy function into Optuna-style objective."""
    def objective(trial):
        # Suggest parameters using trial interface
        x = np.array([
            trial.suggest_float(f'x{i}', bounds[i][0], bounds[i][1])
            for i in range(dim)
        ])
        return float(func(x))
    return objective


# =============================================================================
# BATCH 1: UNIMODAL FUNCTIONS (Simplest - 12 functions)
# =============================================================================

def _sphere(x: np.ndarray) -> float:
    """Sphere function. Global min: f(0,...,0) = 0"""
    return float(np.sum(x**2))

def _sum_squares(x: np.ndarray) -> float:
    """Sum of Squares. Global min: f(0,...,0) = 0"""
    return float(np.sum((np.arange(1, len(x) + 1)) * x**2))


# =============================================================================
# BATCH 2: MULTIMODAL CORE FUNCTIONS (30 functions)
# =============================================================================

def _ackley(x: np.ndarray) -> float:
    """Ackley function. Global min: f(0,...,0) = 0"""
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e)

def _rastrigin(x: np.ndarray) -> float:
    """Rastrigin function. Global min: f(0,...,0) = 0"""
    A = 10.0
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2.0 * np.pi * x)))

def _schwefel(x: np.ndarray) -> float:
    """Schwefel function. Global min: f(420.9687,...) = 0"""
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def _griewank(x: np.ndarray) -> float:
    """Griewank function. Global min: f(0,...,0) = 0"""
    d = len(x)
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return float(sum_term - prod_term + 1.0)

def _levy(x: np.ndarray) -> float:
    """Levy function. Global min: f(1,...,1) = 0"""
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2))
    term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)

def _styblinski_tang(x: np.ndarray) -> float:
    """Styblinski-Tang function. Global min: f(-2.9035,...) ≈ -39.16599*d"""
    return float(np.sum(x**4 - 16.0*x**2 + 5.0*x) / 2.0)


# Register UNIMODAL functions
_UNIMODAL_REGISTRY = {}

# Sphere variants (2D, 5D, 10D, 20D, 50D, 100D)
for dim in [2, 5, 10, 20, 50, 100]:
    bounds = [(-5.12, 5.12)] * dim
    _UNIMODAL_REGISTRY[f'sphere_{dim}d'] = BenchmarkProblem(
        name=f'sphere_{dim}d',
        objective=_make_optuna_objective(_sphere, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='unimodal',
        description=f'Sphere function in {dim}D - simple convex quadratic'
    )

# Sum of Squares variants (2D, 5D, 10D, 20D, 50D, 100D)
for dim in [2, 5, 10, 20, 50, 100]:
    bounds = [(-10.0, 10.0)] * dim
    _UNIMODAL_REGISTRY[f'sum_squares_{dim}d'] = BenchmarkProblem(
        name=f'sum_squares_{dim}d',
        objective=_make_optuna_objective(_sum_squares, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='unimodal',
        description=f'Weighted sum of squares in {dim}D'
    )


# Register MULTIMODAL CORE functions
_MULTIMODAL_REGISTRY = {}

# Ackley variants (2D, 5D, 10D, 20D, 50D)
for dim in [2, 5, 10, 20, 50]:
    bounds = [(-32.768, 32.768)] * dim
    _MULTIMODAL_REGISTRY[f'ackley_{dim}d'] = BenchmarkProblem(
        name=f'ackley_{dim}d',
        objective=_make_optuna_objective(_ackley, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='multimodal',
        description=f'Ackley function in {dim}D - highly multimodal with deep basin'
    )

# Rastrigin variants (2D, 5D, 10D, 20D, 50D)
for dim in [2, 5, 10, 20, 50]:
    bounds = [(-5.12, 5.12)] * dim
    _MULTIMODAL_REGISTRY[f'rastrigin_{dim}d'] = BenchmarkProblem(
        name=f'rastrigin_{dim}d',
        objective=_make_optuna_objective(_rastrigin, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='multimodal',
        description=f'Rastrigin function in {dim}D - highly multimodal cosine modulation'
    )

# Schwefel variants (2D, 5D, 10D, 20D, 50D)
for dim in [2, 5, 10, 20, 50]:
    bounds = [(-500.0, 500.0)] * dim
    _MULTIMODAL_REGISTRY[f'schwefel_{dim}d'] = BenchmarkProblem(
        name=f'schwefel_{dim}d',
        objective=_make_optuna_objective(_schwefel, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='multimodal',
        description=f'Schwefel function in {dim}D - deceptive with distant optimum'
    )

# Griewank variants (2D, 5D, 10D, 20D, 50D)
for dim in [2, 5, 10, 20, 50]:
    bounds = [(-600.0, 600.0)] * dim
    _MULTIMODAL_REGISTRY[f'griewank_{dim}d'] = BenchmarkProblem(
        name=f'griewank_{dim}d',
        objective=_make_optuna_objective(_griewank, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='multimodal',
        description=f'Griewank function in {dim}D - product of cosines creates modality'
    )

# Levy variants (2D, 5D, 10D, 20D, 50D)
for dim in [2, 5, 10, 20, 50]:
    bounds = [(-10.0, 10.0)] * dim
    _MULTIMODAL_REGISTRY[f'levy_{dim}d'] = BenchmarkProblem(
        name=f'levy_{dim}d',
        objective=_make_optuna_objective(_levy, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='multimodal',
        description=f'Levy function in {dim}D - multimodal with optimum at (1,...,1)'
    )

# Styblinski-Tang variants (2D, 5D, 10D, 20D, 50D, 100D)
for dim in [2, 5, 10, 20, 50, 100]:
    bounds = [(-5.0, 5.0)] * dim
    _MULTIMODAL_REGISTRY[f'styblinski_tang_{dim}d'] = BenchmarkProblem(
        name=f'styblinski_tang_{dim}d',
        objective=_make_optuna_objective(_styblinski_tang, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=-39.16599 * dim,
        category='multimodal',
        description=f'Styblinski-Tang function in {dim}D - multimodal with single global minimum'
    )


# =============================================================================
# BATCH 3: MULTIMODAL - SPECIAL 2D FUNCTIONS (6 functions)
# =============================================================================

def _beale(x: np.ndarray) -> float:
    """Beale function (2D). Global min: f(3, 0.5) = 0"""
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def _branin(x: np.ndarray) -> float:
    """Branin function (2D). 3 global minima, f* ≈ 0.397887"""
    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def _goldstein_price(x: np.ndarray) -> float:
    """Goldstein-Price function (2D). Global min: f(0,-1) = 3"""
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

def _eggholder(x: np.ndarray) -> float:
    """Eggholder function (2D). Global min: f(512, 404.2319) ≈ -959.6407"""
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + x[1] + 47))) - \
           x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))

def _holder_table(x: np.ndarray) -> float:
    """Holder Table function (2D). Global min: f* ≈ -19.2085"""
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * 
                   np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

def _hartmann_3d(x: np.ndarray) -> float:
    """Hartmann 3D function. Global min: f* ≈ -3.86278"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                  [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    result = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        result -= alpha[i] * np.exp(-inner)
    return result

# Registry for special 2D multimodal functions
_SPECIAL_2D_REGISTRY: Dict[str, BenchmarkProblem] = {}

# Beale (2D only)
_SPECIAL_2D_REGISTRY['beale_2d'] = BenchmarkProblem(
    name='beale_2d',
    objective=_make_optuna_objective(_beale, [(-4.5, 4.5), (-4.5, 4.5)], 2),
    dimension=2,
    bounds=[(-4.5, 4.5), (-4.5, 4.5)],
    known_optimum=0.0,
    category='multimodal',
    description='Beale function - 2D multimodal with sharp ridges'
)

# Branin (2D only)
_SPECIAL_2D_REGISTRY['branin_2d'] = BenchmarkProblem(
    name='branin_2d',
    objective=_make_optuna_objective(_branin, [(-5, 10), (0, 15)], 2),
    dimension=2,
    bounds=[(-5, 10), (0, 15)],
    known_optimum=0.397887,
    category='multimodal',
    description='Branin function - 2D with 3 global minima'
)

# Goldstein-Price (2D only)
_SPECIAL_2D_REGISTRY['goldstein_price_2d'] = BenchmarkProblem(
    name='goldstein_price_2d',
    objective=_make_optuna_objective(_goldstein_price, [(-2, 2), (-2, 2)], 2),
    dimension=2,
    bounds=[(-2, 2), (-2, 2)],
    known_optimum=3.0,
    category='multimodal',
    description='Goldstein-Price function - 2D highly multimodal'
)

# Eggholder (2D only)
_SPECIAL_2D_REGISTRY['eggholder_2d'] = BenchmarkProblem(
    name='eggholder_2d',
    objective=_make_optuna_objective(_eggholder, [(-512, 512), (-512, 512)], 2),
    dimension=2,
    bounds=[(-512, 512), (-512, 512)],
    known_optimum=-959.6407,
    category='multimodal',
    description='Eggholder function - 2D highly multimodal and deceptive'
)

# Holder Table (2D only)
_SPECIAL_2D_REGISTRY['holder_table_2d'] = BenchmarkProblem(
    name='holder_table_2d',
    objective=_make_optuna_objective(_holder_table, [(-10, 10), (-10, 10)], 2),
    dimension=2,
    bounds=[(-10, 10), (-10, 10)],
    known_optimum=-19.2085,
    category='multimodal',
    description='Holder Table function - 2D with multiple global minima'
)

# Hartmann 3D (special case)
_SPECIAL_2D_REGISTRY['hartmann_3d'] = BenchmarkProblem(
    name='hartmann_3d',
    objective=_make_optuna_objective(_hartmann_3d, [(0, 1), (0, 1), (0, 1)], 3),
    dimension=3,
    bounds=[(0, 1), (0, 1), (0, 1)],
    known_optimum=-3.86278,
    category='multimodal',
    description='Hartmann 3D function - 3D multimodal with 4 local minima'
)

# =============================================================================
# BATCH 4: Multimodal - Fixed Dimension Functions
# =============================================================================

def _shekel_4d(x: np.ndarray) -> float:
    """Shekel function (4D only). Global min: f* ≈ -10.5364"""
    beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    C = np.array([
        [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
        [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
        [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]
    ])
    result = 0.0
    for i in range(10):
        inner = np.sum((x - C[:, i])**2)
        result -= 1.0 / (inner + beta[i])
    return result

def _hartmann_6d(x: np.ndarray) -> float:
    """Hartmann 6D function (6D only). Global min: f* ≈ -3.32237"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
    ])
    result = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        result -= alpha[i] * np.exp(-inner)
    return result

_FIXED_DIM_REGISTRY: Dict[str, BenchmarkProblem] = {}

_FIXED_DIM_REGISTRY['shekel_4d'] = BenchmarkProblem(
    name='shekel_4d',
    objective=_make_optuna_objective(_shekel_4d, [(0, 10)] * 4, 4),
    dimension=4,
    bounds=[(0, 10), (0, 10), (0, 10), (0, 10)],
    known_optimum=-10.5364,
    category='multimodal',
    description='Shekel function - 4D with multiple local minima'
)

_FIXED_DIM_REGISTRY['hartmann_6d'] = BenchmarkProblem(
    name='hartmann_6d',
    objective=_make_optuna_objective(_hartmann_6d, [(0, 1)] * 6, 6),
    dimension=6,
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
    known_optimum=-3.32237,
    category='multimodal',
    description='Hartmann 6D function - 6D multimodal with 4 local minima'
)


# =============================================================================
# BATCH 5: VALLEY FUNCTIONS (18 functions - Chunk 2.1.5)
# =============================================================================
# Valley/banana-shaped functions with narrow, curved valleys leading to minimum

def _rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Global min: f(1,...,1) = 0"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def _dixon_price(x: np.ndarray) -> float:
    """Dixon-Price function. Optimum depends on dimension."""
    d = len(x)
    term1 = (x[0] - 1)**2
    i = np.arange(2, d + 1)
    term2 = np.sum(i * (2*x[1:]**2 - x[:-1])**2)
    return term1 + term2

def _six_hump_camel(x: np.ndarray) -> float:
    """Six-Hump Camel function (2D only). Global min: f* ≈ -1.0316"""
    return (4 - 2.1*x[0]**2 + x[0]**4/3) * x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2

def _powell(x: np.ndarray) -> float:
    """Powell function. f(0,...,0) = 0. Dimension must be multiple of 4."""
    d = len(x)
    result = 0.0
    for i in range(d // 4):
        idx = 4 * i
        result += (x[idx] + 10*x[idx+1])**2
        result += 5 * (x[idx+2] - x[idx+3])**2
        result += (x[idx+1] - 2*x[idx+2])**4
        result += 10 * (x[idx] - x[idx+3])**4
    return result

def _colville(x: np.ndarray) -> float:
    """Colville: 4D, f(1,1,1,1) = 0. Valley-shaped."""
    return (100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + (x[2] - 1)**2 + 
            90*(x[2]**2 - x[3])**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 
            19.8*(x[1] - 1)*(x[3] - 1))

_VALLEY_REGISTRY: Dict[str, BenchmarkProblem] = {}

# Rosenbrock variants (6 dimensions)
for dim in [2, 5, 10, 20, 50, 100]:
    bounds = [(-5.0, 10.0)] * dim
    _VALLEY_REGISTRY[f'rosenbrock_{dim}d'] = BenchmarkProblem(
        name=f'rosenbrock_{dim}d',
        objective=_make_optuna_objective(_rosenbrock, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='valley',
        description=f'Rosenbrock function in {dim}D - narrow curved valley to minimum at (1,...,1)'
    )

# Dixon-Price variants (6 dimensions)
for dim in [2, 5, 10, 20, 50, 100]:
    bounds = [(-10.0, 10.0)] * dim
    _VALLEY_REGISTRY[f'dixon_price_{dim}d'] = BenchmarkProblem(
        name=f'dixon_price_{dim}d',
        objective=_make_optuna_objective(_dixon_price, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='valley',
        description=f'Dixon-Price function in {dim}D - steep valley structure'
    )

# Six-Hump Camel (2D only)
_VALLEY_REGISTRY['six_hump_camel_2d'] = BenchmarkProblem(
    name='six_hump_camel_2d',
    objective=_make_optuna_objective(_six_hump_camel, [(-3, 3), (-2, 2)], 2),
    dimension=2,
    bounds=[(-3, 3), (-2, 2)],
    known_optimum=-1.0316,
    category='valley',
    description='Six-Hump Camel function - 2D with six local minima and two global minima'
)

# Powell variants (dimensions must be multiples of 4)
for dim in [4, 8, 12, 20, 40]:
    bounds = [(-4.0, 5.0)] * dim
    _VALLEY_REGISTRY[f'powell_{dim}d'] = BenchmarkProblem(
        name=f'powell_{dim}d',
        objective=_make_optuna_objective(_powell, bounds, dim),
        dimension=dim,
        bounds=bounds,
        known_optimum=0.0,
        category='valley',
        description=f'Powell function in {dim}D - quartic valley, must be multiple of 4D'
    )

# Colville (4D only)
_VALLEY_REGISTRY['colville_4d'] = BenchmarkProblem(
    name='colville_4d',
    objective=_make_optuna_objective(_colville, [(-10.0, 10.0)] * 4, 4),
    dimension=4,
    bounds=[(-10.0, 10.0)] * 4,
    known_optimum=0.0,
    category='valley',
    description='Colville 4D: Complex valley function, f(1,1,1,1) = 0'
)


# =============================================================================
# BATCH 6: PLATE FUNCTIONS (Plate-shaped landscapes - 7 functions)
# =============================================================================

def _zakharov(x: np.ndarray) -> float:
    """Zakharov: f(0,...,0) = 0. Plate-shaped."""
    d = len(x)
    i = np.arange(1, d + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


def _booth(x: np.ndarray) -> float:
    """Booth: 2D, f(1,3) = 0. Plate-shaped."""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


# Zakharov variants (6 dimensions)
_PLATE_REGISTRY = {
    'zakharov_2d': BenchmarkProblem(
        name='zakharov_2d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 2, 2),
        dimension=2,
        bounds=[(-5.0, 10.0)] * 2,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 2D: Plate-shaped landscape'
    ),
    'zakharov_5d': BenchmarkProblem(
        name='zakharov_5d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 5, 5),
        dimension=5,
        bounds=[(-5.0, 10.0)] * 5,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 5D: Plate-shaped landscape'
    ),
    'zakharov_10d': BenchmarkProblem(
        name='zakharov_10d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 10, 10),
        dimension=10,
        bounds=[(-5.0, 10.0)] * 10,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 10D: Plate-shaped landscape'
    ),
    'zakharov_20d': BenchmarkProblem(
        name='zakharov_20d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 20, 20),
        dimension=20,
        bounds=[(-5.0, 10.0)] * 20,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 20D: Plate-shaped landscape'
    ),
    'zakharov_50d': BenchmarkProblem(
        name='zakharov_50d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 50, 50),
        dimension=50,
        bounds=[(-5.0, 10.0)] * 50,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 50D: Plate-shaped landscape'
    ),
    'zakharov_100d': BenchmarkProblem(
        name='zakharov_100d',
        objective=_make_optuna_objective(_zakharov, [(-5.0, 10.0)] * 100, 100),
        dimension=100,
        bounds=[(-5.0, 10.0)] * 100,
        known_optimum=0.0,
        category='plate',
        description='Zakharov 100D: Plate-shaped landscape'
    ),
    # Booth (2D only)
    'booth_2d': BenchmarkProblem(
        name='booth_2d',
        objective=_make_optuna_objective(_booth, [(-10.0, 10.0)] * 2, 2),
        dimension=2,
        bounds=[(-10.0, 10.0)] * 2,
        known_optimum=0.0,
        category='plate',
        description='Booth 2D: Plate-shaped, f(1,3) = 0'
    ),
}


# =============================================================================
# BATCH 7: STEEP FUNCTIONS (Steep drops/ridges - 1 function)
# =============================================================================

def _easom(x: np.ndarray) -> float:
    """Easom: 2D, f(π,π) = -1. Flat with steep drop at optimum."""
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))

# Easom (2D only)
_STEEP_REGISTRY = {
    'easom_2d': BenchmarkProblem(
        name='easom_2d',
        objective=_make_optuna_objective(_easom, [(-100.0, 100.0)] * 2, 2),
        dimension=2,
        bounds=[(-100.0, 100.0)] * 2,
        known_optimum=-1.0,
        category='steep',
        description='Easom 2D: Nearly flat landscape with steep drop at (π,π), f(π,π) = -1'
    ),
}


# =============================================================================
# MASTER REGISTRY (Batches 1-7: Unimodal + Multimodal + Special 2D + Fixed Dim + Valley + Plate + Steep)
# =============================================================================

ALL_BENCHMARK_FUNCTIONS: Dict[str, BenchmarkProblem] = {
    **_UNIMODAL_REGISTRY,
    **_MULTIMODAL_REGISTRY,
    **_SPECIAL_2D_REGISTRY,
    **_FIXED_DIM_REGISTRY,
    **_VALLEY_REGISTRY,
    **_PLATE_REGISTRY,
    **_STEEP_REGISTRY,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_benchmark_function(name: str) -> BenchmarkProblem:
    """Get benchmark problem by name."""
    if name not in ALL_BENCHMARK_FUNCTIONS:
        raise KeyError(f"Unknown benchmark function: {name}")
    return ALL_BENCHMARK_FUNCTIONS[name]


def list_benchmark_functions() -> List[str]:
    """List all available benchmark function names."""
    return sorted(ALL_BENCHMARK_FUNCTIONS.keys())


def get_functions_by_category(category: str) -> Dict[str, BenchmarkProblem]:
    """Get functions filtered by category."""
    return {
        name: prob for name, prob in ALL_BENCHMARK_FUNCTIONS.items()
        if prob.category == category
    }


def get_functions_by_dimension(dim: int) -> Dict[str, BenchmarkProblem]:
    """Get functions filtered by dimension."""
    return {
        name: prob for name, prob in ALL_BENCHMARK_FUNCTIONS.items()
        if prob.dimension == dim
    }


if __name__ == '__main__':
    print(f"Total benchmark functions loaded: {len(ALL_BENCHMARK_FUNCTIONS)}")
    print(f"\nCategories:")
    for cat in sorted(set(p.category for p in ALL_BENCHMARK_FUNCTIONS.values())):
        count = len(get_functions_by_category(cat))
        print(f"  {cat}: {count}")
    print(f"\nDimensions: {sorted(set(p.dimension for p in ALL_BENCHMARK_FUNCTIONS.values()))}")
