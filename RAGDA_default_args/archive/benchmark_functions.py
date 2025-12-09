"""
Comprehensive Benchmark Functions Library

Part 1: Synthetic Test Functions from SFU Virtual Library
Reference: https://www.sfu.ca/~ssurjano/optimization.html

47+ standard optimization test functions organized by landscape type.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TestFunction:
    """Container for a test function."""
    name: str
    func: Callable
    bounds: np.ndarray
    dim: int
    optimum: float
    optimum_location: Optional[np.ndarray]
    category: str  # 'multimodal', 'unimodal', 'valley', 'plate', 'steep'
    description: str


# =============================================================================
# MANY LOCAL MINIMA (Multimodal)
# =============================================================================

def ackley(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
    """Ackley function. Global min: f(0,...,0) = 0"""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function. Global min: f(0,...,0) = 0"""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def schwefel(x: np.ndarray) -> float:
    """Schwefel function. Global min: f(420.9687,...) = 0"""
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def griewank(x: np.ndarray) -> float:
    """Griewank function. Global min: f(0,...,0) = 0"""
    d = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return sum_term - prod_term + 1

def levy(x: np.ndarray) -> float:
    """Levy function. Global min: f(1,...,1) = 0"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz(x: np.ndarray, m: float = 10) -> float:
    """Michalewicz function. Optimum depends on dimension."""
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2*m))

def drop_wave(x: np.ndarray) -> float:
    """Drop-Wave function (2D). Global min: f(0,0) = -1"""
    norm_sq = x[0]**2 + x[1]**2
    return -(1 + np.cos(12 * np.sqrt(norm_sq))) / (0.5 * norm_sq + 2)

def shubert(x: np.ndarray) -> float:
    """Shubert function (2D). 18 global minima, f* ≈ -186.7309"""
    sum1 = sum(i * np.cos((i+1)*x[0] + i) for i in range(1, 6))
    sum2 = sum(i * np.cos((i+1)*x[1] + i) for i in range(1, 6))
    return sum1 * sum2

def eggholder(x: np.ndarray) -> float:
    """Eggholder function (2D). f(512, 404.2319) ≈ -959.6407"""
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + x[1] + 47))) - \
           x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))

def holder_table(x: np.ndarray) -> float:
    """Holder Table function (2D). f* ≈ -19.2085"""
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * 
                   np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

def langermann(x: np.ndarray) -> float:
    """Langermann function (2D). f* ≈ -5.1621"""
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    c = np.array([1, 2, 5, 2, 3])
    m = 5
    result = 0
    for i in range(m):
        inner = np.sum((x - A[i])**2)
        result += c[i] * np.exp(-inner/np.pi) * np.cos(np.pi * inner)
    return result

def schaffer_n2(x: np.ndarray) -> float:
    """Schaffer N.2 function (2D). f(0,0) = 0"""
    num = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + num / den

def schaffer_n4(x: np.ndarray) -> float:
    """Schaffer N.4 function (2D). f* ≈ 0.292579"""
    num = np.cos(np.sin(np.abs(x[0]**2 - x[1]**2)))**2 - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + num / den


# =============================================================================
# BOWL-SHAPED (Unimodal)
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function. f(0,...,0) = 0"""
    return np.sum(x**2)

def sum_squares(x: np.ndarray) -> float:
    """Sum Squares function. f(0,...,0) = 0"""
    i = np.arange(1, len(x) + 1)
    return np.sum(i * x**2)

def sum_different_powers(x: np.ndarray) -> float:
    """Sum of Different Powers. f(0,...,0) = 0"""
    d = len(x)
    i = np.arange(2, d + 2)
    return np.sum(np.abs(x)**i)

def rotated_hyper_ellipsoid(x: np.ndarray) -> float:
    """Rotated Hyper-Ellipsoid. f(0,...,0) = 0"""
    d = len(x)
    result = 0
    for i in range(d):
        result += np.sum(x[:i+1])**2
    return result

def trid(x: np.ndarray) -> float:
    """Trid function. Optimum depends on d."""
    d = len(x)
    return np.sum((x - 1)**2) - np.sum(x[1:] * x[:-1])

def perm_0_d_beta(x: np.ndarray, beta: float = 10) -> float:
    """Perm 0,d,β function. f(1,1/2,1/3,...,1/d) = 0"""
    d = len(x)
    result = 0
    for i in range(1, d + 1):
        inner = 0
        for j in range(1, d + 1):
            inner += (j + beta) * (x[j-1]**i - (1/j)**i)
        result += inner**2
    return result


# =============================================================================
# PLATE-SHAPED
# =============================================================================

def booth(x: np.ndarray) -> float:
    """Booth function (2D). f(1,3) = 0"""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def matyas(x: np.ndarray) -> float:
    """Matyas function (2D). f(0,0) = 0"""
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def mccormick(x: np.ndarray) -> float:
    """McCormick function (2D). f(-0.54719,-1.54719) ≈ -1.9133"""
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def zakharov(x: np.ndarray) -> float:
    """Zakharov function. f(0,...,0) = 0"""
    d = len(x)
    i = np.arange(1, d + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


# =============================================================================
# VALLEY-SHAPED
# =============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. f(1,...,1) = 0"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def dixon_price(x: np.ndarray) -> float:
    """Dixon-Price function. Optimum depends on d."""
    d = len(x)
    term1 = (x[0] - 1)**2
    i = np.arange(2, d + 1)
    term2 = np.sum(i * (2*x[1:]**2 - x[:-1])**2)
    return term1 + term2

def six_hump_camel(x: np.ndarray) -> float:
    """Six-Hump Camel function (2D). f* ≈ -1.0316"""
    return (4 - 2.1*x[0]**2 + x[0]**4/3) * x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2

def three_hump_camel(x: np.ndarray) -> float:
    """Three-Hump Camel function (2D). f(0,0) = 0"""
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


# =============================================================================
# STEEP RIDGES/DROPS
# =============================================================================

def easom(x: np.ndarray) -> float:
    """Easom function (2D). f(π,π) = -1"""
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))

def de_jong_n5(x: np.ndarray) -> float:
    """De Jong N.5 / Shekel's Foxholes (2D). f* ≈ 0.998"""
    a = np.array([[-32,-16,0,16,32]*5, [-32]*5+[-16]*5+[0]*5+[16]*5+[32]*5])
    result = 0.002
    for i in range(25):
        inner = i + 1 + (x[0] - a[0,i])**6 + (x[1] - a[1,i])**6
        result += 1 / inner
    return 1 / result


# =============================================================================
# OTHER CLASSIC FUNCTIONS
# =============================================================================

def beale(x: np.ndarray) -> float:
    """Beale function (2D). f(3, 0.5) = 0"""
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def branin(x: np.ndarray, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)) -> float:
    """Branin function (2D). 3 global minima, f* ≈ 0.397887"""
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def goldstein_price(x: np.ndarray) -> float:
    """Goldstein-Price function (2D). f(0,-1) = 3"""
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

def hartmann_3d(x: np.ndarray) -> float:
    """Hartmann 3D function. f* ≈ -3.86278"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
                  [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    result = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        result -= alpha[i] * np.exp(-inner)
    return result

def hartmann_6d(x: np.ndarray) -> float:
    """Hartmann 6D function. f* ≈ -3.32237"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    result = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        result -= alpha[i] * np.exp(-inner)
    return result

def powell(x: np.ndarray) -> float:
    """Powell function. f(0,...,0) = 0. Dimension must be multiple of 4."""
    d = len(x)
    result = 0
    for i in range(d // 4):
        idx = 4 * i
        result += (x[idx] + 10*x[idx+1])**2
        result += 5 * (x[idx+2] - x[idx+3])**2
        result += (x[idx+1] - 2*x[idx+2])**4
        result += 10 * (x[idx] - x[idx+3])**4
    return result

def styblinski_tang(x: np.ndarray) -> float:
    """Styblinski-Tang function. f(-2.9035,...) ≈ -39.16599*d"""
    return np.sum(x**4 - 16*x**2 + 5*x) / 2

def colville(x: np.ndarray) -> float:
    """Colville function (4D). f(1,1,1,1) = 0"""
    return (100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + (x[2] - 1)**2 + 
            90*(x[2]**2 - x[3])**2 + 10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 
            19.8*(x[1] - 1)*(x[3] - 1))

def shekel(x: np.ndarray, m: int = 10) -> float:
    """Shekel function (4D). f* depends on m."""
    beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])[:m]
    C = np.array([[4,1,8,6,3,2,5,8,6,7], [4,1,8,6,7,9,3,1,2,3.6],
                  [4,1,8,6,3,2,5,8,6,7], [4,1,8,6,7,9,3,1,2,3.6]])[:,:m]
    result = 0
    for i in range(m):
        inner = np.sum((x - C[:,i])**2) + beta[i]
        result -= 1 / inner
    return result


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

def get_all_functions() -> Dict[str, TestFunction]:
    """Get dictionary of all test functions with metadata."""
    
    functions = {}
    
    # Multimodal functions
    for d in [2, 5, 10, 20, 50]:
        functions[f'ackley_{d}d'] = TestFunction(
            name=f'Ackley-{d}D', func=ackley,
            bounds=np.array([[-32.768, 32.768]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='multimodal', description='Many local minima with deep basin'
        )
        functions[f'rastrigin_{d}d'] = TestFunction(
            name=f'Rastrigin-{d}D', func=rastrigin,
            bounds=np.array([[-5.12, 5.12]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='multimodal', description='Highly multimodal with regular structure'
        )
        functions[f'griewank_{d}d'] = TestFunction(
            name=f'Griewank-{d}D', func=griewank,
            bounds=np.array([[-600, 600]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='multimodal', description='Product term coupling'
        )
        functions[f'levy_{d}d'] = TestFunction(
            name=f'Levy-{d}D', func=levy,
            bounds=np.array([[-10, 10]] * d),
            dim=d, optimum=0.0, optimum_location=np.ones(d),
            category='multimodal', description='Many local minima'
        )
        functions[f'schwefel_{d}d'] = TestFunction(
            name=f'Schwefel-{d}D', func=schwefel,
            bounds=np.array([[-500, 500]] * d),
            dim=d, optimum=0.0, optimum_location=np.full(d, 420.9687),
            category='multimodal', description='Deceptive with distant optimum'
        )
    
    # Unimodal functions (various dims)
    for d in [2, 5, 10, 20, 50, 100]:
        functions[f'sphere_{d}d'] = TestFunction(
            name=f'Sphere-{d}D', func=sphere,
            bounds=np.array([[-5.12, 5.12]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='unimodal', description='Simple quadratic bowl'
        )
        functions[f'rosenbrock_{d}d'] = TestFunction(
            name=f'Rosenbrock-{d}D', func=rosenbrock,
            bounds=np.array([[-5, 10]] * d),
            dim=d, optimum=0.0, optimum_location=np.ones(d),
            category='valley', description='Narrow curved valley'
        )
        functions[f'sum_squares_{d}d'] = TestFunction(
            name=f'SumSquares-{d}D', func=sum_squares,
            bounds=np.array([[-10, 10]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='unimodal', description='Weighted quadratic'
        )
        functions[f'zakharov_{d}d'] = TestFunction(
            name=f'Zakharov-{d}D', func=zakharov,
            bounds=np.array([[-5, 10]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='plate', description='Plate-shaped'
        )
        functions[f'dixon_price_{d}d'] = TestFunction(
            name=f'DixonPrice-{d}D', func=dixon_price,
            bounds=np.array([[-10, 10]] * d),
            dim=d, optimum=0.0, optimum_location=None,
            category='valley', description='Valley-shaped'
        )
        functions[f'styblinski_tang_{d}d'] = TestFunction(
            name=f'StyblinskiTang-{d}D', func=styblinski_tang,
            bounds=np.array([[-5, 5]] * d),
            dim=d, optimum=-39.16599*d, optimum_location=np.full(d, -2.903534),
            category='multimodal', description='Many local minima'
        )
    
    # Fixed dimension functions (2D)
    functions['booth_2d'] = TestFunction(
        name='Booth-2D', func=booth, bounds=np.array([[-10, 10], [-10, 10]]),
        dim=2, optimum=0.0, optimum_location=np.array([1, 3]),
        category='plate', description='Plate-shaped'
    )
    functions['beale_2d'] = TestFunction(
        name='Beale-2D', func=beale, bounds=np.array([[-4.5, 4.5], [-4.5, 4.5]]),
        dim=2, optimum=0.0, optimum_location=np.array([3, 0.5]),
        category='multimodal', description='Sharp ridge'
    )
    functions['branin_2d'] = TestFunction(
        name='Branin-2D', func=branin, bounds=np.array([[-5, 10], [0, 15]]),
        dim=2, optimum=0.397887, optimum_location=np.array([np.pi, 2.275]),
        category='multimodal', description='3 global minima'
    )
    functions['goldstein_price_2d'] = TestFunction(
        name='GoldsteinPrice-2D', func=goldstein_price,
        bounds=np.array([[-2, 2], [-2, 2]]),
        dim=2, optimum=3.0, optimum_location=np.array([0, -1]),
        category='multimodal', description='Flat regions with sharp peaks'
    )
    functions['six_hump_camel_2d'] = TestFunction(
        name='SixHumpCamel-2D', func=six_hump_camel,
        bounds=np.array([[-3, 3], [-2, 2]]),
        dim=2, optimum=-1.0316, optimum_location=np.array([0.0898, -0.7126]),
        category='valley', description='6 local minima, 2 global'
    )
    functions['easom_2d'] = TestFunction(
        name='Easom-2D', func=easom, bounds=np.array([[-100, 100], [-100, 100]]),
        dim=2, optimum=-1.0, optimum_location=np.array([np.pi, np.pi]),
        category='steep', description='Flat with steep drop'
    )
    functions['eggholder_2d'] = TestFunction(
        name='Eggholder-2D', func=eggholder,
        bounds=np.array([[-512, 512], [-512, 512]]),
        dim=2, optimum=-959.6407, optimum_location=np.array([512, 404.2319]),
        category='multimodal', description='Many local minima'
    )
    functions['holder_table_2d'] = TestFunction(
        name='HolderTable-2D', func=holder_table,
        bounds=np.array([[-10, 10], [-10, 10]]),
        dim=2, optimum=-19.2085, optimum_location=np.array([8.05502, 9.66459]),
        category='multimodal', description='4 identical global minima'
    )
    
    # Fixed dimension (3D, 4D, 6D)
    functions['hartmann_3d'] = TestFunction(
        name='Hartmann-3D', func=hartmann_3d,
        bounds=np.array([[0, 1], [0, 1], [0, 1]]),
        dim=3, optimum=-3.86278, optimum_location=np.array([0.114614, 0.555649, 0.852547]),
        category='multimodal', description='4 local minima'
    )
    functions['colville_4d'] = TestFunction(
        name='Colville-4D', func=colville,
        bounds=np.array([[-10, 10]] * 4),
        dim=4, optimum=0.0, optimum_location=np.ones(4),
        category='valley', description='Valley-shaped in 4D'
    )
    functions['shekel_4d'] = TestFunction(
        name='Shekel-4D', func=shekel,
        bounds=np.array([[0, 10]] * 4),
        dim=4, optimum=-10.5364, optimum_location=np.array([4, 4, 4, 4]),
        category='multimodal', description='10 local minima'
    )
    functions['hartmann_6d'] = TestFunction(
        name='Hartmann-6D', func=hartmann_6d,
        bounds=np.array([[0, 1]] * 6),
        dim=6, optimum=-3.32237, 
        optimum_location=np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
        category='multimodal', description='6 local minima'
    )
    
    # Powell (must be multiple of 4)
    for d in [4, 8, 12, 20, 40]:
        functions[f'powell_{d}d'] = TestFunction(
            name=f'Powell-{d}D', func=powell,
            bounds=np.array([[-4, 5]] * d),
            dim=d, optimum=0.0, optimum_location=np.zeros(d),
            category='valley', description='Sum of quartics'
        )
    
    return functions


def get_functions_by_category(category: str) -> Dict[str, TestFunction]:
    """Get functions filtered by category."""
    all_funcs = get_all_functions()
    return {k: v for k, v in all_funcs.items() if v.category == category}


def get_functions_by_dimension(min_dim: int, max_dim: int) -> Dict[str, TestFunction]:
    """Get functions filtered by dimension range."""
    all_funcs = get_all_functions()
    return {k: v for k, v in all_funcs.items() if min_dim <= v.dim <= max_dim}


if __name__ == '__main__':
    funcs = get_all_functions()
    print(f"Total test functions: {len(funcs)}")
    
    # Summary by category
    categories = {}
    for f in funcs.values():
        categories[f.category] = categories.get(f.category, 0) + 1
    
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Summary by dimension
    dims = {}
    for f in funcs.values():
        dims[f.dim] = dims.get(f.dim, 0) + 1
    
    print("\nBy dimension:")
    for d, count in sorted(dims.items()):
        print(f"  {d}D: {count}")
