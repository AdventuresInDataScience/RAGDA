"""
Example: Using Scipy-style API with RAGDA.

Shows minimize() and maximize() functions for array-based objectives.
"""

from ragda import minimize, maximize
import numpy as np


def main():
    print("=" * 70)
    print("RAGDA Example: Scipy-Style API")
    print("=" * 70)
    
    # Example 1: Minimize Rosenbrock
    print("\n1. Minimizing Rosenbrock function")
    print("   Global minimum: f(1, 1) = 0")
    
    def rosenbrock(x):
        """Classic optimization test function."""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    result = minimize(
        rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        options={'maxiter': 200, 'random_state': 42}
    )
    
    print(f"\n   Result:")
    print(f"   - Minimum value: {result.fun:.6f}")
    print(f"   - Location: x = {result.x}")
    print(f"   - Success: {result.success}")
    print(f"   - Function evals: {result.nfev}")
    
    # Example 2: Minimize Ackley (multimodal)
    print("\n2. Minimizing Ackley function (multimodal)")
    print("   Global minimum: f(0, 0) = 0")
    
    def ackley(x):
        """Highly multimodal test function."""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e
    
    result = minimize(
        ackley,
        bounds=[(-5, 5), (-5, 5)],
        options={'maxiter': 150, 'random_state': 42}
    )
    
    print(f"\n   Result:")
    print(f"   - Minimum value: {result.fun:.6f}")
    print(f"   - Location: x = {result.x}")
    
    # Example 3: Maximize a function
    print("\n3. Maximizing negative sphere")
    print("   Global maximum: f(0, 0) = 0")
    
    def neg_sphere(x):
        """Negative sphere - maximum at origin."""
        return -np.sum(x**2)
    
    result = maximize(
        neg_sphere,
        bounds=[(-5, 5), (-5, 5)],
        options={'maxiter': 100, 'random_state': 42}
    )
    
    print(f"\n   Result:")
    print(f"   - Maximum value: {result.fun:.6f}")
    print(f"   - Location: x = {result.x}")
    
    # Example 4: Higher dimensions
    print("\n4. High-dimensional sphere (20D)")
    
    def sphere(x):
        return np.sum(x**2)
    
    result = minimize(
        sphere,
        bounds=[(-5, 5)] * 20,
        options={'maxiter': 200, 'random_state': 42}
    )
    
    print(f"\n   Result:")
    print(f"   - Minimum value: {result.fun:.6f}")
    print(f"   - Dimensionality: {len(result.x)}D")
    print(f"   - All components near 0: {np.all(np.abs(result.x) < 1.0)}")


if __name__ == "__main__":
    main()
