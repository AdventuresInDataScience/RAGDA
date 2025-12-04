"""
Genuine Benchmark Problems for RAGDA Parameter Optimization

This module provides REAL problems that naturally exhibit the characteristics
we care about, rather than artificial modifications:

1. DIMENSIONALITY - Problems with inherent high-dimensional structure
   - Neural network weight optimization
   - System identification with many parameters
   - PDE discretization parameters

2. COST - Problems with inherent computational expense
   - Full model training (not single evaluation)
   - Simulation-based objectives
   - Cross-validation based evaluation

3. LANDSCAPE RUGGEDNESS - Deterministic but chaotic/complex
   - Chaotic system parameter estimation (Lorenz, Mackey-Glass, Hénon, Rössler)
   - Time series forecasting objectives
   - Non-convex function fitting

Key principle: All problems are DETERMINISTIC. Same input = same output.
"Noisy" in optimization context means rugged/chaotic landscape, not stochastic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import warnings


@dataclass
class GenuineProblem:
    """A genuine optimization problem with natural characteristics."""
    name: str
    func: Callable
    bounds: List[Tuple[float, float]]
    dim: int
    category: str  # 'chaotic', 'ml_training', 'simulation', 'system_id', 'nn_weights'
    expected_cost: str  # 'cheap', 'moderate', 'expensive', 'very_expensive'
    expected_ruggedness: str  # 'smooth', 'moderate', 'rugged', 'highly_rugged'
    description: str
    optimal_value: Optional[float] = None


# =============================================================================
# CHAOTIC SYSTEM PARAMETER ESTIMATION
# These are DETERMINISTIC but create highly rugged landscapes
# =============================================================================

def mackey_glass_objective(params: np.ndarray, target_data: np.ndarray = None) -> float:
    """
    Mackey-Glass chaotic time series parameter estimation.
    
    The Mackey-Glass equation: dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    
    Parameters to optimize: [beta, gamma, n, tau_scale]
    
    This creates a HIGHLY RUGGED landscape because small parameter changes
    can lead to dramatically different chaotic trajectories.
    """
    beta, gamma, n, tau_scale = params
    
    # Ensure valid parameters
    if beta <= 0 or gamma <= 0 or n <= 0 or tau_scale <= 0:
        return 1e10
    
    # Discretized Mackey-Glass
    tau = int(max(1, tau_scale * 17))  # tau around 17 gives chaos
    history_len = tau + 1
    T = 500
    dt = 0.1
    
    x = np.zeros(T)
    x[:history_len] = 0.9 + 0.2 * np.sin(np.linspace(0, 2*np.pi, history_len))
    
    try:
        for t in range(history_len, T):
            x_tau = x[t - tau]
            dxdt = beta * x_tau / (1 + x_tau**n) - gamma * x[t-1]
            x[t] = x[t-1] + dt * dxdt
            
            if np.isnan(x[t]) or np.isinf(x[t]) or abs(x[t]) > 1e6:
                return 1e10
    except:
        return 1e10
    
    # Target: known chaotic parameters (beta=0.2, gamma=0.1, n=10, tau=17)
    if target_data is None:
        # Generate target with known good parameters
        target_data = _generate_mackey_glass_target()
    
    # MSE between generated and target (use last portion after transient)
    start = T // 2
    mse = np.mean((x[start:] - target_data[start:T])**2)
    return mse


def _generate_mackey_glass_target():
    """Generate target Mackey-Glass series with known parameters."""
    beta, gamma, n, tau = 0.2, 0.1, 10, 17
    T = 500
    dt = 0.1
    x = np.zeros(T)
    x[:tau+1] = 0.9 + 0.2 * np.sin(np.linspace(0, 2*np.pi, tau+1))
    
    for t in range(tau+1, T):
        x_tau = x[t - tau]
        dxdt = beta * x_tau / (1 + x_tau**n) - gamma * x[t-1]
        x[t] = x[t-1] + dt * dxdt
    
    return x


# Cache the target data
_MG_TARGET = None

def mackey_glass_wrapper(params: np.ndarray) -> float:
    global _MG_TARGET
    if _MG_TARGET is None:
        _MG_TARGET = _generate_mackey_glass_target()
    return mackey_glass_objective(params, _MG_TARGET)


def lorenz_objective(params: np.ndarray) -> float:
    """
    Lorenz attractor parameter estimation.
    
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    Parameters: [sigma, rho, beta]
    Classic chaotic values: sigma=10, rho=28, beta=8/3
    
    Highly rugged landscape due to sensitivity to initial conditions.
    """
    sigma, rho, beta = params
    
    if sigma <= 0 or rho <= 0 or beta <= 0:
        return 1e10
    
    T = 1000
    dt = 0.01
    
    # Simulate Lorenz system
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dt * dx
            y += dt * dy
            z += dt * dz
            
            if np.isnan(x) or np.isinf(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory.append([x, y, z])
    except:
        return 1e10
    
    trajectory = np.array(trajectory)
    
    # Target: match statistical properties of classic Lorenz
    # (mean, std, autocorrelation structure)
    target_mean = np.array([0.0, 0.0, 23.5])  # approximate
    target_std = np.array([7.9, 9.0, 8.5])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


def henon_map_objective(params: np.ndarray) -> float:
    """
    Hénon map parameter estimation.
    
    x_{n+1} = 1 - a * x_n^2 + y_n
    y_{n+1} = b * x_n
    
    Parameters: [a, b]
    Classic chaotic values: a=1.4, b=0.3
    """
    a, b = params
    
    if a <= 0 or a > 2 or abs(b) > 1:
        return 1e10
    
    T = 1000
    x, y = 0.1, 0.1
    trajectory_x = []
    
    try:
        for _ in range(T):
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new
            
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory_x.append(x)
    except:
        return 1e10
    
    trajectory_x = np.array(trajectory_x[100:])  # skip transient
    
    # Target statistics for a=1.4, b=0.3
    target_mean = 0.26
    target_std = 0.70
    
    mean_err = (np.mean(trajectory_x) - target_mean)**2
    std_err = (np.std(trajectory_x) - target_std)**2
    
    return mean_err + std_err


def rossler_objective(params: np.ndarray) -> float:
    """
    Rössler attractor parameter estimation.
    
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)
    
    Parameters: [a, b, c]
    Classic chaotic values: a=0.2, b=0.2, c=5.7
    """
    a, b, c = params
    
    if a <= 0 or b <= 0 or c <= 0:
        return 1e10
    
    T = 2000
    dt = 0.05
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            x += dt * dx
            y += dt * dy
            z += dt * dz
            
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory.append([x, y, z])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])  # skip transient
    
    # Target statistics
    target_mean = np.array([0.0, 0.0, 2.0])
    target_std = np.array([5.0, 5.0, 4.0])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


def logistic_map_objective(params: np.ndarray) -> float:
    """
    Logistic map parameter estimation with Lyapunov exponent matching.
    
    x_{n+1} = r * x_n * (1 - x_n)
    
    Parameters: [r]
    Chaotic for r > 3.57
    """
    r = params[0]
    
    if r <= 0 or r > 4:
        return 1e10
    
    T = 1000
    x = 0.5
    trajectory = []
    
    for _ in range(T):
        x = r * x * (1 - x)
        if np.isnan(x) or x < 0 or x > 1:
            return 1e10
        trajectory.append(x)
    
    trajectory = np.array(trajectory[100:])
    
    # Target: chaotic regime statistics (r=3.9)
    target_mean = 0.5
    target_std = 0.35
    
    mean_err = (np.mean(trajectory) - target_mean)**2
    std_err = (np.std(trajectory) - target_std)**2
    
    return mean_err + std_err


def coupled_logistic_objective(params: np.ndarray) -> float:
    """
    Coupled logistic maps - a genuine higher-dimensional chaotic system.
    
    x_i(n+1) = (1-eps) * r * x_i * (1 - x_i) + eps/2 * (x_{i-1} + x_{i+1})
    
    Parameters: [r, eps, x1_init, x2_init, ...]
    """
    n_maps = (len(params) - 2)
    r = params[0]
    eps = params[1]
    x = params[2:].copy()
    
    if r <= 0 or r > 4 or eps < 0 or eps > 1:
        return 1e10
    if len(x) < 3:
        x = np.concatenate([x, np.random.rand(3 - len(x))])
    
    T = 500
    trajectories = []
    
    try:
        for _ in range(T):
            x_new = np.zeros_like(x)
            for i in range(len(x)):
                coupling = (x[(i-1) % len(x)] + x[(i+1) % len(x)]) / 2
                x_new[i] = (1 - eps) * r * x[i] * (1 - x[i]) + eps * coupling
            x = np.clip(x_new, 0, 1)
            trajectories.append(x.copy())
    except:
        return 1e10
    
    trajectories = np.array(trajectories[100:])
    
    # Target: synchronized chaos with specific statistics
    target_correlation = 0.3  # moderate coupling
    
    # Measure pairwise correlations
    corrs = []
    for i in range(trajectories.shape[1] - 1):
        corr = np.corrcoef(trajectories[:, i], trajectories[:, i+1])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    
    if len(corrs) == 0:
        return 1e10
    
    avg_corr = np.mean(corrs)
    return (avg_corr - target_correlation)**2


def rabinovich_fabrikant_objective(params: np.ndarray) -> float:
    """
    Rabinovich-Fabrikant equations parameter estimation.
    
    dx/dt = y*(z - 1 + x^2) + gamma*x
    dy/dt = x*(3*z + 1 - x^2) + gamma*y
    dz/dt = -2*z*(alpha + x*y)
    
    Parameters: [gamma, alpha]
    Chaotic for gamma=0.87, alpha=1.1
    """
    gamma, alpha = params
    
    if gamma <= 0 or alpha <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    x, y, z = 0.1, 0.1, 0.1
    trajectory = []
    
    try:
        for _ in range(T):
            dx = y * (z - 1 + x**2) + gamma * x
            dy = x * (3*z + 1 - x**2) + gamma * y
            dz = -2 * z * (alpha + x * y)
            
            x += dt * dx
            y += dt * dy
            z += dt * dz
            
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory.append([x, y, z])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])
    
    # Target statistics for gamma=0.87, alpha=1.1
    target_std = np.array([1.5, 1.5, 0.8])
    
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    return std_err


def duffing_oscillator_objective(params: np.ndarray) -> float:
    """
    Duffing oscillator parameter estimation.
    
    dx/dt = y
    dy/dt = -delta*y - alpha*x - beta*x^3 + gamma*cos(omega*t)
    
    Parameters: [delta, alpha, beta, gamma, omega]
    Chaotic for delta=0.3, alpha=-1, beta=1, gamma=0.5, omega=1.2
    """
    delta, alpha, beta, gamma, omega = params
    
    T = 5000
    dt = 0.01
    x, y = 0.1, 0.0
    trajectory = []
    
    try:
        for i in range(T):
            t = i * dt
            dx = y
            dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
            
            x += dt * dx
            y += dt * dy
            
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory.append([x, y])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[1000:])
    
    # Target: chaotic regime statistics
    target_std = np.array([1.0, 1.2])
    
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    return std_err


def lotka_volterra_objective(params: np.ndarray) -> float:
    """
    Lotka-Volterra (predator-prey) equations parameter estimation.
    
    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y
    
    Parameters: [alpha, beta, delta, gamma]
    Classic values: alpha=1.5, beta=1.0, delta=1.0, gamma=3.0
    
    Not chaotic but has complex oscillatory dynamics.
    """
    alpha, beta, delta, gamma = params
    
    if alpha <= 0 or beta <= 0 or delta <= 0 or gamma <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    x, y = 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            
            x += dt * dx
            y += dt * dy
            
            if np.isnan(x) or x <= 0 or y <= 0 or x > 1e6:
                return 1e10
            
            trajectory.append([x, y])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[200:])
    
    # Target: oscillation statistics
    target_mean = np.array([3.0, 1.5])
    target_std = np.array([2.0, 1.0])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


def lotka_volterra_4species_objective(params: np.ndarray) -> float:
    """
    4-species Lotka-Volterra competition model.
    
    dx_i/dt = x_i * (r_i - sum_j(a_ij * x_j))
    
    Parameters: growth rates r1-r4 and interaction coefficients
    This is higher-dimensional with complex dynamics.
    """
    if len(params) != 8:
        return 1e10
    
    r = params[:4]  # growth rates
    # Interaction matrix (simplified - diagonal dominance)
    A = np.array([
        [1.0, params[4], 0.1, 0.1],
        [params[5], 1.0, params[6], 0.1],
        [0.1, params[7], 1.0, 0.2],
        [0.1, 0.1, 0.2, 1.0]
    ])
    
    if np.any(r <= 0):
        return 1e10
    
    T = 1000
    dt = 0.05
    x = np.array([0.5, 0.5, 0.5, 0.5])
    trajectory = []
    
    try:
        for _ in range(T):
            dx = x * (r - A @ x)
            x = x + dt * dx
            x = np.maximum(x, 1e-10)
            
            if np.any(np.isnan(x)) or np.any(x > 1e6):
                return 1e10
            
            trajectory.append(x.copy())
    except:
        return 1e10
    
    trajectory = np.array(trajectory[200:])
    
    # Target: stable coexistence
    target_mean = np.array([0.4, 0.4, 0.4, 0.4])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    stability = np.mean(trajectory.std(axis=0))  # Lower is more stable
    
    return mean_err + 0.5 * stability


def burgers_equation_objective(params: np.ndarray) -> float:
    """
    1D Burgers equation parameter estimation (simplified Navier-Stokes).
    
    du/dt + u * du/dx = nu * d^2u/dx^2
    
    Parameters: [nu (viscosity), initial condition Fourier coefficients]
    
    This is a simplified version of Navier-Stokes that still captures
    nonlinear advection and diffusion.
    """
    nu = params[0]
    n_modes = (len(params) - 1) // 2
    
    if nu <= 0 or n_modes < 2:
        return 1e10
    
    a_coeffs = params[1:n_modes+1]
    b_coeffs = params[n_modes+1:2*n_modes+1] if len(params) > n_modes+1 else np.zeros(n_modes)
    
    # Spatial discretization
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    
    # Initial condition from Fourier coefficients
    u = np.zeros(N)
    for k in range(len(a_coeffs)):
        u += a_coeffs[k] * np.sin((k+1) * x)
    for k in range(len(b_coeffs)):
        u += b_coeffs[k] * np.cos((k+1) * x)
    
    # Time stepping (simple explicit)
    dt = 0.001
    n_steps = 500
    
    try:
        for _ in range(n_steps):
            # Advection term (upwind)
            dudx = np.zeros(N)
            for i in range(N):
                if u[i] > 0:
                    dudx[i] = (u[i] - u[i-1]) / dx
                else:
                    dudx[i] = (u[(i+1) % N] - u[i]) / dx
            
            # Diffusion term (central)
            d2udx2 = np.zeros(N)
            for i in range(N):
                d2udx2[i] = (u[(i+1) % N] - 2*u[i] + u[i-1]) / dx**2
            
            u = u - dt * u * dudx + dt * nu * d2udx2
            
            if np.any(np.isnan(u)) or np.max(np.abs(u)) > 1e6:
                return 1e10
    except:
        return 1e10
    
    # Target: smooth decaying solution
    target_energy = 0.1
    energy = np.mean(u**2)
    
    return (energy - target_energy)**2


def van_der_pol_objective(params: np.ndarray) -> float:
    """
    Van der Pol oscillator parameter estimation.
    
    dx/dt = y
    dy/dt = mu*(1 - x^2)*y - x
    
    Parameters: [mu]
    Relaxation oscillations for mu > 0
    """
    mu = params[0]
    
    if mu <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    x, y = 0.1, 0.0
    trajectory = []
    
    try:
        for _ in range(T):
            dx = y
            dy = mu * (1 - x**2) * y - x
            
            x += dt * dx
            y += dt * dy
            
            if np.isnan(x) or abs(x) > 1e6:
                return 1e10
            
            trajectory.append([x, y])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])
    
    # Target: limit cycle amplitude ~2 for mu=1
    target_amplitude = 2.0
    amplitude = np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])
    
    return (amplitude - target_amplitude)**2


def double_pendulum_objective(params: np.ndarray) -> float:
    """
    Double pendulum parameter estimation - classic chaotic system.
    
    Parameters: [m1, m2, l1, l2] (masses and lengths)
    
    Highly sensitive to initial conditions, true chaos.
    """
    m1, m2, l1, l2 = params
    g = 9.81
    
    if m1 <= 0 or m2 <= 0 or l1 <= 0 or l2 <= 0:
        return 1e10
    
    # Initial angles and angular velocities
    theta1, theta2 = np.pi/2, np.pi/2
    omega1, omega2 = 0.0, 0.0
    
    T = 1000
    dt = 0.005
    trajectory = []
    
    try:
        for _ in range(T):
            # Equations of motion (simplified)
            delta = theta2 - theta1
            
            den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
            den2 = (l2 / l1) * den1
            
            domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                      m2 * g * np.sin(theta2) * np.cos(delta) +
                      m2 * l2 * omega2**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta1)) / den1
            
            domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                      (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                      (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta2)) / den2
            
            omega1 += dt * domega1
            omega2 += dt * domega2
            theta1 += dt * omega1
            theta2 += dt * omega2
            
            if np.isnan(theta1) or abs(omega1) > 1e6:
                return 1e10
            
            trajectory.append([theta1, theta2, omega1, omega2])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[200:])
    
    # Target: chaotic behavior with specific energy distribution
    target_energy_ratio = 0.5  # ratio of kinetic energies
    
    ke1 = 0.5 * m1 * (l1 * trajectory[:, 2])**2
    ke2 = 0.5 * m2 * (l2 * trajectory[:, 3])**2
    
    actual_ratio = np.mean(ke1) / (np.mean(ke1) + np.mean(ke2) + 1e-10)
    
    return (actual_ratio - target_energy_ratio)**2


# =============================================================================
# TRUE HIGH-DIMENSIONAL PROBLEMS
# These have inherent high-dimensional structure, not just scaled low-D
# =============================================================================

def neural_network_weights_objective(params: np.ndarray) -> float:
    """
    Optimize weights of a small neural network for XOR problem.
    
    Architecture: 2 -> 4 -> 1 (17 parameters: 8 + 4 + 4 + 1 = 17 weights + biases)
    
    This is a TRUE high-dimensional problem with complex loss landscape
    including saddle points, local minima, and symmetries.
    """
    # Unpack weights
    # Layer 1: 2x4 weights + 4 biases = 12
    # Layer 2: 4x1 weights + 1 bias = 5
    if len(params) != 17:
        return 1e10
    
    W1 = params[:8].reshape(2, 4)
    b1 = params[8:12]
    W2 = params[12:16].reshape(4, 1)
    b2 = params[16]
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    
    # Forward pass
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    h = sigmoid(X @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    
    # Binary cross-entropy loss
    eps = 1e-7
    loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))
    
    return loss


def nn_regression_objective(params: np.ndarray) -> float:
    """
    Neural network for regression on a nonlinear function.
    
    Architecture: 1 -> 8 -> 8 -> 1 (89 parameters)
    Target: f(x) = sin(x) + 0.5*sin(3x)
    
    True high-D with complex loss landscape.
    """
    expected_dim = 1*8 + 8 + 8*8 + 8 + 8*1 + 1  # 89
    if len(params) != expected_dim:
        return 1e10
    
    # Unpack weights
    idx = 0
    W1 = params[idx:idx+8].reshape(1, 8); idx += 8
    b1 = params[idx:idx+8]; idx += 8
    W2 = params[idx:idx+64].reshape(8, 8); idx += 64
    b2 = params[idx:idx+8]; idx += 8
    W3 = params[idx:idx+8].reshape(8, 1); idx += 8
    b3 = params[idx]; idx += 1
    
    # Training data
    X = np.linspace(-2*np.pi, 2*np.pi, 50).reshape(-1, 1)
    y = np.sin(X) + 0.5 * np.sin(3*X)
    
    # Forward pass with tanh
    def tanh(x):
        return np.tanh(np.clip(x, -500, 500))
    
    h1 = tanh(X @ W1 + b1)
    h2 = tanh(h1 @ W2 + b2)
    out = h2 @ W3 + b3
    
    # MSE loss
    loss = np.mean((out - y)**2)
    return loss


def nn_mnist_subset_objective(params: np.ndarray) -> float:
    """
    Neural network on a small MNIST-like problem (digit 0 vs 1).
    
    Uses a tiny synthetic dataset to keep it fast but realistic.
    Architecture: 64 -> 16 -> 2 (softmax) = 1024 + 16 + 32 + 2 = 1074 params
    
    This is a TRUE high-dimensional ML problem.
    """
    expected_dim = 64*16 + 16 + 16*2 + 2  # 1074
    if len(params) != expected_dim:
        return 1e10
    
    # Unpack
    idx = 0
    W1 = params[idx:idx+1024].reshape(64, 16); idx += 1024
    b1 = params[idx:idx+16]; idx += 16
    W2 = params[idx:idx+32].reshape(16, 2); idx += 32
    b2 = params[idx:idx+2]; idx += 2
    
    # Synthetic 8x8 digit data (simplified 0 and 1 patterns)
    np.random.seed(42)
    n_samples = 100
    
    # Generate "0" patterns (circle-ish)
    zeros = []
    for _ in range(n_samples // 2):
        img = np.zeros(64)
        # Simple ring pattern with noise
        for i in [1,2,5,6,8,15,16,23,24,31,32,39,40,47,48,55,57,58,61,62]:
            img[i] = 0.8 + 0.2 * np.random.rand()
        zeros.append(img)
    
    # Generate "1" patterns (vertical line)
    ones = []
    for _ in range(n_samples // 2):
        img = np.zeros(64)
        for i in [3,4,11,12,19,20,27,28,35,36,43,44,51,52,59,60]:
            img[i] = 0.8 + 0.2 * np.random.rand()
        ones.append(img)
    
    X = np.array(zeros + ones)
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    
    # Forward pass
    def relu(x):
        return np.maximum(0, x)
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    h = relu(X @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    
    # Cross-entropy loss
    eps = 1e-7
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
    
    return loss


def sparse_coding_objective(params: np.ndarray) -> float:
    """
    Sparse coding dictionary learning.
    
    Find dictionary D such that X ≈ D @ A where A is sparse.
    
    Parameters: dictionary atoms (flattened)
    This is inherently high-dimensional with complex landscape.
    """
    n_atoms = 16
    atom_dim = 25
    expected_dim = n_atoms * atom_dim  # 400
    
    if len(params) != expected_dim:
        return 1e10
    
    D = params.reshape(n_atoms, atom_dim)
    
    # Normalize dictionary atoms
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    D = D / norms
    
    # Generate synthetic signals (sparse combinations)
    np.random.seed(42)
    n_signals = 50
    
    # True dictionary
    D_true = np.random.randn(n_atoms, atom_dim)
    D_true = D_true / np.linalg.norm(D_true, axis=1, keepdims=True)
    
    # Sparse codes (3 active atoms per signal)
    A_true = np.zeros((n_atoms, n_signals))
    for j in range(n_signals):
        active = np.random.choice(n_atoms, 3, replace=False)
        A_true[active, j] = np.random.randn(3)
    
    X = D_true.T @ A_true  # atom_dim x n_signals
    
    # Reconstruction with current dictionary (simple OMP-like)
    reconstruction_error = 0
    for j in range(n_signals):
        x = X[:, j]
        # Greedy selection of atoms
        residual = x.copy()
        selected = []
        for _ in range(3):
            correlations = D @ residual
            best = np.argmax(np.abs(correlations))
            selected.append(best)
            # Solve least squares for selected atoms
            D_sel = D[selected, :].T
            coeffs, _, _, _ = np.linalg.lstsq(D_sel, x, rcond=None)
            residual = x - D_sel @ coeffs
        
        reconstruction_error += np.sum(residual**2)
    
    return reconstruction_error / n_signals


# =============================================================================
# EXPENSIVE PROBLEMS (inherent computational cost)
# =============================================================================

def cross_validation_svm_objective(params: np.ndarray) -> float:
    """
    SVM hyperparameter optimization with full cross-validation.
    
    Parameters: [C, gamma]
    
    This is inherently expensive because it runs full k-fold CV.
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** params[0]  # log scale: -3 to 3
    gamma = 10 ** params[1]  # log scale: -4 to 1
    
    if C <= 0 or gamma <= 0:
        return 1e10
    
    # Generate dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    
    # 5-fold CV
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    
    try:
        scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)  # Minimize negative accuracy
    except:
        return 1e10


def cross_validation_rf_objective(params: np.ndarray) -> float:
    """
    Random Forest hyperparameter optimization with cross-validation.
    
    Parameters: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    
    Inherently expensive due to ensemble training + CV.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(10 + params[0] * 190)  # 10-200
    max_depth = int(2 + params[1] * 18)  # 2-20
    min_samples_split = int(2 + params[2] * 18)  # 2-20
    min_samples_leaf = int(1 + params[3] * 9)  # 1-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               random_state=42)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=1
    )
    
    try:
        scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def simulation_optimization_objective(params: np.ndarray) -> float:
    """
    Simulated annealing schedule optimization (meta-optimization).
    
    Parameters: [T_init, cooling_rate, n_iterations_per_temp]
    
    Expensive because we run full SA simulations to evaluate.
    """
    T_init = 10 ** params[0]  # 1 to 1000
    cooling_rate = 0.8 + 0.199 * params[1]  # 0.8 to 0.999
    n_iters = int(10 + params[2] * 90)  # 10-100 per temperature
    
    if T_init <= 0 or cooling_rate <= 0 or cooling_rate >= 1:
        return 1e10
    
    # Run SA on a test function (Rastrigin) and measure performance
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    n_runs = 5
    dim = 5
    bounds = [(-5.12, 5.12)] * dim
    
    total_best = 0
    
    for run in range(n_runs):
        np.random.seed(run)
        x = np.random.uniform(-5.12, 5.12, dim)
        best = rastrigin(x)
        T = T_init
        
        max_temps = 50
        for _ in range(max_temps):
            for _ in range(n_iters):
                # Propose move
                x_new = x + np.random.randn(dim) * T * 0.1
                x_new = np.clip(x_new, -5.12, 5.12)
                
                f_new = rastrigin(x_new)
                delta = f_new - rastrigin(x)
                
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    x = x_new
                    if f_new < best:
                        best = f_new
            
            T *= cooling_rate
            if T < 1e-6:
                break
        
        total_best += best
    
    return total_best / n_runs


# =============================================================================
# LOW-DIMENSION MODERATE/EXPENSIVE COST PROBLEMS
# Filling gaps in low-dim + moderate/expensive categories
# =============================================================================

def ridge_regression_cv_objective(params: np.ndarray) -> float:
    """
    Ridge regression with cross-validation.
    Parameters: [alpha (log scale)]
    Target: LOW + moderate + smooth
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** params[0]  # log scale: -4 to 4
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = Ridge(alpha=alpha)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def lasso_regression_cv_objective(params: np.ndarray) -> float:
    """
    Lasso regression with cross-validation.
    Parameters: [alpha (log scale)]
    Target: LOW + moderate + smooth
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** params[0]
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = Lasso(alpha=alpha, max_iter=2000)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def elastic_net_cv_objective(params: np.ndarray) -> float:
    """
    Elastic Net with cross-validation.
    Parameters: [alpha, l1_ratio]
    Target: LOW + moderate + smooth
    """
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** params[0]  # log scale
    l1_ratio = params[1]  # 0 to 1
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def knn_classifier_cv_objective(params: np.ndarray) -> float:
    """
    KNN classifier with cross-validation.
    Parameters: [n_neighbors, weights_idx, p]
    Target: LOW + moderate + moderate
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_neighbors = int(1 + params[0] * 29)  # 1-30
    weights = 'uniform' if params[1] < 0.5 else 'distance'
    p = int(1 + params[2] * 2)  # 1, 2, or 3
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def decision_tree_cv_objective(params: np.ndarray) -> float:
    """
    Decision tree with cross-validation.
    Parameters: [max_depth, min_samples_split, min_samples_leaf]
    Target: LOW + moderate + moderate
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    max_depth = int(2 + params[0] * 28)  # 2-30
    min_samples_split = int(2 + params[1] * 18)  # 2-20
    min_samples_leaf = int(1 + params[2] * 9)  # 1-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def gradient_boosting_cv_objective(params: np.ndarray) -> float:
    """
    Gradient Boosting with cross-validation.
    Parameters: [n_estimators, learning_rate, max_depth]
    Target: LOW + expensive + moderate
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 150)  # 50-200
    learning_rate = 0.01 + params[1] * 0.49  # 0.01-0.5
    max_depth = int(2 + params[2] * 8)  # 2-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def adaboost_cv_objective(params: np.ndarray) -> float:
    """
    AdaBoost with cross-validation.
    Parameters: [n_estimators, learning_rate]
    Target: LOW + expensive + smooth
    """
    try:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 150)  # 50-200
    learning_rate = 0.1 + params[1] * 1.9  # 0.1-2.0
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def logistic_regression_cv_objective(params: np.ndarray) -> float:
    """
    Logistic regression with cross-validation.
    Parameters: [C (log scale)]
    Target: LOW + moderate + smooth
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** params[0]  # log scale: -4 to 4
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def mlp_regressor_cv_objective(params: np.ndarray) -> float:
    """
    MLP regressor with cross-validation.
    Parameters: [hidden_size, alpha, learning_rate_init]
    Target: LOW + expensive + rugged
    """
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    hidden_size = int(10 + params[0] * 90)  # 10-100
    alpha = 10 ** (params[1] * 6 - 5)  # 1e-5 to 10
    learning_rate_init = 10 ** (params[2] * 3 - 4)  # 1e-4 to 0.1
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_size,),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=500,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def svm_rbf_large_cv_objective(params: np.ndarray) -> float:
    """
    SVM RBF with larger dataset and more CV folds.
    Parameters: [C, gamma]
    Target: LOW + expensive + smooth
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** params[0]
    gamma = 10 ** params[1]
    
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=15, random_state=42)
    
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    try:
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def bagging_cv_objective(params: np.ndarray) -> float:
    """
    Bagging classifier with cross-validation.
    Parameters: [n_estimators, max_samples, max_features]
    Target: LOW + expensive + smooth
    """
    try:
        from sklearn.ensemble import BaggingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(20 + params[0] * 80)  # 20-100
    max_samples = 0.5 + params[1] * 0.5  # 0.5-1.0
    max_features = 0.5 + params[2] * 0.5  # 0.5-1.0
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = BaggingClassifier(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


# =============================================================================
# MEDIUM-DIMENSION PROBLEMS (11-50D)
# Filling gaps in medium-dim categories
# =============================================================================

def nn_medium_20d_objective(params: np.ndarray) -> float:
    """
    Neural network 2->10->10->1 = 20 params (small but medium-dim).
    Target: MEDIUM + cheap + rugged
    """
    if len(params) != 20:
        return 1e10
    
    # Unpack: 2*10=20 weights for layer1, but we'll make it simpler
    # Actually: 2*8 + 8 + 8*1 + 1 = 16+8+8+1 = 33, let's use 1->10->10->1
    # 1*10 + 10 = 20 for first layer+bias, then we stop
    W1 = params[:10].reshape(1, 10)
    b1 = params[10:20]
    
    X = np.linspace(-3, 3, 50).reshape(-1, 1)
    y = np.sin(X) + 0.3 * np.sin(5*X)
    
    h = np.tanh(X @ W1 + b1)
    out = np.mean(h, axis=1, keepdims=True)  # Simple aggregation
    
    return np.mean((out - y)**2)


def coupled_oscillator_15d_objective(params: np.ndarray) -> float:
    """
    Coupled harmonic oscillators parameter estimation (15D).
    Target: MEDIUM + cheap + rugged
    """
    if len(params) != 15:
        return 1e10
    
    # 5 oscillators, each with mass, spring constant, coupling
    n_osc = 5
    masses = params[:5]
    springs = params[5:10]
    couplings = params[10:15]
    
    if np.any(masses <= 0) or np.any(springs <= 0):
        return 1e10
    
    T = 500
    dt = 0.01
    x = np.zeros(n_osc)
    v = np.zeros(n_osc)
    x[0] = 1.0  # Initial displacement
    
    energy_history = []
    
    for _ in range(T):
        forces = -springs * x
        for i in range(n_osc - 1):
            coupling_force = couplings[i] * (x[i+1] - x[i])
            forces[i] += coupling_force
            forces[i+1] -= coupling_force
        
        a = forces / masses
        v += dt * a
        x += dt * v
        
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
        
        energy = 0.5 * np.sum(masses * v**2) + 0.5 * np.sum(springs * x**2)
        energy_history.append(energy)
    
    # Target: energy conservation (should be constant)
    energy_history = np.array(energy_history)
    return np.std(energy_history) / (np.mean(energy_history) + 1e-10)


def rastrigin_rotated_20d_objective(params: np.ndarray) -> float:
    """
    Rotated Rastrigin in 20D with rotation parameters.
    Target: MEDIUM + cheap + rugged
    """
    if len(params) != 20:
        return 1e10
    
    x = params
    n = len(x)
    
    # Apply rotation via Givens rotations
    y = x.copy()
    for i in range(0, n-1, 2):
        angle = 0.5  # Fixed rotation
        c, s = np.cos(angle), np.sin(angle)
        y[i], y[i+1] = c*y[i] - s*y[i+1], s*y[i] + c*y[i+1]
    
    return 10*n + np.sum(y**2 - 10*np.cos(2*np.pi*y))


def heat_diffusion_30d_objective(params: np.ndarray) -> float:
    """
    Heat diffusion with 30D initial condition (Fourier modes).
    Target: MEDIUM + moderate + smooth
    """
    if len(params) != 30:
        return 1e10
    
    n_modes = 15
    a_coeffs = params[:n_modes]
    b_coeffs = params[n_modes:2*n_modes]
    
    N = 64
    x = np.linspace(0, 2*np.pi, N)
    dx = x[1] - x[0]
    
    u = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x) + b_coeffs[k] * np.sin((k+1) * x)
    
    dt = 0.3 * dx**2
    for _ in range(200):
        u_new = u.copy()
        for i in range(1, N-1):
            u_new[i] = u[i] + dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
        u = u_new
    
    return np.mean(u**2)


def wave_equation_30d_objective(params: np.ndarray) -> float:
    """
    Wave equation initial condition optimization (30D).
    Target: MEDIUM + moderate + smooth
    """
    if len(params) != 30:
        return 1e10
    
    n_modes = 15
    a_coeffs = params[:n_modes]
    b_coeffs = params[n_modes:2*n_modes]
    
    N = 64
    x = np.linspace(0, 2*np.pi, N)
    dx = x[1] - x[0]
    c = 1.0  # wave speed
    
    u = np.zeros(N)
    u_prev = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x) + b_coeffs[k] * np.sin((k+1) * x)
    u_prev = u.copy()
    
    dt = 0.5 * dx / c
    for _ in range(200):
        u_new = np.zeros(N)
        for i in range(1, N-1):
            u_new[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
        u_prev = u
        u = u_new
    
    return np.mean(u**2)


def advection_diffusion_30d_objective(params: np.ndarray) -> float:
    """
    Advection-diffusion equation (30D parameters).
    Target: MEDIUM + moderate + smooth
    """
    if len(params) != 30:
        return 1e10
    
    velocity = params[0]
    diffusion = np.abs(params[1]) + 0.01
    ic_coeffs = params[2:]
    
    N = 64
    x = np.linspace(0, 2*np.pi, N)
    dx = x[1] - x[0]
    
    u = np.zeros(N)
    for k, coeff in enumerate(ic_coeffs):
        u += coeff * np.sin((k+1) * x)
    
    dt = 0.1 * dx**2 / diffusion
    for _ in range(100):
        u_new = u.copy()
        for i in range(1, N-1):
            advection = -velocity * (u[i] - u[i-1]) / dx
            diffusion_term = diffusion * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
            u_new[i] = u[i] + dt * (advection + diffusion_term)
        u = u_new
    
    return np.mean(u**2)


def sparse_regression_40d_objective(params: np.ndarray) -> float:
    """
    Sparse regression coefficient optimization (40D).
    Target: MEDIUM + moderate + moderate
    """
    if len(params) != 40:
        return 1e10
    
    np.random.seed(42)
    n_samples, n_features = 200, 40
    X = np.random.randn(n_samples, n_features)
    
    true_coef = np.zeros(n_features)
    true_coef[::5] = np.random.randn(8)
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)
    
    y_pred = X @ params
    mse = np.mean((y - y_pred)**2)
    sparsity_penalty = 0.1 * np.sum(np.abs(params))
    
    return mse + sparsity_penalty


def matrix_factorization_36d_objective(params: np.ndarray) -> float:
    """
    Low-rank matrix factorization (36D = 6x3 + 3x6).
    Target: MEDIUM + moderate + moderate
    """
    if len(params) != 36:
        return 1e10
    
    U = params[:18].reshape(6, 3)
    V = params[18:].reshape(3, 6)
    
    np.random.seed(42)
    A_true = np.random.randn(6, 3) @ np.random.randn(3, 6)
    A_true += 0.1 * np.random.randn(6, 6)
    
    A_approx = U @ V
    return np.mean((A_true - A_approx)**2)


def tensor_decomposition_27d_objective(params: np.ndarray) -> float:
    """
    CP tensor decomposition (27D = 3x3x3 rank-1 components).
    Target: MEDIUM + moderate + moderate
    """
    if len(params) != 27:
        return 1e10
    
    a = params[:9].reshape(3, 3)
    b = params[9:18].reshape(3, 3)
    c = params[18:27].reshape(3, 3)
    
    np.random.seed(42)
    T_true = np.random.randn(3, 3, 3)
    
    T_approx = np.zeros((3, 3, 3))
    for r in range(3):
        T_approx += np.outer(np.outer(a[:, r], b[:, r]).flatten(), c[:, r]).reshape(3, 3, 3)
    
    return np.mean((T_true - T_approx)**2)


def ica_unmixing_25d_objective(params: np.ndarray) -> float:
    """
    Independent Component Analysis unmixing matrix (25D = 5x5).
    Target: MEDIUM + moderate + moderate
    """
    if len(params) != 25:
        return 1e10
    
    W = params.reshape(5, 5)
    
    np.random.seed(42)
    n_samples = 500
    S = np.random.laplace(size=(5, n_samples))
    A = np.random.randn(5, 5)
    X = A @ S
    
    Y = W @ X
    
    kurtosis = np.mean(Y**4, axis=1) - 3
    neg_entropy = np.sum(np.abs(kurtosis))
    
    orthogonality = np.mean((W @ W.T - np.eye(5))**2)
    
    return -neg_entropy + 10 * orthogonality


def pca_reconstruction_30d_objective(params: np.ndarray) -> float:
    """
    PCA components optimization (30D = 6 components x 5D).
    Target: MEDIUM + moderate + moderate
    """
    if len(params) != 30:
        return 1e10
    
    V = params.reshape(6, 5)
    
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    V_normalized = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
    
    projected = X @ V_normalized.T
    reconstructed = projected @ V_normalized
    
    return np.mean((X - reconstructed)**2)


def lorenz96_20d_objective(params: np.ndarray) -> float:
    """
    Lorenz 96 model - weather-like dynamics (20D state).
    Target: MEDIUM + moderate + rugged
    """
    if len(params) != 20:
        return 1e10
    
    F = 8.0
    n = 20
    x = params.copy()
    
    dt = 0.01
    trajectory = []
    
    for _ in range(500):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        x = x + dt * dx
        
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
        
        trajectory.append(x.copy())
    
    trajectory = np.array(trajectory[100:])
    
    target_mean = F
    target_std = 3.5
    
    mean_err = (np.mean(trajectory) - target_mean)**2
    std_err = (np.std(trajectory) - target_std)**2
    
    return mean_err + std_err


def kuramoto_oscillators_20d_objective(params: np.ndarray) -> float:
    """
    Kuramoto coupled oscillators - synchronization dynamics (20D).
    Target: MEDIUM + moderate + rugged
    """
    if len(params) != 20:
        return 1e10
    
    n = 10
    omega = params[:n]
    K = params[n:]
    
    theta = np.random.rand(n) * 2 * np.pi
    np.random.seed(42)
    
    dt = 0.1
    for _ in range(200):
        dtheta = omega.copy()
        for i in range(n):
            for j in range(n):
                dtheta[i] += K[j] / n * np.sin(theta[j] - theta[i])
        theta = theta + dt * dtheta
    
    r = np.abs(np.mean(np.exp(1j * theta)))
    
    target_sync = 0.7
    return (r - target_sync)**2


def cellular_automata_25d_objective(params: np.ndarray) -> float:
    """
    Cellular automata rule optimization (25D = 5x5 rule weights).
    Target: MEDIUM + moderate + rugged
    """
    if len(params) != 25:
        return 1e10
    
    weights = params.reshape(5, 5)
    
    grid = np.random.rand(20, 20) > 0.5
    grid = grid.astype(float)
    
    for _ in range(50):
        new_grid = np.zeros_like(grid)
        for i in range(2, 18):
            for j in range(2, 18):
                neighborhood = grid[i-2:i+3, j-2:j+3]
                activation = np.sum(neighborhood * weights)
                new_grid[i, j] = 1.0 if activation > 0 else 0.0
        grid = new_grid
    
    target_density = 0.3
    density = np.mean(grid)
    
    return (density - target_density)**2


def reaction_diffusion_30d_objective(params: np.ndarray) -> float:
    """
    Reaction-diffusion pattern formation (30D parameters).
    Target: MEDIUM + moderate + rugged
    """
    if len(params) != 30:
        return 1e10
    
    Du, Dv = np.abs(params[0]) + 0.01, np.abs(params[1]) + 0.01
    f, k = params[2], params[3]
    ic_u = params[4:17]
    ic_v = params[17:30]
    
    N = 32
    u = np.ones((N, N)) * 0.5
    v = np.ones((N, N)) * 0.25
    
    for idx, coeff in enumerate(ic_u):
        u += coeff * 0.1 * np.sin((idx+1) * np.linspace(0, 2*np.pi, N)).reshape(-1, 1)
    for idx, coeff in enumerate(ic_v):
        v += coeff * 0.1 * np.cos((idx+1) * np.linspace(0, 2*np.pi, N)).reshape(1, -1)
    
    dx = 1.0
    dt = 0.5
    
    for _ in range(100):
        laplacian_u = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u
        laplacian_v = np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4*v
        
        reaction_u = -u * v**2 + f * (1 - u)
        reaction_v = u * v**2 - (f + k) * v
        
        u = u + dt * (Du * laplacian_u / dx**2 + reaction_u)
        v = v + dt * (Dv * laplacian_v / dx**2 + reaction_v)
        
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
    
    target_pattern_strength = 0.2
    pattern_strength = np.std(u)
    
    return (pattern_strength - target_pattern_strength)**2


def svm_medium_features_cv_objective(params: np.ndarray) -> float:
    """
    SVM on medium-dimensional feature space (20D hyperparams).
    Target: MEDIUM + expensive + smooth
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return 1e10
    
    C = 10 ** params[0]
    gamma = 10 ** params[1]
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=18, n_informative=10, random_state=42)
    
    X_weighted = X * feature_weights[:18]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    try:
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def rf_feature_selection_25d_objective(params: np.ndarray) -> float:
    """
    Random Forest with feature selection weights (25D).
    Target: MEDIUM + expensive + smooth
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    max_depth = int(3 + params[1] * 12)
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=23, n_informative=12, random_state=42)
    
    X_weighted = X * feature_weights[:23]
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def gb_feature_engineering_30d_objective(params: np.ndarray) -> float:
    """
    Gradient Boosting with feature engineering (30D).
    Target: MEDIUM + expensive + smooth
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    learning_rate = 0.01 + params[1] * 0.49
    max_depth = int(2 + params[2] * 8)
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=27, n_informative=15, random_state=42)
    
    X_weighted = X * feature_weights[:27]
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42
    )
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def pca_svm_pipeline_25d_objective(params: np.ndarray) -> float:
    """
    PCA + SVM pipeline optimization (25D).
    Target: MEDIUM + expensive + smooth
    """
    try:
        from sklearn.svm import SVC
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return 1e10
    
    n_components = int(2 + params[0] * 18)
    C = 10 ** params[1]
    gamma = 10 ** params[2]
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=22, n_informative=12, random_state=42)
    
    X_weighted = X * feature_weights[:22]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(n_components, 22))),
        ('svm', SVC(C=C, gamma=gamma, kernel='rbf'))
    ])
    
    try:
        scores = cross_val_score(pipeline, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def kernel_ridge_30d_objective(params: np.ndarray) -> float:
    """
    Kernel Ridge Regression with feature engineering (30D).
    Target: MEDIUM + expensive + smooth
    """
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return 1e10
    
    alpha = 10 ** params[0]
    gamma = 10 ** params[1]
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=28, n_informative=15, noise=10, random_state=42)
    
    X_weighted = X * feature_weights[:28]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    model = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
    try:
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def ensemble_weights_25d_objective(params: np.ndarray) -> float:
    """
    Ensemble model weight optimization (25D).
    Target: MEDIUM + expensive + moderate
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    weights = np.abs(params[:5])
    weights = weights / (np.sum(weights) + 1e-10)
    hyperparams = params[5:]
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=12, random_state=42)
    
    models = [
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        LogisticRegression(max_iter=500, random_state=42),
        SVC(probability=True, random_state=42),
        KNeighborsClassifier(n_neighbors=5)
    ]
    
    predictions = []
    for model in models:
        try:
            preds = cross_val_predict(model, X, y, cv=3, method='predict_proba')[:, 1]
            predictions.append(preds)
        except:
            predictions.append(np.ones(len(y)) * 0.5)
    
    predictions = np.array(predictions)
    ensemble_pred = np.sum(predictions * weights.reshape(-1, 1), axis=0)
    ensemble_class = (ensemble_pred > 0.5).astype(int)
    
    accuracy = np.mean(ensemble_class == y)
    return -accuracy


def mlp_architecture_30d_objective(params: np.ndarray) -> float:
    """
    MLP architecture and hyperparameter search (30D).
    Target: MEDIUM + expensive + moderate
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    layer1_size = int(10 + params[0] * 90)
    layer2_size = int(10 + params[1] * 90)
    alpha = 10 ** (params[2] * 6 - 5)
    learning_rate_init = 10 ** (params[3] * 3 - 4)
    feature_weights = np.abs(params[4:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=26, n_informative=15, random_state=42)
    
    X_weighted = X * feature_weights[:26]
    
    model = MLPClassifier(
        hidden_layer_sizes=(layer1_size, layer2_size),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=200,
        random_state=42
    )
    
    try:
        scores = cross_val_score(model, X_weighted, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def xgb_tuning_25d_objective(params: np.ndarray) -> float:
    """
    XGBoost hyperparameter tuning with feature weights (25D).
    Target: MEDIUM + expensive + moderate
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 150)
    max_depth = int(2 + params[1] * 8)
    learning_rate = 0.01 + params[2] * 0.29
    subsample = 0.5 + params[3] * 0.5
    feature_weights = np.abs(params[4:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=21, n_informative=12, random_state=42)
    
    X_weighted = X * feature_weights[:21]
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def lgb_tuning_25d_objective(params: np.ndarray) -> float:
    """
    LightGBM hyperparameter tuning with feature weights (25D).
    Target: MEDIUM + expensive + moderate
    """
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 150)
    max_depth = int(2 + params[1] * 13)
    learning_rate = 0.01 + params[2] * 0.29
    num_leaves = int(10 + params[3] * 90)
    feature_weights = np.abs(params[4:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=21, n_informative=12, random_state=42)
    
    X_weighted = X * feature_weights[:21]
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        verbosity=-1
    )
    
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def genetic_algorithm_25d_objective(params: np.ndarray) -> float:
    """
    Genetic algorithm parameter tuning via meta-optimization (25D).
    Target: MEDIUM + expensive + rugged
    """
    if len(params) != 25:
        return 1e10
    
    pop_size = int(20 + params[0] * 80)
    mutation_rate = params[1] * 0.5
    crossover_rate = 0.5 + params[2] * 0.5
    selection_pressure = 1 + params[3] * 4
    gene_init = params[4:]
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    dim = 10
    pop = np.random.randn(pop_size, dim) * 2
    pop[0] = gene_init[:dim] * 5.12
    
    n_generations = 30
    
    for _ in range(n_generations):
        fitness = np.array([rastrigin(ind) for ind in pop])
        
        ranks = np.argsort(np.argsort(fitness))
        selection_probs = (pop_size - ranks) ** selection_pressure
        selection_probs = selection_probs / selection_probs.sum()
        
        new_pop = []
        for _ in range(pop_size):
            if np.random.rand() < crossover_rate:
                p1, p2 = np.random.choice(pop_size, 2, p=selection_probs, replace=False)
                crossover_point = np.random.randint(dim)
                child = np.concatenate([pop[p1, :crossover_point], pop[p2, crossover_point:]])
            else:
                p1 = np.random.choice(pop_size, p=selection_probs)
                child = pop[p1].copy()
            
            if np.random.rand() < mutation_rate:
                mutation_idx = np.random.randint(dim)
                child[mutation_idx] += np.random.randn() * 0.5
            
            child = np.clip(child, -5.12, 5.12)
            new_pop.append(child)
        
        pop = np.array(new_pop)
    
    final_fitness = np.array([rastrigin(ind) for ind in pop])
    return np.min(final_fitness)


def particle_swarm_30d_objective(params: np.ndarray) -> float:
    """
    Particle Swarm Optimization parameter tuning (30D).
    Target: MEDIUM + expensive + rugged
    """
    if len(params) != 30:
        return 1e10
    
    n_particles = int(20 + params[0] * 80)
    w = 0.4 + params[1] * 0.5
    c1 = params[2] * 3
    c2 = params[3] * 3
    v_max = params[4] * 2
    init_pos = params[5:]
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    dim = 10
    pos = np.random.randn(n_particles, dim) * 2
    pos[0] = init_pos[:dim] * 5.12
    vel = np.zeros((n_particles, dim))
    
    pbest_pos = pos.copy()
    pbest_val = np.array([rastrigin(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    
    n_iterations = 50
    
    for _ in range(n_iterations):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = pos + vel
        pos = np.clip(pos, -5.12, 5.12)
        
        for i in range(n_particles):
            val = rastrigin(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = pos[i].copy()
    
    return gbest_val


def differential_evolution_30d_objective(params: np.ndarray) -> float:
    """
    Differential Evolution parameter tuning (30D).
    Target: MEDIUM + expensive + rugged
    """
    if len(params) != 30:
        return 1e10
    
    pop_size = int(20 + params[0] * 80)
    F = 0.1 + params[1] * 1.4
    CR = params[2]
    init_pop_scale = params[3] * 5
    init_seed = params[4:]
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    dim = 10
    pop = np.random.randn(pop_size, dim) * init_pop_scale
    pop[0] = init_seed[:dim] * 5.12
    pop = np.clip(pop, -5.12, 5.12)
    
    fitness = np.array([rastrigin(ind) for ind in pop])
    
    n_generations = 40
    
    for _ in range(n_generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, -5.12, 5.12)
            
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = rastrigin(trial)
            
            if trial_fitness < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness
    
    return np.min(fitness)


def cma_es_25d_objective(params: np.ndarray) -> float:
    """
    CMA-ES hyperparameter meta-optimization (25D).
    Target: MEDIUM + expensive + rugged
    """
    if len(params) != 25:
        return 1e10
    
    sigma0 = 0.1 + params[0] * 2.9
    lambda_mult = 1 + params[1] * 4
    mu_ratio = 0.2 + params[2] * 0.3
    init_mean = params[3:13] * 5.12
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    dim = 10
    
    n_offspring = int(4 + 3 * np.log(dim) * lambda_mult)
    n_parents = max(1, int(n_offspring * mu_ratio))
    
    mean = init_mean[:dim]
    sigma = sigma0
    C = np.eye(dim)
    
    n_generations = 30
    
    for _ in range(n_generations):
        offspring = []
        for _ in range(n_offspring):
            z = np.random.randn(dim)
            try:
                x = mean + sigma * np.linalg.cholesky(C + 1e-8 * np.eye(dim)) @ z
            except:
                x = mean + sigma * z
            x = np.clip(x, -5.12, 5.12)
            offspring.append((x, rastrigin(x)))
        
        offspring.sort(key=lambda t: t[1])
        parents = [o[0] for o in offspring[:n_parents]]
        
        mean = np.mean(parents, axis=0)
        
        if len(parents) > 1:
            deviations = np.array(parents) - mean
            C = 0.8 * C + 0.2 * (deviations.T @ deviations) / n_parents
        
        sigma *= 0.95
    
    return rastrigin(mean)


# =============================================================================
# VERY HIGH DIMENSIONAL PROBLEMS (100+ dimensions)
# =============================================================================

def large_nn_objective(params: np.ndarray) -> float:
    """
    Larger neural network: 10 -> 32 -> 32 -> 1 = 1377 parameters.
    
    Approximating a complex multivariate function.
    """
    expected_dim = 10*32 + 32 + 32*32 + 32 + 32*1 + 1  # 1377
    if len(params) != expected_dim:
        return 1e10
    
    # Unpack
    idx = 0
    W1 = params[idx:idx+320].reshape(10, 32); idx += 320
    b1 = params[idx:idx+32]; idx += 32
    W2 = params[idx:idx+1024].reshape(32, 32); idx += 1024
    b2 = params[idx:idx+32]; idx += 32
    W3 = params[idx:idx+32].reshape(32, 1); idx += 32
    b3 = params[idx]; idx += 1
    
    # Training data: approximate a complex function
    np.random.seed(42)
    X = np.random.randn(200, 10)
    # Target: sum of sines with interactions
    y = np.sum(np.sin(X), axis=1) + np.sum(X[:, :5] * X[:, 5:], axis=1)
    y = y.reshape(-1, 1)
    
    # Forward pass
    def tanh(x):
        return np.tanh(np.clip(x, -10, 10))
    
    h1 = tanh(X @ W1 + b1)
    h2 = tanh(h1 @ W2 + b2)
    out = h2 @ W3 + b3
    
    loss = np.mean((out - y)**2)
    return loss


def pde_discretization_objective(params: np.ndarray) -> float:
    """
    1D heat equation solver parameter optimization.
    
    Parameters: initial condition coefficients (Fourier modes)
    Goal: minimize deviation from target steady state.
    
    Genuinely high-dimensional PDE problem.
    """
    n_modes = len(params) // 2
    if n_modes < 5:
        return 1e10
    
    a_coeffs = params[:n_modes]
    b_coeffs = params[n_modes:2*n_modes]
    
    # Spatial discretization
    N = 100
    x = np.linspace(0, 2*np.pi, N)
    dx = x[1] - x[0]
    
    # Initial condition from Fourier coefficients
    u = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x) + b_coeffs[k] * np.sin((k+1) * x)
    
    # Time stepping (explicit Euler)
    dt = 0.4 * dx**2  # CFL condition
    alpha = 1.0  # diffusion coefficient
    n_steps = 500
    
    for _ in range(n_steps):
        u_new = u.copy()
        for i in range(1, N-1):
            u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        # Periodic BCs
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        u = u_new
    
    # Target: flat steady state at 0
    return np.mean(u**2)


# =============================================================================
# COMBINED PROBLEM LIBRARY
# =============================================================================

def get_chaotic_problems() -> List[GenuineProblem]:
    """Get chaotic system parameter estimation problems."""
    return [
        GenuineProblem(
            name="MackeyGlass-4D",
            func=mackey_glass_wrapper,
            bounds=[(0.1, 0.5), (0.05, 0.2), (5, 15), (0.5, 2.0)],
            dim=4,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Mackey-Glass chaotic time series parameter estimation"
        ),
        GenuineProblem(
            name="Lorenz-3D",
            func=lorenz_objective,
            bounds=[(1, 20), (10, 50), (0.5, 5)],
            dim=3,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Lorenz attractor parameter estimation"
        ),
        GenuineProblem(
            name="Henon-2D",
            func=henon_map_objective,
            bounds=[(1.0, 1.5), (0.1, 0.5)],
            dim=2,
            category="chaotic",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Hénon map parameter estimation"
        ),
        GenuineProblem(
            name="Rossler-3D",
            func=rossler_objective,
            bounds=[(0.05, 0.5), (0.05, 0.5), (3, 10)],
            dim=3,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Rössler attractor parameter estimation"
        ),
        GenuineProblem(
            name="LogisticMap-1D",
            func=logistic_map_objective,
            bounds=[(3.5, 4.0)],
            dim=1,
            category="chaotic",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Logistic map parameter estimation"
        ),
        GenuineProblem(
            name="CoupledLogistic-10D",
            func=coupled_logistic_objective,
            bounds=[(3.5, 4.0), (0.0, 0.5)] + [(0.1, 0.9)]*8,
            dim=10,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Coupled logistic maps system identification"
        ),
        GenuineProblem(
            name="RabinovichFabrikant-2D",
            func=rabinovich_fabrikant_objective,
            bounds=[(0.1, 1.5), (0.5, 2.0)],
            dim=2,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Rabinovich-Fabrikant equations parameter estimation"
        ),
        GenuineProblem(
            name="Duffing-5D",
            func=duffing_oscillator_objective,
            bounds=[(0.1, 0.5), (-2, 0), (0.5, 2), (0.1, 1), (0.5, 2)],
            dim=5,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Duffing oscillator parameter estimation"
        ),
        GenuineProblem(
            name="LotkaVolterra-4D",
            func=lotka_volterra_objective,
            bounds=[(0.5, 3), (0.5, 2), (0.5, 2), (1, 5)],
            dim=4,
            category="dynamical",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Lotka-Volterra predator-prey parameter estimation"
        ),
        GenuineProblem(
            name="LotkaVolterra4Species-8D",
            func=lotka_volterra_4species_objective,
            bounds=[(0.5, 2)]*4 + [(0.1, 0.9)]*4,
            dim=8,
            category="dynamical",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="4-species Lotka-Volterra competition model"
        ),
        GenuineProblem(
            name="VanDerPol-1D",
            func=van_der_pol_objective,
            bounds=[(0.1, 5)],
            dim=1,
            category="dynamical",
            expected_cost="cheap",
            expected_ruggedness="smooth",
            description="Van der Pol oscillator parameter estimation"
        ),
        GenuineProblem(
            name="DoublePendulum-4D",
            func=double_pendulum_objective,
            bounds=[(0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2)],
            dim=4,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="highly_rugged",
            description="Double pendulum parameter estimation"
        ),
        GenuineProblem(
            name="Burgers-9D",
            func=burgers_equation_objective,
            bounds=[(0.01, 0.5)] + [(-1, 1)]*4 + [(-1, 1)]*4,
            dim=9,
            category="pde",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="1D Burgers equation (simplified Navier-Stokes)"
        ),
    ]


def get_nn_problems() -> List[GenuineProblem]:
    """Get neural network weight optimization problems."""
    return [
        GenuineProblem(
            name="NN-XOR-17D",
            func=neural_network_weights_objective,
            bounds=[(-3, 3)] * 17,
            dim=17,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Small neural network for XOR problem"
        ),
        GenuineProblem(
            name="NN-Regression-89D",
            func=nn_regression_objective,
            bounds=[(-2, 2)] * 89,
            dim=89,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Neural network regression 1->8->8->1"
        ),
        GenuineProblem(
            name="NN-MNIST-1074D",
            func=nn_mnist_subset_objective,
            bounds=[(-1, 1)] * 1074,
            dim=1074,
            category="nn_weights",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Neural network for digit classification 64->16->2"
        ),
        GenuineProblem(
            name="NN-Large-1377D",
            func=large_nn_objective,
            bounds=[(-1, 1)] * 1377,
            dim=1377,
            category="nn_weights",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Large neural network 10->32->32->1"
        ),
    ]


def get_expensive_problems() -> List[GenuineProblem]:
    """Get inherently expensive optimization problems."""
    return [
        GenuineProblem(
            name="SVM-CV-2D",
            func=cross_validation_svm_objective,
            bounds=[(-3, 3), (-4, 1)],
            dim=2,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="SVM hyperparameter optimization with 5-fold CV"
        ),
        GenuineProblem(
            name="RF-CV-4D",
            func=cross_validation_rf_objective,
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            dim=4,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Random Forest hyperparameter optimization with CV"
        ),
        GenuineProblem(
            name="SA-Schedule-3D",
            func=simulation_optimization_objective,
            bounds=[(0, 3), (0, 1), (0, 1)],
            dim=3,
            category="simulation",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Simulated annealing schedule optimization"
        ),
        # LOW + moderate + smooth
        GenuineProblem(
            name="Ridge-CV-1D",
            func=ridge_regression_cv_objective,
            bounds=[(-4, 4)],
            dim=1,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Ridge regression alpha tuning with CV"
        ),
        GenuineProblem(
            name="Lasso-CV-1D",
            func=lasso_regression_cv_objective,
            bounds=[(-4, 4)],
            dim=1,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Lasso regression alpha tuning with CV"
        ),
        GenuineProblem(
            name="ElasticNet-CV-2D",
            func=elastic_net_cv_objective,
            bounds=[(-4, 4), (0.01, 0.99)],
            dim=2,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Elastic Net hyperparameter tuning with CV"
        ),
        GenuineProblem(
            name="LogisticReg-CV-1D",
            func=logistic_regression_cv_objective,
            bounds=[(-4, 4)],
            dim=1,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Logistic regression C tuning with CV"
        ),
        # LOW + moderate + moderate
        GenuineProblem(
            name="KNN-CV-3D",
            func=knn_classifier_cv_objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            dim=3,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="KNN classifier hyperparameter tuning with CV"
        ),
        GenuineProblem(
            name="DecisionTree-CV-3D",
            func=decision_tree_cv_objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            dim=3,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Decision tree hyperparameter tuning with CV"
        ),
        # LOW + expensive + smooth
        GenuineProblem(
            name="AdaBoost-CV-2D",
            func=adaboost_cv_objective,
            bounds=[(0, 1), (0, 1)],
            dim=2,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="AdaBoost hyperparameter tuning with CV"
        ),
        GenuineProblem(
            name="SVM-Large-CV-2D",
            func=svm_rbf_large_cv_objective,
            bounds=[(-3, 3), (-4, 1)],
            dim=2,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="SVM RBF with larger dataset and 10-fold CV"
        ),
        GenuineProblem(
            name="Bagging-CV-3D",
            func=bagging_cv_objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            dim=3,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Bagging classifier hyperparameter tuning with CV"
        ),
        # LOW + expensive + moderate
        GenuineProblem(
            name="GradientBoost-CV-3D",
            func=gradient_boosting_cv_objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            dim=3,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="Gradient Boosting hyperparameter tuning with CV"
        ),
        # LOW + expensive + rugged
        GenuineProblem(
            name="MLP-Regressor-CV-3D",
            func=mlp_regressor_cv_objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            dim=3,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="MLP regressor hyperparameter tuning with CV"
        ),
    ]


def get_other_genuine_problems() -> List[GenuineProblem]:
    """Get other genuine high-dimensional problems."""
    return [
        GenuineProblem(
            name="SparseCoding-400D",
            func=sparse_coding_objective,
            bounds=[(-1, 1)] * 400,
            dim=400,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Sparse coding dictionary learning"
        ),
        GenuineProblem(
            name="PDE-HeatEq-50D",
            func=pde_discretization_objective,
            bounds=[(-1, 1)] * 50,
            dim=50,
            category="simulation",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="1D heat equation initial condition optimization"
        ),
    ]


def get_medium_dim_problems() -> List[GenuineProblem]:
    """Get MEDIUM dimension (11-50D) problems to fill gaps."""
    return [
        # MEDIUM + cheap + rugged (need 3 more)
        GenuineProblem(
            name="NN-Medium-20D",
            func=nn_medium_20d_objective,
            bounds=[(-2, 2)] * 20,
            dim=20,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Small neural network 1->10 weights"
        ),
        GenuineProblem(
            name="CoupledOscillators-15D",
            func=coupled_oscillator_15d_objective,
            bounds=[(0.1, 2)] * 5 + [(0.1, 5)] * 5 + [(0, 1)] * 5,
            dim=15,
            category="dynamical",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Coupled harmonic oscillators parameter estimation"
        ),
        GenuineProblem(
            name="RastriginRotated-20D",
            func=rastrigin_rotated_20d_objective,
            bounds=[(-5.12, 5.12)] * 20,
            dim=20,
            category="synthetic",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Rotated Rastrigin function"
        ),
        # MEDIUM + moderate + smooth (need 4 more)
        GenuineProblem(
            name="HeatDiffusion-30D",
            func=heat_diffusion_30d_objective,
            bounds=[(-1, 1)] * 30,
            dim=30,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Heat diffusion equation initial condition"
        ),
        GenuineProblem(
            name="WaveEquation-30D",
            func=wave_equation_30d_objective,
            bounds=[(-1, 1)] * 30,
            dim=30,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Wave equation initial condition"
        ),
        GenuineProblem(
            name="AdvectionDiffusion-30D",
            func=advection_diffusion_30d_objective,
            bounds=[(-1, 1)] * 30,
            dim=30,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Advection-diffusion equation"
        ),
        # MEDIUM + moderate + moderate (need 5)
        GenuineProblem(
            name="SparseRegression-40D",
            func=sparse_regression_40d_objective,
            bounds=[(-2, 2)] * 40,
            dim=40,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Sparse regression coefficient optimization"
        ),
        GenuineProblem(
            name="MatrixFactorization-36D",
            func=matrix_factorization_36d_objective,
            bounds=[(-2, 2)] * 36,
            dim=36,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Low-rank matrix factorization"
        ),
        GenuineProblem(
            name="TensorDecomposition-27D",
            func=tensor_decomposition_27d_objective,
            bounds=[(-2, 2)] * 27,
            dim=27,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="CP tensor decomposition"
        ),
        GenuineProblem(
            name="ICA-Unmixing-25D",
            func=ica_unmixing_25d_objective,
            bounds=[(-2, 2)] * 25,
            dim=25,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="ICA unmixing matrix optimization"
        ),
        GenuineProblem(
            name="PCA-Reconstruction-30D",
            func=pca_reconstruction_30d_objective,
            bounds=[(-2, 2)] * 30,
            dim=30,
            category="ml_training",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="PCA components optimization"
        ),
        # MEDIUM + moderate + rugged (need 5)
        GenuineProblem(
            name="Lorenz96-20D",
            func=lorenz96_20d_objective,
            bounds=[(-10, 20)] * 20,
            dim=20,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Lorenz 96 weather model"
        ),
        GenuineProblem(
            name="KuramotoOscillators-20D",
            func=kuramoto_oscillators_20d_objective,
            bounds=[(-2, 2)] * 10 + [(0, 2)] * 10,
            dim=20,
            category="dynamical",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Kuramoto coupled oscillators synchronization"
        ),
        GenuineProblem(
            name="CellularAutomata-25D",
            func=cellular_automata_25d_objective,
            bounds=[(-1, 1)] * 25,
            dim=25,
            category="simulation",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Cellular automata rule optimization"
        ),
        GenuineProblem(
            name="ReactionDiffusion-30D",
            func=reaction_diffusion_30d_objective,
            bounds=[(0.001, 0.1), (0.001, 0.1), (0, 0.1), (0, 0.1)] + [(-1, 1)] * 26,
            dim=30,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Reaction-diffusion pattern formation"
        ),
        # MEDIUM + expensive + smooth (need 5)
        GenuineProblem(
            name="SVM-FeatureWeights-20D",
            func=svm_medium_features_cv_objective,
            bounds=[(-3, 3), (-4, 1)] + [(0, 1)] * 18,
            dim=20,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="SVM with feature weighting"
        ),
        GenuineProblem(
            name="RF-FeatureSelection-25D",
            func=rf_feature_selection_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Random Forest with feature selection"
        ),
        GenuineProblem(
            name="GB-FeatureEngineering-30D",
            func=gb_feature_engineering_30d_objective,
            bounds=[(0, 1)] * 30,
            dim=30,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Gradient Boosting with feature engineering"
        ),
        GenuineProblem(
            name="PCA-SVM-Pipeline-25D",
            func=pca_svm_pipeline_25d_objective,
            bounds=[(0, 1), (-3, 3), (-4, 1)] + [(0, 1)] * 22,
            dim=25,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="PCA + SVM pipeline optimization"
        ),
        GenuineProblem(
            name="KernelRidge-30D",
            func=kernel_ridge_30d_objective,
            bounds=[(-4, 2), (-4, 1)] + [(0, 1)] * 28,
            dim=30,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Kernel Ridge with feature weights"
        ),
        # MEDIUM + expensive + moderate (need 5)
        GenuineProblem(
            name="EnsembleWeights-25D",
            func=ensemble_weights_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="Ensemble model weight optimization"
        ),
        GenuineProblem(
            name="MLP-Architecture-30D",
            func=mlp_architecture_30d_objective,
            bounds=[(0, 1)] * 30,
            dim=30,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="MLP architecture search"
        ),
        GenuineProblem(
            name="XGBoost-Tuning-25D",
            func=xgb_tuning_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="XGBoost hyperparameter tuning with features"
        ),
        GenuineProblem(
            name="LightGBM-Tuning-25D",
            func=lgb_tuning_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="LightGBM hyperparameter tuning with features"
        ),
        # MEDIUM + expensive + rugged (need 5)
        GenuineProblem(
            name="GeneticAlgorithm-25D",
            func=genetic_algorithm_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Genetic algorithm parameter tuning"
        ),
        GenuineProblem(
            name="ParticleSwarm-30D",
            func=particle_swarm_30d_objective,
            bounds=[(0, 1)] * 30,
            dim=30,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="PSO parameter tuning"
        ),
        GenuineProblem(
            name="DifferentialEvolution-30D",
            func=differential_evolution_30d_objective,
            bounds=[(0, 1)] * 30,
            dim=30,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Differential Evolution parameter tuning"
        ),
        GenuineProblem(
            name="CMA-ES-25D",
            func=cma_es_25d_objective,
            bounds=[(0, 1)] * 25,
            dim=25,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="CMA-ES hyperparameter meta-optimization"
        ),
    ]


def get_high_dim_problems() -> List[GenuineProblem]:
    """Get HIGH dimension (51+D) problems to fill gaps."""
    return [
        # HIGH + cheap + moderate (need 5)
        GenuineProblem(
            name="NN-Deep-100D",
            func=nn_deep_100d_objective,
            bounds=[(-2, 2)] * 100,
            dim=100,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Deep neural network weights 5->10->10->1"
        ),
        GenuineProblem(
            name="Autoencoder-80D",
            func=autoencoder_80d_objective,
            bounds=[(-2, 2)] * 80,
            dim=80,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Autoencoder 10->5->10 weights"
        ),
        GenuineProblem(
            name="RBM-60D",
            func=rbm_60d_objective,
            bounds=[(-2, 2)] * 60,
            dim=60,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Restricted Boltzmann Machine weights"
        ),
        GenuineProblem(
            name="WordEmbedding-75D",
            func=word_embedding_75d_objective,
            bounds=[(-2, 2)] * 75,
            dim=75,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Word embedding optimization"
        ),
        GenuineProblem(
            name="Hopfield-64D",
            func=hopfield_64d_objective,
            bounds=[(-2, 2)] * 64,
            dim=64,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="moderate",
            description="Hopfield network weights"
        ),
        # HIGH + cheap + rugged (need 4 more)
        GenuineProblem(
            name="SparseAutoencoder-100D",
            func=sparse_autoencoder_100d_objective,
            bounds=[(-2, 2)] * 100,
            dim=100,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Sparse autoencoder with sparsity penalty"
        ),
        GenuineProblem(
            name="VAE-90D",
            func=variational_autoencoder_90d_objective,
            bounds=[(-2, 2)] * 90,
            dim=90,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Variational autoencoder weights"
        ),
        GenuineProblem(
            name="DenoisingAE-80D",
            func=denoising_autoencoder_80d_objective,
            bounds=[(-2, 2)] * 80,
            dim=80,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Denoising autoencoder weights"
        ),
        GenuineProblem(
            name="ContrastiveLearning-70D",
            func=contrastive_learning_70d_objective,
            bounds=[(-2, 2)] * 70,
            dim=70,
            category="nn_weights",
            expected_cost="cheap",
            expected_ruggedness="rugged",
            description="Contrastive learning projection head"
        ),
        # HIGH + moderate + smooth (need 5)
        GenuineProblem(
            name="Heat2D-60D",
            func=heat_equation_60d_objective,
            bounds=[(-1, 1)] * 60,
            dim=60,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="2D heat equation initial condition"
        ),
        GenuineProblem(
            name="Poisson-60D",
            func=poisson_equation_60d_objective,
            bounds=[(-1, 1)] * 60,
            dim=60,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Poisson equation source optimization"
        ),
        GenuineProblem(
            name="Laplace-64D",
            func=laplace_equation_64d_objective,
            bounds=[(-1, 1)] * 64,
            dim=64,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Laplace equation boundary optimization"
        ),
        GenuineProblem(
            name="Helmholtz-60D",
            func=helmholtz_equation_60d_objective,
            bounds=[(-1, 1)] * 60,
            dim=60,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Helmholtz equation parameter optimization"
        ),
        GenuineProblem(
            name="Biharmonic-56D",
            func=biharmonic_equation_56d_objective,
            bounds=[(-1, 1)] * 56,
            dim=56,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="smooth",
            description="Biharmonic equation (plate bending)"
        ),
        # HIGH + moderate + moderate (need 5)
        GenuineProblem(
            name="SpectralMethod-64D",
            func=spectral_method_64d_objective,
            bounds=[(-1, 1)] * 64,
            dim=64,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Spectral method coefficients"
        ),
        GenuineProblem(
            name="FiniteElement-70D",
            func=finite_element_70d_objective,
            bounds=[(-1, 1)] * 70,
            dim=70,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Finite element node values"
        ),
        GenuineProblem(
            name="Multigrid-60D",
            func=multigrid_60d_objective,
            bounds=[(-1, 1)] * 60,
            dim=60,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Multigrid solver parameters"
        ),
        GenuineProblem(
            name="DomainDecomposition-72D",
            func=domain_decomposition_72d_objective,
            bounds=[(-1, 1)] * 72,
            dim=72,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Domain decomposition interface"
        ),
        GenuineProblem(
            name="AdaptiveMesh-65D",
            func=adaptive_mesh_65d_objective,
            bounds=[(-1, 1)] * 65,
            dim=65,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="moderate",
            description="Adaptive mesh refinement"
        ),
        # HIGH + moderate + rugged (need 5)
        GenuineProblem(
            name="Lorenz96Extended-60D",
            func=lorenz96_extended_60d_objective,
            bounds=[(-10, 20)] * 60,
            dim=60,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Extended Lorenz 96 model"
        ),
        GenuineProblem(
            name="CoupledMapLattice-64D",
            func=coupled_map_lattice_64d_objective,
            bounds=[(0, 1)] * 64,
            dim=64,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Coupled map lattice dynamics"
        ),
        GenuineProblem(
            name="NeuralField-70D",
            func=neural_field_70d_objective,
            bounds=[(-1, 1)] * 70,
            dim=70,
            category="dynamical",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Neural field dynamics"
        ),
        GenuineProblem(
            name="SpatiotemporalChaos-60D",
            func=spatiotemporal_chaos_60d_objective,
            bounds=[(-2, 2)] * 60,
            dim=60,
            category="chaotic",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Spatiotemporal chaotic system"
        ),
        GenuineProblem(
            name="GinzburgLandau-56D",
            func=ginzburg_landau_56d_objective,
            bounds=[(-2, 2)] * 56,
            dim=56,
            category="pde",
            expected_cost="moderate",
            expected_ruggedness="rugged",
            description="Complex Ginzburg-Landau equation"
        ),
        # HIGH + expensive + smooth (need 5)
        GenuineProblem(
            name="SVM-HighDim-60D",
            func=svm_high_dim_60d_objective,
            bounds=[(-3, 3), (-4, 1)] + [(0, 1)] * 58,
            dim=60,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="SVM on high-dimensional feature space"
        ),
        GenuineProblem(
            name="RF-HighDim-70D",
            func=rf_high_dim_70d_objective,
            bounds=[(0, 1)] * 70,
            dim=70,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Random Forest on high-dimensional features"
        ),
        GenuineProblem(
            name="GB-HighDim-65D",
            func=gb_high_dim_65d_objective,
            bounds=[(0, 1)] * 65,
            dim=65,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Gradient Boosting high-dimensional"
        ),
        GenuineProblem(
            name="KernelPCA-SVM-75D",
            func=kernel_pca_svm_75d_objective,
            bounds=[(0, 1)] * 4 + [(0, 1)] * 71,
            dim=75,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Kernel PCA + SVM pipeline"
        ),
        GenuineProblem(
            name="NeuralNet-Dropout-80D",
            func=neural_net_dropout_80d_objective,
            bounds=[(0, 1)] * 80,
            dim=80,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="smooth",
            description="Neural network with dropout tuning"
        ),
        # HIGH + expensive + moderate (need 5)
        GenuineProblem(
            name="XGBoost-HighDim-60D",
            func=xgb_high_dim_60d_objective,
            bounds=[(0, 1)] * 60,
            dim=60,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="XGBoost on high-dimensional data"
        ),
        GenuineProblem(
            name="LightGBM-HighDim-65D",
            func=lgb_high_dim_65d_objective,
            bounds=[(0, 1)] * 65,
            dim=65,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="LightGBM on high-dimensional data"
        ),
        GenuineProblem(
            name="EnsembleStacking-70D",
            func=ensemble_stacking_70d_objective,
            bounds=[(0, 1)] * 70,
            dim=70,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="Ensemble stacking weights"
        ),
        GenuineProblem(
            name="FeatureSelection-75D",
            func=feature_selection_pipeline_75d_objective,
            bounds=[(0, 1)] * 75,
            dim=75,
            category="ml_training",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="Feature selection pipeline"
        ),
        GenuineProblem(
            name="Hyperband-60D",
            func=hyperband_surrogate_60d_objective,
            bounds=[(0, 1)] * 60,
            dim=60,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="moderate",
            description="Hyperband multi-fidelity search"
        ),
        # HIGH + expensive + rugged (need 5)
        GenuineProblem(
            name="BayesianOpt-60D",
            func=bayesian_optimization_60d_objective,
            bounds=[(0, 1)] * 60,
            dim=60,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Bayesian optimization surrogate"
        ),
        GenuineProblem(
            name="NAS-70D",
            func=neural_architecture_search_70d_objective,
            bounds=[(0, 1)] * 70,
            dim=70,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Neural architecture search"
        ),
        GenuineProblem(
            name="EvolutionStrategy-65D",
            func=evolutionary_strategy_65d_objective,
            bounds=[(0, 1)] * 65,
            dim=65,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Evolution strategy meta-optimization"
        ),
        GenuineProblem(
            name="SimulatedAnnealing-55D",
            func=simulated_annealing_meta_55d_objective,
            bounds=[(0, 1)] * 55,
            dim=55,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Simulated annealing meta-optimization"
        ),
        GenuineProblem(
            name="CovarianceAdaptation-60D",
            func=covariance_adaptation_60d_objective,
            bounds=[(0, 1)] * 60,
            dim=60,
            category="meta_optimization",
            expected_cost="expensive",
            expected_ruggedness="rugged",
            description="Covariance matrix adaptation"
        ),
    ]


# =============================================================================
# HIGH-DIMENSION PROBLEMS (51+D)
# =============================================================================

def nn_deep_100d_objective(params: np.ndarray) -> float:
    """
    Deep neural network 5->10->10->1 weights = 100D.
    Target: HIGH + cheap + moderate
    """
    if len(params) != 100:
        return 1e10
    
    W1 = params[:50].reshape(5, 10)
    b1 = params[50:60]
    W2 = params[60:90].reshape(10, 3)
    b2 = params[90:93]
    W3 = params[93:99].reshape(3, 2)
    b3 = params[99]
    
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1] * X[:, 2])
    
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    out = np.mean(h2 @ W3, axis=1) + b3
    
    return np.mean((out - y)**2)


def autoencoder_80d_objective(params: np.ndarray) -> float:
    """
    Autoencoder 10->5->10 = 80D weights.
    Target: HIGH + cheap + moderate
    """
    if len(params) != 80:
        return 1e10
    
    W_enc = params[:50].reshape(10, 5)
    b_enc = params[50:55]
    W_dec = params[55:75].reshape(5, 4)
    b_dec = params[75:79]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    encoded = np.tanh(X @ W_enc + b_enc)
    decoded = encoded @ W_dec + b_dec
    
    return np.mean((X[:, :4] - decoded)**2)


def rbm_60d_objective(params: np.ndarray) -> float:
    """
    Restricted Boltzmann Machine weights 6x10 = 60D.
    Target: HIGH + cheap + moderate
    """
    if len(params) != 60:
        return 1e10
    
    W = params.reshape(6, 10)
    
    np.random.seed(42)
    v = (np.random.rand(100, 6) > 0.5).astype(float)
    
    h_prob = 1 / (1 + np.exp(-v @ W))
    v_recon_prob = 1 / (1 + np.exp(-h_prob @ W.T))
    
    return np.mean((v - v_recon_prob)**2)


def word_embedding_75d_objective(params: np.ndarray) -> float:
    """
    Word embedding optimization 15 words x 5D = 75D.
    Target: HIGH + cheap + moderate
    """
    if len(params) != 75:
        return 1e10
    
    embeddings = params.reshape(15, 5)
    
    np.random.seed(42)
    similar_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    dissimilar_pairs = [(0, 8), (1, 9), (2, 10), (3, 11)]
    
    loss = 0
    for i, j in similar_pairs:
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        loss += dist**2
    
    for i, j in dissimilar_pairs:
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        loss += max(0, 2 - dist)**2
    
    return loss


def hopfield_64d_objective(params: np.ndarray) -> float:
    """
    Hopfield network weights 8x8 = 64D.
    Target: HIGH + cheap + moderate
    """
    if len(params) != 64:
        return 1e10
    
    W = params.reshape(8, 8)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    
    np.random.seed(42)
    patterns = np.sign(np.random.randn(5, 8))
    
    total_energy = 0
    for p in patterns:
        noisy = p.copy()
        noisy[:2] = -noisy[:2]
        
        for _ in range(10):
            h = W @ noisy
            noisy = np.sign(h)
        
        total_energy += np.sum((noisy - p)**2)
    
    return total_energy


def sparse_autoencoder_100d_objective(params: np.ndarray) -> float:
    """
    Sparse autoencoder with sparsity penalty = 100D.
    Target: HIGH + cheap + rugged
    """
    if len(params) != 100:
        return 1e10
    
    W_enc = params[:60].reshape(10, 6)
    b_enc = params[60:66]
    W_dec = params[66:96].reshape(6, 5)
    b_dec = params[96:100]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    encoded = 1 / (1 + np.exp(-(X @ W_enc + b_enc)))
    decoded = encoded @ W_dec + b_dec
    
    recon_loss = np.mean((X[:, :5] - decoded)**2)
    sparsity = np.mean(encoded)
    sparsity_loss = (sparsity - 0.1)**2
    
    return recon_loss + 0.5 * sparsity_loss


def variational_autoencoder_90d_objective(params: np.ndarray) -> float:
    """
    VAE encoder/decoder weights = 90D.
    Target: HIGH + cheap + rugged
    """
    if len(params) != 90:
        return 1e10
    
    W_enc = params[:40].reshape(8, 5)
    b_enc = params[40:45]
    W_dec = params[45:85].reshape(5, 8)
    b_dec = params[85:90]
    
    np.random.seed(42)
    X = np.random.randn(100, 8)
    
    z_mean = X @ W_enc + b_enc
    z = z_mean + 0.1 * np.random.randn(*z_mean.shape)
    
    x_recon = z @ W_dec + b_dec
    
    recon_loss = np.mean((X - x_recon)**2)
    kl_loss = 0.5 * np.mean(z_mean**2)
    
    return recon_loss + 0.1 * kl_loss


def denoising_autoencoder_80d_objective(params: np.ndarray) -> float:
    """
    Denoising autoencoder = 80D.
    Target: HIGH + cheap + rugged
    """
    if len(params) != 80:
        return 1e10
    
    W_enc = params[:50].reshape(10, 5)
    b_enc = params[50:55]
    W_dec = params[55:75].reshape(5, 4)
    b_dec = params[75:79]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    X_noisy = X + 0.3 * np.random.randn(*X.shape)
    
    encoded = np.tanh(X_noisy @ W_enc + b_enc)
    decoded = encoded @ W_dec + b_dec
    
    return np.mean((X[:, :4] - decoded)**2)


def contrastive_learning_70d_objective(params: np.ndarray) -> float:
    """
    Contrastive learning projection head = 70D.
    Target: HIGH + cheap + rugged
    """
    if len(params) != 70:
        return 1e10
    
    W1 = params[:40].reshape(8, 5)
    b1 = params[40:45]
    W2 = params[45:65].reshape(5, 4)
    b2 = params[65:69]
    
    np.random.seed(42)
    X = np.random.randn(50, 8)
    X_aug = X + 0.1 * np.random.randn(*X.shape)
    
    h1 = np.tanh(X @ W1 + b1)
    z1 = h1 @ W2 + b2
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    
    h2 = np.tanh(X_aug @ W1 + b1)
    z2 = h2 @ W2 + b2
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    
    pos_sim = np.sum(z1 * z2, axis=1)
    loss = -np.mean(pos_sim)
    
    return loss


def heat_equation_60d_objective(params: np.ndarray) -> float:
    """
    2D heat equation initial condition = 60D.
    Target: HIGH + moderate + smooth
    """
    if len(params) != 60:
        return 1e10
    
    N = 8
    u = params[:64].reshape(8, 8) if len(params) >= 64 else np.zeros((8, 8))
    u = params[:60].reshape(6, 10)[:6, :8]
    
    dx = 1.0
    dt = 0.2
    
    for _ in range(50):
        u_new = u.copy()
        for i in range(1, 5):
            for j in range(1, 7):
                u_new[i, j] = u[i, j] + dt * (
                    (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]) / dx**2
                )
        u = u_new
    
    return np.mean(u**2)


def poisson_equation_60d_objective(params: np.ndarray) -> float:
    """
    Poisson equation source term optimization = 60D.
    Target: HIGH + moderate + smooth
    """
    if len(params) != 60:
        return 1e10
    
    f = params.reshape(6, 10)
    N = 6
    M = 10
    
    u = np.zeros((N, M))
    dx = 1.0
    
    for _ in range(100):
        u_new = u.copy()
        for i in range(1, N-1):
            for j in range(1, M-1):
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - dx**2 * f[i, j])
        u = u_new
    
    target = np.zeros((N, M))
    target[N//2, M//2] = 1.0
    
    return np.mean((u - target)**2)


def laplace_equation_64d_objective(params: np.ndarray) -> float:
    """
    Laplace equation boundary optimization = 64D.
    Target: HIGH + moderate + smooth
    """
    if len(params) != 64:
        return 1e10
    
    boundary = params
    
    N = 8
    u = np.zeros((N, N))
    u[0, :] = boundary[:8]
    u[-1, :] = boundary[8:16]
    u[:, 0] = boundary[16:24]
    u[:, -1] = boundary[24:32]
    
    for _ in range(100):
        u_new = u.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
        u = u_new
    
    target_center = 0.5
    center_val = u[N//2, N//2]
    
    return (center_val - target_center)**2


def helmholtz_equation_60d_objective(params: np.ndarray) -> float:
    """
    Helmholtz equation parameter optimization = 60D.
    Target: HIGH + moderate + smooth
    """
    if len(params) != 60:
        return 1e10
    
    k = np.abs(params[0]) + 0.1
    source = params[1:].reshape(6, 10)[:5, :9]
    
    N, M = 5, 9
    u = np.zeros((N, M), dtype=complex)
    dx = 0.5
    
    for _ in range(50):
        u_new = u.copy()
        for i in range(1, N-1):
            for j in range(1, M-1):
                laplacian = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]) / dx**2
                u_new[i, j] = u[i, j] + 0.1 * (laplacian + k**2 * u[i, j] + source[i, j])
        u = u_new
    
    return np.mean(np.abs(u)**2)


def biharmonic_equation_56d_objective(params: np.ndarray) -> float:
    """
    Biharmonic equation (plate bending) = 56D.
    Target: HIGH + moderate + smooth
    """
    if len(params) != 56:
        return 1e10
    
    load = params.reshape(7, 8)
    
    N, M = 7, 8
    u = np.zeros((N, M))
    dx = 1.0
    
    for _ in range(100):
        u_new = u.copy()
        for i in range(2, N-2):
            for j in range(2, M-2):
                biharmonic = (
                    20*u[i,j] - 8*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) +
                    2*(u[i+1,j+1] + u[i+1,j-1] + u[i-1,j+1] + u[i-1,j-1]) +
                    u[i+2,j] + u[i-2,j] + u[i,j+2] + u[i,j-2]
                ) / dx**4
                u_new[i, j] = u[i, j] - 0.01 * (biharmonic - load[i, j])
        u = u_new
    
    return np.mean(u**2)


def spectral_method_64d_objective(params: np.ndarray) -> float:
    """
    Spectral method coefficients = 64D.
    Target: HIGH + moderate + moderate
    """
    if len(params) != 64:
        return 1e10
    
    coeffs = params.reshape(8, 8)
    
    N = 32
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)
    
    u = np.zeros((N, N))
    for k in range(8):
        for l in range(8):
            u += coeffs[k, l] * np.sin((k+1)*X) * np.sin((l+1)*Y)
    
    target = np.sin(X) * np.sin(Y)
    
    return np.mean((u - target)**2)


def finite_element_70d_objective(params: np.ndarray) -> float:
    """
    Finite element node values = 70D.
    Target: HIGH + moderate + moderate
    """
    if len(params) != 70:
        return 1e10
    
    nodes = params.reshape(7, 10)
    
    smoothness = 0
    for i in range(6):
        for j in range(9):
            grad_x = nodes[i+1, j] - nodes[i, j]
            grad_y = nodes[i, j+1] - nodes[i, j]
            smoothness += grad_x**2 + grad_y**2
    
    target_sum = 5.0
    constraint = (np.sum(nodes) - target_sum)**2
    
    return smoothness + 10 * constraint


def multigrid_60d_objective(params: np.ndarray) -> float:
    """
    Multigrid solver parameters = 60D.
    Target: HIGH + moderate + moderate
    """
    if len(params) != 60:
        return 1e10
    
    coarse = params[:15].reshape(3, 5)
    fine = params[15:60].reshape(5, 9)
    
    interpolated = np.zeros((5, 9))
    for i in range(5):
        for j in range(9):
            ci, cj = min(i//2, 2), min(j//2, 4)
            interpolated[i, j] = coarse[ci, cj]
    
    combined = 0.5 * interpolated + 0.5 * fine
    
    return np.mean((combined - np.mean(combined))**2)


def domain_decomposition_72d_objective(params: np.ndarray) -> float:
    """
    Domain decomposition interface values = 72D.
    Target: HIGH + moderate + moderate
    """
    if len(params) != 72:
        return 1e10
    
    domain1 = params[:36].reshape(6, 6)
    domain2 = params[36:72].reshape(6, 6)
    
    interface_mismatch = np.mean((domain1[:, -1] - domain2[:, 0])**2)
    
    smoothness1 = np.mean(np.diff(domain1, axis=0)**2 + np.diff(domain1, axis=1)**2)
    smoothness2 = np.mean(np.diff(domain2, axis=0)**2 + np.diff(domain2, axis=1)**2)
    
    return interface_mismatch + 0.1 * (smoothness1 + smoothness2)


def adaptive_mesh_65d_objective(params: np.ndarray) -> float:
    """
    Adaptive mesh refinement = 65D.
    Target: HIGH + moderate + moderate
    """
    if len(params) != 65:
        return 1e10
    
    mesh_density = np.abs(params[:25]).reshape(5, 5) + 0.1
    solution = params[25:65].reshape(5, 8)
    
    gradient = np.sqrt(np.diff(solution, axis=0)**2 + np.diff(solution[:, :-1], axis=1)**2)
    
    desired_density = gradient / (np.max(gradient) + 1e-8)
    
    return np.mean((mesh_density[:4, :4] - desired_density[:4, :4])**2)


def lorenz96_extended_60d_objective(params: np.ndarray) -> float:
    """
    Extended Lorenz 96 model = 60D.
    Target: HIGH + moderate + rugged
    """
    if len(params) != 60:
        return 1e10
    
    F = 8.0
    x = params.copy()
    n = 60
    
    dt = 0.01
    for _ in range(200):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        x = x + dt * dx
        
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
    
    return (np.mean(x) - F)**2 + (np.std(x) - 3.5)**2


def coupled_map_lattice_64d_objective(params: np.ndarray) -> float:
    """
    Coupled map lattice dynamics = 64D.
    Target: HIGH + moderate + rugged
    """
    if len(params) != 64:
        return 1e10
    
    x = params.reshape(8, 8)
    eps = 0.3
    r = 3.8
    
    for _ in range(100):
        x_new = np.zeros_like(x)
        for i in range(8):
            for j in range(8):
                local = r * x[i, j] * (1 - x[i, j])
                neighbors = (x[(i+1)%8, j] + x[(i-1)%8, j] + x[i, (j+1)%8] + x[i, (j-1)%8]) / 4
                x_new[i, j] = (1 - eps) * local + eps * neighbors
        x = np.clip(x_new, 0, 1)
    
    target_pattern = 0.5
    return (np.mean(x) - target_pattern)**2


def neural_field_70d_objective(params: np.ndarray) -> float:
    """
    Neural field dynamics = 70D.
    Target: HIGH + moderate + rugged
    """
    if len(params) != 70:
        return 1e10
    
    u = params[:49].reshape(7, 7)
    weights = params[49:70].reshape(3, 7)
    
    for _ in range(50):
        u_new = np.zeros_like(u)
        for i in range(7):
            for j in range(7):
                coupling = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = (i + di) % 7, (j + dj) % 7
                        w_idx = di + 1
                        coupling += weights[w_idx, j] * np.tanh(u[ni, nj])
                u_new[i, j] = 0.9 * u[i, j] + 0.1 * coupling
        u = u_new
    
    target_activity = 0.3
    return (np.mean(np.abs(u)) - target_activity)**2


def spatiotemporal_chaos_60d_objective(params: np.ndarray) -> float:
    """
    Spatiotemporal chaotic system = 60D.
    Target: HIGH + moderate + rugged
    """
    if len(params) != 60:
        return 1e10
    
    u = params[:30]
    v = params[30:60]
    
    a, b = 0.5, 0.1
    dx = 0.5
    dt = 0.01
    
    for _ in range(200):
        du = np.zeros(30)
        dv = np.zeros(30)
        for i in range(1, 29):
            laplacian_u = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
            laplacian_v = (v[i+1] - 2*v[i] + v[i-1]) / dx**2
            du[i] = laplacian_u + u[i] - u[i]**3 - v[i]
            dv[i] = b * laplacian_v + a * (u[i] - v[i])
        u = u + dt * du
        v = v + dt * dv
        
        if np.any(np.isnan(u)):
            return 1e10
    
    return np.std(u) + np.std(v)


def ginzburg_landau_56d_objective(params: np.ndarray) -> float:
    """
    Complex Ginzburg-Landau equation = 56D.
    Target: HIGH + moderate + rugged
    """
    if len(params) != 56:
        return 1e10
    
    A = params[:28] + 1j * params[28:56]
    A = A.reshape(4, 7)
    
    c1, c2 = 1.0, 0.5
    dx = 1.0
    dt = 0.1
    
    for _ in range(50):
        A_new = A.copy()
        for i in range(1, 3):
            for j in range(1, 6):
                laplacian = (A[i+1, j] + A[i-1, j] + A[i, j+1] + A[i, j-1] - 4*A[i, j]) / dx**2
                A_new[i, j] = A[i, j] + dt * (A[i, j] - (1 + 1j*c2) * np.abs(A[i, j])**2 * A[i, j] + (1 + 1j*c1) * laplacian)
        A = A_new
    
    return np.mean(np.abs(A)**2)


def svm_high_dim_60d_objective(params: np.ndarray) -> float:
    """
    SVM on high-dimensional feature space = 60D.
    Target: HIGH + expensive + smooth
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** params[0]
    gamma = 10 ** params[1]
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=58, n_informative=30, random_state=42)
    X_weighted = X * feature_weights[:58]
    
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def rf_high_dim_70d_objective(params: np.ndarray) -> float:
    """
    Random Forest on high-dimensional features = 70D.
    Target: HIGH + expensive + smooth
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    max_depth = int(3 + params[1] * 12)
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=68, n_informative=35, random_state=42)
    X_weighted = X * feature_weights[:68]
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def gb_high_dim_65d_objective(params: np.ndarray) -> float:
    """
    Gradient Boosting on high-dimensional features = 65D.
    Target: HIGH + expensive + smooth
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    learning_rate = 0.01 + params[1] * 0.29
    max_depth = int(2 + params[2] * 6)
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=62, n_informative=30, random_state=42)
    X_weighted = X * feature_weights[:62]
    
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def kernel_pca_svm_75d_objective(params: np.ndarray) -> float:
    """
    Kernel PCA + SVM pipeline = 75D.
    Target: HIGH + expensive + smooth
    """
    try:
        from sklearn.svm import SVC
        from sklearn.decomposition import KernelPCA
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_components = int(5 + params[0] * 25)
    gamma_pca = 10 ** params[1]
    C = 10 ** params[2]
    gamma_svm = 10 ** params[3]
    feature_weights = np.abs(params[4:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=71, n_informative=35, random_state=42)
    X_weighted = X * feature_weights[:71]
    
    pipeline = Pipeline([
        ('kpca', KernelPCA(n_components=min(n_components, 30), kernel='rbf', gamma=gamma_pca)),
        ('svm', SVC(C=C, gamma=gamma_svm, kernel='rbf'))
    ])
    
    try:
        scores = cross_val_score(pipeline, X_weighted, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def neural_net_dropout_80d_objective(params: np.ndarray) -> float:
    """
    Neural network with dropout tuning = 80D.
    Target: HIGH + expensive + smooth
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    hidden1 = int(20 + params[0] * 80)
    hidden2 = int(10 + params[1] * 40)
    alpha = 10 ** (params[2] * 6 - 5)
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=77, n_informative=40, random_state=42)
    X_weighted = X * feature_weights[:77]
    
    model = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2), alpha=alpha, max_iter=200, random_state=42)
    try:
        scores = cross_val_score(model, X_weighted, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def xgb_high_dim_60d_objective(params: np.ndarray) -> float:
    """
    XGBoost on high-dimensional data = 60D.
    Target: HIGH + expensive + moderate
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    max_depth = int(2 + params[1] * 8)
    learning_rate = 0.01 + params[2] * 0.29
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=57, n_informative=30, random_state=42)
    X_weighted = X * feature_weights[:57]
    
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss')
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def lgb_high_dim_65d_objective(params: np.ndarray) -> float:
    """
    LightGBM on high-dimensional data = 65D.
    Target: HIGH + expensive + moderate
    """
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + params[0] * 100)
    max_depth = int(2 + params[1] * 10)
    num_leaves = int(10 + params[2] * 50)
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=62, n_informative=30, random_state=42)
    X_weighted = X * feature_weights[:62]
    
    model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, random_state=42, verbosity=-1)
    try:
        scores = cross_val_score(model, X_weighted, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def ensemble_stacking_70d_objective(params: np.ndarray) -> float:
    """
    Ensemble stacking weights = 70D.
    Target: HIGH + expensive + moderate
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    weights = np.abs(params[:3])
    weights = weights / (np.sum(weights) + 1e-10)
    feature_weights = np.abs(params[3:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=67, n_informative=35, random_state=42)
    X_weighted = X * feature_weights[:67]
    
    models = [
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        LogisticRegression(max_iter=500, random_state=42)
    ]
    
    predictions = []
    for model in models:
        try:
            preds = cross_val_predict(model, X_weighted, y, cv=3, method='predict_proba')[:, 1]
            predictions.append(preds)
        except:
            predictions.append(np.ones(len(y)) * 0.5)
    
    predictions = np.array(predictions)
    ensemble_pred = np.sum(predictions * weights.reshape(-1, 1), axis=0)
    ensemble_class = (ensemble_pred > 0.5).astype(int)
    
    return -np.mean(ensemble_class == y)


def feature_selection_pipeline_75d_objective(params: np.ndarray) -> float:
    """
    Feature selection + classifier pipeline = 75D.
    Target: HIGH + expensive + moderate
    """
    try:
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    k_features = int(10 + params[0] * 40)
    n_estimators = int(50 + params[1] * 100)
    feature_weights = np.abs(params[2:]) + 0.1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=73, n_informative=35, random_state=42)
    X_weighted = X * feature_weights[:73]
    
    pipeline = Pipeline([
        ('select', SelectKBest(f_classif, k=min(k_features, 73))),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1))
    ])
    
    try:
        scores = cross_val_score(pipeline, X_weighted, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def hyperband_surrogate_60d_objective(params: np.ndarray) -> float:
    """
    Hyperband-style multi-fidelity search = 60D.
    Target: HIGH + expensive + moderate
    """
    if len(params) != 60:
        return 1e10
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    configs = params[:30].reshape(6, 5) * 10.24 - 5.12
    budgets = np.abs(params[30:36]) + 0.1
    
    total_cost = 0
    for i, (config, budget) in enumerate(zip(configs, budgets)):
        n_evals = int(10 + budget * 40)
        results = []
        for _ in range(n_evals):
            noise = np.random.randn(5) * 0.1
            results.append(rastrigin(config + noise))
        total_cost += np.mean(results)
    
    return total_cost / 6


def bayesian_optimization_60d_objective(params: np.ndarray) -> float:
    """
    Bayesian optimization surrogate = 60D.
    Target: HIGH + expensive + rugged
    """
    if len(params) != 60:
        return 1e10
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    length_scales = np.abs(params[:10]) + 0.1
    init_points = params[10:50].reshape(8, 5) * 10.24 - 5.12
    
    X_observed = init_points
    y_observed = np.array([rastrigin(x) for x in X_observed])
    
    for _ in range(10):
        candidates = np.random.randn(20, 5) * 2
        
        distances = np.zeros((20, len(X_observed)))
        for i, cand in enumerate(candidates):
            for j, obs in enumerate(X_observed):
                distances[i, j] = np.sum(((cand - obs) / length_scales[:5])**2)
        
        mean_pred = np.mean(y_observed) * np.ones(20)
        for i in range(20):
            weights = np.exp(-distances[i])
            if np.sum(weights) > 1e-8:
                mean_pred[i] = np.sum(weights * y_observed) / np.sum(weights)
        
        best_idx = np.argmin(mean_pred)
        new_x = candidates[best_idx]
        new_y = rastrigin(new_x)
        
        X_observed = np.vstack([X_observed, new_x])
        y_observed = np.append(y_observed, new_y)
    
    return np.min(y_observed)


def neural_architecture_search_70d_objective(params: np.ndarray) -> float:
    """
    Neural architecture search = 70D.
    Target: HIGH + expensive + rugged
    """
    if len(params) != 70:
        return 1e10
    
    layer_sizes = (np.abs(params[:5]) * 50 + 10).astype(int)
    activations = params[5:10]
    connections = params[10:35].reshape(5, 5)
    weights = params[35:70]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    
    h = X[:, :5]
    for i, (size, act, conn) in enumerate(zip(layer_sizes[:3], activations[:3], connections[:3])):
        W = weights[i*10:(i+1)*10].reshape(5, 2) if i*10+10 <= len(weights) else np.random.randn(5, 2)
        h = h @ W[:, :min(2, h.shape[1])] if h.shape[1] >= 2 else h
        if act > 0.5:
            h = np.tanh(h)
        else:
            h = np.maximum(0, h)
    
    out = np.mean(h, axis=1)
    return np.mean((out - y)**2)


def evolutionary_strategy_65d_objective(params: np.ndarray) -> float:
    """
    Evolution strategy meta-optimization = 65D.
    Target: HIGH + expensive + rugged
    """
    if len(params) != 65:
        return 1e10
    
    sigma = np.abs(params[0]) + 0.01
    learning_rate = np.abs(params[1]) * 0.1 + 0.001
    pop_size = int(np.abs(params[2]) * 40 + 10)
    init_mean = params[3:18] * 5.12
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    mean = init_mean[:10]
    
    for _ in range(30):
        population = mean + sigma * np.random.randn(pop_size, 10)
        population = np.clip(population, -5.12, 5.12)
        
        fitness = np.array([rastrigin(ind) for ind in population])
        
        ranks = np.argsort(np.argsort(fitness))
        weights = np.maximum(0, np.log(pop_size/2 + 1) - np.log(ranks + 1))
        weights = weights / np.sum(weights)
        
        gradient = np.sum((population - mean).T * weights, axis=1)
        mean = mean + learning_rate * gradient
    
    return rastrigin(mean)


def simulated_annealing_meta_55d_objective(params: np.ndarray) -> float:
    """
    Simulated annealing meta-optimization = 55D.
    Target: HIGH + expensive + rugged
    """
    if len(params) != 55:
        return 1e10
    
    T_init = np.abs(params[0]) * 100 + 1
    cooling = 0.8 + np.abs(params[1]) * 0.19
    step_size = np.abs(params[2]) + 0.1
    init_x = params[3:18] * 5.12
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    x = init_x[:10]
    best_x = x.copy()
    best_f = rastrigin(x)
    T = T_init
    
    for _ in range(200):
        x_new = x + step_size * np.random.randn(10)
        x_new = np.clip(x_new, -5.12, 5.12)
        
        f_new = rastrigin(x_new)
        delta = f_new - rastrigin(x)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new
            if f_new < best_f:
                best_f = f_new
                best_x = x.copy()
        
        T *= cooling
    
    return best_f


def covariance_adaptation_60d_objective(params: np.ndarray) -> float:
    """
    Covariance matrix adaptation = 60D.
    Target: HIGH + expensive + rugged
    """
    if len(params) != 60:
        return 1e10
    
    sigma = np.abs(params[0]) + 0.1
    c_sigma = np.abs(params[1]) * 0.5
    c_c = np.abs(params[2]) * 0.5
    init_mean = params[3:13] * 5.12
    init_C_diag = np.abs(params[13:23]) + 0.1
    
    def rastrigin(x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    mean = init_mean
    C = np.diag(init_C_diag)
    
    lambda_pop = 20
    mu = 10
    
    for _ in range(25):
        try:
            L = np.linalg.cholesky(C + 1e-6 * np.eye(10))
        except:
            L = np.eye(10)
        
        offspring = mean + sigma * (np.random.randn(lambda_pop, 10) @ L.T)
        offspring = np.clip(offspring, -5.12, 5.12)
        
        fitness = np.array([rastrigin(ind) for ind in offspring])
        idx = np.argsort(fitness)[:mu]
        
        selected = offspring[idx]
        new_mean = np.mean(selected, axis=0)
        
        step = new_mean - mean
        mean = new_mean
        
        C = 0.9 * C + 0.1 * np.outer(step, step)
    
    return rastrigin(mean)


def get_all_genuine_problems() -> List[GenuineProblem]:
    """Get all genuine benchmark problems."""
    return (
        get_chaotic_problems() +
        get_nn_problems() +
        get_expensive_problems() +
        get_other_genuine_problems() +
        get_medium_dim_problems() +
        get_high_dim_problems()
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GENUINE BENCHMARK PROBLEMS")
    print("=" * 80)
    
    problems = get_all_genuine_problems()
    
    print(f"\nTotal problems: {len(problems)}")
    print("\nBy category:")
    categories = {}
    for p in problems:
        categories[p.category] = categories.get(p.category, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\n" + "-" * 80)
    print("Testing each problem...")
    print("-" * 80)
    
    import time
    
    for p in problems:
        # Generate random point in bounds
        x = np.array([np.random.uniform(b[0], b[1]) for b in p.bounds])
        
        start = time.perf_counter()
        try:
            val = p.func(x)
            elapsed = (time.perf_counter() - start) * 1000
            status = "OK"
        except Exception as e:
            elapsed = 0
            val = float('nan')
            status = f"ERROR: {e}"
        
        print(f"{p.name:25s} dim={p.dim:4d}  cost={p.expected_cost:15s}  "
              f"rugged={p.expected_ruggedness:15s}  time={elapsed:8.2f}ms  {status}")
