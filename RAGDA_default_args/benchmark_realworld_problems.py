"""
Real-World Benchmark Problems for RAGDA Parameter Optimization

This module provides real-world optimization problems with natural characteristics:
- Chaotic system parameter estimation
- Dynamical systems
- Neural network training
- ML hyperparameter optimization
- PDE problems
- Control and meta-optimization

All problems use Optuna-style API for compatibility with MARsOpt and cross-optimizer benchmarking.
All problems are DETERMINISTIC - same input always gives same output.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any


@dataclass
class BenchmarkProblem:
    """Real-world benchmark problem definition (Optuna-compatible)."""
    name: str
    objective: Callable  # Takes Optuna trial, returns float
    dimension: int
    bounds: List[Tuple[float, float]]
    known_optimum: Optional[float]  # None for real-world problems (unknown optimum)
    category: str  # 'chaotic', 'dynamical', 'nn_weights', 'ml_training', 'pde', etc.
    description: str


def _make_optuna_objective(func: Callable, bounds: List[Tuple[float, float]], dim: int) -> Callable:
    """Convert a raw function f(x) to Optuna objective f(trial)."""
    def optuna_objective(trial):
        x = np.array([
            trial.suggest_float(f'x{i}', bounds[i][0], bounds[i][1])
            for i in range(dim)
        ])
        return func(x)
    return optuna_objective


# =============================================================================
# BATCH 1: CHAOTIC SYSTEMS (16 functions)
# Parameter estimation for chaotic dynamical systems - highly rugged landscapes
# =============================================================================

# -------------------------
# Mackey-Glass System (4D)
# -------------------------

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


_MG_TARGET = _generate_mackey_glass_target()


def _mackey_glass(x: np.ndarray) -> float:
    """Mackey-Glass chaotic time series parameter estimation."""
    beta, gamma, n, tau_scale = x
    
    if beta <= 0 or gamma <= 0 or n <= 0 or tau_scale <= 0:
        return 1e10
    
    tau = int(max(1, tau_scale * 17))
    history_len = tau + 1
    T = 500
    dt = 0.1
    
    u = np.zeros(T)
    u[:history_len] = 0.9 + 0.2 * np.sin(np.linspace(0, 2*np.pi, history_len))
    
    try:
        for t in range(history_len, T):
            u_tau = u[t - tau]
            dudt = beta * u_tau / (1 + u_tau**n) - gamma * u[t-1]
            u[t] = u[t-1] + dt * dudt
            
            if np.isnan(u[t]) or np.isinf(u[t]) or abs(u[t]) > 1e6:
                return 1e10
    except:
        return 1e10
    
    start = T // 2
    mse = np.mean((u[start:] - _MG_TARGET[start:T])**2)
    return mse


# -------------------------
# Lorenz Attractor (3D)
# -------------------------

def _lorenz(x: np.ndarray) -> float:
    """Lorenz attractor parameter estimation."""
    sigma, rho, beta = x
    
    if sigma <= 0 or rho <= 0 or beta <= 0:
        return 1e10
    
    T = 1000
    dt = 0.01
    
    u, v, w = 1.0, 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            du = sigma * (v - u)
            dv = u * (rho - w) - v
            dw = u * v - beta * w
            
            u += dt * du
            v += dt * dv
            w += dt * dw
            
            if np.isnan(u) or np.isinf(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory.append([u, v, w])
    except:
        return 1e10
    
    trajectory = np.array(trajectory)
    
    target_mean = np.array([0.0, 0.0, 23.5])
    target_std = np.array([7.9, 9.0, 8.5])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


# -------------------------
# Hénon Map (2D)
# -------------------------

def _henon(x: np.ndarray) -> float:
    """Hénon map parameter estimation."""
    a, b = x
    
    if a <= 0 or a > 2 or abs(b) > 1:
        return 1e10
    
    T = 1000
    u, v = 0.1, 0.1
    trajectory_u = []
    
    try:
        for _ in range(T):
            u_new = 1 - a * u**2 + v
            v_new = b * u
            u, v = u_new, v_new
            
            if np.isnan(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory_u.append(u)
    except:
        return 1e10
    
    trajectory_u = np.array(trajectory_u[100:])
    
    target_mean = 0.26
    target_std = 0.70
    
    mean_err = (np.mean(trajectory_u) - target_mean)**2
    std_err = (np.std(trajectory_u) - target_std)**2
    
    return mean_err + std_err


# -------------------------
# Rössler Attractor (3D)
# -------------------------

def _rossler(x: np.ndarray) -> float:
    """Rössler attractor parameter estimation."""
    a, b, c = x
    
    if a <= 0 or b <= 0 or c <= 0:
        return 1e10
    
    T = 2000
    dt = 0.05
    u, v, w = 1.0, 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            du = -v - w
            dv = u + a * v
            dw = b + w * (u - c)
            
            u += dt * du
            v += dt * dv
            w += dt * dw
            
            if np.isnan(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory.append([u, v, w])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])
    
    target_mean = np.array([0.0, 0.0, 2.0])
    target_std = np.array([5.0, 5.0, 4.0])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


# -------------------------
# Logistic Map (1D)
# -------------------------

def _logistic_map(x: np.ndarray) -> float:
    """Logistic map parameter estimation."""
    r = x[0]
    
    if r <= 0 or r > 4:
        return 1e10
    
    T = 1000
    u = 0.5
    trajectory = []
    
    for _ in range(T):
        u = r * u * (1 - u)
        if np.isnan(u) or u < 0 or u > 1:
            return 1e10
        trajectory.append(u)
    
    trajectory = np.array(trajectory[100:])
    
    target_mean = 0.5
    target_std = 0.35
    
    mean_err = (np.mean(trajectory) - target_mean)**2
    std_err = (np.std(trajectory) - target_std)**2
    
    return mean_err + std_err


# -------------------------
# Coupled Logistic Maps (10D)
# -------------------------

def _coupled_logistic(x: np.ndarray) -> float:
    """Coupled logistic maps system."""
    r = x[0]
    eps = x[1]
    u = x[2:].copy()
    
    if r <= 0 or r > 4 or eps < 0 or eps > 1:
        return 1e10
    if len(u) < 3:
        u = np.concatenate([u, np.random.rand(3 - len(u))])
    
    T = 500
    trajectories = []
    
    try:
        for _ in range(T):
            u_new = np.zeros_like(u)
            for i in range(len(u)):
                coupling = (u[(i-1) % len(u)] + u[(i+1) % len(u)]) / 2
                u_new[i] = (1 - eps) * r * u[i] * (1 - u[i]) + eps * coupling
            u = np.clip(u_new, 0, 1)
            trajectories.append(u.copy())
    except:
        return 1e10
    
    trajectories = np.array(trajectories[100:])
    
    target_correlation = 0.3
    
    corrs = []
    for i in range(trajectories.shape[1] - 1):
        corr = np.corrcoef(trajectories[:, i], trajectories[:, i+1])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    
    if len(corrs) == 0:
        return 1e10
    
    avg_corr = np.mean(corrs)
    return (avg_corr - target_correlation)**2


# -------------------------
# Rabinovich-Fabrikant (2D)
# -------------------------

def _rabinovich_fabrikant(x: np.ndarray) -> float:
    """Rabinovich-Fabrikant equations parameter estimation."""
    gamma, alpha = x
    
    if gamma <= 0 or alpha <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    u, v, w = 0.1, 0.1, 0.1
    trajectory = []
    
    try:
        for _ in range(T):
            du = v * (w - 1 + u**2) + gamma * u
            dv = u * (3*w + 1 - u**2) + gamma * v
            dw = -2 * w * (alpha + u * v)
            
            u += dt * du
            v += dt * dv
            w += dt * dw
            
            if np.isnan(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory.append([u, v, w])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])
    
    target_std = np.array([1.5, 1.5, 0.8])
    
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    return std_err


# -------------------------
# Duffing Oscillator (5D)
# -------------------------

def _duffing(x: np.ndarray) -> float:
    """Duffing oscillator parameter estimation."""
    delta, alpha, beta, gamma, omega = x
    
    T = 5000
    dt = 0.01
    u, v = 0.1, 0.0
    trajectory = []
    
    try:
        for i in range(T):
            t = i * dt
            du = v
            dv = -delta * v - alpha * u - beta * u**3 + gamma * np.cos(omega * t)
            
            u += dt * du
            v += dt * dv
            
            if np.isnan(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory.append([u, v])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[1000:])
    
    target_std = np.array([1.0, 1.2])
    
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    return std_err


# -------------------------
# Double Pendulum (4D)
# -------------------------

def _double_pendulum(x: np.ndarray) -> float:
    """Double pendulum parameter estimation."""
    m1, m2, l1, l2 = x
    g = 9.81
    
    if m1 <= 0 or m2 <= 0 or l1 <= 0 or l2 <= 0:
        return 1e10
    
    theta1, theta2 = np.pi/2, np.pi/2
    omega1, omega2 = 0.0, 0.0
    
    T = 1000
    dt = 0.005
    trajectory = []
    
    try:
        for _ in range(T):
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
    
    target_energy_ratio = 0.5
    
    ke1 = 0.5 * m1 * (l1 * trajectory[:, 2])**2
    ke2 = 0.5 * m2 * (l2 * trajectory[:, 3])**2
    
    actual_ratio = np.mean(ke1) / (np.mean(ke1) + np.mean(ke2) + 1e-10)
    
    return (actual_ratio - target_energy_ratio)**2


# -------------------------
# Lorenz96 (20D)
# -------------------------

def _lorenz96_20d(x: np.ndarray) -> float:
    """Lorenz 96 model - weather-like dynamics (20D state)."""
    if len(x) != 20:
        return 1e10
    
    F = 8.0
    n = 20
    u = x.copy()
    
    dt = 0.01
    trajectory = []
    
    for _ in range(500):
        du = np.zeros(n)
        for i in range(n):
            du[i] = (u[(i+1) % n] - u[(i-2) % n]) * u[(i-1) % n] - u[i] + F
        u = u + dt * du
        
        if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e6):
            return 1e10
        
        trajectory.append(u.copy())
    
    trajectory = np.array(trajectory[100:])
    
    target_mean = F
    target_std = 3.5
    
    mean_err = (np.mean(trajectory) - target_mean)**2
    std_err = (np.std(trajectory) - target_std)**2
    
    return mean_err + std_err


# -------------------------
# Lorenz96 Extended (60D)
# -------------------------

def _lorenz96_60d(x: np.ndarray) -> float:
    """Extended Lorenz 96 model (60D)."""
    if len(x) != 60:
        return 1e10
    
    F = 8.0
    n = 60
    u = x.copy()
    
    dt = 0.01
    trajectory = []
    
    for _ in range(300):
        du = np.zeros(n)
        for i in range(n):
            du[i] = (u[(i+1) % n] - u[(i-2) % n]) * u[(i-1) % n] - u[i] + F
        u = u + dt * du
        
        if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e6):
            return 1e10
        
        trajectory.append(u.copy())
    
    trajectory = np.array(trajectory[50:])
    
    target_mean = F
    target_std = 3.5
    
    mean_err = (np.mean(trajectory) - target_mean)**2
    std_err = (np.std(trajectory) - target_std)**2
    
    return mean_err + std_err


# -------------------------
# Coupled Map Lattice (64D)
# -------------------------

def _coupled_map_lattice(x: np.ndarray) -> float:
    """Coupled map lattice (64D)."""
    if len(x) != 64:
        return 1e10
    
    r = 3.8
    eps = 0.3
    u = np.clip(x, 0, 1)
    
    T = 200
    trajectory = []
    
    try:
        for _ in range(T):
            u_new = np.zeros_like(u)
            for i in range(len(u)):
                local = r * u[i] * (1 - u[i])
                coupling = 0.5 * (u[(i-1) % len(u)] + u[(i+1) % len(u)])
                u_new[i] = (1 - eps) * local + eps * coupling
            u = np.clip(u_new, 0, 1)
            trajectory.append(u.copy())
    except:
        return 1e10
    
    trajectory = np.array(trajectory[50:])
    
    target_std = 0.25
    actual_std = np.std(trajectory)
    
    return (actual_std - target_std)**2


# -------------------------
# Hénon Extended (20D)
# -------------------------

def _henon_20d(x: np.ndarray) -> float:
    """Extended Hénon map (20D coupled system)."""
    if len(x) != 20:
        return 1e10
    
    a, b = 1.4, 0.3
    eps = 0.1
    u = x[:10].copy()
    v = x[10:].copy()
    
    T = 500
    trajectory = []
    
    try:
        for _ in range(T):
            u_new = np.zeros_like(u)
            v_new = np.zeros_like(v)
            for i in range(len(u)):
                coupling_u = eps * 0.5 * (u[(i-1) % len(u)] + u[(i+1) % len(u)])
                coupling_v = eps * 0.5 * (v[(i-1) % len(v)] + v[(i+1) % len(v)])
                u_new[i] = (1 - eps) * (1 - a * u[i]**2 + v[i]) + coupling_u
                v_new[i] = (1 - eps) * (b * u[i]) + coupling_v
            u, v = u_new, v_new
            
            if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e6):
                return 1e10
            
            trajectory.append(np.concatenate([u, v]))
    except:
        return 1e10
    
    trajectory = np.array(trajectory[100:])
    
    target_std = 0.6
    actual_std = np.std(trajectory)
    
    return (actual_std - target_std)**2


# -------------------------
# Coupled Logistic Maps (100D)
# -------------------------

def _coupled_logistic_100d(x: np.ndarray) -> float:
    """Large-scale coupled logistic maps (100D)."""
    if len(x) != 100:
        return 1e10
    
    r = 3.9
    eps = 0.2
    u = np.clip(x, 0, 1)
    
    T = 150
    trajectory = []
    
    try:
        for _ in range(T):
            u_new = np.zeros_like(u)
            for i in range(len(u)):
                local = r * u[i] * (1 - u[i])
                coupling = 0.5 * (u[(i-1) % len(u)] + u[(i+1) % len(u)])
                u_new[i] = (1 - eps) * local + eps * coupling
            u = np.clip(u_new, 0, 1)
            trajectory.append(u.copy())
    except:
        return 1e10
    
    trajectory = np.array(trajectory[50:])
    
    target_std = 0.3
    actual_std = np.std(trajectory)
    
    return (actual_std - target_std)**2


# -------------------------
# Spatiotemporal Chaos (60D)
# -------------------------

def _spatiotemporal_chaos(x: np.ndarray) -> float:
    """Spatiotemporal chaotic system (60D)."""
    if len(x) != 60:
        return 1e10
    
    a = 1.8
    eps = 0.4
    u = x.copy()
    
    T = 200
    trajectory = []
    
    try:
        for _ in range(T):
            u_new = np.zeros_like(u)
            for i in range(len(u)):
                laplacian = u[(i-1) % len(u)] + u[(i+1) % len(u)] - 2*u[i]
                u_new[i] = a * np.tanh(u[i]) + eps * laplacian
            u = u_new
            
            if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e6):
                return 1e10
            
            trajectory.append(u.copy())
    except:
        return 1e10
    
    trajectory = np.array(trajectory[50:])
    
    target_std = 1.0
    actual_std = np.std(trajectory)
    
    return (actual_std - target_std)**2


# -------------------------
# Standard Map Chain (30D)
# -------------------------

def _standard_map_chain(x: np.ndarray) -> float:
    """Chain of coupled standard maps (30D)."""
    if len(x) != 30:
        return 1e10
    
    K = 0.5
    eps = 0.1
    
    n_maps = 15
    p = x[:n_maps].copy()
    q = x[n_maps:].copy()
    
    T = 300
    trajectory = []
    
    try:
        for _ in range(T):
            p_new = p.copy()
            q_new = q.copy()
            
            for i in range(n_maps):
                coupling_p = eps * 0.5 * (p[(i-1) % n_maps] + p[(i+1) % n_maps])
                p_new[i] = (p[i] + K * np.sin(q[i]) + coupling_p) % (2 * np.pi)
                q_new[i] = (q[i] + p_new[i]) % (2 * np.pi)
            
            p, q = p_new, q_new
            trajectory.append(np.concatenate([p, q]))
    except:
        return 1e10
    
    trajectory = np.array(trajectory[100:])
    
    target_std = 1.5
    actual_std = np.std(trajectory)
    
    return (actual_std - target_std)**2


# =============================================================================
# BATCH 2: DYNAMICAL SYSTEMS (6 functions)
# Non-chaotic but complex dynamical behavior - oscillations, synchronization
# =============================================================================

# -------------------------
# Lotka-Volterra (4D)
# -------------------------

def _lotka_volterra(x: np.ndarray) -> float:
    """Lotka-Volterra (predator-prey) equations parameter estimation."""
    alpha, beta, delta, gamma = x
    
    if alpha <= 0 or beta <= 0 or delta <= 0 or gamma <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    u, v = 1.0, 1.0
    trajectory = []
    
    try:
        for _ in range(T):
            du = alpha * u - beta * u * v
            dv = delta * u * v - gamma * v
            
            u += dt * du
            v += dt * dv
            
            if np.isnan(u) or u <= 0 or v <= 0 or u > 1e6:
                return 1e10
            
            trajectory.append([u, v])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[200:])
    
    target_mean = np.array([3.0, 1.5])
    target_std = np.array([2.0, 1.0])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    std_err = np.mean((trajectory.std(axis=0) - target_std)**2)
    
    return mean_err + std_err


# -------------------------
# Lotka-Volterra 4 Species (8D)
# -------------------------

def _lotka_volterra_4species(x: np.ndarray) -> float:
    """4-species Lotka-Volterra competition model."""
    if len(x) != 8:
        return 1e10
    
    r = x[:4]
    A = np.array([
        [1.0, x[4], 0.1, 0.1],
        [x[5], 1.0, x[6], 0.1],
        [0.1, x[7], 1.0, 0.2],
        [0.1, 0.1, 0.2, 1.0]
    ])
    
    if np.any(r <= 0):
        return 1e10
    
    T = 1000
    dt = 0.05
    u = np.array([0.5, 0.5, 0.5, 0.5])
    trajectory = []
    
    try:
        for _ in range(T):
            du = u * (r - A @ u)
            u = u + dt * du
            u = np.maximum(u, 1e-10)
            
            if np.any(np.isnan(u)) or np.any(u > 1e6):
                return 1e10
            
            trajectory.append(u.copy())
    except:
        return 1e10
    
    trajectory = np.array(trajectory[200:])
    
    target_mean = np.array([0.4, 0.4, 0.4, 0.4])
    
    mean_err = np.mean((trajectory.mean(axis=0) - target_mean)**2)
    stability = np.mean(trajectory.std(axis=0))
    
    return mean_err + 0.5 * stability


# -------------------------
# Van der Pol (1D)
# -------------------------

def _van_der_pol(x: np.ndarray) -> float:
    """Van der Pol oscillator parameter estimation."""
    mu = x[0]
    
    if mu <= 0:
        return 1e10
    
    T = 2000
    dt = 0.01
    u, v = 0.1, 0.0
    trajectory = []
    
    try:
        for _ in range(T):
            du = v
            dv = mu * (1 - u**2) * v - u
            
            u += dt * du
            v += dt * dv
            
            if np.isnan(u) or abs(u) > 1e6:
                return 1e10
            
            trajectory.append([u, v])
    except:
        return 1e10
    
    trajectory = np.array(trajectory[500:])
    
    target_amplitude = 2.0
    amplitude = np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])
    
    return (amplitude - target_amplitude)**2


# -------------------------
# Coupled Oscillators (15D)
# -------------------------

def _coupled_oscillators(x: np.ndarray) -> float:
    """Coupled harmonic oscillators parameter estimation (15D)."""
    if len(x) != 15:
        return 1e10
    
    n_osc = 5
    masses = x[:5]
    springs = x[5:10]
    couplings = x[10:15]
    
    if np.any(masses <= 0) or np.any(springs <= 0):
        return 1e10
    
    T = 500
    dt = 0.01
    u = np.zeros(n_osc)
    v = np.zeros(n_osc)
    u[0] = 1.0
    
    energy_history = []
    
    for _ in range(T):
        forces = -springs * u
        for i in range(n_osc - 1):
            coupling_force = couplings[i] * (u[i+1] - u[i])
            forces[i] += coupling_force
            forces[i+1] -= coupling_force
        
        a = forces / masses
        v += dt * a
        u += dt * v
        
        if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e6):
            return 1e10
        
        energy = 0.5 * np.sum(masses * v**2) + 0.5 * np.sum(springs * u**2)
        energy_history.append(energy)
    
    energy_history = np.array(energy_history)
    return np.std(energy_history) / (np.mean(energy_history) + 1e-10)


# -------------------------
# Kuramoto Oscillators (20D)
# -------------------------

def _kuramoto_oscillators(x: np.ndarray) -> float:
    """Kuramoto coupled oscillators - synchronization dynamics (20D)."""
    if len(x) != 20:
        return 1e10
    
    n = 10
    omega = x[:n]
    K = x[n:]
    
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


# -------------------------
# Neural Field (70D)
# -------------------------

def _neural_field(x: np.ndarray) -> float:
    """Neural field dynamics (70D)."""
    if len(x) != 70:
        return 1e10
    
    u = x[:49].reshape(7, 7)
    weights = x[49:70].reshape(3, 7)
    
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


# =============================================================================
# REGISTRY: CHAOTIC SYSTEMS (Batch 1)
# =============================================================================

_CHAOTIC_REGISTRY = {
    'MackeyGlass-4D': BenchmarkProblem(
        name='MackeyGlass-4D',
        objective=_make_optuna_objective(_mackey_glass, [(0.1, 0.5), (0.05, 0.2), (5, 15), (0.5, 2.0)], 4),
        dimension=4,
        bounds=[(0.1, 0.5), (0.05, 0.2), (5, 15), (0.5, 2.0)],
        known_optimum=None,
        category='chaotic',
        description='Mackey-Glass chaotic time series parameter estimation'
    ),
    'Lorenz-3D': BenchmarkProblem(
        name='Lorenz-3D',
        objective=_make_optuna_objective(_lorenz, [(1, 20), (10, 50), (0.5, 5)], 3),
        dimension=3,
        bounds=[(1, 20), (10, 50), (0.5, 5)],
        known_optimum=None,
        category='chaotic',
        description='Lorenz attractor parameter estimation'
    ),
    'Henon-2D': BenchmarkProblem(
        name='Henon-2D',
        objective=_make_optuna_objective(_henon, [(1.0, 1.5), (0.1, 0.5)], 2),
        dimension=2,
        bounds=[(1.0, 1.5), (0.1, 0.5)],
        known_optimum=None,
        category='chaotic',
        description='Hénon map parameter estimation'
    ),
    'Rossler-3D': BenchmarkProblem(
        name='Rossler-3D',
        objective=_make_optuna_objective(_rossler, [(0.05, 0.5), (0.05, 0.5), (3, 10)], 3),
        dimension=3,
        bounds=[(0.05, 0.5), (0.05, 0.5), (3, 10)],
        known_optimum=None,
        category='chaotic',
        description='Rössler attractor parameter estimation'
    ),
    'LogisticMap-1D': BenchmarkProblem(
        name='LogisticMap-1D',
        objective=_make_optuna_objective(_logistic_map, [(3.5, 4.0)], 1),
        dimension=1,
        bounds=[(3.5, 4.0)],
        known_optimum=None,
        category='chaotic',
        description='Logistic map parameter estimation'
    ),
    'CoupledLogistic-10D': BenchmarkProblem(
        name='CoupledLogistic-10D',
        objective=_make_optuna_objective(_coupled_logistic, [(3.5, 4.0), (0.0, 0.5)] + [(0.1, 0.9)]*8, 10),
        dimension=10,
        bounds=[(3.5, 4.0), (0.0, 0.5)] + [(0.1, 0.9)]*8,
        known_optimum=None,
        category='chaotic',
        description='Coupled logistic maps system identification'
    ),
    'RabinovichFabrikant-2D': BenchmarkProblem(
        name='RabinovichFabrikant-2D',
        objective=_make_optuna_objective(_rabinovich_fabrikant, [(0.1, 1.5), (0.5, 2.0)], 2),
        dimension=2,
        bounds=[(0.1, 1.5), (0.5, 2.0)],
        known_optimum=None,
        category='chaotic',
        description='Rabinovich-Fabrikant equations parameter estimation'
    ),
    'Duffing-5D': BenchmarkProblem(
        name='Duffing-5D',
        objective=_make_optuna_objective(_duffing, [(0.1, 0.5), (-2, 0), (0.5, 2), (0.1, 1), (0.5, 2)], 5),
        dimension=5,
        bounds=[(0.1, 0.5), (-2, 0), (0.5, 2), (0.1, 1), (0.5, 2)],
        known_optimum=None,
        category='chaotic',
        description='Duffing oscillator parameter estimation'
    ),
    'DoublePendulum-4D': BenchmarkProblem(
        name='DoublePendulum-4D',
        objective=_make_optuna_objective(_double_pendulum, [(0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2)], 4),
        dimension=4,
        bounds=[(0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2)],
        known_optimum=None,
        category='chaotic',
        description='Double pendulum parameter estimation'
    ),
    'Lorenz96-20D': BenchmarkProblem(
        name='Lorenz96-20D',
        objective=_make_optuna_objective(_lorenz96_20d, [(-10, 10)]*20, 20),
        dimension=20,
        bounds=[(-10, 10)]*20,
        known_optimum=None,
        category='chaotic',
        description='Lorenz 96 weather-like dynamics (20D)'
    ),
    'Lorenz96Extended-60D': BenchmarkProblem(
        name='Lorenz96Extended-60D',
        objective=_make_optuna_objective(_lorenz96_60d, [(-10, 10)]*60, 60),
        dimension=60,
        bounds=[(-10, 10)]*60,
        known_optimum=None,
        category='chaotic',
        description='Extended Lorenz 96 model (60D)'
    ),
    'CoupledMapLattice-64D': BenchmarkProblem(
        name='CoupledMapLattice-64D',
        objective=_make_optuna_objective(_coupled_map_lattice, [(0, 1)]*64, 64),
        dimension=64,
        bounds=[(0, 1)]*64,
        known_optimum=None,
        category='chaotic',
        description='Coupled map lattice (64D)'
    ),
    'HenonExtended-20D': BenchmarkProblem(
        name='HenonExtended-20D',
        objective=_make_optuna_objective(_henon_20d, [(-2, 2)]*20, 20),
        dimension=20,
        bounds=[(-2, 2)]*20,
        known_optimum=None,
        category='chaotic',
        description='Extended Hénon map (20D coupled system)'
    ),
    'CoupledLogisticMaps-100D': BenchmarkProblem(
        name='CoupledLogisticMaps-100D',
        objective=_make_optuna_objective(_coupled_logistic_100d, [(0, 1)]*100, 100),
        dimension=100,
        bounds=[(0, 1)]*100,
        known_optimum=None,
        category='chaotic',
        description='Large-scale coupled logistic maps (100D)'
    ),
    'SpatiotemporalChaos-60D': BenchmarkProblem(
        name='SpatiotemporalChaos-60D',
        objective=_make_optuna_objective(_spatiotemporal_chaos, [(-2, 2)]*60, 60),
        dimension=60,
        bounds=[(-2, 2)]*60,
        known_optimum=None,
        category='chaotic',
        description='Spatiotemporal chaotic system (60D)'
    ),
    'StandardMapChain-30D': BenchmarkProblem(
        name='StandardMapChain-30D',
        objective=_make_optuna_objective(_standard_map_chain, [(0, 2*np.pi)]*30, 30),
        dimension=30,
        bounds=[(0, 2*np.pi)]*30,
        known_optimum=None,
        category='chaotic',
        description='Chain of coupled standard maps (30D)'
    ),
}

# =============================================================================
# REGISTRY: DYNAMICAL SYSTEMS (Batch 2)
# =============================================================================

_DYNAMICAL_REGISTRY = {
    'LotkaVolterra-4D': BenchmarkProblem(
        name='LotkaVolterra-4D',
        objective=_make_optuna_objective(_lotka_volterra, [(0.5, 3), (0.5, 2), (0.5, 2), (1, 5)], 4),
        dimension=4,
        bounds=[(0.5, 3), (0.5, 2), (0.5, 2), (1, 5)],
        known_optimum=None,
        category='dynamical',
        description='Lotka-Volterra predator-prey parameter estimation'
    ),
    'LotkaVolterra4Species-8D': BenchmarkProblem(
        name='LotkaVolterra4Species-8D',
        objective=_make_optuna_objective(_lotka_volterra_4species, [(0.5, 2)]*4 + [(0.1, 0.9)]*4, 8),
        dimension=8,
        bounds=[(0.5, 2)]*4 + [(0.1, 0.9)]*4,
        known_optimum=None,
        category='dynamical',
        description='4-species Lotka-Volterra competition model'
    ),
    'VanDerPol-1D': BenchmarkProblem(
        name='VanDerPol-1D',
        objective=_make_optuna_objective(_van_der_pol, [(0.1, 5)], 1),
        dimension=1,
        bounds=[(0.1, 5)],
        known_optimum=None,
        category='dynamical',
        description='Van der Pol oscillator parameter estimation'
    ),
    'CoupledOscillators-15D': BenchmarkProblem(
        name='CoupledOscillators-15D',
        objective=_make_optuna_objective(_coupled_oscillators, [(0.1, 2)]*5 + [(0.1, 2)]*5 + [(0, 1)]*5, 15),
        dimension=15,
        bounds=[(0.1, 2)]*5 + [(0.1, 2)]*5 + [(0, 1)]*5,
        known_optimum=None,
        category='dynamical',
        description='Coupled harmonic oscillators parameter estimation'
    ),
    'KuramotoOscillators-20D': BenchmarkProblem(
        name='KuramotoOscillators-20D',
        objective=_make_optuna_objective(_kuramoto_oscillators, [(-2, 2)]*10 + [(0, 2)]*10, 20),
        dimension=20,
        bounds=[(-2, 2)]*10 + [(0, 2)]*10,
        known_optimum=None,
        category='dynamical',
        description='Kuramoto coupled oscillators synchronization dynamics'
    ),
    'NeuralField-70D': BenchmarkProblem(
        name='NeuralField-70D',
        objective=_make_optuna_objective(_neural_field, [(-1, 1)]*49 + [(-0.5, 0.5)]*21, 70),
        dimension=70,
        bounds=[(-1, 1)]*49 + [(-0.5, 0.5)]*21,
        known_optimum=None,
        category='dynamical',
        description='Neural field dynamics (70D)'
    ),
}

# =============================================================================
# BATCH 3: Neural Network Weight Optimization (16 functions)
# =============================================================================
# Neural networks trained on various tasks - true high-dimensional problems
# with complex loss landscapes including saddle points, local minima, symmetries

def _nn_xor_17d(x: np.ndarray) -> float:
    """Small NN for XOR problem (2->4->1 = 17 params)."""
    if len(x) != 17:
        return 1e10
    
    W1 = x[:8].reshape(2, 4)
    b1 = x[8:12]
    W2 = x[12:16].reshape(4, 1)
    b2 = x[16]
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    
    # Forward pass
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    h = sigmoid(X @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    
    # Binary cross-entropy
    eps = 1e-7
    loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _nn_regression_89d(x: np.ndarray) -> float:
    """NN for regression (1->8->8->1 without b2 = 89 params)."""
    # Correct calc: 1*8 + 8 + 8*8 + 0 + 8*1 + 1 = 8+8+64+8+1 = 89
    if len(x) != 89:
        return 1e10
    
    idx = 0
    W1 = x[idx:idx+8].reshape(1, 8); idx += 8
    b1 = x[idx:idx+8]; idx += 8
    W2 = x[idx:idx+64].reshape(8, 8); idx += 64
    # Skip b2 to match 89 params
    W3 = x[idx:idx+8].reshape(8, 1); idx += 8
    b3 = x[idx]; idx += 1
    
    # Training data
    X_train = np.linspace(-2*np.pi, 2*np.pi, 50).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.5 * np.sin(3*X_train)
    
    def tanh(z):
        return np.tanh(np.clip(z, -500, 500))
    
    h1 = tanh(X_train @ W1 + b1)
    h2 = tanh(h1 @ W2)  # No b2
    out = h2 @ W3 + b3
    
    loss = np.mean((out - y_train)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _nn_mnist_1074d(x: np.ndarray) -> float:
    """NN for digit classification (64->16->2 = 1074 params)."""
    expected_dim = 64*16 + 16 + 16*2 + 2  # 1074
    if len(x) != expected_dim:
        return 1e10
    
    idx = 0
    W1 = x[idx:idx+1024].reshape(64, 16); idx += 1024
    b1 = x[idx:idx+16]; idx += 16
    W2 = x[idx:idx+32].reshape(16, 2); idx += 32
    b2 = x[idx:idx+2]; idx += 2
    
    # Simplified MNIST-like data (8x8 images)
    np.random.seed(42)
    X = np.random.randn(100, 64) * 0.5
    y = (np.sum(X[:, :32], axis=1) > 0).astype(int)
    y_onehot = np.eye(2)[y]
    
    def relu(z):
        return np.maximum(0, np.clip(z, -500, 500))
    
    def softmax(z):
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(np.clip(z - z_max, -500, 500))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
    
    h = relu(X @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    
    # Cross-entropy loss
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-7), axis=1))
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _nn_large_1377d(x: np.ndarray) -> float:
    """Large NN (10->32->31->1 = 1377 params)."""
    # 10*32 + 32 + 32*31 + 2 + 31*1 = 320+32+992+2+31 = 1377
    if len(x) != 1377:
        return 1e10
    
    idx = 0
    W1 = x[idx:idx+320].reshape(10, 32); idx += 320
    b1 = x[idx:idx+32]; idx += 32
    W2 = x[idx:idx+992].reshape(32, 31); idx += 992
    b2 = x[idx:idx+2]; idx += 2  # Only 2 bias terms for dimension matching
    W3 = x[idx:idx+31].reshape(31, 1); idx += 31
    # No b3
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.sum(np.sin(X), axis=1) + np.sum(X[:, :5] * X[:, 5:], axis=1)
    y = y.reshape(-1, 1)
    
    def relu(z):
        return np.maximum(0, np.clip(z, -500, 500))
    
    h1 = relu(X @ W1 + b1)
    # Only add b2 to first 2 elements
    h2_raw = h1 @ W2
    h2_raw[:, :2] += b2
    h2 = relu(h2_raw)
    out = h2 @ W3  # No b3
    
    loss = np.mean((out - y)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _nn_medium_20d(x: np.ndarray) -> float:
    """Medium NN (1->10 = 20 params)."""
    if len(x) != 20:
        return 1e10
    
    W1 = x[:10].reshape(1, 10)
    b1 = x[10:20]
    
    X = np.linspace(-3, 3, 50).reshape(-1, 1)
    y = np.sin(X) + 0.3 * np.sin(5*X)
    
    h = np.tanh(X @ W1 + b1)
    out = np.mean(h, axis=1, keepdims=True)
    
    loss = np.mean((out - y)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _nn_deep_100d(x: np.ndarray) -> float:
    """Deep NN (5->10->10->1 = 100 params)."""
    if len(x) != 100:
        return 1e10
    
    W1 = x[:50].reshape(5, 10)
    b1 = x[50:60]
    W2 = x[60:90].reshape(10, 3)
    b2 = x[90:93]
    W3 = x[93:99].reshape(3, 2)
    b3 = x[99]
    
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1] * X[:, 2])
    
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    out = np.mean(h2 @ W3, axis=1) + b3
    
    loss = np.mean((out - y)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _autoencoder_80d(x: np.ndarray) -> float:
    """Autoencoder (10->5->10 = 80 params)."""
    if len(x) != 80:
        return 1e10
    
    W_enc = x[:50].reshape(10, 5)
    b_enc = x[50:55]
    W_dec = x[55:75].reshape(5, 4)
    b_dec = x[75:79]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    encoded = np.tanh(X @ W_enc + b_enc)
    decoded = encoded @ W_dec + b_dec
    
    loss = np.mean((X[:, :4] - decoded)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _rbm_60d(x: np.ndarray) -> float:
    """Restricted Boltzmann Machine (6x10 = 60 params)."""
    if len(x) != 60:
        return 1e10
    
    W = x.reshape(6, 10)
    
    np.random.seed(42)
    v = (np.random.rand(100, 6) > 0.5).astype(float)
    
    h_prob = 1 / (1 + np.exp(-np.clip(v @ W, -500, 500)))
    v_recon_prob = 1 / (1 + np.exp(-np.clip(h_prob @ W.T, -500, 500)))
    
    loss = np.mean((v - v_recon_prob)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _word_embedding_75d(x: np.ndarray) -> float:
    """Word embedding optimization (15 words x 5D = 75 params)."""
    if len(x) != 75:
        return 1e10
    
    embeddings = x.reshape(15, 5)
    
    similar_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    dissimilar_pairs = [(0, 8), (1, 9), (2, 10), (3, 11)]
    
    loss = 0
    for i, j in similar_pairs:
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        loss += dist**2
    
    for i, j in dissimilar_pairs:
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        loss += max(0, 2 - dist)**2
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _hopfield_64d(x: np.ndarray) -> float:
    """Hopfield network (8x8 = 64 params)."""
    if len(x) != 64:
        return 1e10
    
    W = x.reshape(8, 8)
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
    
    if np.isnan(total_energy) or np.isinf(total_energy):
        return 1e10
    
    return total_energy


def _sparse_autoencoder_100d(x: np.ndarray) -> float:
    """Sparse autoencoder with sparsity penalty (100 params)."""
    if len(x) != 100:
        return 1e10
    
    # 10->5->10: W_enc(10x5=50), b_enc(5), W_dec(5x10=50), b_dec(10) but that's 115
    # Let's use 10->6->10: W_enc(10x6=60), b_enc(6), W_dec(6x4=24), b_dec(4) = 94, not 100
    # Use 10->5->10: W_enc(50), b_enc(5), W_dec(5x9=45), no b_dec = 100
    W_enc = x[:50].reshape(10, 5)
    b_enc = x[50:55]
    W_dec = x[55:100].reshape(5, 9)
    # No b_dec to make it exactly 100
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    encoded = 1 / (1 + np.exp(-np.clip(X @ W_enc + b_enc, -500, 500)))
    decoded = encoded @ W_dec  # No bias
    
    recon_loss = np.mean((X[:, :9] - decoded)**2)
    sparsity = np.mean(encoded)
    sparsity_loss = (sparsity - 0.1)**2
    
    loss = recon_loss + 0.5 * sparsity_loss
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _vae_90d(x: np.ndarray) -> float:
    """Variational autoencoder (90 params)."""
    if len(x) != 90:
        return 1e10
    
    W_enc = x[:40].reshape(8, 5)
    b_enc = x[40:45]
    W_dec = x[45:85].reshape(5, 8)
    b_dec = x[85:90]
    
    np.random.seed(42)
    X = np.random.randn(100, 8)
    
    z_mean = X @ W_enc + b_enc
    z = z_mean + 0.1 * np.random.randn(*z_mean.shape)
    
    x_recon = z @ W_dec + b_dec
    
    recon_loss = np.mean((X - x_recon)**2)
    kl_loss = 0.5 * np.mean(z_mean**2)
    
    loss = recon_loss + 0.1 * kl_loss
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _denoising_ae_80d(x: np.ndarray) -> float:
    """Denoising autoencoder (80 params)."""
    if len(x) != 80:
        return 1e10
    
    W_enc = x[:50].reshape(10, 5)
    b_enc = x[50:55]
    W_dec = x[55:75].reshape(5, 4)
    b_dec = x[75:79]
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    X_noisy = X + 0.3 * np.random.randn(*X.shape)
    
    encoded = np.tanh(X_noisy @ W_enc + b_enc)
    decoded = encoded @ W_dec + b_dec
    
    loss = np.mean((X[:, :4] - decoded)**2)
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _contrastive_learning_70d(x: np.ndarray) -> float:
    """Contrastive learning projection head (70 params)."""
    if len(x) != 70:
        return 1e10
    
    W1 = x[:40].reshape(8, 5)
    b1 = x[40:45]
    W2 = x[45:65].reshape(5, 4)
    b2 = x[65:69]
    
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
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _neural_hessian_80d(x: np.ndarray) -> float:
    """Neural Hessian conditioning (80 params, expensive)."""
    if len(x) != 80:
        return 1e10
    
    layer_size = 6  # 2 * 6 * 6 = 72, close to 80
    n_weights = 2 * layer_size * layer_size
    
    if len(x) < n_weights:
        x_padded = np.concatenate([x, np.zeros(n_weights - len(x))])
    else:
        x_padded = x[:n_weights]
    
    W1 = x_padded[:layer_size*layer_size].reshape(layer_size, layer_size)
    W2 = x_padded[layer_size*layer_size:2*layer_size*layer_size].reshape(layer_size, layer_size)
    
    np.random.seed(111)
    X = np.random.randn(100, layer_size)
    y = np.random.randn(100, layer_size)
    
    h = np.tanh(X @ W1)
    y_pred = h @ W2
    
    residual = y_pred - y
    J = h
    H_approx = J.T @ J / len(X) + 0.01 * np.eye(layer_size)
    
    try:
        eigvals = np.linalg.eigvalsh(H_approx)
        eigvals = np.maximum(eigvals, 1e-10)
        condition_number = eigvals[-1] / eigvals[0]
        loss = np.mean(residual**2) + 0.01 * condition_number
    except:
        return 1e10
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


def _neural_hessian_100d(x: np.ndarray) -> float:
    """Large neural Hessian conditioning (100 params, expensive)."""
    if len(x) != 100:
        return 1e10
    
    layer_size = 7  # 2 * 7 * 7 = 98, close to 100
    n_weights = 2 * layer_size * layer_size
    
    if len(x) < n_weights:
        x_padded = np.concatenate([x, np.zeros(n_weights - len(x))])
    else:
        x_padded = x[:n_weights]
    
    W1 = x_padded[:layer_size*layer_size].reshape(layer_size, layer_size)
    W2 = x_padded[layer_size*layer_size:2*layer_size*layer_size].reshape(layer_size, layer_size)
    
    np.random.seed(111)
    X = np.random.randn(100, layer_size)
    y = np.random.randn(100, layer_size)
    
    h = np.tanh(X @ W1)
    y_pred = h @ W2
    
    residual = y_pred - y
    J = h
    H_approx = J.T @ J / len(X) + 0.01 * np.eye(layer_size)
    
    try:
        eigvals = np.linalg.eigvalsh(H_approx)
        eigvals = np.maximum(eigvals, 1e-10)
        condition_number = eigvals[-1] / eigvals[0]
        loss = np.mean(residual**2) + 0.01 * condition_number
    except:
        return 1e10
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    
    return loss


# Registry for NN weight problems
_NN_WEIGHTS_REGISTRY: Dict[str, BenchmarkProblem] = {
    'NN-XOR-17D': BenchmarkProblem(
        name='NN-XOR-17D',
        objective=_make_optuna_objective(_nn_xor_17d, [(-3, 3)] * 17, 17),
        dimension=17,
        bounds=[(-3, 3)] * 17,
        known_optimum=None,
        category='nn_weights',
        description='Small neural network for XOR problem (2->4->1)'
    ),
    'NN-Regression-89D': BenchmarkProblem(
        name='NN-Regression-89D',
        objective=_make_optuna_objective(_nn_regression_89d, [(-2, 2)] * 89, 89),
        dimension=89,
        bounds=[(-2, 2)] * 89,
        known_optimum=None,
        category='nn_weights',
        description='Neural network regression 1->8->8->1'
    ),
    'NN-MNIST-1074D': BenchmarkProblem(
        name='NN-MNIST-1074D',
        objective=_make_optuna_objective(_nn_mnist_1074d, [(-1, 1)] * 1074, 1074),
        dimension=1074,
        bounds=[(-1, 1)] * 1074,
        known_optimum=None,
        category='nn_weights',
        description='Neural network for digit classification 64->16->2'
    ),
    'NN-Large-1377D': BenchmarkProblem(
        name='NN-Large-1377D',
        objective=_make_optuna_objective(_nn_large_1377d, [(-1, 1)] * 1377, 1377),
        dimension=1377,
        bounds=[(-1, 1)] * 1377,
        known_optimum=None,
        category='nn_weights',
        description='Large neural network 10->32->32->1'
    ),
    'NN-Medium-20D': BenchmarkProblem(
        name='NN-Medium-20D',
        objective=_make_optuna_objective(_nn_medium_20d, [(-2, 2)] * 20, 20),
        dimension=20,
        bounds=[(-2, 2)] * 20,
        known_optimum=None,
        category='nn_weights',
        description='Medium neural network 1->10 weights'
    ),
    'NN-Deep-100D': BenchmarkProblem(
        name='NN-Deep-100D',
        objective=_make_optuna_objective(_nn_deep_100d, [(-2, 2)] * 100, 100),
        dimension=100,
        bounds=[(-2, 2)] * 100,
        known_optimum=None,
        category='nn_weights',
        description='Deep neural network 5->10->10->1'
    ),
    'Autoencoder-80D': BenchmarkProblem(
        name='Autoencoder-80D',
        objective=_make_optuna_objective(_autoencoder_80d, [(-2, 2)] * 80, 80),
        dimension=80,
        bounds=[(-2, 2)] * 80,
        known_optimum=None,
        category='nn_weights',
        description='Autoencoder 10->5->10 weights'
    ),
    'RBM-60D': BenchmarkProblem(
        name='RBM-60D',
        objective=_make_optuna_objective(_rbm_60d, [(-2, 2)] * 60, 60),
        dimension=60,
        bounds=[(-2, 2)] * 60,
        known_optimum=None,
        category='nn_weights',
        description='Restricted Boltzmann Machine weights'
    ),
    'WordEmbedding-75D': BenchmarkProblem(
        name='WordEmbedding-75D',
        objective=_make_optuna_objective(_word_embedding_75d, [(-2, 2)] * 75, 75),
        dimension=75,
        bounds=[(-2, 2)] * 75,
        known_optimum=None,
        category='nn_weights',
        description='Word embedding optimization 15x5'
    ),
    'Hopfield-64D': BenchmarkProblem(
        name='Hopfield-64D',
        objective=_make_optuna_objective(_hopfield_64d, [(-2, 2)] * 64, 64),
        dimension=64,
        bounds=[(-2, 2)] * 64,
        known_optimum=None,
        category='nn_weights',
        description='Hopfield network 8x8 weights'
    ),
    'SparseAutoencoder-100D': BenchmarkProblem(
        name='SparseAutoencoder-100D',
        objective=_make_optuna_objective(_sparse_autoencoder_100d, [(-2, 2)] * 100, 100),
        dimension=100,
        bounds=[(-2, 2)] * 100,
        known_optimum=None,
        category='nn_weights',
        description='Sparse autoencoder with sparsity penalty'
    ),
    'VAE-90D': BenchmarkProblem(
        name='VAE-90D',
        objective=_make_optuna_objective(_vae_90d, [(-2, 2)] * 90, 90),
        dimension=90,
        bounds=[(-2, 2)] * 90,
        known_optimum=None,
        category='nn_weights',
        description='Variational autoencoder weights'
    ),
    'DenoisingAE-80D': BenchmarkProblem(
        name='DenoisingAE-80D',
        objective=_make_optuna_objective(_denoising_ae_80d, [(-2, 2)] * 80, 80),
        dimension=80,
        bounds=[(-2, 2)] * 80,
        known_optimum=None,
        category='nn_weights',
        description='Denoising autoencoder weights'
    ),
    'ContrastiveLearning-70D': BenchmarkProblem(
        name='ContrastiveLearning-70D',
        objective=_make_optuna_objective(_contrastive_learning_70d, [(-2, 2)] * 70, 70),
        dimension=70,
        bounds=[(-2, 2)] * 70,
        known_optimum=None,
        category='nn_weights',
        description='Contrastive learning projection head'
    ),
    'NeuralHessian-80D': BenchmarkProblem(
        name='NeuralHessian-80D',
        objective=_make_optuna_objective(_neural_hessian_80d, [(-2, 2)] * 80, 80),
        dimension=80,
        bounds=[(-2, 2)] * 80,
        known_optimum=None,
        category='nn_weights',
        description='Neural Hessian conditioning (expensive)'
    ),
    'NeuralHessian-100D': BenchmarkProblem(
        name='NeuralHessian-100D',
        objective=_make_optuna_objective(_neural_hessian_100d, [(-2, 2)] * 100, 100),
        dimension=100,
        bounds=[(-2, 2)] * 100,
        known_optimum=None,
        category='nn_weights',
        description='Large neural Hessian conditioning (expensive)'
    ),
}

# =============================================================================
# MASTER REGISTRY (Batches 1-3)
# =============================================================================

ALL_REALWORLD_PROBLEMS: Dict[str, BenchmarkProblem] = {
    **_CHAOTIC_REGISTRY,     # 16 functions
    **_DYNAMICAL_REGISTRY,   # 6 functions
    **_NN_WEIGHTS_REGISTRY,  # 16 functions
}


def get_problem(name: str) -> BenchmarkProblem:
    """Get a real-world problem by name."""
    if name not in ALL_REALWORLD_PROBLEMS:
        raise KeyError(f"Unknown problem: {name}. Available: {list(ALL_REALWORLD_PROBLEMS.keys())}")
    return ALL_REALWORLD_PROBLEMS[name]


def list_all_problems() -> List[str]:
    """List all real-world problem names."""
    return sorted(ALL_REALWORLD_PROBLEMS.keys())


def get_problems_by_category(category: str) -> Dict[str, BenchmarkProblem]:
    """Get all problems in a category."""
    return {
        name: prob for name, prob in ALL_REALWORLD_PROBLEMS.items()
        if prob.category == category
    }
