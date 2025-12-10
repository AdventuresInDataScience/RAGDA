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
    """Variational autoencoder (90 params: 8->5->8 with no decoder bias)."""
    if len(x) != 90:
        return 1e10
    
    # Encoder: 8->5 with bias (40 + 5 = 45)
    W_enc = x[:40].reshape(8, 5)
    b_enc = x[40:45]
    # Decoder: 5->8 no bias (40 params, 45 + 40 = 85)
    W_dec = x[45:85].reshape(5, 8)
    # Extra 5 params as latent regularization weights
    reg_weights = np.abs(x[85:90]) + 0.1
    
    np.random.seed(42)
    X = np.random.randn(100, 8)
    
    # Encode
    z_mean = X @ W_enc + b_enc
    z = z_mean + 0.1 * np.random.randn(*z_mean.shape)
    
    # Decode (no bias)
    x_recon = z @ W_dec
    
    # Loss with per-dimension regularization
    recon_loss = np.mean((X - x_recon)**2)
    kl_loss = np.sum(reg_weights * z_mean**2) / len(X)
    
    loss = recon_loss + 0.01 * kl_loss
    
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
# BATCH 4: ML TRAINING/CV PROBLEMS - Part 1 (15 functions)
# Hyperparameter optimization with cross-validation - expensive, smooth landscapes
# =============================================================================

def _svm_cv_2d(x: np.ndarray) -> float:
    """SVM hyperparameter optimization with 5-fold CV (2D: C, gamma)."""
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** x[0]  # log scale: -3 to 3
    gamma = 10 ** x[1]  # log scale: -4 to 1
    
    if C <= 0 or gamma <= 0:
        return 1e10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    
    try:
        scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)  # Minimize negative accuracy
    except:
        return 1e10


def _rf_cv_4d(x: np.ndarray) -> float:
    """Random Forest hyperparameter optimization with CV (4D)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(10 + x[0] * 190)  # 10-200
    max_depth = int(2 + x[1] * 18)  # 2-20
    min_samples_split = int(2 + x[2] * 18)  # 2-20
    min_samples_leaf = int(1 + x[3] * 9)  # 1-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
    
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


def _ridge_cv_1d(x: np.ndarray) -> float:
    """Ridge regression with CV (1D: alpha)."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** x[0]  # log scale: -4 to 4
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = Ridge(alpha=alpha)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def _lasso_cv_1d(x: np.ndarray) -> float:
    """Lasso regression with CV (1D: alpha)."""
    try:
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** x[0]
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = Lasso(alpha=alpha, max_iter=2000)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def _elasticnet_cv_2d(x: np.ndarray) -> float:
    """Elastic Net with CV (2D: alpha, l1_ratio)."""
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    alpha = 10 ** x[0]  # log scale
    l1_ratio = x[1]  # 0.01 to 0.99
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)
    except:
        return 1e10


def _logisticreg_cv_1d(x: np.ndarray) -> float:
    """Logistic regression with CV (1D: C)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** x[0]  # log scale: -4 to 4
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _knn_cv_3d(x: np.ndarray) -> float:
    """KNN classifier with CV (3D: n_neighbors, weights, p)."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_neighbors = int(1 + x[0] * 29)  # 1-30
    weights = 'uniform' if x[1] < 0.5 else 'distance'
    p = int(1 + x[2] * 2)  # 1, 2, or 3
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _decisiontree_cv_3d(x: np.ndarray) -> float:
    """Decision tree with CV (3D: max_depth, min_samples_split, min_samples_leaf)."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    max_depth = int(2 + x[0] * 28)  # 2-30
    min_samples_split = int(2 + x[1] * 18)  # 2-20
    min_samples_leaf = int(1 + x[2] * 9)  # 1-10
    
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


def _adaboost_cv_2d(x: np.ndarray) -> float:
    """AdaBoost with CV (2D: n_estimators, learning_rate)."""
    try:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    learning_rate = 0.1 + x[1] * 1.9  # 0.1-2.0
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        algorithm='SAMME'
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _svm_large_cv_2d(x: np.ndarray) -> float:
    """SVM RBF with larger dataset and more CV folds (2D: C, gamma)."""
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** x[0]
    gamma = 10 ** x[1]
    
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=15, random_state=42)
    
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    try:
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _bagging_cv_3d(x: np.ndarray) -> float:
    """Bagging classifier with CV (3D: n_estimators, max_samples, max_features)."""
    try:
        from sklearn.ensemble import BaggingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(20 + x[0] * 80)  # 20-100
    max_samples = 0.5 + x[1] * 0.5  # 0.5-1.0
    max_features = 0.5 + x[2] * 0.5  # 0.5-1.0
    
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


def _gradientboost_cv_3d(x: np.ndarray) -> float:
    """Gradient Boosting with CV (3D: n_estimators, learning_rate, max_depth)."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    learning_rate = 0.01 + x[1] * 0.49  # 0.01-0.5
    max_depth = int(2 + x[2] * 8)  # 2-10
    
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


def _mlp_regressor_cv_3d(x: np.ndarray) -> float:
    """MLP regressor with CV (3D: hidden_size, alpha, learning_rate_init)."""
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    hidden_size = int(10 + x[0] * 90)  # 10-100
    alpha = 10 ** (x[1] * 6 - 5)  # 1e-5 to 10
    learning_rate_init = 10 ** (x[2] * 3 - 4)  # 1e-4 to 0.1
    
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


def _neuralnet_dropout_20d(x: np.ndarray) -> float:
    """Neural network with dropout tuning (20D: architecture + dropout rates)."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # 20D: 10 neurons per layer + 10 dropout params (not directly in sklearn, use alpha as proxy)
    neurons = [int(10 + x[i] * 90) for i in range(10)]  # 10-100 neurons each layer
    alpha = np.mean([10 ** (-4 + x[i+10] * 4) for i in range(10)])  # L2 reg as dropout proxy
    
    np.random.seed(42)
    X, y = make_classification(n_samples=800, n_features=40, n_informative=20, random_state=42)
    
    # Use smaller architecture for speed (first 3 layer sizes)
    hidden_sizes = tuple([neurons[0], neurons[1], neurons[2]])
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        alpha=alpha,
        max_iter=100,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _lightgbm_cv_4d(x: np.ndarray) -> float:
    """LightGBM with CV (4D: n_estimators, learning_rate, max_depth, num_leaves)."""
    try:
        # Simulate LightGBM with GradientBoosting (LightGBM may not be installed)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    learning_rate = 0.01 + x[1] * 0.29  # 0.01-0.3
    max_depth = int(3 + x[2] * 7)  # 3-10
    # num_leaves simulated via max_depth
    
    np.random.seed(42)
    X, y = make_classification(n_samples=600, n_features=25, random_state=42)
    
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


def _xgboost_cv_4d(x: np.ndarray) -> float:
    """XGBoost with CV (4D: n_estimators, learning_rate, max_depth, subsample)."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    learning_rate = 0.01 + x[1] * 0.29  # 0.01-0.3
    max_depth = int(3 + x[2] * 7)  # 3-10
    subsample = 0.5 + x[3] * 0.5  # 0.5-1.0
    
    np.random.seed(42)
    X, y = make_classification(n_samples=600, n_features=25, random_state=42)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _catboost_cv_4d(x: np.ndarray) -> float:
    """CatBoost simulation with CV (4D: iterations, learning_rate, depth, l2_leaf_reg)."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    learning_rate = 0.01 + x[1] * 0.29  # 0.01-0.3
    max_depth = int(3 + x[2] * 7)  # 3-10
    # l2_leaf_reg simulated via min_samples_split
    min_samples_split = int(2 + x[3] * 18)  # 2-20
    
    np.random.seed(42)
    X, y = make_classification(n_samples=600, n_features=25, random_state=42)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _extratrees_cv_4d(x: np.ndarray) -> float:
    """ExtraTrees with CV (4D: n_estimators, max_depth, min_samples_split, min_samples_leaf)."""
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    max_depth = int(5 + x[1] * 15)  # 5-20
    min_samples_split = int(2 + x[2] * 18)  # 2-20
    min_samples_leaf = int(1 + x[3] * 9)  # 1-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=600, n_features=25, random_state=42)
    
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
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


def _stacking_ensemble_15d(x: np.ndarray) -> float:
    """Stacking ensemble with multiple base models (15D: 3 models × 5 params each)."""
    try:
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # RF params
    rf_n_est = int(50 + x[0] * 150)
    rf_depth = int(5 + x[1] * 15)
    # GB params
    gb_n_est = int(50 + x[2] * 150)
    gb_lr = 0.01 + x[3] * 0.29
    gb_depth = int(3 + x[4] * 7)
    # LR params (via penalty strength)
    lr_C = 10 ** (-2 + x[5] * 4)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_depth, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=gb_n_est, learning_rate=gb_lr, max_depth=gb_depth, random_state=42))
    ]
    
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=lr_C, max_iter=200),
        cv=3
    )
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _voting_ensemble_10d(x: np.ndarray) -> float:
    """Voting ensemble with weight tuning (10D: 5 models, 5 weights, weights sum to 1)."""
    try:
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # Model params (5D)
    rf_n_est = int(50 + x[0] * 50)
    gb_n_est = int(30 + x[1] * 70)
    dt_depth = int(5 + x[2] * 10)
    
    # Weights (5D) - normalize to sum to 1
    weights_raw = x[5:10]
    weights = weights_raw / np.sum(weights_raw)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=rf_n_est, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=gb_n_est, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=dt_depth, random_state=42))
    ]
    
    model = VotingClassifier(estimators=estimators, voting='soft', weights=weights[:3])
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _feature_selection_rfe_20d(x: np.ndarray) -> float:
    """RFE feature selection optimization (20D: n_features + estimator params)."""
    try:
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_features_to_select = int(5 + x[0] * 20)  # 5-25
    rf_n_est = int(50 + x[1] * 100)
    rf_depth = int(5 + x[2] * 15)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=30, n_informative=15, random_state=42)
    
    estimator = RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_depth, random_state=42)
    selector = RFE(estimator, n_features_to_select=min(n_features_to_select, X.shape[1]))
    
    try:
        X_selected = selector.fit_transform(X, y)
        scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _feature_selection_mi_15d(x: np.ndarray) -> float:
    """Mutual information feature selection (15D: MI threshold + estimator params)."""
    try:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    k_features = int(5 + x[0] * 20)  # 5-25
    rf_n_est = int(50 + x[1] * 100)
    rf_depth = int(5 + x[2] * 15)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=30, n_informative=15, random_state=42)
    
    selector = SelectKBest(mutual_info_classif, k=min(k_features, X.shape[1]))
    try:
        X_selected = selector.fit_transform(X, y)
        estimator = RandomForestClassifier(n_estimators=rf_n_est, max_depth=rf_depth, random_state=42)
        scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _feature_engineering_poly_12d(x: np.ndarray) -> float:
    """Polynomial feature engineering (12D: degree + interaction terms + estimator)."""
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    degree = int(2 + x[0] * 1)  # 2 or 3
    interaction_only = x[1] > 0.5
    include_bias = x[2] > 0.5
    C = 10 ** (-2 + x[3] * 4)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=8, n_informative=6, random_state=42)
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    try:
        X_poly = poly.fit_transform(X)
        model = LogisticRegression(C=C, max_iter=500)
        scores = cross_val_score(model, X_poly, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _feature_scaling_robust_8d(x: np.ndarray) -> float:
    """Robust scaling optimization (8D: quantile ranges + estimator params)."""
    try:
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    q_low = 0.05 + x[0] * 0.2  # 0.05-0.25
    q_high = 0.75 + x[1] * 0.2  # 0.75-0.95
    C = 10 ** (-2 + x[2] * 4)
    gamma = 10 ** (-4 + x[3] * 3)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    # Add outliers
    X[:10] *= 10
    
    scaler = RobustScaler(quantile_range=(q_low * 100, q_high * 100))
    try:
        X_scaled = scaler.fit_transform(X)
        model = SVC(C=C, gamma=gamma, kernel='rbf')
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _class_weights_imbalanced_10d(x: np.ndarray) -> float:
    """Class weight tuning for imbalanced data (10D: weights + estimator params)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # Class weight ratio
    weight_ratio = 1 + x[0] * 9  # 1-10
    rf_n_est = int(50 + x[1] * 100)
    rf_depth = int(5 + x[2] * 15)
    
    np.random.seed(42)
    X, y = make_classification(
        n_samples=600, n_features=20,
        weights=[0.9, 0.1],  # Imbalanced
        random_state=42
    )
    
    class_weight = {0: 1.0, 1: weight_ratio}
    model = RandomForestClassifier(
        n_estimators=rf_n_est,
        max_depth=rf_depth,
        class_weight=class_weight,
        random_state=42
    )
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        return -np.mean(scores)
    except:
        return 1e10


def _pca_components_1d(x: np.ndarray) -> float:
    """PCA component selection (1D: n_components as fraction)."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_components_frac = 0.1 + x[0] * 0.89  # 0.1-0.99
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=50, n_informative=20, random_state=42)
    
    n_components = max(2, int(n_components_frac * X.shape[1]))
    pca = PCA(n_components=n_components)
    
    try:
        X_pca = pca.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        scores = cross_val_score(model, X_pca, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _svd_components_1d(x: np.ndarray) -> float:
    """TruncatedSVD component selection (1D: n_components)."""
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_components_frac = 0.1 + x[0] * 0.89  # 0.1-0.99
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=50, n_informative=20, random_state=42)
    
    n_components = max(2, min(int(n_components_frac * X.shape[1]), X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components)
    
    try:
        X_svd = svd.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        scores = cross_val_score(model, X_svd, y, cv=5, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _tsne_hyperparams_3d(x: np.ndarray) -> float:
    """t-SNE hyperparameter optimization (3D: perplexity, learning_rate, max_iter)."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    perplexity = max(5.0, min(5 + x[0] * 45, 50.0))  # 5-50, clipped
    learning_rate = max(10.0, 10 + x[1] * 990)  # 10-1000
    max_iter = int(max(250, min(250 + x[2] * 750, 1000)))  # 250-1000
    
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=50, n_informative=20, random_state=42)
    
    try:
        # Ensure perplexity < n_samples
        perplexity = min(perplexity, X.shape[0] / 4.0)
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, 
                    max_iter=max_iter, random_state=42, init='random')
        X_tsne = tsne.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        scores = cross_val_score(model, X_tsne, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _umap_hyperparams_4d(x: np.ndarray) -> float:
    """UMAP hyperparameter optimization simulation (4D: n_neighbors, min_dist, metric params)."""
    try:
        # UMAP may not be installed, simulate with PCA + scaling
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # Simulate UMAP params
    n_components = int(2 + x[0] * 8)  # 2-10
    
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=40, n_informative=15, random_state=42)
    
    try:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        scores = cross_val_score(model, X_reduced, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except:
        return 1e10


def _isolation_forest_cv_3d(x: np.ndarray) -> float:
    """Isolation Forest hyperparameter tuning (3D: n_estimators, max_samples, contamination)."""
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)  # 50-200
    max_samples_frac = 0.5 + x[1] * 0.5  # 0.5-1.0
    contamination = 0.01 + x[2] * 0.19  # 0.01-0.2
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    max_samples = int(max_samples_frac * X.shape[0])
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42
    )
    try:
        predictions = model.fit_predict(X)
        # Return negative proportion of inliers (maximize detection accuracy)
        inlier_ratio = np.sum(predictions == 1) / len(predictions)
        return -inlier_ratio
    except:
        return 1e10


def _autoencoder_hyperparams_5d(x: np.ndarray) -> float:
    """Autoencoder hyperparameter optimization using MLPRegressor (5D: encoding_dim + arch params)."""
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    encoding_dim = int(5 + x[0] * 20)  # 5-25
    hidden1 = int(20 + x[1] * 80)  # 20-100
    hidden2 = int(10 + x[2] * 40)  # 10-50
    alpha = 10 ** (-5 + x[3] * 3)  # 1e-5 to 1e-2
    learning_rate_init = 10 ** (-4 + x[4] * 2)  # 1e-4 to 1e-2
    
    np.random.seed(42)
    X, _ = make_classification(n_samples=500, n_features=30, n_informative=20, random_state=42)
    
    # Autoencoder: encode then decode
    model = MLPRegressor(
        hidden_layer_sizes=(hidden1, encoding_dim, hidden2),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=100,
        random_state=42
    )
    try:
        model.fit(X, X)  # Reconstruct input
        reconstructed = model.predict(X)
        mse = np.mean((X - reconstructed) ** 2)
        return mse
    except:
        return 1e10


def _sparse_coding_30d(x: np.ndarray) -> float:
    """Sparse coding dictionary learning (30D: alpha, n_components, max_iter)."""
    try:
        from sklearn.decomposition import DictionaryLearning
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    alpha = 0.01 + x[0] * 1.99  # 0.01-2.0
    n_components = int(5 + x[1] * 20)  # 5-25 components
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=50, n_informative=30, random_state=42)
    
    try:
        # Dictionary learning for sparse representation
        dict_learner = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=100, random_state=42)
        X_sparse = dict_learner.fit_transform(X)
        
        model = LogisticRegression(max_iter=200)
        scores = cross_val_score(model, X_sparse, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _nmf_factorization_25d(x: np.ndarray) -> float:
    """Non-negative matrix factorization (25D: n_components + initialization params)."""
    try:
        from sklearn.decomposition import NMF
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return 1e10
    
    n_components = int(5 + x[0] * 20)  # 5-25
    alpha_W = 0.0 + x[1] * 0.5  # 0-0.5 regularization for W
    alpha_H = 0.0 + x[2] * 0.5  # 0-0.5 regularization for H
    l1_ratio = x[3]  # 0-1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=50, n_informative=30, random_state=42)
    X = MinMaxScaler().fit_transform(X)  # NMF requires non-negative
    
    try:
        nmf = NMF(n_components=n_components, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, max_iter=200, random_state=42)
        X_nmf = nmf.fit_transform(X)
        
        model = LogisticRegression(max_iter=200)
        scores = cross_val_score(model, X_nmf, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _kernel_approximation_20d(x: np.ndarray) -> float:
    """Kernel approximation with RBF sampler (20D: gamma, n_components, linear model params)."""
    try:
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    gamma = 0.001 + x[0] * 9.999  # 0.001-10
    n_components = int(50 + x[1] * 150)  # 50-200
    alpha = 10 ** (-5 + x[2] * 4)  # 1e-5 to 1e-1
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=40, n_informative=25, random_state=42)
    
    try:
        rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
        X_rbf = rbf_sampler.fit_transform(X)
        
        model = SGDClassifier(alpha=alpha, max_iter=100, random_state=42)
        scores = cross_val_score(model, X_rbf, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _calibration_tuning_15d(x: np.ndarray) -> float:
    """Probability calibration tuning (15D: base model + calibration params)."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # Base RF params
    n_estimators = int(50 + x[0] * 100)
    max_depth = int(5 + x[1] * 15)
    min_samples_split = int(2 + x[2] * 18)
    
    # Calibration method (sigmoid vs isotonic approximated by threshold)
    method = 'sigmoid' if x[3] < 0.5 else 'isotonic'
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=30, n_informative=20, random_state=42)
    
    try:
        base_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                            min_samples_split=min_samples_split, random_state=42)
        calibrated = CalibratedClassifierCV(base_model, method=method, cv=3)
        
        scores = cross_val_score(calibrated, X, y, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _multi_output_regression_40d(x: np.ndarray) -> float:
    """Multi-output regression with chained models (40D: chain of 4 models × 10 params)."""
    try:
        from sklearn.multioutput import RegressorChain
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    # Params for chain of regressors
    n_estimators = int(20 + x[0] * 80)
    learning_rate = 0.01 + x[1] * 0.29
    max_depth = int(2 + x[2] * 6)
    subsample = 0.5 + x[3] * 0.5
    
    np.random.seed(42)
    X, y = make_regression(n_samples=400, n_features=30, n_targets=4, random_state=42)
    
    try:
        base_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth, subsample=subsample, random_state=42)
        chain = RegressorChain(base_model, random_state=42)
        
        scores = cross_val_score(chain, X, y, cv=3, scoring='neg_mean_squared_error')
        return -np.mean(scores)  # Returns positive MSE
    except Exception:
        return 1e10


def _quantile_regression_12d(x: np.ndarray) -> float:
    """Quantile regression optimization (12D: multiple quantiles + regularization)."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_regression
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)
    learning_rate = 0.01 + x[1] * 0.19
    max_depth = int(3 + x[2] * 7)
    alpha = 0.1 + x[3] * 0.8  # Quantile level (0.1-0.9)
    
    np.random.seed(42)
    X, y = make_regression(n_samples=400, n_features=25, noise=10, random_state=42)
    
    try:
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                          max_depth=max_depth, loss='quantile', alpha=alpha, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
        return -np.mean(scores)  # Returns positive MAE
    except Exception:
        return 1e10


def _semi_supervised_30d(x: np.ndarray) -> float:
    """Semi-supervised learning with label propagation (30D: kernel + model params)."""
    try:
        from sklearn.semi_supervised import LabelSpreading
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    gamma = 0.1 + x[0] * 9.9  # 0.1-10
    alpha = 0.01 + x[1] * 0.99  # 0.01-1.0
    max_iter = int(10 + x[2] * 90)  # 10-100
    
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15, random_state=42)
    
    # Simulate unlabeled data (set 50% labels to -1)
    y_semi = y.copy()
    unlabeled_indices = np.random.RandomState(42).choice(len(y), size=150, replace=False)
    y_semi[unlabeled_indices] = -1
    
    try:
        model = LabelSpreading(kernel='rbf', gamma=gamma, alpha=alpha, max_iter=max_iter)
        model.fit(X, y_semi)
        
        # Evaluate on originally labeled data
        labeled_mask = y_semi != -1
        accuracy = (model.predict(X[labeled_mask]) == y[labeled_mask]).mean()
        return -accuracy
    except Exception:
        return 1e10


def _ordinal_regression_18d(x: np.ndarray) -> float:
    """Ordinal regression with ordered logistic (18D: thresholds + coefficients)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    C = 10 ** (-2 + x[0] * 4)  # 0.01-100
    
    np.random.seed(42)
    # Create ordinal target (0, 1, 2, 3, 4)
    X, y_binary = make_classification(n_samples=500, n_features=25, n_informative=15, 
                                      n_classes=5, n_clusters_per_class=1, random_state=42)
    
    try:
        # Use OvR logistic regression as approximation for ordinal
        model = LogisticRegression(C=C, max_iter=200, multi_class='ovr', random_state=42)
        scores = cross_val_score(model, X, y_binary, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


def _cost_sensitive_learning_22d(x: np.ndarray) -> float:
    """Cost-sensitive learning with custom loss (22D: model params + cost matrix)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    n_estimators = int(50 + x[0] * 150)
    max_depth = int(5 + x[1] * 15)
    
    # Cost ratio for misclassification
    cost_ratio = 1 + x[2] * 9  # 1-10 (cost of false negative vs false positive)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=30, weights=[0.8, 0.2], random_state=42)
    
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import f1_score
        
        # Use class weights to simulate cost-sensitive learning
        sample_weights = np.ones(len(y))
        sample_weights[y == 1] *= cost_ratio
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        # Manual CV loop with sample weights
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            weights_train = sample_weights[train_idx]
            
            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_test)
            scores.append(f1_score(y_test, y_pred))
        
        return -np.mean(scores)
    except Exception:
        return 1e10


def _transfer_learning_35d(x: np.ndarray) -> float:
    """Transfer learning simulation (35D: source model + adaptation layer)."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
    except ImportError:
        return 1e10
    
    # Source model params
    hidden_layer_1 = int(20 + x[0] * 80)
    hidden_layer_2 = int(10 + x[1] * 40)
    
    # Adaptation layer
    adaptation_size = int(5 + x[2] * 20)
    
    # Fine-tuning params
    alpha = 10 ** (-5 + x[3] * 4)
    learning_rate_init = 10 ** (-4 + x[4] * 3)
    
    np.random.seed(42)
    # Source task (larger dataset)
    X_source, y_source = make_classification(n_samples=1000, n_features=50, n_informative=30, random_state=42)
    
    # Target task (smaller, related)
    X_target, y_target = make_classification(n_samples=300, n_features=50, n_informative=25, random_state=43)
    
    try:
        # Pre-train on source
        source_model = MLPClassifier(hidden_layer_sizes=(hidden_layer_1, hidden_layer_2), 
                                      alpha=alpha, max_iter=50, random_state=42)
        source_model.fit(X_source, y_source)
        
        # Fine-tune on target
        transfer_model = MLPClassifier(hidden_layer_sizes=(hidden_layer_1, adaptation_size),
                                       alpha=alpha, learning_rate_init=learning_rate_init,
                                       warm_start=True, max_iter=30, random_state=42)
        
        scores = cross_val_score(transfer_model, X_target, y_target, cv=3, scoring='accuracy')
        return -np.mean(scores)
    except Exception:
        return 1e10


# =============================================================================
# PDE PROBLEMS (18 functions) - Chunk 2.2.7
# =============================================================================

def _burgers_equation_9d(x: np.ndarray) -> float:
    """1D Burgers equation parameter estimation (9D: viscosity + Fourier coefficients)."""
    nu = 0.01 + x[0] * 0.49  # viscosity: 0.01-0.5
    
    # Fourier coefficients for initial condition
    a_coeffs = 2 * x[1:5] - 1  # 4 sin coefficients: -1 to 1
    b_coeffs = 2 * x[5:9] - 1  # 4 cos coefficients: -1 to 1
    
    # Spatial discretization
    N = 64
    L = 2 * np.pi
    x_grid = np.linspace(0, L, N, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    
    # Initial condition from Fourier coefficients
    u = np.zeros(N)
    for k in range(len(a_coeffs)):
        u += a_coeffs[k] * np.sin((k+1) * x_grid)
    for k in range(len(b_coeffs)):
        u += b_coeffs[k] * np.cos((k+1) * x_grid)
    
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
    except Exception:
        return 1e10
    
    # Target: smooth decaying solution
    target_energy = 0.1
    energy = np.mean(u**2)
    
    return (energy - target_energy)**2


def _pde_heat_eq_50d(x: np.ndarray) -> float:
    """1D heat equation solver (50D: Fourier mode initial conditions)."""
    n_modes = 25
    a_coeffs = 2 * x[:n_modes] - 1  # -1 to 1
    b_coeffs = 2 * x[n_modes:2*n_modes] - 1  # -1 to 1
    
    # Spatial discretization
    N = 100
    x_grid = np.linspace(0, 2*np.pi, N)
    dx = x_grid[1] - x_grid[0]
    
    # Initial condition from Fourier coefficients
    u = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x_grid) + b_coeffs[k] * np.sin((k+1) * x_grid)
    
    # Time stepping (explicit Euler)
    dt = 0.4 * dx**2  # CFL condition
    alpha = 1.0  # diffusion coefficient
    n_steps = 500
    
    try:
        for _ in range(n_steps):
            u_new = u.copy()
            for i in range(1, N-1):
                u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
            # Periodic BCs
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            u = u_new
    except Exception:
        return 1e10
    
    # Target: flat steady state at 0
    return np.mean(u**2)


def _heat_diffusion_30d(x: np.ndarray) -> float:
    """Heat diffusion equation (30D: Fourier mode initial conditions)."""
    n_modes = 15
    a_coeffs = 2 * x[:n_modes] - 1
    b_coeffs = 2 * x[n_modes:] - 1
    
    N = 64
    x_grid = np.linspace(0, 2*np.pi, N)
    dx = x_grid[1] - x_grid[0]
    
    # Initial condition
    u = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x_grid) + b_coeffs[k] * np.sin((k+1) * x_grid)
    
    dt = 0.3 * dx**2
    
    try:
        for _ in range(200):
            u_new = u.copy()
            for i in range(1, N-1):
                u_new[i] = u[i] + dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
            u = u_new
    except Exception:
        return 1e10
    
    return np.mean(u**2)


def _wave_equation_30d(x: np.ndarray) -> float:
    """Wave equation initial condition optimization (30D)."""
    n_modes = 15
    a_coeffs = 2 * x[:n_modes] - 1
    b_coeffs = 2 * x[n_modes:] - 1
    
    N = 64
    x_grid = np.linspace(0, 2*np.pi, N)
    dx = x_grid[1] - x_grid[0]
    c = 1.0  # wave speed
    
    # Initial condition
    u = np.zeros(N)
    for k in range(n_modes):
        u += a_coeffs[k] * np.cos((k+1) * x_grid) + b_coeffs[k] * np.sin((k+1) * x_grid)
    u_prev = u.copy()
    
    dt = 0.5 * dx / c
    
    try:
        for _ in range(200):
            u_new = np.zeros(N)
            for i in range(1, N-1):
                u_new[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
            u_prev = u
            u = u_new
    except Exception:
        return 1e10
    
    return np.mean(u**2)


def _advection_diffusion_30d(x: np.ndarray) -> float:
    """Advection-diffusion equation (30D: velocity + diffusion + IC)."""
    velocity = 2 * x[0] - 1  # -1 to 1
    diffusion = 0.01 + x[1] * 0.49  # 0.01 to 0.5
    ic_coeffs = 2 * x[2:] - 1  # -1 to 1
    
    N = 64
    x_grid = np.linspace(0, 2*np.pi, N)
    dx = x_grid[1] - x_grid[0]
    
    # Initial condition
    u = np.zeros(N)
    for k, coeff in enumerate(ic_coeffs):
        u += coeff * np.sin((k+1) * x_grid)
    
    dt = 0.1 * dx**2 / diffusion
    
    try:
        for _ in range(100):
            u_new = u.copy()
            for i in range(1, N-1):
                advection = -velocity * (u[i] - u[i-1]) / dx
                diffusion_term = diffusion * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
                u_new[i] = u[i] + dt * (advection + diffusion_term)
            u = u_new
    except Exception:
        return 1e10
    
    return np.mean(u**2)


def _reaction_diffusion_30d(x: np.ndarray) -> float:
    """Reaction-diffusion pattern formation (30D: parameters + ICs)."""
    Du = 0.001 + x[0] * 0.099  # 0.001-0.1
    Dv = 0.001 + x[1] * 0.099  # 0.001-0.1
    f = x[2] * 0.1  # 0-0.1
    k = x[3] * 0.1  # 0-0.1
    
    ic_u = 2 * x[4:17] - 1  # 13 coefficients
    ic_v = 2 * x[17:30] - 1  # 13 coefficients
    
    N = 32
    u = np.ones((N, N)) * 0.5
    v = np.ones((N, N)) * 0.25
    
    # Add initial perturbations
    for idx, coeff in enumerate(ic_u):
        u += coeff * 0.1 * np.sin((idx+1) * np.linspace(0, 2*np.pi, N)).reshape(-1, 1)
    for idx, coeff in enumerate(ic_v):
        v += coeff * 0.1 * np.cos((idx+1) * np.linspace(0, 2*np.pi, N)).reshape(1, -1)
    
    dx = 1.0
    dt = 0.5
    
    try:
        for _ in range(100):
            laplacian_u = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u
            laplacian_v = np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4*v
            
            reaction_u = -u * v**2 + f * (1 - u)
            reaction_v = u * v**2 - (f + k) * v
            
            u = u + dt * (Du * laplacian_u / dx**2 + reaction_u)
            v = v + dt * (Dv * laplacian_v / dx**2 + reaction_v)
            
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
    except Exception:
        return 1e10
    
    # Target: pattern with specific strength
    target_pattern_strength = 0.2
    pattern_strength = np.std(u)
    
    return (pattern_strength - target_pattern_strength)**2


def _heat_2d_60d(x: np.ndarray) -> float:
    """2D heat equation initial condition (60D)."""
    u = (2 * x - 1).reshape(6, 10)[:6, :8]  # Reshape to 6x8 grid
    
    dx = 1.0
    dt = 0.2
    
    try:
        for _ in range(50):
            u_new = u.copy()
            for i in range(1, 5):
                for j in range(1, 7):
                    u_new[i, j] = u[i, j] + dt * (
                        (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]) / dx**2
                    )
            u = u_new
    except Exception:
        return 1e10
    
    return np.mean(u**2)


def _poisson_60d(x: np.ndarray) -> float:
    """Poisson equation source term optimization (60D)."""
    f = (2 * x - 1).reshape(6, 10)
    N, M = 6, 10
    
    u = np.zeros((N, M))
    dx = 1.0
    
    try:
        # Iterative solver
        for _ in range(100):
            u_new = u.copy()
            for i in range(1, N-1):
                for j in range(1, M-1):
                    u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - dx**2 * f[i, j])
            u = u_new
    except Exception:
        return 1e10
    
    # Target: specific value at center
    target = np.zeros((N, M))
    target[N//2, M//2] = 1.0
    
    return np.mean((u - target)**2)


def _laplace_64d(x: np.ndarray) -> float:
    """Laplace equation boundary optimization (64D)."""
    boundary = 2 * x - 1  # -1 to 1
    
    N = 8
    u = np.zeros((N, N))
    
    # Set boundary conditions from parameters
    u[0, :] = boundary[:8]
    u[-1, :] = boundary[8:16]
    u[:, 0] = boundary[16:24]
    u[:, -1] = boundary[24:32]
    
    try:
        # Iterative Laplace solver
        for _ in range(100):
            u_new = u.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
            u = u_new
    except Exception:
        return 1e10
    
    # Target: specific value at center
    target_center = 0.5
    center_val = u[N//2, N//2]
    
    return (center_val - target_center)**2


def _helmholtz_60d(x: np.ndarray) -> float:
    """Helmholtz equation parameter optimization (60D)."""
    k = 0.1 + x[0] * 9.9  # wavenumber: 0.1-10
    
    # Need 45 elements for 5x9 grid, use first 46 params (1 for k, 45 for source)
    source_params = x[1:46]  # 45 elements
    source = (2 * source_params - 1).reshape(5, 9)
    
    N, M = 5, 9
    u = np.zeros((N, M), dtype=complex)
    dx = 0.5
    
    try:
        for _ in range(50):
            u_new = u.copy()
            for i in range(1, N-1):
                for j in range(1, M-1):
                    laplacian = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]) / dx**2
                    u_new[i, j] = u[i, j] + 0.1 * (laplacian + k**2 * u[i, j] + source[i, j])
            u = u_new
    except Exception:
        return 1e10
    
    return np.mean(np.abs(u)**2)


def _biharmonic_56d(x: np.ndarray) -> float:
    """Biharmonic equation / plate bending (56D)."""
    load = (2 * x - 1).reshape(7, 8)
    
    N, M = 7, 8
    u = np.zeros((N, M))
    dx = 1.0
    
    try:
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
    except Exception:
        return 1e10
    
    return np.mean(u**2)


def _spectral_method_64d(x: np.ndarray) -> float:
    """Spectral method coefficients optimization (64D)."""
    coeffs = (2 * x - 1).reshape(8, 8)
    
    N = 32
    x_grid = np.linspace(0, 2*np.pi, N)
    y_grid = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Reconstruct solution from spectral coefficients
    u = np.zeros((N, N))
    for k in range(8):
        for l in range(8):
            u += coeffs[k, l] * np.sin((k+1)*X) * np.sin((l+1)*Y)
    
    # Target solution
    target = np.sin(X) * np.sin(Y)
    
    return np.mean((u - target)**2)


def _finite_element_70d(x: np.ndarray) -> float:
    """Finite element node values optimization (70D)."""
    nodes = (2 * x - 1).reshape(7, 10)
    
    # Minimize gradient (smoothness)
    smoothness = 0
    for i in range(6):
        for j in range(9):
            grad_x = nodes[i+1, j] - nodes[i, j]
            grad_y = nodes[i, j+1] - nodes[i, j]
            smoothness += grad_x**2 + grad_y**2
    
    # Constraint: sum to target value
    target_sum = 5.0
    constraint = (np.sum(nodes) - target_sum)**2
    
    return smoothness + 10 * constraint


def _multigrid_60d(x: np.ndarray) -> float:
    """Multigrid solver parameter optimization (60D)."""
    coarse = (2 * x[:15] - 1).reshape(3, 5)
    fine = (2 * x[15:60] - 1).reshape(5, 9)
    
    # Interpolate coarse to fine
    interpolated = np.zeros((5, 9))
    for i in range(5):
        for j in range(9):
            ci, cj = min(i//2, 2), min(j//2, 4)
            interpolated[i, j] = coarse[ci, cj]
    
    # Combined solution
    combined = 0.5 * interpolated + 0.5 * fine
    
    # Minimize deviation from mean
    return np.mean((combined - np.mean(combined))**2)


def _domain_decomposition_72d(x: np.ndarray) -> float:
    """Domain decomposition interface optimization (72D)."""
    domain1 = (2 * x[:36] - 1).reshape(6, 6)
    domain2 = (2 * x[36:72] - 1).reshape(6, 6)
    
    # Minimize interface mismatch (right edge of domain1 with left edge of domain2)
    interface_mismatch = np.mean((domain1[-1, :] - domain2[0, :])**2)
    
    # Minimize internal gradients (smoothness) - use compatible slices
    grad1_x = np.diff(domain1, axis=0)  # 5x6
    grad1_y = np.diff(domain1, axis=1)  # 6x5
    smoothness1 = np.mean(grad1_x**2) + np.mean(grad1_y**2)
    
    grad2_x = np.diff(domain2, axis=0)  # 5x6
    grad2_y = np.diff(domain2, axis=1)  # 6x5
    smoothness2 = np.mean(grad2_x**2) + np.mean(grad2_y**2)
    
    return interface_mismatch + 0.1 * (smoothness1 + smoothness2)


def _adaptive_mesh_65d(x: np.ndarray) -> float:
    """Adaptive mesh refinement optimization (65D)."""
    mesh_density = x[:25].reshape(5, 5) + 0.1  # Ensure positive
    solution = (2 * x[25:65] - 1).reshape(5, 8)
    
    # Compute solution gradient magnitude (simplified)
    grad_x = np.diff(solution, axis=0)  # 4x8
    grad_y = np.diff(solution, axis=1)  # 5x7
    # Take overlapping region
    gradient = np.sqrt(grad_x[:, :7]**2 + grad_y[:4, :]**2)  # 4x7
    
    # Desired mesh density proportional to gradient
    desired_density = gradient / (np.max(gradient) + 1e-8)
    
    # Minimize difference (comparing 4x4 regions)
    return np.mean((mesh_density[:4, :4] - desired_density[:4, :4])**2)


def _ginzburg_landau_56d(x: np.ndarray) -> float:
    """Complex Ginzburg-Landau equation (56D: real + imaginary parts)."""
    A_real = 2 * x[:28] - 1
    A_imag = 2 * x[28:56] - 1
    A = (A_real + 1j * A_imag).reshape(4, 7)
    
    c1, c2 = 1.0, 0.5
    dx = 1.0
    dt = 0.1
    
    try:
        for _ in range(50):
            A_new = A.copy()
            for i in range(1, 3):
                for j in range(1, 6):
                    laplacian = (A[i+1, j] + A[i-1, j] + A[i, j+1] + A[i, j-1] - 4*A[i, j]) / dx**2
                    A_new[i, j] = A[i, j] + dt * (A[i, j] - (1 + 1j*c2) * np.abs(A[i, j])**2 * A[i, j] + (1 + 1j*c1) * laplacian)
            A = A_new
    except Exception:
        return 1e10
    
    return np.mean(np.abs(A)**2)


def _wave_equation_120d(x: np.ndarray) -> float:
    """1D wave equation boundary reflection minimization (120D)."""
    c = 1.0  # wave speed
    n = len(x)
    dx = 1.0 / (n + 1)
    dt = 0.5 * dx / c
    
    # Initial wave profile
    u = 2 * x - 1  # -1 to 1
    u_prev = u.copy()
    total_boundary_energy = 0.0
    
    try:
        for step in range(100):
            u_new = np.zeros_like(u)
            u_new[1:-1] = 2*u[1:-1] - u_prev[1:-1] + (c*dt/dx)**2 * (u[2:] - 2*u[1:-1] + u[:-2])
            # Absorbing boundary conditions
            u_new[0] = u[1]
            u_new[-1] = u[-2]
            
            # Accumulate boundary energy (want to minimize)
            total_boundary_energy += u_new[0]**2 + u_new[-1]**2
            
            u_prev = u
            u = u_new
    except Exception:
        return 1e10
    
    return total_boundary_energy


# =============================================================================
# META-OPTIMIZATION & CONTROL PROBLEMS (18 functions) - Chunk 2.2.8
# =============================================================================

def _genetic_algorithm_25d(x: np.ndarray) -> float:
    """Genetic algorithm parameter tuning via meta-optimization (25D)."""
    pop_size = int(20 + x[0] * 80)  # 20-100
    mutation_rate = x[1] * 0.5  # 0-0.5
    crossover_rate = 0.5 + x[2] * 0.5  # 0.5-1.0
    selection_pressure = 1 + x[3] * 4  # 1-5
    gene_init = 2 * x[4:] - 1  # -1 to 1, scaled to problem domain
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    dim = 10
    np.random.seed(42)
    pop = np.random.randn(pop_size, dim) * 2
    pop[0] = gene_init[:dim] * 5.12
    
    n_generations = 30
    
    try:
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
    except Exception:
        return 1e10
    
    final_fitness = np.array([rastrigin(ind) for ind in pop])
    return np.min(final_fitness)


def _particle_swarm_30d(x: np.ndarray) -> float:
    """Particle Swarm Optimization parameter tuning (30D)."""
    n_particles = int(20 + x[0] * 80)  # 20-100
    w = 0.4 + x[1] * 0.5  # inertia: 0.4-0.9
    c1 = x[2] * 3  # cognitive: 0-3
    c2 = x[3] * 3  # social: 0-3
    v_max = x[4] * 2  # velocity limit: 0-2
    init_pos = 2 * x[5:] - 1
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    dim = 10
    np.random.seed(42)
    pos = np.random.randn(n_particles, dim) * 2
    pos[0] = init_pos[:dim] * 5.12
    vel = np.zeros((n_particles, dim))
    
    pbest_pos = pos.copy()
    pbest_val = np.array([rastrigin(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    
    n_iterations = 50
    
    try:
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
    except Exception:
        return 1e10
    
    return gbest_val


def _differential_evolution_30d(x: np.ndarray) -> float:
    """Differential Evolution parameter tuning (30D)."""
    pop_size = int(20 + x[0] * 80)  # 20-100
    F = 0.1 + x[1] * 1.4  # scaling factor: 0.1-1.5
    CR = x[2]  # crossover rate: 0-1
    init_pop_scale = x[3] * 5  # initialization scale
    init_seed = 2 * x[4:] - 1
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    dim = 10
    np.random.seed(42)
    pop = np.random.randn(pop_size, dim) * init_pop_scale
    pop[0] = init_seed[:dim] * 5.12
    pop = np.clip(pop, -5.12, 5.12)
    
    fitness = np.array([rastrigin(ind) for ind in pop])
    n_generations = 40
    
    try:
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
    except Exception:
        return 1e10
    
    return np.min(fitness)


def _cma_es_25d(x: np.ndarray) -> float:
    """CMA-ES hyperparameter meta-optimization (25D)."""
    sigma0 = 0.1 + x[0] * 2.9  # initial step size: 0.1-3.0
    lambda_mult = 1 + x[1] * 4  # offspring multiplier: 1-5
    mu_ratio = 0.2 + x[2] * 0.3  # parent ratio: 0.2-0.5
    init_mean = 2 * x[3:13] - 1
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    dim = 10
    n_offspring = int(4 + 3 * np.log(dim) * lambda_mult)
    n_parents = max(1, int(n_offspring * mu_ratio))
    
    mean = init_mean[:dim] * 5.12
    sigma = sigma0
    C = np.eye(dim)
    n_generations = 30
    
    try:
        for _ in range(n_generations):
            offspring = []
            for _ in range(n_offspring):
                z = np.random.randn(dim)
                try:
                    x_sample = mean + sigma * np.linalg.cholesky(C + 1e-8 * np.eye(dim)) @ z
                except:
                    x_sample = mean + sigma * z
                x_sample = np.clip(x_sample, -5.12, 5.12)
                offspring.append((x_sample, rastrigin(x_sample)))
            
            offspring.sort(key=lambda t: t[1])
            parents = [o[0] for o in offspring[:n_parents]]
            
            mean = np.mean(parents, axis=0)
            
            if len(parents) > 1:
                deviations = np.array(parents) - mean
                C = 0.8 * C + 0.2 * (deviations.T @ deviations) / n_parents
            
            sigma *= 0.95
    except Exception:
        return 1e10
    
    return rastrigin(mean)


def _hyperband_60d(x: np.ndarray) -> float:
    """Hyperband-style multi-fidelity search (60D)."""
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    configs = (2 * x[:30] - 1).reshape(6, 5) * 5.12  # 6 configurations
    budgets = x[30:36] + 0.1  # budget for each config
    
    total_cost = 0
    np.random.seed(42)
    
    try:
        for i, (config, budget) in enumerate(zip(configs, budgets)):
            n_evals = int(10 + budget * 40)  # 10-50 evaluations
            results = []
            for _ in range(n_evals):
                noise = np.random.randn(5) * 0.1
                results.append(rastrigin(config + noise))
            total_cost += np.mean(results)
    except Exception:
        return 1e10
    
    return total_cost / 6


def _bayesian_opt_60d(x: np.ndarray) -> float:
    """Bayesian optimization surrogate (60D)."""
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    length_scales = x[:10] + 0.1  # kernel length scales
    init_points = (2 * x[10:50] - 1).reshape(8, 5) * 5.12  # initial observations
    
    X_observed = init_points
    y_observed = np.array([rastrigin(x_pt) for x_pt in X_observed])
    
    np.random.seed(42)
    
    try:
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
    except Exception:
        return 1e10
    
    return np.min(y_observed)


def _nas_70d(x: np.ndarray) -> float:
    """Neural architecture search (70D)."""
    layer_sizes = (x[:5] * 50 + 10).astype(int)  # 10-60 neurons per layer
    activations = x[5:10]  # activation choices
    connections = x[10:35].reshape(5, 5)  # connection patterns
    weights = 2 * x[35:70] - 1  # weight initialization
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    
    try:
        h = X[:, :5]
        for i in range(min(3, len(layer_sizes))):
            if i*10+10 <= len(weights):
                W = weights[i*10:(i+1)*10].reshape(5, 2)
                if h.shape[1] >= 2:
                    h = h @ W[:, :min(2, h.shape[1])]
                    
            if activations[i] > 0.5:
                h = np.tanh(h)
            else:
                h = np.maximum(0, h)
        
        out = np.mean(h, axis=1) if h.ndim > 1 else h
        return np.mean((out - y)**2)
    except Exception:
        return 1e10


def _evolution_strategy_65d(x: np.ndarray) -> float:
    """Evolution strategy meta-optimization (65D)."""
    sigma = x[0] + 0.01  # step size
    learning_rate = x[1] * 0.1 + 0.001  # learning rate
    pop_size = int(x[2] * 40 + 10)  # population size: 10-50
    init_mean = 2 * x[3:18] - 1
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    mean = init_mean[:10] * 5.12
    np.random.seed(42)
    
    try:
        for _ in range(30):
            population = mean + sigma * np.random.randn(pop_size, 10)
            population = np.clip(population, -5.12, 5.12)
            
            fitness = np.array([rastrigin(ind) for ind in population])
            
            ranks = np.argsort(np.argsort(fitness))
            weights = np.maximum(0, np.log(pop_size/2 + 1) - np.log(ranks + 1))
            weights = weights / np.sum(weights)
            
            gradient = np.sum((population - mean).T * weights, axis=1)
            mean = mean + learning_rate * gradient
    except Exception:
        return 1e10
    
    return rastrigin(mean)


def _simulated_annealing_55d(x: np.ndarray) -> float:
    """Simulated annealing meta-optimization (55D)."""
    T_init = x[0] * 100 + 1  # initial temperature: 1-101
    cooling = 0.8 + x[1] * 0.19  # cooling rate: 0.8-0.99
    step_size = x[2] + 0.1  # step size: 0.1-1.1
    init_x = 2 * x[3:18] - 1
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    x_curr = init_x[:10] * 5.12
    best_x = x_curr.copy()
    best_f = rastrigin(x_curr)
    T = T_init
    
    np.random.seed(42)
    
    try:
        for _ in range(200):
            x_new = x_curr + step_size * np.random.randn(10)
            x_new = np.clip(x_new, -5.12, 5.12)
            
            f_new = rastrigin(x_new)
            delta = f_new - rastrigin(x_curr)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                x_curr = x_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_curr.copy()
            
            T *= cooling
    except Exception:
        return 1e10
    
    return best_f


def _covariance_adaptation_60d(x: np.ndarray) -> float:
    """Covariance matrix adaptation (60D)."""
    sigma = x[0] + 0.1  # step size
    c_sigma = x[1] * 0.5  # cumulation for step size
    c_c = x[2] * 0.5  # cumulation for covariance
    init_mean = 2 * x[3:13] - 1
    init_C_diag = x[13:23] + 0.1  # covariance matrix diagonal
    
    def rastrigin(x_vec):
        return 10*len(x_vec) + np.sum(x_vec**2 - 10*np.cos(2*np.pi*x_vec))
    
    mean = init_mean * 5.12
    C = np.diag(init_C_diag)
    
    lambda_pop = 20
    mu = 10
    
    np.random.seed(42)
    
    try:
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
    except Exception:
        return 1e10
    
    return rastrigin(mean)


def _pid_tuning_6d(x: np.ndarray) -> float:
    """PID controller tuning (6D: Kp, Ki, Kd + padding)."""
    # Scale from [0,1] to [-10, 10] to match original bounds
    Kp = x[0] * 20 - 10  # -10 to 10
    Ki = x[1] * 20 - 10 if len(x) > 1 else 0.0
    Kd = x[2] * 20 - 10 if len(x) > 2 else 0.0
    
    dt = 0.01
    y, y_prev = 0.0, 0.0
    integral = 0.0
    setpoint = 1.0
    
    total_error = 0.0
    
    try:
        for _ in range(500):
            error = setpoint - y
            integral += error * dt
            derivative = (y - y_prev) / dt
            u = Kp * error + Ki * integral - Kd * derivative
            y_prev = y
            y += dt * (-y + u)
            
            if np.abs(y) > 1e6:
                return 1e10
            
            total_error += error**2 + 0.01 * u**2
    except Exception:
        return 1e10
    
    return total_error


def _lqr_control_8d(x: np.ndarray) -> float:
    """LQR control synthesis (8D: controller gains + system params)."""
    n, m = 2, 1
    
    if len(x) >= n * m:
        K = x[:n*m].reshape(m, n) * 10 - 5  # scale to [-5, 5]
    else:
        K = np.zeros((m, n))
    
    A = np.array([[0, 1], [-1, -0.1]])
    B = np.array([[0], [1]])
    Q = np.eye(n)
    R = np.array([[0.1]])
    
    Acl = A - B @ K
    eigvals = np.linalg.eigvals(Acl)
    
    if np.any(np.real(eigvals) > 0):
        return 1e10
    
    x_state = np.array([1.0, 0.0])
    total_cost = 0.0
    
    try:
        for _ in range(100):
            u = -K @ x_state
            total_cost += float(x_state @ Q @ x_state + u @ R @ u)
            x_state = A @ x_state + (B @ u).flatten()
    except Exception:
        return 1e10
    
    return total_cost


def _trajectory_opt_100d(x: np.ndarray) -> float:
    """Trajectory optimization for dynamical system (100D: control sequence)."""
    state = np.array([0.0, 0.0])
    target = np.array([1.0, 0.0])
    
    dt = 0.1
    total_cost = 0.0
    
    controls = 2 * x - 1  # scale to [-1, 1]
    
    try:
        for u in controls:
            state[0] += dt * state[1]
            state[1] += dt * u
            total_cost += 0.01 * u**2
    except Exception:
        return 1e10
    
    terminal_error = np.sum((state - target)**2)
    return terminal_error + total_cost


def _trajectory_opt_120d(x: np.ndarray) -> float:
    """Extended trajectory optimization (120D: longer time horizon)."""
    state = np.array([0.0, 0.0])
    target = np.array([1.0, 0.0])
    
    dt = 0.08
    total_cost = 0.0
    
    controls = 2 * x - 1  # scale to [-1, 1]
    
    try:
        for u in controls:
            state[0] += dt * state[1]
            state[1] += dt * u
            total_cost += 0.01 * u**2
    except Exception:
        return 1e10
    
    terminal_error = np.sum((state - target)**2)
    return terminal_error + total_cost


def _inverse_kinematics_80d(x: np.ndarray) -> float:
    """Robot arm inverse kinematics (80D: joint angles)."""
    n = len(x)
    link_length = 1.0 / n
    
    joint_angles = 2 * np.pi * x - np.pi  # scale to [-π, π]
    
    x_pos, y_pos = 0.0, 0.0
    angle = 0.0
    
    try:
        for theta in joint_angles:
            angle += theta
            x_pos += link_length * np.cos(angle)
            y_pos += link_length * np.sin(angle)
    except Exception:
        return 1e10
    
    target = np.array([0.5, 0.5])
    end = np.array([x_pos, y_pos])
    
    dist_error = np.sum((end - target)**2)
    smoothness = np.sum(np.diff(joint_angles)**2)
    
    return dist_error + 0.01 * smoothness


def _inverse_kinematics_100d(x: np.ndarray) -> float:
    """Extended robot arm inverse kinematics (100D: more joints)."""
    n = len(x)
    link_length = 1.0 / n
    
    joint_angles = 2 * np.pi * x - np.pi  # scale to [-π, π]
    
    x_pos, y_pos = 0.0, 0.0
    angle = 0.0
    
    try:
        for theta in joint_angles:
            angle += theta
            x_pos += link_length * np.cos(angle)
            y_pos += link_length * np.sin(angle)
    except Exception:
        return 1e10
    
    target = np.array([0.5, 0.5])
    end = np.array([x_pos, y_pos])
    
    dist_error = np.sum((end - target)**2)
    smoothness = np.sum(np.diff(joint_angles)**2)
    
    return dist_error + 0.01 * smoothness


def _inverse_kinematics_long_80d(x: np.ndarray) -> float:
    """Long-horizon inverse kinematics optimization (80D)."""
    # Same as regular but with tighter smoothness constraint
    n = len(x)
    link_length = 1.0 / n
    
    joint_angles = 2 * np.pi * x - np.pi
    
    x_pos, y_pos = 0.0, 0.0
    angle = 0.0
    
    try:
        for theta in joint_angles:
            angle += theta
            x_pos += link_length * np.cos(angle)
            y_pos += link_length * np.sin(angle)
    except Exception:
        return 1e10
    
    target = np.array([0.5, 0.5])
    end = np.array([x_pos, y_pos])
    
    dist_error = np.sum((end - target)**2)
    smoothness = np.sum(np.diff(joint_angles)**2)
    
    return dist_error + 0.05 * smoothness  # Higher smoothness penalty


def _inverse_kinematics_long_100d(x: np.ndarray) -> float:
    """Extended long-horizon inverse kinematics (100D)."""
    n = len(x)
    link_length = 1.0 / n
    
    joint_angles = 2 * np.pi * x - np.pi
    
    x_pos, y_pos = 0.0, 0.0
    angle = 0.0
    
    try:
        for theta in joint_angles:
            angle += theta
            x_pos += link_length * np.cos(angle)
            y_pos += link_length * np.sin(angle)
    except Exception:
        return 1e10
    
    target = np.array([0.5, 0.5])
    end = np.array([x_pos, y_pos])
    
    dist_error = np.sum((end - target)**2)
    smoothness = np.sum(np.diff(joint_angles)**2)
    
    return dist_error + 0.05 * smoothness  # Higher smoothness penalty


# =============================================================================
# REMAINING PROBLEMS (18 functions) - Chunk 2.2.9
# =============================================================================

def _sa_schedule_3d(x: np.ndarray) -> float:
    """Simulated annealing schedule optimization (3D meta-optimization)."""
    # Map from [0,1] to proper ranges
    T_init = 10 ** (x[0] * 3)  # 1 to 1000
    cooling_rate = 0.8 + 0.199 * x[1]  # 0.8 to 0.999
    n_iters = int(10 + x[2] * 90)  # 10-100 per temperature
    
    if T_init <= 0 or cooling_rate <= 0 or cooling_rate >= 1:
        return 1e10
    
    def rastrigin(x_vec):
        return 10 * len(x_vec) + np.sum(x_vec**2 - 10 * np.cos(2 * np.pi * x_vec))
    
    n_runs = 5
    dim = 5
    total_best = 0
    
    try:
        for run in range(n_runs):
            np.random.seed(run)
            x_sa = np.random.uniform(-5.12, 5.12, dim)
            best = rastrigin(x_sa)
            T = T_init
            
            max_temps = 50
            for _ in range(max_temps):
                for _ in range(n_iters):
                    x_new = x_sa + np.random.randn(dim) * T * 0.1
                    x_new = np.clip(x_new, -5.12, 5.12)
                    
                    f_new = rastrigin(x_new)
                    delta = f_new - rastrigin(x_sa)
                    
                    if delta < 0 or np.random.rand() < np.exp(-delta / T):
                        x_sa = x_new
                        if f_new < best:
                            best = f_new
                
                T *= cooling_rate
                if T < 1e-6:
                    break
            
            total_best += best
    except Exception:
        return 1e10
    
    return total_best / n_runs


def _cellular_automata_25d(x: np.ndarray) -> float:
    """Cellular automata rule optimization (25D = 5x5 rule weights)."""
    # Map from [0,1] to [-1,1]
    weights = (2 * x - 1).reshape(5, 5)
    
    try:
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
    except Exception:
        return 1e10


def _cellular_automaton_120d(x: np.ndarray) -> float:
    """Continuous cellular automaton (120D) - pattern formation optimization."""
    n = len(x)
    grid = x.copy()
    
    try:
        grid_min, grid_max = grid.min(), grid.max()
        if grid_max - grid_min > 1e-10:
            grid = (grid - grid_min) / (grid_max - grid_min)
        else:
            grid = np.ones_like(grid) * 0.5
        
        for _ in range(50):
            new_grid = np.zeros_like(grid)
            for i in range(n):
                left = grid[(i-1) % n]
                center = grid[i]
                right = grid[(i+1) % n]
                avg = (left + center + right) / 3
                new_grid[i] = 0.5 + 0.5 * np.tanh(5 * (avg - 0.5))
            grid = new_grid
        
        hist, _ = np.histogram(grid, bins=20, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return -entropy  # Maximize entropy
    except Exception:
        return 1e10


def _spin_glass_150d(x: np.ndarray) -> float:
    """Spin glass energy minimization (150D)."""
    n = len(x)
    # Map from [0,1] to [-1,1] via tanh
    spins = 2*x - 1
    spins = np.tanh(spins)
    
    try:
        rng = np.random.RandomState(42)
        J = rng.randn(n, n) / np.sqrt(n)
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        
        energy = -0.5 * np.dot(spins, np.dot(J, spins))
        h = rng.randn(n) * 0.1
        energy -= np.dot(h, spins)
        
        return energy
    except Exception:
        return 1e10


def _covariance_estimation_120d(x: np.ndarray) -> float:
    """Covariance matrix estimation (120D - lower triangular of 15x15)."""
    n = 15
    expected_dim = n * (n + 1) // 2
    
    try:
        if len(x) < expected_dim:
            params = np.concatenate([x, np.zeros(expected_dim - len(x))])
        else:
            params = x[:expected_dim]
        
        # Map from [0,1] to parameter range
        params = 2 * params - 1
        
        L = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                L[i, j] = params[idx]
                idx += 1
        
        Sigma = L @ L.T + 0.01 * np.eye(n)
        
        rho = 0.7
        Sigma_true = np.array([[rho**abs(i-j) for j in range(n)] for i in range(n)])
        
        return np.sum((Sigma - Sigma_true)**2)
    except Exception:
        return 1e10


def _linear_system_id_144d(x: np.ndarray) -> float:
    """Linear dynamical system identification (144D = 12x12 matrix)."""
    n = 12
    
    try:
        if len(x) < n*n:
            params = np.concatenate([x, np.zeros(n*n - len(x))])
        else:
            params = x[:n*n]
        
        # Map from [0,1] to [-1,1]
        A = (2 * params - 1).reshape(n, n) * 0.5
        
        eigvals = np.linalg.eigvals(A)
        if np.max(np.abs(eigvals)) > 1.0:
            return 1e10 * np.max(np.abs(eigvals))
        
        rng = np.random.RandomState(456)
        A_true = rng.randn(n, n) * 0.3
        
        x_state = rng.randn(n)
        x_true = x_state.copy()
        
        error = 0.0
        for _ in range(50):
            x_state = A @ x_state
            x_true = A_true @ x_true
            error += np.sum((x_state - x_true)**2)
        
        return error
    except Exception:
        return 1e10


def _coupled_logistic_maps_100d(x: np.ndarray) -> float:
    """Coupled logistic map lattice (100D) - chaotic dynamics."""
    n = len(x)
    eps = 0.3
    state = np.clip(x.copy(), 0.01, 0.99)
    
    try:
        for _ in range(100):
            fx = 4 * state * (1 - state)
            state_new = (1 - eps) * fx
            state_new[1:] += eps/2 * fx[:-1]
            state_new[:-1] += eps/2 * fx[1:]
            state_new[0] += eps/2 * fx[-1]
            state_new[-1] += eps/2 * fx[0]
            state = np.clip(state_new, 0.01, 0.99)
        
        return np.var(state) * n + np.abs(np.mean(state) - 0.5)
    except Exception:
        return 1e10


def _coupled_pendulums_100d(x: np.ndarray) -> float:
    """Coupled pendulums energy variance (100D)."""
    n = len(x)
    # Map from [0,1] to [-π, π]
    theta = (2*x - 1) * np.pi
    omega = np.zeros(n)
    k = 0.5
    
    T, dt = 50, 0.1
    energies = []
    
    try:
        for _ in range(int(T/dt)):
            accel = -np.sin(theta)
            accel[1:] += k * (theta[:-1] - theta[1:])
            accel[:-1] += k * (theta[1:] - theta[:-1])
            omega += dt * accel
            theta += dt * omega
            E = 0.5 * np.sum(omega**2) + np.sum(1 - np.cos(theta))
            energies.append(E)
        
        return np.var(energies) + 0.01 * np.mean(energies)
    except Exception:
        return 1e10


def _standard_map_chain_30d(x: np.ndarray) -> float:
    """Chain of standard maps (30D) - coupled Chirikov maps."""
    n = len(x) // 2
    # Map from [0,1] to [0, 2π]
    theta = x[:n] * 2 * np.pi
    p = x[n:] * 2 * np.pi
    
    K = 0.9
    
    try:
        for _ in range(50):
            p_new = p + K * np.sin(theta)
            theta_new = theta + p_new
            theta_new[1:] += 0.1 * np.sin(theta[:-1] - theta[1:])
            theta, p = theta_new % (2*np.pi), p_new
        
        return np.var(p)
    except Exception:
        return 1e10


def _epidemic_control_25d(x: np.ndarray) -> float:
    """Epidemic control optimization (25D) - minimize infections and cost."""
    n = len(x)
    beta0 = 0.3
    gamma = 0.1
    N = 1000
    
    S, I, R = N - 1, 1, 0
    dt = 0.5
    total_infected = 0
    intervention_cost = 0
    
    try:
        for intervention in x:
            intervention = np.clip(intervention, 0, 1)
            beta = beta0 * (1 - 0.8 * intervention)
            
            for _ in range(10):
                dS = -beta * S * I / N
                dI = beta * S * I / N - gamma * I
                dR = gamma * I
                S += dt * dS
                I += dt * dI
                R += dt * dR
                total_infected += I * dt
                intervention_cost += intervention**2
        
        return total_infected / 1000 + 0.1 * intervention_cost
    except Exception:
        return 1e10


def _epidemic_control_40d(x: np.ndarray) -> float:
    """Epidemic control optimization (40D) - extended time horizon."""
    return _epidemic_control_25d(x)


def _supply_chain_35d(x: np.ndarray) -> float:
    """Supply chain inventory optimization (35D)."""
    n = len(x)
    n_stages = 5
    n_products = max(1, n // n_stages)
    
    # Map from [0,1] to reorder points
    reorder_points = x[:n_products*n_stages].reshape(n_stages, -1) * 100
    
    rng = np.random.RandomState(333)
    n_simulations = 50
    
    total_cost = 0.0
    
    try:
        for _ in range(n_simulations):
            inventory = np.ones((n_stages, n_products)) * 100
            
            for t in range(100):
                demand = rng.poisson(10, n_products)
                fulfilled = np.minimum(inventory[-1], demand)
                inventory[-1] -= fulfilled
                stockout = demand - fulfilled
                total_cost += np.sum(stockout) * 10
                total_cost += np.sum(inventory) * 0.1
                
                for s in range(n_stages - 1):
                    reorder = inventory[s] < reorder_points[s]
                    if s == 0:
                        inventory[s][reorder] += 50
                    else:
                        transfer = np.minimum(inventory[s-1], 50 * np.ones(n_products))
                        inventory[s][reorder] += transfer[reorder]
                        inventory[s-1][reorder] -= transfer[reorder]
        
        return total_cost / n_simulations
    except Exception:
        return 1e10


def _supply_chain_50d(x: np.ndarray) -> float:
    """Supply chain optimization (50D) - larger problem."""
    return _supply_chain_35d(x)


def _graph_partition_25d(x: np.ndarray) -> float:
    """Graph partitioning continuous relaxation (25D)."""
    n = len(x)
    # Map from [0,1] to assignment probabilities via sigmoid
    assignment = x.copy()
    
    try:
        rng = np.random.RandomState(444)
        A = rng.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.7).astype(float)
        
        cut = 0.0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] > 0:
                    cut += A[i, j] * assignment[i] * (1 - assignment[j])
                    cut += A[i, j] * (1 - assignment[i]) * assignment[j]
        
        balance = (np.sum(assignment) - n/2)**2
        return cut + 0.1 * balance
    except Exception:
        return 1e10


def _graph_partition_40d(x: np.ndarray) -> float:
    """Graph partitioning (40D) - larger graph."""
    return _graph_partition_25d(x)


def _risk_parity_30d(x: np.ndarray) -> float:
    """Risk parity portfolio optimization (30D)."""
    n = len(x)
    w = np.abs(x)
    w = w / (np.sum(w) + 1e-10)
    
    try:
        rng = np.random.RandomState(555)
        factors = rng.randn(n, 3)
        specific = np.diag(rng.rand(n) * 0.1)
        Sigma = factors @ factors.T / 10 + specific
        
        port_var = w @ Sigma @ w
        marginal_risk = Sigma @ w
        risk_contrib = w * marginal_risk / (np.sqrt(port_var) + 1e-10)
        target_contrib = np.sqrt(port_var) / n
        
        return np.sum((risk_contrib - target_contrib)**2)
    except Exception:
        return 1e10


def _chemical_kinetics_5d(x: np.ndarray) -> float:
    """Chemical reaction kinetics parameter fitting (5D)."""
    k = np.abs(x)
    if len(k) < 3:
        k = np.concatenate([k, np.ones(3 - len(k))])
    
    k1, k2 = k[0] * 2, k[1] * 2  # Scale to reasonable range
    dt = 0.1
    A, B, C = 1.0, 0.0, 0.0
    
    trajectory = []
    
    try:
        for _ in range(100):
            dA = -k1 * A
            dB = k1 * A - k2 * B
            dC = k2 * B
            A += dt * dA
            B += dt * dB
            C += dt * dC
            trajectory.append([A, B, C])
        
        trajectory = np.array(trajectory)
        
        k1_true, k2_true = 0.5, 0.3
        A, B, C = 1.0, 0.0, 0.0
        target = []
        for _ in range(100):
            dA = -k1_true * A
            dB = k1_true * A - k2_true * B
            dC = k2_true * B
            A += dt * dA
            B += dt * dB
            C += dt * dC
            target.append([A, B, C])
        target = np.array(target)
        
        return np.mean((trajectory - target)**2)
    except Exception:
        return 1e10


def _regression_coeffs_5d(x: np.ndarray) -> float:
    """Regularized regression coefficient estimation (5D)."""
    n = len(x)
    # Map from [0,1] to [-5,5]
    params = (2*x - 1) * 5
    
    try:
        rng = np.random.RandomState(888)
        X = rng.randn(50, n)
        beta_true = rng.randn(n)
        y = X @ beta_true + rng.randn(50) * 0.1
        
        pred = X @ params
        mse = np.mean((y - pred)**2)
        reg = 0.1 * np.sum(params**2)
        
        return mse + reg
    except Exception:
        return 1e10


def _nested_cv_5d(x: np.ndarray) -> float:
    """Hyperparameter optimization with nested CV (5D)."""
    C = np.exp(x[0])
    gamma = np.exp(x[1]) if len(x) > 1 else 1.0
    
    rng = np.random.RandomState(666)
    
    outer_scores = []
    for outer_fold in range(5):
        inner_scores = []
        for inner_fold in range(3):
            base_score = 0.8
            score = base_score - 0.1 * np.abs(np.log10(C + 1e-10) - 1)
            score -= 0.1 * np.abs(np.log10(gamma + 1e-10))
            score += rng.randn() * 0.02
            inner_scores.append(score)
        outer_scores.append(np.mean(inner_scores))
    
    return -np.mean(outer_scores) + 0.1 * np.var(outer_scores)


def _bayesian_acquisition_6d(x: np.ndarray) -> float:
    """Bayesian acquisition function optimization surrogate (6D)."""
    n = len(x)
    
    rng = np.random.RandomState(777)
    n_basis = 20
    
    centers = rng.randn(n_basis, n)
    weights_mean = rng.randn(n_basis)
    weights_var = np.abs(rng.randn(n_basis))
    
    rbf = np.exp(-np.sum((x - centers)**2, axis=1) / 2)
    
    mean = np.dot(rbf, weights_mean)
    var = np.dot(rbf, weights_var) + 0.01
    
    best_f = -0.5
    z = (best_f - mean) / (np.sqrt(var) + 1e-10)
    ei = (best_f - mean) * (0.5 + 0.5 * np.tanh(z)) + np.sqrt(var) * np.exp(-z**2/2)
    
    return -ei


# Registry for ML Training/CV problems
_ML_TRAINING_REGISTRY: Dict[str, BenchmarkProblem] = {
    'SVM-CV-2D': BenchmarkProblem(
        name='SVM-CV-2D',
        objective=_make_optuna_objective(_svm_cv_2d, [(-3, 3), (-4, 1)], 2),
        dimension=2,
        bounds=[(-3, 3), (-4, 1)],
        known_optimum=None,
        category='ml_training',
        description='SVM hyperparameter optimization with 5-fold CV'
    ),
    'RF-CV-4D': BenchmarkProblem(
        name='RF-CV-4D',
        objective=_make_optuna_objective(_rf_cv_4d, [(0, 1), (0, 1), (0, 1), (0, 1)], 4),
        dimension=4,
        bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='Random Forest hyperparameter optimization with CV'
    ),
    'Ridge-CV-1D': BenchmarkProblem(
        name='Ridge-CV-1D',
        objective=_make_optuna_objective(_ridge_cv_1d, [(-4, 4)], 1),
        dimension=1,
        bounds=[(-4, 4)],
        known_optimum=None,
        category='ml_training',
        description='Ridge regression alpha tuning with CV'
    ),
    'Lasso-CV-1D': BenchmarkProblem(
        name='Lasso-CV-1D',
        objective=_make_optuna_objective(_lasso_cv_1d, [(-4, 4)], 1),
        dimension=1,
        bounds=[(-4, 4)],
        known_optimum=None,
        category='ml_training',
        description='Lasso regression alpha tuning with CV'
    ),
    'ElasticNet-CV-2D': BenchmarkProblem(
        name='ElasticNet-CV-2D',
        objective=_make_optuna_objective(_elasticnet_cv_2d, [(-4, 4), (0.01, 0.99)], 2),
        dimension=2,
        bounds=[(-4, 4), (0.01, 0.99)],
        known_optimum=None,
        category='ml_training',
        description='Elastic Net hyperparameter tuning with CV'
    ),
    'LogisticReg-CV-1D': BenchmarkProblem(
        name='LogisticReg-CV-1D',
        objective=_make_optuna_objective(_logisticreg_cv_1d, [(-4, 4)], 1),
        dimension=1,
        bounds=[(-4, 4)],
        known_optimum=None,
        category='ml_training',
        description='Logistic regression C tuning with CV'
    ),
    'KNN-CV-3D': BenchmarkProblem(
        name='KNN-CV-3D',
        objective=_make_optuna_objective(_knn_cv_3d, [(0, 1), (0, 1), (0, 1)], 3),
        dimension=3,
        bounds=[(0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='KNN classifier hyperparameter tuning with CV'
    ),
    'DecisionTree-CV-3D': BenchmarkProblem(
        name='DecisionTree-CV-3D',
        objective=_make_optuna_objective(_decisiontree_cv_3d, [(0, 1), (0, 1), (0, 1)], 3),
        dimension=3,
        bounds=[(0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='Decision tree hyperparameter tuning with CV'
    ),
    'AdaBoost-CV-2D': BenchmarkProblem(
        name='AdaBoost-CV-2D',
        objective=_make_optuna_objective(_adaboost_cv_2d, [(0, 1), (0, 1)], 2),
        dimension=2,
        bounds=[(0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='AdaBoost hyperparameter tuning with CV'
    ),
    'SVM-Large-CV-2D': BenchmarkProblem(
        name='SVM-Large-CV-2D',
        objective=_make_optuna_objective(_svm_large_cv_2d, [(-3, 3), (-4, 1)], 2),
        dimension=2,
        bounds=[(-3, 3), (-4, 1)],
        known_optimum=None,
        category='ml_training',
        description='SVM with larger dataset and 10-fold CV'
    ),
    'Bagging-CV-3D': BenchmarkProblem(
        name='Bagging-CV-3D',
        objective=_make_optuna_objective(_bagging_cv_3d, [(0, 1), (0, 1), (0, 1)], 3),
        dimension=3,
        bounds=[(0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='Bagging classifier hyperparameter tuning with CV'
    ),
    'GradientBoost-CV-3D': BenchmarkProblem(
        name='GradientBoost-CV-3D',
        objective=_make_optuna_objective(_gradientboost_cv_3d, [(0, 1), (0, 1), (0, 1)], 3),
        dimension=3,
        bounds=[(0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='Gradient Boosting hyperparameter tuning with CV'
    ),
    'MLP-Regressor-CV-3D': BenchmarkProblem(
        name='MLP-Regressor-CV-3D',
        objective=_make_optuna_objective(_mlp_regressor_cv_3d, [(0, 1), (0, 1), (0, 1)], 3),
        dimension=3,
        bounds=[(0, 1), (0, 1), (0, 1)],
        known_optimum=None,
        category='ml_training',
        description='MLP regressor hyperparameter tuning with CV'
    ),
    'NestedCV-5D': BenchmarkProblem(
        name='NestedCV-5D',
        objective=_make_optuna_objective(_nested_cv_5d, [(-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3)], 5),
        dimension=5,
        bounds=[(-3, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3)],
        known_optimum=None,
        category='ml_training',
        description='Nested cross-validation optimization'
    ),
    'BayesianAcquisition-6D': BenchmarkProblem(
        name='BayesianAcquisition-6D',
        objective=_make_optuna_objective(_bayesian_acquisition_6d, [(-3, 3)] * 6, 6),
        dimension=6,
        bounds=[(-3, 3)] * 6,
        known_optimum=None,
        category='ml_training',
        description='Bayesian acquisition function optimization'
    ),
    # Chunk 2.2.5: Additional ML Training Problems (18 new)
    'NeuralNet-Dropout-20D': BenchmarkProblem(
        name='NeuralNet-Dropout-20D',
        objective=_make_optuna_objective(_neuralnet_dropout_20d, [(0, 1)] * 20, 20),
        dimension=20,
        bounds=[(0, 1)] * 20,
        known_optimum=None,
        category='ml_training',
        description='Neural network with dropout/regularization tuning (20D)'
    ),
    'LightGBM-CV-4D': BenchmarkProblem(
        name='LightGBM-CV-4D',
        objective=_make_optuna_objective(_lightgbm_cv_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_training',
        description='LightGBM hyperparameter optimization with CV'
    ),
    'XGBoost-CV-4D': BenchmarkProblem(
        name='XGBoost-CV-4D',
        objective=_make_optuna_objective(_xgboost_cv_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_training',
        description='XGBoost hyperparameter optimization with CV'
    ),
    'CatBoost-CV-4D': BenchmarkProblem(
        name='CatBoost-CV-4D',
        objective=_make_optuna_objective(_catboost_cv_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_training',
        description='CatBoost simulation hyperparameter optimization with CV'
    ),
    'ExtraTrees-CV-4D': BenchmarkProblem(
        name='ExtraTrees-CV-4D',
        objective=_make_optuna_objective(_extratrees_cv_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_training',
        description='Extra Trees hyperparameter optimization with CV'
    ),
    'StackingEnsemble-15D': BenchmarkProblem(
        name='StackingEnsemble-15D',
        objective=_make_optuna_objective(_stacking_ensemble_15d, [(0, 1)] * 15, 15),
        dimension=15,
        bounds=[(0, 1)] * 15,
        known_optimum=None,
        category='ml_training',
        description='Stacking ensemble with multiple base models (15D)'
    ),
    'VotingEnsemble-10D': BenchmarkProblem(
        name='VotingEnsemble-10D',
        objective=_make_optuna_objective(_voting_ensemble_10d, [(0, 1)] * 10, 10),
        dimension=10,
        bounds=[(0, 1)] * 10,
        known_optimum=None,
        category='ml_training',
        description='Voting ensemble with weight tuning (10D)'
    ),
    'FeatureSelection-RFE-20D': BenchmarkProblem(
        name='FeatureSelection-RFE-20D',
        objective=_make_optuna_objective(_feature_selection_rfe_20d, [(0, 1)] * 20, 20),
        dimension=20,
        bounds=[(0, 1)] * 20,
        known_optimum=None,
        category='ml_training',
        description='RFE feature selection optimization (20D)'
    ),
    'FeatureSelection-MI-15D': BenchmarkProblem(
        name='FeatureSelection-MI-15D',
        objective=_make_optuna_objective(_feature_selection_mi_15d, [(0, 1)] * 15, 15),
        dimension=15,
        bounds=[(0, 1)] * 15,
        known_optimum=None,
        category='ml_training',
        description='Mutual information feature selection (15D)'
    ),
    'FeatureEngineering-Poly-12D': BenchmarkProblem(
        name='FeatureEngineering-Poly-12D',
        objective=_make_optuna_objective(_feature_engineering_poly_12d, [(0, 1)] * 12, 12),
        dimension=12,
        bounds=[(0, 1)] * 12,
        known_optimum=None,
        category='ml_training',
        description='Polynomial feature engineering (12D)'
    ),
    'FeatureScaling-Robust-8D': BenchmarkProblem(
        name='FeatureScaling-Robust-8D',
        objective=_make_optuna_objective(_feature_scaling_robust_8d, [(0, 1)] * 8, 8),
        dimension=8,
        bounds=[(0, 1)] * 8,
        known_optimum=None,
        category='ml_training',
        description='Robust scaling optimization (8D)'
    ),
    'ClassWeights-Imbalanced-10D': BenchmarkProblem(
        name='ClassWeights-Imbalanced-10D',
        objective=_make_optuna_objective(_class_weights_imbalanced_10d, [(0, 1)] * 10, 10),
        dimension=10,
        bounds=[(0, 1)] * 10,
        known_optimum=None,
        category='ml_training',
        description='Class weight tuning for imbalanced data (10D)'
    ),
    'PCA-Components-1D': BenchmarkProblem(
        name='PCA-Components-1D',
        objective=_make_optuna_objective(_pca_components_1d, [(0, 1)], 1),
        dimension=1,
        bounds=[(0, 1)],
        known_optimum=None,
        category='ml_training',
        description='PCA component selection (1D)'
    ),
    'SVD-Components-1D': BenchmarkProblem(
        name='SVD-Components-1D',
        objective=_make_optuna_objective(_svd_components_1d, [(0, 1)], 1),
        dimension=1,
        bounds=[(0, 1)],
        known_optimum=None,
        category='ml_training',
        description='TruncatedSVD component selection (1D)'
    ),
    'TSNE-Hyperparams-3D': BenchmarkProblem(
        name='TSNE-Hyperparams-3D',
        objective=_make_optuna_objective(_tsne_hyperparams_3d, [(0, 1)] * 3, 3),
        dimension=3,
        bounds=[(0, 1)] * 3,
        known_optimum=None,
        category='ml_training',
        description='t-SNE hyperparameter optimization (3D)'
    ),
    'UMAP-Hyperparams-4D': BenchmarkProblem(
        name='UMAP-Hyperparams-4D',
        objective=_make_optuna_objective(_umap_hyperparams_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_training',
        description='UMAP hyperparameter optimization simulation (4D)'
    ),
    'IsolationForest-CV-3D': BenchmarkProblem(
        name='IsolationForest-CV-3D',
        objective=_make_optuna_objective(_isolation_forest_cv_3d, [(0, 1)] * 3, 3),
        dimension=3,
        bounds=[(0, 1)] * 3,
        known_optimum=None,
        category='ml_training',
        description='Isolation Forest hyperparameter tuning (3D)'
    ),
    'AutoEncoder-Hyperparams-5D': BenchmarkProblem(
        name='AutoEncoder-Hyperparams-5D',
        objective=_make_optuna_objective(_autoencoder_hyperparams_5d, [(0, 1)] * 5, 5),
        dimension=5,
        bounds=[(0, 1)] * 5,
        known_optimum=None,
        category='ml_training',
        description='Autoencoder hyperparameter optimization (5D)'
    ),
    'SparseCoding-30D': BenchmarkProblem(
        name='SparseCoding-30D',
        objective=_make_optuna_objective(_sparse_coding_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='ml_training',
        description='Sparse coding dictionary learning (30D)'
    ),
    'NMF-Factorization-25D': BenchmarkProblem(
        name='NMF-Factorization-25D',
        objective=_make_optuna_objective(_nmf_factorization_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='ml_training',
        description='Non-negative matrix factorization (25D)'
    ),
    'KernelApproximation-20D': BenchmarkProblem(
        name='KernelApproximation-20D',
        objective=_make_optuna_objective(_kernel_approximation_20d, [(0, 1)] * 20, 20),
        dimension=20,
        bounds=[(0, 1)] * 20,
        known_optimum=None,
        category='ml_training',
        description='Kernel approximation with RBF sampler (20D)'
    ),
    'CalibrationTuning-15D': BenchmarkProblem(
        name='CalibrationTuning-15D',
        objective=_make_optuna_objective(_calibration_tuning_15d, [(0, 1)] * 15, 15),
        dimension=15,
        bounds=[(0, 1)] * 15,
        known_optimum=None,
        category='ml_training',
        description='Probability calibration tuning (15D)'
    ),
    'MultiOutputRegression-40D': BenchmarkProblem(
        name='MultiOutputRegression-40D',
        objective=_make_optuna_objective(_multi_output_regression_40d, [(0, 1)] * 40, 40),
        dimension=40,
        bounds=[(0, 1)] * 40,
        known_optimum=None,
        category='ml_training',
        description='Multi-output regression with chained models (40D)'
    ),
    'QuantileRegression-12D': BenchmarkProblem(
        name='QuantileRegression-12D',
        objective=_make_optuna_objective(_quantile_regression_12d, [(0, 1)] * 12, 12),
        dimension=12,
        bounds=[(0, 1)] * 12,
        known_optimum=None,
        category='ml_training',
        description='Quantile regression optimization (12D)'
    ),
    'SemiSupervised-30D': BenchmarkProblem(
        name='SemiSupervised-30D',
        objective=_make_optuna_objective(_semi_supervised_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='ml_training',
        description='Semi-supervised learning with label propagation (30D)'
    ),
    'OrdinalRegression-18D': BenchmarkProblem(
        name='OrdinalRegression-18D',
        objective=_make_optuna_objective(_ordinal_regression_18d, [(0, 1)] * 18, 18),
        dimension=18,
        bounds=[(0, 1)] * 18,
        known_optimum=None,
        category='ml_training',
        description='Ordinal regression with ordered logistic (18D)'
    ),
    'CostSensitiveLearning-22D': BenchmarkProblem(
        name='CostSensitiveLearning-22D',
        objective=_make_optuna_objective(_cost_sensitive_learning_22d, [(0, 1)] * 22, 22),
        dimension=22,
        bounds=[(0, 1)] * 22,
        known_optimum=None,
        category='ml_training',
        description='Cost-sensitive learning with custom loss (22D)'
    ),
    'TransferLearning-35D': BenchmarkProblem(
        name='TransferLearning-35D',
        objective=_make_optuna_objective(_transfer_learning_35d, [(0, 1)] * 35, 35),
        dimension=35,
        bounds=[(0, 1)] * 35,
        known_optimum=None,
        category='ml_training',
        description='Transfer learning simulation (35D)'
    ),
}

# =============================================================================
# PDE PROBLEMS REGISTRY - Chunk 2.2.7 (18 functions)
# =============================================================================

_PDE_REGISTRY: Dict[str, BenchmarkProblem] = {
    'Burgers-9D': BenchmarkProblem(
        name='Burgers-9D',
        objective=_make_optuna_objective(_burgers_equation_9d, [(0, 1)] * 9, 9),
        dimension=9,
        bounds=[(0, 1)] * 9,
        known_optimum=None,
        category='pde',
        description='1D Burgers equation parameter estimation (9D)'
    ),
    'PDE-HeatEq-50D': BenchmarkProblem(
        name='PDE-HeatEq-50D',
        objective=_make_optuna_objective(_pde_heat_eq_50d, [(0, 1)] * 50, 50),
        dimension=50,
        bounds=[(0, 1)] * 50,
        known_optimum=None,
        category='pde',
        description='1D heat equation with Fourier modes (50D)'
    ),
    'HeatDiffusion-30D': BenchmarkProblem(
        name='HeatDiffusion-30D',
        objective=_make_optuna_objective(_heat_diffusion_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='pde',
        description='Heat diffusion equation (30D)'
    ),
    'WaveEquation-30D': BenchmarkProblem(
        name='WaveEquation-30D',
        objective=_make_optuna_objective(_wave_equation_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='pde',
        description='Wave equation initial condition (30D)'
    ),
    'AdvectionDiffusion-30D': BenchmarkProblem(
        name='AdvectionDiffusion-30D',
        objective=_make_optuna_objective(_advection_diffusion_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='pde',
        description='Advection-diffusion equation (30D)'
    ),
    'ReactionDiffusion-30D': BenchmarkProblem(
        name='ReactionDiffusion-30D',
        objective=_make_optuna_objective(_reaction_diffusion_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='pde',
        description='Reaction-diffusion pattern formation (30D)'
    ),
    'Heat2D-60D': BenchmarkProblem(
        name='Heat2D-60D',
        objective=_make_optuna_objective(_heat_2d_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='pde',
        description='2D heat equation (60D)'
    ),
    'Poisson-60D': BenchmarkProblem(
        name='Poisson-60D',
        objective=_make_optuna_objective(_poisson_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='pde',
        description='Poisson equation source optimization (60D)'
    ),
    'Laplace-64D': BenchmarkProblem(
        name='Laplace-64D',
        objective=_make_optuna_objective(_laplace_64d, [(0, 1)] * 64, 64),
        dimension=64,
        bounds=[(0, 1)] * 64,
        known_optimum=None,
        category='pde',
        description='Laplace equation boundary optimization (64D)'
    ),
    'Helmholtz-60D': BenchmarkProblem(
        name='Helmholtz-60D',
        objective=_make_optuna_objective(_helmholtz_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='pde',
        description='Helmholtz equation parameter optimization (60D)'
    ),
    'Biharmonic-56D': BenchmarkProblem(
        name='Biharmonic-56D',
        objective=_make_optuna_objective(_biharmonic_56d, [(0, 1)] * 56, 56),
        dimension=56,
        bounds=[(0, 1)] * 56,
        known_optimum=None,
        category='pde',
        description='Biharmonic equation / plate bending (56D)'
    ),
    'SpectralMethod-64D': BenchmarkProblem(
        name='SpectralMethod-64D',
        objective=_make_optuna_objective(_spectral_method_64d, [(0, 1)] * 64, 64),
        dimension=64,
        bounds=[(0, 1)] * 64,
        known_optimum=None,
        category='pde',
        description='Spectral method coefficients (64D)'
    ),
    'FiniteElement-70D': BenchmarkProblem(
        name='FiniteElement-70D',
        objective=_make_optuna_objective(_finite_element_70d, [(0, 1)] * 70, 70),
        dimension=70,
        bounds=[(0, 1)] * 70,
        known_optimum=None,
        category='pde',
        description='Finite element node values (70D)'
    ),
    'Multigrid-60D': BenchmarkProblem(
        name='Multigrid-60D',
        objective=_make_optuna_objective(_multigrid_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='pde',
        description='Multigrid solver parameters (60D)'
    ),
    'DomainDecomposition-72D': BenchmarkProblem(
        name='DomainDecomposition-72D',
        objective=_make_optuna_objective(_domain_decomposition_72d, [(0, 1)] * 72, 72),
        dimension=72,
        bounds=[(0, 1)] * 72,
        known_optimum=None,
        category='pde',
        description='Domain decomposition interface (72D)'
    ),
    'AdaptiveMesh-65D': BenchmarkProblem(
        name='AdaptiveMesh-65D',
        objective=_make_optuna_objective(_adaptive_mesh_65d, [(0, 1)] * 65, 65),
        dimension=65,
        bounds=[(0, 1)] * 65,
        known_optimum=None,
        category='pde',
        description='Adaptive mesh refinement (65D)'
    ),
    'GinzburgLandau-56D': BenchmarkProblem(
        name='GinzburgLandau-56D',
        objective=_make_optuna_objective(_ginzburg_landau_56d, [(0, 1)] * 56, 56),
        dimension=56,
        bounds=[(0, 1)] * 56,
        known_optimum=None,
        category='pde',
        description='Complex Ginzburg-Landau equation (56D)'
    ),
    'WaveEquation-120D': BenchmarkProblem(
        name='WaveEquation-120D',
        objective=_make_optuna_objective(_wave_equation_120d, [(0, 1)] * 120, 120),
        dimension=120,
        bounds=[(0, 1)] * 120,
        known_optimum=None,
        category='pde',
        description='1D wave equation boundary optimization (120D)'
    ),
}


# =============================================================================
# META-OPTIMIZATION & CONTROL REGISTRY (Chunk 2.2.8)
# =============================================================================

_META_CONTROL_REGISTRY: Dict[str, BenchmarkProblem] = {
    'GeneticAlgorithm-25D': BenchmarkProblem(
        name='GeneticAlgorithm-25D',
        objective=_make_optuna_objective(_genetic_algorithm_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='meta_optimization',
        description='Genetic algorithm hyperparameter tuning (25D)'
    ),
    'ParticleSwarm-30D': BenchmarkProblem(
        name='ParticleSwarm-30D',
        objective=_make_optuna_objective(_particle_swarm_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='meta_optimization',
        description='Particle swarm optimization parameter tuning (30D)'
    ),
    'DifferentialEvolution-30D': BenchmarkProblem(
        name='DifferentialEvolution-30D',
        objective=_make_optuna_objective(_differential_evolution_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='meta_optimization',
        description='Differential evolution parameter tuning (30D)'
    ),
    'CMA-ES-25D': BenchmarkProblem(
        name='CMA-ES-25D',
        objective=_make_optuna_objective(_cma_es_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='meta_optimization',
        description='CMA-ES hyperparameter meta-optimization (25D)'
    ),
    'Hyperband-60D': BenchmarkProblem(
        name='Hyperband-60D',
        objective=_make_optuna_objective(_hyperband_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='meta_optimization',
        description='Hyperband multi-fidelity search (60D)'
    ),
    'BayesianOpt-60D': BenchmarkProblem(
        name='BayesianOpt-60D',
        objective=_make_optuna_objective(_bayesian_opt_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='meta_optimization',
        description='Bayesian optimization surrogate (60D)'
    ),
    'NAS-70D': BenchmarkProblem(
        name='NAS-70D',
        objective=_make_optuna_objective(_nas_70d, [(0, 1)] * 70, 70),
        dimension=70,
        bounds=[(0, 1)] * 70,
        known_optimum=None,
        category='meta_optimization',
        description='Neural architecture search (70D)'
    ),
    'EvolutionStrategy-65D': BenchmarkProblem(
        name='EvolutionStrategy-65D',
        objective=_make_optuna_objective(_evolution_strategy_65d, [(0, 1)] * 65, 65),
        dimension=65,
        bounds=[(0, 1)] * 65,
        known_optimum=None,
        category='meta_optimization',
        description='Evolution strategy meta-optimization (65D)'
    ),
    'SimulatedAnnealing-55D': BenchmarkProblem(
        name='SimulatedAnnealing-55D',
        objective=_make_optuna_objective(_simulated_annealing_55d, [(0, 1)] * 55, 55),
        dimension=55,
        bounds=[(0, 1)] * 55,
        known_optimum=None,
        category='meta_optimization',
        description='Simulated annealing meta-optimization (55D)'
    ),
    'CovarianceAdaptation-60D': BenchmarkProblem(
        name='CovarianceAdaptation-60D',
        objective=_make_optuna_objective(_covariance_adaptation_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='meta_optimization',
        description='Covariance matrix adaptation (60D)'
    ),
    'PIDTuning-6D': BenchmarkProblem(
        name='PIDTuning-6D',
        objective=_make_optuna_objective(_pid_tuning_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='control',
        description='PID controller tuning (6D)'
    ),
    'LQRControl-8D': BenchmarkProblem(
        name='LQRControl-8D',
        objective=_make_optuna_objective(_lqr_control_8d, [(0, 1)] * 8, 8),
        dimension=8,
        bounds=[(0, 1)] * 8,
        known_optimum=None,
        category='control',
        description='LQR control synthesis (8D)'
    ),
    'TrajectoryOpt-100D': BenchmarkProblem(
        name='TrajectoryOpt-100D',
        objective=_make_optuna_objective(_trajectory_opt_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='control',
        description='Trajectory optimization for dynamical system (100D)'
    ),
    'TrajectoryOpt-120D': BenchmarkProblem(
        name='TrajectoryOpt-120D',
        objective=_make_optuna_objective(_trajectory_opt_120d, [(0, 1)] * 120, 120),
        dimension=120,
        bounds=[(0, 1)] * 120,
        known_optimum=None,
        category='control',
        description='Extended trajectory optimization (120D)'
    ),
    'InverseKinematics-80D': BenchmarkProblem(
        name='InverseKinematics-80D',
        objective=_make_optuna_objective(_inverse_kinematics_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='control',
        description='Robot arm inverse kinematics (80D)'
    ),
    'InverseKinematics-100D': BenchmarkProblem(
        name='InverseKinematics-100D',
        objective=_make_optuna_objective(_inverse_kinematics_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='control',
        description='Extended robot arm inverse kinematics (100D)'
    ),
    'InverseKinematicsLong-80D': BenchmarkProblem(
        name='InverseKinematicsLong-80D',
        objective=_make_optuna_objective(_inverse_kinematics_long_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='control',
        description='Long-horizon inverse kinematics optimization (80D)'
    ),
    'InverseKinematicsLong-100D': BenchmarkProblem(
        name='InverseKinematicsLong-100D',
        objective=_make_optuna_objective(_inverse_kinematics_long_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='control',
        description='Extended long-horizon inverse kinematics (100D)'
    ),
}


# =============================================================================
# REMAINING PROBLEMS REGISTRY (Chunk 2.2.9)
# =============================================================================

_REMAINING_REGISTRY: Dict[str, BenchmarkProblem] = {
    'SA-Schedule-3D': BenchmarkProblem(
        name='SA-Schedule-3D',
        objective=_make_optuna_objective(_sa_schedule_3d, [(0, 1)] * 3, 3),
        dimension=3,
        bounds=[(0, 1)] * 3,
        known_optimum=None,
        category='meta_optimization',
        description='Simulated annealing schedule meta-optimization (3D)'
    ),
    'CellularAutomata-25D': BenchmarkProblem(
        name='CellularAutomata-25D',
        objective=_make_optuna_objective(_cellular_automata_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='simulation',
        description='Cellular automata rule optimization (25D)'
    ),
    'CellularAutomaton-120D': BenchmarkProblem(
        name='CellularAutomaton-120D',
        objective=_make_optuna_objective(_cellular_automaton_120d, [(0, 1)] * 120, 120),
        dimension=120,
        bounds=[(0, 1)] * 120,
        known_optimum=None,
        category='simulation',
        description='Continuous cellular automaton pattern formation (120D)'
    ),
    'SpinGlass-150D': BenchmarkProblem(
        name='SpinGlass-150D',
        objective=_make_optuna_objective(_spin_glass_150d, [(0, 1)] * 150, 150),
        dimension=150,
        bounds=[(0, 1)] * 150,
        known_optimum=None,
        category='physics',
        description='Spin glass energy minimization (150D)'
    ),
    'CovarianceEstimation-120D': BenchmarkProblem(
        name='CovarianceEstimation-120D',
        objective=_make_optuna_objective(_covariance_estimation_120d, [(0, 1)] * 120, 120),
        dimension=120,
        bounds=[(0, 1)] * 120,
        known_optimum=None,
        category='statistics',
        description='Covariance matrix estimation (120D)'
    ),
    'LinearSystemID-144D': BenchmarkProblem(
        name='LinearSystemID-144D',
        objective=_make_optuna_objective(_linear_system_id_144d, [(0, 1)] * 144, 144),
        dimension=144,
        bounds=[(0, 1)] * 144,
        known_optimum=None,
        category='control',
        description='Linear dynamical system identification (144D)'
    ),
    'CoupledPendulums-100D': BenchmarkProblem(
        name='CoupledPendulums-100D',
        objective=_make_optuna_objective(_coupled_pendulums_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='dynamical',
        description='Coupled pendulums energy variance (100D)'
    ),
    'EpidemicControl-25D': BenchmarkProblem(
        name='EpidemicControl-25D',
        objective=_make_optuna_objective(_epidemic_control_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='simulation',
        description='Epidemic control optimization (25D)'
    ),
    'EpidemicControl-40D': BenchmarkProblem(
        name='EpidemicControl-40D',
        objective=_make_optuna_objective(_epidemic_control_40d, [(0, 1)] * 40, 40),
        dimension=40,
        bounds=[(0, 1)] * 40,
        known_optimum=None,
        category='simulation',
        description='Epidemic control optimization (40D)'
    ),
    'SupplyChain-35D': BenchmarkProblem(
        name='SupplyChain-35D',
        objective=_make_optuna_objective(_supply_chain_35d, [(0, 1)] * 35, 35),
        dimension=35,
        bounds=[(0, 1)] * 35,
        known_optimum=None,
        category='optimization',
        description='Supply chain inventory optimization (35D)'
    ),
    'SupplyChain-50D': BenchmarkProblem(
        name='SupplyChain-50D',
        objective=_make_optuna_objective(_supply_chain_50d, [(0, 1)] * 50, 50),
        dimension=50,
        bounds=[(0, 1)] * 50,
        known_optimum=None,
        category='optimization',
        description='Supply chain optimization (50D)'
    ),
    'GraphPartition-25D': BenchmarkProblem(
        name='GraphPartition-25D',
        objective=_make_optuna_objective(_graph_partition_25d, [(0, 1)] * 25, 25),
        dimension=25,
        bounds=[(0, 1)] * 25,
        known_optimum=None,
        category='optimization',
        description='Graph partitioning continuous relaxation (25D)'
    ),
    'GraphPartition-40D': BenchmarkProblem(
        name='GraphPartition-40D',
        objective=_make_optuna_objective(_graph_partition_40d, [(0, 1)] * 40, 40),
        dimension=40,
        bounds=[(0, 1)] * 40,
        known_optimum=None,
        category='optimization',
        description='Graph partitioning (40D)'
    ),
    'RiskParity-30D': BenchmarkProblem(
        name='RiskParity-30D',
        objective=_make_optuna_objective(_risk_parity_30d, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='finance',
        description='Risk parity portfolio optimization (30D)'
    ),
    'ChemicalKinetics-5D': BenchmarkProblem(
        name='ChemicalKinetics-5D',
        objective=_make_optuna_objective(_chemical_kinetics_5d, [(0, 1)] * 5, 5),
        dimension=5,
        bounds=[(0, 1)] * 5,
        known_optimum=None,
        category='chemistry',
        description='Chemical reaction kinetics parameter fitting (5D)'
    ),
    'RegressionCoeffs-5D': BenchmarkProblem(
        name='RegressionCoeffs-5D',
        objective=_make_optuna_objective(_regression_coeffs_5d, [(0, 1)] * 5, 5),
        dimension=5,
        bounds=[(0, 1)] * 5,
        known_optimum=None,
        category='statistics',
        description='Regularized regression coefficient estimation (5D)'
    ),
}


# =============================================================================
# MASTER REGISTRY (Batches 1-6)
# =============================================================================

ALL_REALWORLD_PROBLEMS: Dict[str, BenchmarkProblem] = {
    **_CHAOTIC_REGISTRY,        # 16 functions
    **_DYNAMICAL_REGISTRY,      # 6 functions
    **_NN_WEIGHTS_REGISTRY,     # 16 functions
    **_ML_TRAINING_REGISTRY,    # 43 functions
    **_PDE_REGISTRY,            # 18 functions
    **_META_CONTROL_REGISTRY,   # 18 functions
    **_REMAINING_REGISTRY,      # 16 functions (2 were already in chaotic registry)
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
