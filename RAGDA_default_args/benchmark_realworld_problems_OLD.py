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
# MASTER REGISTRY (Currently: Batch 1 only)
# =============================================================================

ALL_REALWORLD_PROBLEMS: Dict[str, BenchmarkProblem] = {
    **_CHAOTIC_REGISTRY,  # 16 functions
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
