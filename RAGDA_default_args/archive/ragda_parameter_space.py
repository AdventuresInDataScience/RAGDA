"""
RAGDA Parameter Space Definition for Meta-Optimization

This module defines the complete, correct parameter space for RAGDA optimization,
with proper bounds, types, defaults, and constraint handling.

Based on exhaustive audit of ragda/optimizer.py and ragda/highdim.py.

Parameters are split into:
1. __init__ parameters (optimizer-level)
2. optimize() parameters (run-level)

Constraints are documented and enforced via penalty functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Callable
from enum import Enum
import numpy as np


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

@dataclass
class ParameterDef:
    """Definition of a single RAGDA parameter."""
    name: str
    param_type: str  # 'int', 'float', 'bool', 'categorical'
    default: Any
    low: Optional[float] = None  # For numeric types
    high: Optional[float] = None  # For numeric types
    log_scale: bool = False  # Use log-scale for optimization
    choices: Optional[List[Any]] = None  # For categorical
    description: str = ""
    location: str = "optimize"  # 'init' or 'optimize'
    
    # Constraint info
    constraint_notes: str = ""
    depends_on: Optional[List[str]] = None  # Parameters this depends on


# =============================================================================
# COMPLETE RAGDA PARAMETER SPACE
# Audited from ragda/optimizer.py
# =============================================================================

RAGDA_PARAMETERS: Dict[str, ParameterDef] = {
    # =========================================================================
    # __init__ PARAMETERS (optimizer-level)
    # =========================================================================
    
    "n_workers": ParameterDef(
        name="n_workers",
        param_type="int",
        default=None,  # Defaults to cpu_count() // 2
        low=1,
        high=16,  # Practical upper limit; more workers have diminishing returns
        description="Number of parallel workers for exploration",
        location="init",
        constraint_notes="Must be >= 1. Higher values increase exploration but also overhead."
    ),
    
    # Note: highdim_threshold, variance_threshold, reduction_method are init params
    # but they control automatic high-dim detection. For meta-optimization of defaults,
    # we focus on the optimize() parameters since high-dim path is automatic.
    
    # =========================================================================
    # optimize() PARAMETERS - Core
    # =========================================================================
    
    "n_trials": ParameterDef(
        name="n_trials",
        param_type="int",
        default=500,
        low=50,
        high=2000,
        description="Number of iterations per worker (budget parameter)",
        location="optimize",
        constraint_notes="This is the optimization budget. Higher = more thorough but slower."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Population & Sampling (Lambda/Sigma schedules)
    # =========================================================================
    
    "lambda_start": ParameterDef(
        name="lambda_start",
        param_type="int",
        default=50,
        low=10,
        high=200,
        description="Initial batch size (samples per iteration)",
        location="optimize",
        constraint_notes="Higher values = more samples but slower. Must be >= lambda_end."
    ),
    
    "lambda_end": ParameterDef(
        name="lambda_end",
        param_type="int",
        default=10,
        low=5,
        high=100,
        description="Final batch size at end of optimization",
        location="optimize",
        constraint_notes="Must be <= lambda_start. Lower values for fine-tuning."
    ),
    
    "lambda_decay_rate": ParameterDef(
        name="lambda_decay_rate",
        param_type="float",
        default=5.0,
        low=0.5,
        high=20.0,
        description="Exponential decay rate for batch size schedule",
        location="optimize",
        constraint_notes="Higher = faster decay from lambda_start to lambda_end."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Sample Space (Sigma)
    # =========================================================================
    
    "sigma_init": ParameterDef(
        name="sigma_init",
        param_type="float",
        default=0.3,
        low=0.05,
        high=0.8,  # Must be <= 1.0 per validation
        description="Initial sampling radius in [0,1] unit space",
        location="optimize",
        constraint_notes="Must be in (0, 1]. Larger = more exploration. 0.3 is conservative."
    ),
    
    "sigma_final_fraction": ParameterDef(
        name="sigma_final_fraction",
        param_type="float",
        default=0.2,
        low=0.05,
        high=0.5,  # Must be <= 1.0 per validation
        description="Final sigma as fraction of sigma_init",
        location="optimize",
        constraint_notes="Must be in (0, 1]. Actual final sigma = sigma_init * sigma_final_fraction."
    ),
    
    "sigma_decay_schedule": ParameterDef(
        name="sigma_decay_schedule",
        param_type="categorical",
        default="exponential",
        choices=["exponential", "linear", "cosine"],
        description="How sigma decays over iterations",
        location="optimize",
        constraint_notes="exponential is aggressive, cosine is smooth, linear is predictable."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Adaptive Shrinking
    # =========================================================================
    
    "shrink_factor": ParameterDef(
        name="shrink_factor",
        param_type="float",
        default=0.9,
        low=0.5,
        high=0.99,  # Must be < 1.0 per validation
        description="Multiply sigma by this on stagnation",
        location="optimize",
        constraint_notes="Must be in (0, 1). Lower = more aggressive shrinking."
    ),
    
    "shrink_patience": ParameterDef(
        name="shrink_patience",
        param_type="int",
        default=10,
        low=3,
        high=50,
        description="Iterations without improvement before shrinking",
        location="optimize",
        constraint_notes="Must be > 0. Lower = shrink faster on plateaus."
    ),
    
    "shrink_threshold": ParameterDef(
        name="shrink_threshold",
        param_type="float",
        default=1e-6,
        low=1e-10,
        high=1e-3,
        log_scale=True,
        description="Minimum relative improvement to count as progress",
        location="optimize",
        constraint_notes="Tiny improvements below this don't reset patience counter."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Top-N Weighting (Worker Diversity)
    # =========================================================================
    
    "use_improvement_weights": ParameterDef(
        name="use_improvement_weights",
        param_type="bool",
        default=True,
        description="Use only improving samples for gradient estimation",
        location="optimize",
        constraint_notes="True is recommended. False uses all samples (less focused)."
    ),
    
    "top_n_min": ParameterDef(
        name="top_n_min",
        param_type="float",
        default=0.2,
        low=0.05,
        high=0.5,  # Must be <= 1.0 and <= top_n_max
        description="Minimum top_n fraction across workers (most selective)",
        location="optimize",
        constraint_notes="Must be in (0, 1] and <= top_n_max. Lower = more selective workers.",
        depends_on=["top_n_max"]
    ),
    
    "top_n_max": ParameterDef(
        name="top_n_max",
        param_type="float",
        default=1.0,
        low=0.3,
        high=1.0,  # Must be <= 1.0
        description="Maximum top_n fraction across workers (least selective)",
        location="optimize",
        constraint_notes="Must be in (0, 1] and >= top_n_min. Higher = more diverse workers.",
        depends_on=["top_n_min"]
    ),
    
    "weight_decay": ParameterDef(
        name="weight_decay",
        param_type="float",
        default=0.95,
        low=0.5,
        high=0.999,
        description="Exponential decay for rank-based weights",
        location="optimize",
        constraint_notes="Higher = more uniform weights. Lower = emphasize top samples."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Worker Synchronization
    # =========================================================================
    
    "sync_frequency": ParameterDef(
        name="sync_frequency",
        param_type="int",
        default=100,
        low=0,  # 0 = never sync
        high=500,
        description="How often workers synchronize (0 = never)",
        location="optimize",
        constraint_notes="0 disables sync. Lower values = more frequent knowledge sharing."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Worker Strategy
    # =========================================================================
    
    "worker_strategy": ParameterDef(
        name="worker_strategy",
        param_type="categorical",
        default="greedy",
        choices=["greedy", "dynamic"],
        description="How workers share and restart",
        location="optimize",
        constraint_notes="greedy: all reset to best. dynamic: elite survive, others restart."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Dynamic Worker Strategy (only when strategy='dynamic')
    # =========================================================================
    
    "elite_fraction": ParameterDef(
        name="elite_fraction",
        param_type="float",
        default=0.5,
        low=0.1,
        high=1.0,  # Must be in (0, 1]
        description="Fraction of top workers to keep at sync (dynamic strategy)",
        location="optimize",
        constraint_notes="Must be in (0, 1]. Only used when worker_strategy='dynamic'.",
        depends_on=["worker_strategy"]
    ),
    
    "restart_mode": ParameterDef(
        name="restart_mode",
        param_type="categorical",
        default="adaptive",
        choices=["elite", "random", "adaptive"],
        description="How non-elite workers restart (dynamic strategy)",
        location="optimize",
        constraint_notes="elite: from elite positions. random: fresh start. adaptive: mix.",
        depends_on=["worker_strategy"]
    ),
    
    "restart_elite_prob_start": ParameterDef(
        name="restart_elite_prob_start",
        param_type="float",
        default=0.3,
        low=0.0,
        high=1.0,
        description="Initial probability of restarting from elite (adaptive mode)",
        location="optimize",
        constraint_notes="Only used when restart_mode='adaptive'. Start with more random.",
        depends_on=["worker_strategy", "restart_mode"]
    ),
    
    "restart_elite_prob_end": ParameterDef(
        name="restart_elite_prob_end",
        param_type="float",
        default=0.8,
        low=0.0,
        high=1.0,
        description="Final probability of restarting from elite (adaptive mode)",
        location="optimize",
        constraint_notes="Only used when restart_mode='adaptive'. End with more elite restarts.",
        depends_on=["worker_strategy", "restart_mode"]
    ),
    
    "enable_worker_decay": ParameterDef(
        name="enable_worker_decay",
        param_type="bool",
        default=False,
        description="Reduce number of active workers over time",
        location="optimize",
        constraint_notes="Only used when worker_strategy='dynamic'.",
        depends_on=["worker_strategy"]
    ),
    
    "min_workers": ParameterDef(
        name="min_workers",
        param_type="int",
        default=2,
        low=1,
        high=8,
        description="Minimum workers to keep when using worker decay",
        location="optimize",
        constraint_notes="Must be >= 1 and <= n_workers. Only used with enable_worker_decay=True.",
        depends_on=["worker_strategy", "enable_worker_decay", "n_workers"]
    ),
    
    "worker_decay_rate": ParameterDef(
        name="worker_decay_rate",
        param_type="float",
        default=0.5,
        low=0.1,
        high=0.9,
        description="How aggressively to decay workers (0-1)",
        location="optimize",
        constraint_notes="Must be in [0, 1]. 0.5 = reduce to ~50% by end.",
        depends_on=["worker_strategy", "enable_worker_decay"]
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Mini-batch (for data-driven objectives)
    # =========================================================================
    
    "use_minibatch": ParameterDef(
        name="use_minibatch",
        param_type="bool",
        default=False,
        description="Enable mini-batch evaluation for data-driven objectives",
        location="optimize",
        constraint_notes="Only useful for objectives that accept batch_size parameter."
    ),
    
    "minibatch_start": ParameterDef(
        name="minibatch_start",
        param_type="int",
        default=32,
        low=16,
        high=256,
        description="Starting batch size for mini-batch mode",
        location="optimize",
        constraint_notes="Only used when use_minibatch=True. Smaller = faster, noisier.",
        depends_on=["use_minibatch"]
    ),
    
    "minibatch_end": ParameterDef(
        name="minibatch_end",
        param_type="int",
        default=1000,
        low=100,
        high=10000,
        description="Final batch size for mini-batch mode",
        location="optimize",
        constraint_notes="Only used when use_minibatch=True. Must be >= minibatch_start.",
        depends_on=["use_minibatch", "minibatch_start"]
    ),
    
    "minibatch_schedule": ParameterDef(
        name="minibatch_schedule",
        param_type="categorical",
        default="inverse_decay",
        choices=["constant", "linear", "exponential", "inverse_decay", "step"],
        description="How batch size grows over iterations",
        location="optimize",
        constraint_notes="Only used when use_minibatch=True.",
        depends_on=["use_minibatch"]
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - ADAM Optimizer
    # =========================================================================
    
    "adam_learning_rate": ParameterDef(
        name="adam_learning_rate",
        param_type="float",
        default=0.001,
        low=1e-5,
        high=0.1,
        log_scale=True,
        description="ADAM learning rate for pseudo-gradient updates",
        location="optimize",
        constraint_notes="Must be > 0. Lower = more stable, slower. Higher = faster, may overshoot."
    ),
    
    "adam_beta1": ParameterDef(
        name="adam_beta1",
        param_type="float",
        default=0.9,
        low=0.5,
        high=0.99,  # Must be < 1.0
        description="ADAM exponential decay rate for first moment",
        location="optimize",
        constraint_notes="Must be in [0, 1). Higher = more momentum."
    ),
    
    "adam_beta2": ParameterDef(
        name="adam_beta2",
        param_type="float",
        default=0.999,
        low=0.9,
        high=0.9999,  # Must be < 1.0
        description="ADAM exponential decay rate for second moment",
        location="optimize",
        constraint_notes="Must be in [0, 1). Higher = more stable learning rate scaling."
    ),
    
    "adam_epsilon": ParameterDef(
        name="adam_epsilon",
        param_type="float",
        default=1e-8,
        low=1e-12,
        high=1e-4,
        log_scale=True,
        description="ADAM numerical stability constant",
        location="optimize",
        constraint_notes="Must be > 0. Prevents division by zero."
    ),
    
    # =========================================================================
    # optimize() PARAMETERS - Early Stopping
    # =========================================================================
    
    "early_stop_threshold": ParameterDef(
        name="early_stop_threshold",
        param_type="float",
        default=1e-12,
        low=1e-15,
        high=1e-6,
        log_scale=True,
        description="Stop if best value below this threshold",
        location="optimize",
        constraint_notes="For minimization. Set very low to effectively disable."
    ),
    
    "early_stop_patience": ParameterDef(
        name="early_stop_patience",
        param_type="int",
        default=50,
        low=10,
        high=200,
        description="Stop if no improvement for this many iterations",
        location="optimize",
        constraint_notes="Must be > 0. Lower = stop earlier on plateaus."
    ),
}


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================

@dataclass
class Constraint:
    """Definition of a parameter constraint."""
    name: str
    description: str
    check_fn: Callable[[Dict[str, Any]], bool]  # Returns True if VALID
    penalty: float = 1000.0  # Penalty if violated


def check_lambda_order(params: Dict[str, Any]) -> bool:
    """lambda_end must be <= lambda_start"""
    return params.get("lambda_end", 10) <= params.get("lambda_start", 50)


def check_top_n_order(params: Dict[str, Any]) -> bool:
    """top_n_min must be <= top_n_max"""
    return params.get("top_n_min", 0.2) <= params.get("top_n_max", 1.0)


def check_minibatch_order(params: Dict[str, Any]) -> bool:
    """minibatch_end must be >= minibatch_start when use_minibatch=True"""
    if not params.get("use_minibatch", False):
        return True  # Constraint doesn't apply
    return params.get("minibatch_end", 1000) >= params.get("minibatch_start", 32)


def check_min_workers_bound(params: Dict[str, Any]) -> bool:
    """min_workers must be <= n_workers when enable_worker_decay=True"""
    if not params.get("enable_worker_decay", False):
        return True  # Constraint doesn't apply
    return params.get("min_workers", 2) <= params.get("n_workers", 4)


def check_sigma_init_valid(params: Dict[str, Any]) -> bool:
    """sigma_init must be in (0, 1]"""
    sigma = params.get("sigma_init", 0.3)
    return 0 < sigma <= 1.0


def check_sigma_final_fraction_valid(params: Dict[str, Any]) -> bool:
    """sigma_final_fraction must be in (0, 1]"""
    frac = params.get("sigma_final_fraction", 0.2)
    return 0 < frac <= 1.0


def check_shrink_factor_valid(params: Dict[str, Any]) -> bool:
    """shrink_factor must be in (0, 1)"""
    sf = params.get("shrink_factor", 0.9)
    return 0 < sf < 1.0


def check_adam_beta1_valid(params: Dict[str, Any]) -> bool:
    """adam_beta1 must be in [0, 1)"""
    b1 = params.get("adam_beta1", 0.9)
    return 0 <= b1 < 1.0


def check_adam_beta2_valid(params: Dict[str, Any]) -> bool:
    """adam_beta2 must be in [0, 1)"""
    b2 = params.get("adam_beta2", 0.999)
    return 0 <= b2 < 1.0


def check_top_n_min_valid(params: Dict[str, Any]) -> bool:
    """top_n_min must be in (0, 1]"""
    tn = params.get("top_n_min", 0.2)
    return 0 < tn <= 1.0


def check_top_n_max_valid(params: Dict[str, Any]) -> bool:
    """top_n_max must be in (0, 1]"""
    tn = params.get("top_n_max", 1.0)
    return 0 < tn <= 1.0


def check_elite_fraction_valid(params: Dict[str, Any]) -> bool:
    """elite_fraction must be in (0, 1]"""
    ef = params.get("elite_fraction", 0.5)
    return 0 < ef <= 1.0


CONSTRAINTS: List[Constraint] = [
    Constraint(
        name="lambda_order",
        description="lambda_end <= lambda_start",
        check_fn=check_lambda_order,
        penalty=1000.0
    ),
    Constraint(
        name="top_n_order",
        description="top_n_min <= top_n_max",
        check_fn=check_top_n_order,
        penalty=1000.0
    ),
    Constraint(
        name="minibatch_order",
        description="minibatch_end >= minibatch_start (when use_minibatch=True)",
        check_fn=check_minibatch_order,
        penalty=1000.0
    ),
    Constraint(
        name="min_workers_bound",
        description="min_workers <= n_workers (when enable_worker_decay=True)",
        check_fn=check_min_workers_bound,
        penalty=1000.0
    ),
    Constraint(
        name="sigma_init_valid",
        description="sigma_init in (0, 1]",
        check_fn=check_sigma_init_valid,
        penalty=1000.0
    ),
    Constraint(
        name="sigma_final_fraction_valid",
        description="sigma_final_fraction in (0, 1]",
        check_fn=check_sigma_final_fraction_valid,
        penalty=1000.0
    ),
    Constraint(
        name="shrink_factor_valid",
        description="shrink_factor in (0, 1)",
        check_fn=check_shrink_factor_valid,
        penalty=1000.0
    ),
    Constraint(
        name="adam_beta1_valid",
        description="adam_beta1 in [0, 1)",
        check_fn=check_adam_beta1_valid,
        penalty=1000.0
    ),
    Constraint(
        name="adam_beta2_valid",
        description="adam_beta2 in [0, 1)",
        check_fn=check_adam_beta2_valid,
        penalty=1000.0
    ),
    Constraint(
        name="top_n_min_valid",
        description="top_n_min in (0, 1]",
        check_fn=check_top_n_min_valid,
        penalty=1000.0
    ),
    Constraint(
        name="top_n_max_valid",
        description="top_n_max in (0, 1]",
        check_fn=check_top_n_max_valid,
        penalty=1000.0
    ),
    Constraint(
        name="elite_fraction_valid",
        description="elite_fraction in (0, 1]",
        check_fn=check_elite_fraction_valid,
        penalty=1000.0
    ),
]


def compute_constraint_penalty(params: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Compute total penalty for constraint violations.
    
    Returns:
        (total_penalty, list_of_violations)
    """
    total_penalty = 0.0
    violations = []
    
    for constraint in CONSTRAINTS:
        if not constraint.check_fn(params):
            total_penalty += constraint.penalty
            violations.append(constraint.name)
    
    return total_penalty, violations


def is_valid_config(params: Dict[str, Any]) -> bool:
    """Check if a parameter configuration is valid (no constraint violations)."""
    penalty, _ = compute_constraint_penalty(params)
    return penalty == 0.0


# =============================================================================
# PARAMETER SUBSETS FOR DIFFERENT OPTIMIZATION SCENARIOS
# =============================================================================

# Core parameters that should ALWAYS be tuned
CORE_PARAMS = [
    "n_workers",
    "n_trials",
    "lambda_start",
    "lambda_end",
    "lambda_decay_rate",
    "sigma_init",
    "sigma_final_fraction",
    "sigma_decay_schedule",
    "shrink_factor",
    "shrink_patience",
    "shrink_threshold",
    "adam_learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
]

# Worker strategy parameters (tune when exploring strategies)
WORKER_STRATEGY_PARAMS = [
    "worker_strategy",
    "sync_frequency",
    "elite_fraction",
    "restart_mode",
    "restart_elite_prob_start",
    "restart_elite_prob_end",
    "enable_worker_decay",
    "min_workers",
    "worker_decay_rate",
]

# Weighting parameters
WEIGHTING_PARAMS = [
    "use_improvement_weights",
    "top_n_min",
    "top_n_max",
    "weight_decay",
]

# Early stopping parameters
EARLY_STOP_PARAMS = [
    "early_stop_threshold",
    "early_stop_patience",
]

# Mini-batch parameters (only for data-driven objectives)
MINIBATCH_PARAMS = [
    "use_minibatch",
    "minibatch_start",
    "minibatch_end",
    "minibatch_schedule",
]

# All tunable parameters (ALL of them as requested)
ALL_TUNABLE_PARAMS = CORE_PARAMS + WORKER_STRATEGY_PARAMS + WEIGHTING_PARAMS + EARLY_STOP_PARAMS + MINIBATCH_PARAMS

# Parameters for "simple" mode (fewer params, faster meta-optimization)
SIMPLE_PARAMS = [
    "n_workers",
    "n_trials",
    "lambda_start",
    "lambda_end",
    "sigma_init",
    "sigma_final_fraction",
    "shrink_factor",
    "adam_learning_rate",
    "worker_strategy",
]


# =============================================================================
# MARSOPT SPACE GENERATOR
# =============================================================================

def generate_marsopt_space(
    param_names: Optional[List[str]] = None,
    include_minibatch: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a MARSOpt-compatible search space.
    
    Args:
        param_names: List of parameter names to include. 
                     If None, uses ALL_TUNABLE_PARAMS.
        include_minibatch: Whether to include minibatch parameters.
    
    Returns:
        List of dicts in MARSOpt format.
    """
    if param_names is None:
        param_names = ALL_TUNABLE_PARAMS.copy()
        if include_minibatch:
            param_names.extend(MINIBATCH_PARAMS)
    
    space = []
    
    for name in param_names:
        if name not in RAGDA_PARAMETERS:
            raise ValueError(f"Unknown parameter: {name}")
        
        param = RAGDA_PARAMETERS[name]
        
        if param.param_type == "int":
            space.append({
                "name": name,
                "type": "int",
                "low": int(param.low),
                "high": int(param.high),
            })
        
        elif param.param_type == "float":
            param_def = {
                "name": name,
                "type": "float",
                "low": param.low,
                "high": param.high,
            }
            if param.log_scale:
                param_def["log"] = True
            space.append(param_def)
        
        elif param.param_type == "bool":
            space.append({
                "name": name,
                "type": "categorical",
                "choices": [True, False],
            })
        
        elif param.param_type == "categorical":
            space.append({
                "name": name,
                "type": "categorical",
                "choices": param.choices,
            })
    
    return space


def get_default_params(param_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get default values for specified parameters.
    
    Args:
        param_names: List of parameter names. If None, returns all defaults.
    
    Returns:
        Dict of parameter defaults.
    """
    if param_names is None:
        param_names = list(RAGDA_PARAMETERS.keys())
    
    return {name: RAGDA_PARAMETERS[name].default for name in param_names 
            if name in RAGDA_PARAMETERS}


def split_params_by_location(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split parameters into __init__ and optimize() groups.
    
    Args:
        params: Dict of all parameters.
    
    Returns:
        (init_params, optimize_params)
    """
    init_params = {}
    optimize_params = {}
    
    for name, value in params.items():
        if name not in RAGDA_PARAMETERS:
            continue
        
        if RAGDA_PARAMETERS[name].location == "init":
            init_params[name] = value
        else:
            optimize_params[name] = value
    
    return init_params, optimize_params


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_parameter_summary():
    """Print a summary of all parameters for documentation."""
    print("=" * 80)
    print("RAGDA PARAMETER SUMMARY")
    print("=" * 80)
    
    # Group by location
    init_params = [p for p in RAGDA_PARAMETERS.values() if p.location == "init"]
    opt_params = [p for p in RAGDA_PARAMETERS.values() if p.location == "optimize"]
    
    print("\n__init__ PARAMETERS:")
    print("-" * 40)
    for p in init_params:
        bounds = f"[{p.low}, {p.high}]" if p.low is not None else str(p.choices)
        print(f"  {p.name}: {p.param_type} = {p.default} {bounds}")
    
    print("\noptimize() PARAMETERS:")
    print("-" * 40)
    for p in opt_params:
        bounds = f"[{p.low}, {p.high}]" if p.low is not None else str(p.choices)
        log_str = " (log)" if p.log_scale else ""
        print(f"  {p.name}: {p.param_type} = {p.default} {bounds}{log_str}")
    
    print("\nCONSTRAINTS:")
    print("-" * 40)
    for c in CONSTRAINTS:
        print(f"  {c.name}: {c.description}")


if __name__ == "__main__":
    print_parameter_summary()
    
    print("\n\nMARSOpt Space (ALL_TUNABLE_PARAMS):")
    print("-" * 40)
    space = generate_marsopt_space()
    for p in space:
        print(f"  {p}")
