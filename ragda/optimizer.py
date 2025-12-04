"""
RAGDA Optimizer - High-Performance Derivative-Free Optimization

Pure Cython/C implementation with:
- Parallel multi-worker exploration
- ADAM-based pseudo-gradient descent
- Mixed variable types (continuous, ordinal, categorical)
- Mini-batch evaluation for data-driven objectives
- Early stopping and convergence detection
- Automatic high-dimensional optimization via dimensionality reduction
- Dynamic worker management with elite selection and adaptive restarts
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, List, Union, Tuple, Literal
from multiprocessing import cpu_count
import warnings
import inspect
import math

# Use loky for robust Windows multiprocessing (handles cloudpickle)
from loky import get_reusable_executor

from .space import SearchSpace
from .result import OptimizationResult, Trial
from . import core  # Pure Cython - no fallback

# Try to import high-dim core (may not be built yet)
try:
    from . import highdim_core
    HIGHDIM_AVAILABLE = True
except ImportError:
    HIGHDIM_AVAILABLE = False


# =============================================================================
# Worker Task (for loky/cloudpickle)
# =============================================================================

class _WorkerTask:
    """Encapsulates all data needed for a worker task.
    
    Using a class instead of tuple allows cloudpickle to serialize
    the objective function along with the task data.
    """
    __slots__ = (
        'x0_cont', 'x0_cat', 'cat_n_values', 'bounds', 'max_iter',
        'lambda_schedule', 'mu_schedule', 'sigma_schedule', 'minibatch_schedule',
        'use_minibatch', 'top_n_fraction', 'adam_lr', 'adam_beta1', 'adam_beta2',
        'adam_epsilon', 'shrink_factor', 'shrink_patience', 'shrink_threshold',
        'use_improvement_weights', 'random_seed', 'worker_id', 'weight_decay',
        'early_stop_threshold', 'early_stop_patience', 'objective', 'space_params',
        'direction'
    )
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _run_worker_task_loky(task: _WorkerTask):
    """Execute a worker task. Called by loky in a subprocess."""
    from ragda import core as worker_core
    from ragda.space import SearchSpace
    
    # Set up the evaluation function with task-local data
    space = SearchSpace(task.space_params)
    objective = task.objective
    direction = task.direction
    sign = -1.0 if direction == 'maximize' else 1.0
    
    # Check if objective supports batch_size
    sig = inspect.signature(objective)
    supports_batch = 'batch_size' in sig.parameters
    
    def evaluate(x_cont, x_cat, batch_size):
        try:
            params = space.from_split_vectors(x_cont, x_cat)
            if supports_batch and batch_size > 0:
                value = objective(params, batch_size=batch_size)
            else:
                value = objective(params)
            return float(sign * value)
        except Exception:
            return 1e10
    
    return worker_core.optimize_worker_core(
        task.x0_cont, task.x0_cat, task.cat_n_values, task.bounds,
        evaluate,
        task.max_iter, task.lambda_schedule, task.mu_schedule,
        task.sigma_schedule, task.minibatch_schedule,
        task.use_minibatch, task.top_n_fraction,
        task.adam_lr, task.adam_beta1, task.adam_beta2, task.adam_epsilon,
        task.shrink_factor, task.shrink_patience, task.shrink_threshold,
        task.use_improvement_weights,
        task.random_seed, task.worker_id,
        None, None, 0,
        task.weight_decay, task.early_stop_threshold, task.early_stop_patience
    )


# =============================================================================
# Objective Wrapper
# =============================================================================
class _ObjectiveWrapper:
    """Wraps user objective for internal use."""
    
    __slots__ = ('objective', 'space', 'direction', '_sign', 'supports_batch_size')
    
    def __init__(self, objective: Callable, space: SearchSpace, direction: str):
        self.objective = objective
        self.space = space
        self.direction = direction
        self._sign = -1.0 if direction == 'maximize' else 1.0
        
        sig = inspect.signature(objective)
        self.supports_batch_size = 'batch_size' in sig.parameters
    
    def __call__(self, x_cont: np.ndarray, x_cat: np.ndarray, batch_size: int) -> float:
        """Evaluate objective."""
        try:
            params = self.space.from_split_vectors(x_cont, x_cat)
            
            if self.supports_batch_size and batch_size > 0:
                value = self.objective(params, batch_size=batch_size)
            else:
                value = self.objective(params)
            
            return float(self._sign * value)
        except Exception:
            return 1e10


# =============================================================================
# Parameter Validation
# =============================================================================
def _validate_parameters(
    n_trials, lambda_start, lambda_end, lambda_decay_rate,
    sigma_init, sigma_final_fraction, shrink_factor, shrink_patience,
    shrink_threshold, top_n_min, top_n_max,
    adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
    use_minibatch, minibatch_start, minibatch_end
):
    """Validate optimization parameters."""
    
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    
    if lambda_start <= 0:
        raise ValueError(f"lambda_start must be positive, got {lambda_start}")
    
    if lambda_end <= 0 or lambda_end > lambda_start:
        raise ValueError(f"lambda_end must be in (0, lambda_start], got {lambda_end}")
    
    if lambda_decay_rate <= 0:
        raise ValueError(f"lambda_decay_rate must be positive, got {lambda_decay_rate}")
    
    if not (0 < sigma_init <= 1.0):
        raise ValueError(f"sigma_init must be in (0, 1], got {sigma_init}")
    
    if not (0 < sigma_final_fraction <= 1.0):
        raise ValueError(f"sigma_final_fraction must be in (0, 1], got {sigma_final_fraction}")
    
    if not (0 < shrink_factor < 1.0):
        raise ValueError(f"shrink_factor must be in (0, 1), got {shrink_factor}")
    
    if shrink_patience <= 0:
        raise ValueError(f"shrink_patience must be positive, got {shrink_patience}")
    
    if not (0 < top_n_min <= 1.0):
        raise ValueError(f"top_n_min must be in (0, 1], got {top_n_min}")
    
    if not (0 < top_n_max <= 1.0):
        raise ValueError(f"top_n_max must be in (0, 1], got {top_n_max}")
    
    if top_n_min > top_n_max:
        raise ValueError(f"top_n_min must be <= top_n_max")
    
    if adam_learning_rate <= 0:
        raise ValueError(f"adam_learning_rate must be positive, got {adam_learning_rate}")
    
    if not (0 <= adam_beta1 < 1.0):
        raise ValueError(f"adam_beta1 must be in [0, 1), got {adam_beta1}")
    
    if not (0 <= adam_beta2 < 1.0):
        raise ValueError(f"adam_beta2 must be in [0, 1), got {adam_beta2}")
    
    if adam_epsilon <= 0:
        raise ValueError(f"adam_epsilon must be positive, got {adam_epsilon}")
    
    if use_minibatch:
        if minibatch_start is None or minibatch_end is None:
            raise ValueError("use_minibatch=True requires minibatch_start and minibatch_end")
        if minibatch_start <= 0:
            raise ValueError(f"minibatch_start must be positive, got {minibatch_start}")
        if minibatch_end < minibatch_start:
            raise ValueError(f"minibatch_end must be >= minibatch_start")


# =============================================================================
# Main Optimizer Class
# =============================================================================
class RAGDAOptimizer:
    """
    RAGDA (Roughly Approximated Gradient Descent Algorithm) Optimizer.
    
    High-performance derivative-free optimization with:
    - Parallel multi-worker exploration with diverse strategies
    - ADAM-based pseudo-gradient descent using only improving samples
    - Support for continuous, ordinal, and categorical variables
    - Mini-batch evaluation for data-driven objectives (ML, portfolios)
    - Adaptive step-size shrinking on stagnation
    - Early stopping on convergence
    - Dynamic worker management with elite selection (optional)
    
    Algorithm:
    1. W workers initialized with different top_n fractions
    2. Each worker samples N points around current position
    3. Only IMPROVING samples (f < f_current) contribute to gradient
    4. Gradient = weighted sum of directions to improving samples
    5. Weight decay applied by rank (best = 1.0, rank i = decay^i)
    6. ADAM update determines step size
    7. Worker synchronization based on strategy:
       - 'greedy': ALL workers reset to global best (original behavior)
       - 'dynamic': Top % elite workers survive, others restart adaptively
    8. Categorical: weighted mode of top improving samples
    
    Example:
        >>> from ragda import RAGDAOptimizer
        >>> 
        >>> def objective(params):
        ...     return (params['x'] - 2)**2 + (params['y'] - 3)**2
        >>> 
        >>> space = [
        ...     {'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]},
        ...     {'name': 'y', 'type': 'continuous', 'bounds': [-5, 5]}
        ... ]
        >>> 
        >>> optimizer = RAGDAOptimizer(space, direction='minimize')
        >>> result = optimizer.optimize(objective, n_trials=100)
        >>> print(f"Best: {result.best_params} = {result.best_value}")
        
        # With dynamic worker strategy for multi-modal problems:
        >>> result = optimizer.optimize(
        ...     objective, n_trials=100,
        ...     worker_strategy='dynamic',
        ...     elite_fraction=0.5,
        ...     enable_worker_decay=True
        ... )
    """
    
    __slots__ = ('space', 'direction', 'random_state', 'n_workers', 'max_parallel_workers',
                 'highdim_threshold', 'variance_threshold', 'reduction_method')
    
    def __init__(
        self,
        space: List[Dict[str, Any]],
        direction: Literal['minimize', 'maximize'] = 'minimize',
        n_workers: Optional[int] = None,
        max_parallel_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        # High-dimensional settings (automatic)
        highdim_threshold: int = 100,
        variance_threshold: float = 0.95,
        reduction_method: Literal['auto', 'kernel_pca', 'incremental_pca', 'random_projection'] = 'auto'
    ):
        """
        Initialize RAGDA optimizer.
        
        Parameters
        ----------
        space : list of dict
            Search space definition. Each dict has:
            - name: str, parameter name
            - type: 'continuous', 'ordinal', or 'categorical'
            - bounds: [lower, upper] for continuous/ordinal
            - values: list for categorical
            - log: bool, log-scale for continuous (optional)
        direction : str
            'minimize' or 'maximize'
        n_workers : int, optional
            Number of logical workers. Can exceed CPU count for noisy objectives.
            Default: CPU_count // 2
        max_parallel_workers : int, optional
            Maximum workers to run simultaneously. Default: CPU_count.
            If n_workers > max_parallel_workers, workers run in waves.
        random_state : int, optional
            Random seed for reproducibility
        highdim_threshold : int
            Use high-dimensional methods when continuous dims >= this (default: 100)
        variance_threshold : float
            Fraction of variance to capture in dimensionality reduction (default: 0.95)
        reduction_method : str
            'auto', 'kernel_pca', 'incremental_pca', or 'random_projection'
        """
        if len(space) == 0:
            raise ValueError("Search space cannot be empty")
        
        self.space = SearchSpace(space)
        self.direction = direction
        self.random_state = random_state
        
        if n_workers is None:
            n_workers = max(1, cpu_count() // 2)
        
        if n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {n_workers}")
        
        self.n_workers = n_workers
        
        # Max parallel workers (for wave-based execution when n_workers > cores)
        if max_parallel_workers is None:
            max_parallel_workers = cpu_count()
        self.max_parallel_workers = max(1, min(max_parallel_workers, n_workers))
        
        # High-dimensional settings
        self.highdim_threshold = highdim_threshold
        self.variance_threshold = variance_threshold
        self.reduction_method = reduction_method
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_schedules(
        self, 
        max_iter: int,
        lambda_start: int,
        lambda_end: int,
        lambda_decay_rate: float,
        sigma_init: float,
        sigma_final_fraction: float,
        sigma_decay_schedule: str,
        minibatch_start: Optional[int],
        minibatch_end: Optional[int],
        minibatch_schedule: Optional[str]
    ) -> Dict[str, np.ndarray]:
        """Compute annealing schedules."""
        t = np.arange(max_iter, dtype=np.float64)
        progress = t / max_iter
        
        # Lambda (batch size) schedule - exponential decay
        decay = np.exp(-lambda_decay_rate * progress)
        lambda_sched = (lambda_end + (lambda_start - lambda_end) * decay).astype(np.int32)
        
        # Mu (selection size) schedule
        truncation = 0.5 + (0.15 - 0.5) * progress
        mu_sched = np.maximum(2, (lambda_sched * truncation).astype(np.int32))
        
        # Sigma (sampling radius) schedule
        if sigma_decay_schedule == 'exponential':
            decay_curve = np.exp(-3 * progress)
        elif sigma_decay_schedule == 'linear':
            decay_curve = 1.0 - progress
        elif sigma_decay_schedule == 'cosine':
            decay_curve = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown sigma_decay_schedule: {sigma_decay_schedule}")
        
        sigma_final = sigma_init * sigma_final_fraction
        sigma_sched = (sigma_final + (sigma_init - sigma_final) * decay_curve).astype(np.float64)
        
        schedules = {
            'lambda': lambda_sched,
            'mu': mu_sched,
            'sigma': sigma_sched
        }
        
        # Mini-batch schedule (batch size GROWS over time)
        if minibatch_start is not None and minibatch_end is not None:
            if minibatch_schedule == 'constant':
                mb_sched = np.full(max_iter, minibatch_start, dtype=np.int32)
            
            elif minibatch_schedule == 'linear':
                mb_sched = (minibatch_start + (minibatch_end - minibatch_start) * progress).astype(np.int32)
            
            elif minibatch_schedule == 'exponential':
                growth = np.exp(3 * progress)
                mb_sched = (minibatch_start + (minibatch_end - minibatch_start) * (growth - 1) / (np.exp(3) - 1)).astype(np.int32)
            
            elif minibatch_schedule == 'inverse_decay':
                # Slow growth at first, faster near end (inverse weight decay)
                decay_rate = 0.95
                inverse_decay = 1.0 - np.power(decay_rate, np.arange(max_iter))
                inverse_decay = inverse_decay / inverse_decay[-1]
                mb_sched = (minibatch_start + (minibatch_end - minibatch_start) * inverse_decay).astype(np.int32)
            
            elif minibatch_schedule == 'step':
                # 5 steps with increasing batch size
                n_steps = 5
                step_size = max_iter // n_steps
                mb_sched = np.zeros(max_iter, dtype=np.int32)
                for step in range(n_steps):
                    start_idx = step * step_size
                    end_idx = (step + 1) * step_size if step < n_steps - 1 else max_iter
                    step_progress = (step + 1) / n_steps
                    batch = int(minibatch_start + (minibatch_end - minibatch_start) * step_progress)
                    mb_sched[start_idx:end_idx] = batch
            
            else:
                raise ValueError(f"Unknown minibatch_schedule: {minibatch_schedule}")
            
            schedules['minibatch'] = mb_sched
        else:
            schedules['minibatch'] = np.full(max_iter, -1, dtype=np.int32)
        
        return schedules
    
    def _generate_worker_strategies(
        self, 
        n_workers: int,
        top_n_min: float,
        top_n_max: float
    ) -> List[Dict]:
        """Generate diverse top_n_fraction for each worker."""
        strategies = []
        for i in range(n_workers):
            if n_workers > 1:
                top_n_fraction = top_n_max - (top_n_max - top_n_min) * (i / (n_workers - 1))
            else:
                top_n_fraction = (top_n_max + top_n_min) / 2
            
            strategies.append({
                'worker_id': i,
                'top_n_fraction': float(top_n_fraction),
            })
        
        return strategies
    
    def _initialize_starting_points(self, x0: Optional[Union[Dict, List[Dict]]]) -> List[Dict]:
        """Initialize starting points using LHS."""
        if x0 is None:
            return self.space.sample(n=self.n_workers, method='lhs')
        
        elif isinstance(x0, dict):
            if not self.space.validate(x0):
                raise ValueError(f"Invalid x0: {x0}")
            
            starting_points = [x0.copy()]
            
            for i in range(1, self.n_workers):
                noisy_x0 = {}
                for param in self.space.parameters:
                    if param.type == 'categorical':
                        if np.random.rand() < 0.3:
                            noisy_x0[param.name] = np.random.choice(param.values)
                        else:
                            noisy_x0[param.name] = x0[param.name]
                    else:
                        unit_val = param.transform_to_unit(x0[param.name])
                        noise = np.random.randn() * 0.1
                        noisy_unit = np.clip(unit_val + noise, 0, 1)
                        noisy_x0[param.name] = param.transform_from_unit(noisy_unit)
                
                starting_points.append(noisy_x0)
            
            return starting_points
        
        elif isinstance(x0, list):
            for x in x0:
                if not self.space.validate(x):
                    raise ValueError(f"Invalid x0 element: {x}")
            
            starting_points = x0.copy()
            
            n_remaining = self.n_workers - len(x0)
            if n_remaining > 0:
                lhs_points = self.space.sample(n=n_remaining, method='lhs')
                starting_points.extend(lhs_points)
            
            return starting_points[:self.n_workers]
        
        else:
            raise ValueError("x0 must be dict, list of dict, or None")
    
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: int = 1000,
        x0: Optional[Union[Dict, List[Dict]]] = None,
        
        # Population & Sampling
        lambda_start: int = 50,
        lambda_end: int = 10,
        lambda_decay_rate: float = 5.0,
        
        # Sample Space
        sigma_init: float = 0.3,
        sigma_final_fraction: float = 0.2,
        sigma_decay_schedule: Literal['exponential', 'linear', 'cosine'] = 'exponential',
        
        # Adaptive Shrinking
        shrink_factor: float = 0.9,
        shrink_patience: int = 10,
        shrink_threshold: float = 1e-6,
        
        # Top-N Weighting (worker diversity)
        use_improvement_weights: bool = True,
        top_n_min: float = 0.2,
        top_n_max: float = 1.0,
        weight_decay: float = 0.95,
        
        # Worker Synchronization
        sync_frequency: int = 100,
        
        # Worker Strategy (NEW)
        worker_strategy: Literal['greedy', 'dynamic'] = 'greedy',
        
        # Dynamic Worker Strategy Settings (NEW - only used when worker_strategy='dynamic')
        elite_fraction: float = 0.5,
        restart_mode: Literal['elite', 'random', 'adaptive'] = 'adaptive',
        restart_elite_prob_start: float = 0.3,
        restart_elite_prob_end: float = 0.8,
        enable_worker_decay: bool = False,
        min_workers: int = 2,
        worker_decay_rate: float = 0.5,
        
        # Mini-batch for Data-Driven Objectives
        use_minibatch: bool = False,
        data_size: Optional[int] = None,
        minibatch_start: Optional[int] = None,
        minibatch_end: Optional[int] = None,
        minibatch_schedule: Literal['constant', 'linear', 'exponential', 'inverse_decay', 'step'] = 'inverse_decay',
        
        # ADAM
        adam_learning_rate: float = 0.001,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        
        # Early Stopping
        early_stop_threshold: float = 1e-12,
        early_stop_patience: int = 50,
        
        # Other
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run RAGDA optimization.
        
        Parameters
        ----------
        objective : callable
            Function to optimize. Takes dict of params, returns float.
            For mini-batch mode, add batch_size parameter:
            def objective(params, batch_size=None): ...
        n_trials : int
            Number of iterations per worker
        x0 : dict or list of dict, optional
            Starting point(s)
        lambda_start, lambda_end : int
            Initial and final batch size (samples per iteration)
        lambda_decay_rate : float
            Exponential decay rate for batch size
        sigma_init : float
            Initial sampling radius (in [0,1] unit space)
        sigma_final_fraction : float
            Final sigma as fraction of initial
        sigma_decay_schedule : str
            'exponential', 'linear', or 'cosine'
        shrink_factor : float
            Multiply sigma by this on stagnation (e.g., 0.9)
        shrink_patience : int
            Iterations without improvement before shrinking
        shrink_threshold : float
            Minimum relative improvement to count as progress
        use_improvement_weights : bool
            Use only improving samples for gradient (recommended)
        top_n_min, top_n_max : float
            Range of top-n fractions across workers (diversity)
        weight_decay : float
            Exponential decay for rank-based weights
        sync_frequency : int
            How often workers sync (0 = never)
        worker_strategy : str
            'greedy': All workers reset to global best at sync (original behavior)
            'dynamic': Top elite_fraction workers survive, others restart adaptively
        elite_fraction : float
            Fraction of top workers to keep (only for worker_strategy='dynamic')
        restart_mode : str
            How to restart non-elite workers (only for worker_strategy='dynamic'):
            - 'elite': Sample from elite worker positions with perturbation
            - 'random': Random restart from search space
            - 'adaptive': Mix that increases elite probability over time
        restart_elite_prob_start : float
            Initial probability of restarting from elite (for restart_mode='adaptive')
        restart_elite_prob_end : float
            Final probability of restarting from elite (for restart_mode='adaptive')
        enable_worker_decay : bool
            Reduce number of active workers over time (only for worker_strategy='dynamic')
        min_workers : int
            Minimum workers to keep when using worker decay
        worker_decay_rate : float
            How aggressively to decay workers (0.0-1.0). 0.5 = reduce to 50% by end
        use_minibatch : bool
            Enable mini-batch evaluation
        data_size : int, optional
            Total dataset size (auto-calculates batch schedule)
        minibatch_start, minibatch_end : int, optional
            Starting and final batch sizes
        minibatch_schedule : str
            'constant', 'linear', 'exponential', 'inverse_decay', 'step'
        adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon : float
            ADAM optimizer parameters
        early_stop_threshold : float
            Stop if best value below this
        early_stop_patience : int
            Stop if no improvement for this many iterations
        verbose : bool
            Print progress information
        
        Returns
        -------
        OptimizationResult
            Contains best_params, best_value, all trials, etc.
        """
        
        # Auto-calculate mini-batch schedule
        if use_minibatch and data_size is not None:
            if minibatch_start is None:
                minibatch_start = max(32, data_size // 20)
            if minibatch_end is None:
                minibatch_end = int(data_size * 0.8)
        
        # Validate
        _validate_parameters(
            n_trials, lambda_start, lambda_end, lambda_decay_rate,
            sigma_init, sigma_final_fraction, shrink_factor, shrink_patience,
            shrink_threshold, top_n_min, top_n_max,
            adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
            use_minibatch, minibatch_start, minibatch_end
        )
        
        # Validate dynamic worker strategy parameters
        if worker_strategy not in ('greedy', 'dynamic'):
            raise ValueError(f"worker_strategy must be 'greedy' or 'dynamic', got {worker_strategy}")
        
        if worker_strategy == 'dynamic':
            if not (0 < elite_fraction <= 1.0):
                raise ValueError(f"elite_fraction must be in (0, 1], got {elite_fraction}")
            if restart_mode not in ('elite', 'random', 'adaptive'):
                raise ValueError(f"restart_mode must be 'elite', 'random', or 'adaptive', got {restart_mode}")
            if not (0 <= restart_elite_prob_start <= 1.0):
                raise ValueError(f"restart_elite_prob_start must be in [0, 1], got {restart_elite_prob_start}")
            if not (0 <= restart_elite_prob_end <= 1.0):
                raise ValueError(f"restart_elite_prob_end must be in [0, 1], got {restart_elite_prob_end}")
            if min_workers < 1:
                raise ValueError(f"min_workers must be >= 1, got {min_workers}")
            if not (0 <= worker_decay_rate <= 1.0):
                raise ValueError(f"worker_decay_rate must be in [0, 1], got {worker_decay_rate}")
        
        if not callable(objective):
            raise ValueError("objective must be callable")
        
        # Check if high-dimensional optimization should be used
        use_highdim = (
            HIGHDIM_AVAILABLE and 
            self.space.n_continuous >= self.highdim_threshold
        )
        
        if use_highdim:
            return self._optimize_highdim(
                objective=objective,
                n_trials=n_trials,
                x0=x0,
                lambda_start=lambda_start,
                lambda_end=lambda_end,
                lambda_decay_rate=lambda_decay_rate,
                sigma_init=sigma_init,
                sigma_final_fraction=sigma_final_fraction,
                sigma_decay_schedule=sigma_decay_schedule,
                shrink_factor=shrink_factor,
                shrink_patience=shrink_patience,
                shrink_threshold=shrink_threshold,
                use_improvement_weights=use_improvement_weights,
                top_n_min=top_n_min,
                top_n_max=top_n_max,
                weight_decay=weight_decay,
                sync_frequency=sync_frequency,
                worker_strategy=worker_strategy,
                elite_fraction=elite_fraction,
                restart_mode=restart_mode,
                restart_elite_prob_start=restart_elite_prob_start,
                restart_elite_prob_end=restart_elite_prob_end,
                enable_worker_decay=enable_worker_decay,
                min_workers=min_workers,
                worker_decay_rate=worker_decay_rate,
                use_minibatch=use_minibatch,
                data_size=data_size,
                minibatch_start=minibatch_start,
                minibatch_end=minibatch_end,
                minibatch_schedule=minibatch_schedule,
                adam_learning_rate=adam_learning_rate,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                adam_epsilon=adam_epsilon,
                early_stop_threshold=early_stop_threshold,
                early_stop_patience=early_stop_patience,
                verbose=verbose
            )
        
        # Standard optimization path
        return self._optimize_standard(
            objective=objective,
            n_trials=n_trials,
            x0=x0,
            lambda_start=lambda_start,
            lambda_end=lambda_end,
            lambda_decay_rate=lambda_decay_rate,
            sigma_init=sigma_init,
            sigma_final_fraction=sigma_final_fraction,
            sigma_decay_schedule=sigma_decay_schedule,
            shrink_factor=shrink_factor,
            shrink_patience=shrink_patience,
            shrink_threshold=shrink_threshold,
            use_improvement_weights=use_improvement_weights,
            top_n_min=top_n_min,
            top_n_max=top_n_max,
            weight_decay=weight_decay,
            sync_frequency=sync_frequency,
            worker_strategy=worker_strategy,
            elite_fraction=elite_fraction,
            restart_mode=restart_mode,
            restart_elite_prob_start=restart_elite_prob_start,
            restart_elite_prob_end=restart_elite_prob_end,
            enable_worker_decay=enable_worker_decay,
            min_workers=min_workers,
            worker_decay_rate=worker_decay_rate,
            use_minibatch=use_minibatch,
            data_size=data_size,
            minibatch_start=minibatch_start,
            minibatch_end=minibatch_end,
            minibatch_schedule=minibatch_schedule,
            adam_learning_rate=adam_learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            verbose=verbose
        )
    
    def _optimize_highdim(
        self,
        objective: Callable,
        n_trials: int,
        x0: Optional[Union[Dict, List[Dict]]],
        verbose: bool,
        **kwargs
    ) -> OptimizationResult:
        """
        High-dimensional optimization using dimensionality reduction.
        
        Two-stage approach:
        1. Sample space, analyze effective dimensionality
        2. If low-dim structure detected: optimize in reduced space + trust region refinement
        3. Otherwise: fall back to standard optimization
        """
        if verbose:
            print(f"{'='*70}")
            print(f"RAGDA High-Dimensional Optimization")
            print(f"{'='*70}")
            print(f"Continuous dimensions: {self.space.n_continuous} (threshold: {self.highdim_threshold})")
            print(f"Reduction method: {self.reduction_method}")
            print(f"Variance threshold: {self.variance_threshold:.1%}")
        
        # Stage 0: Sample initial points for dimensionality analysis
        initial_samples = min(max(100, self.space.n_continuous), n_trials // 2)
        
        if verbose:
            print(f"Sampling {initial_samples} points for dimensionality analysis...")
        
        # Sample points and evaluate
        sample_points = self.space.sample(n=initial_samples, method='lhs')
        X_samples = np.zeros((initial_samples, self.space.n_continuous), dtype=np.float64)
        f_samples = np.zeros(initial_samples, dtype=np.float64)
        
        sign = -1.0 if self.direction == 'maximize' else 1.0
        
        for i, params in enumerate(sample_points):
            x_cont, _, _ = self.space.to_split_vectors(params)
            X_samples[i] = x_cont
            try:
                f_samples[i] = sign * objective(params)
            except Exception:
                f_samples[i] = 1e10
        
        # Analyze effective dimensionality
        eigenvalues = highdim_core.compute_eigenvalues_fast(X_samples)
        dim_result = highdim_core.estimate_effective_dimensionality(
            eigenvalues, self.variance_threshold
        )
        
        effective_dim = dim_result['effective_dim']
        is_low_dim = dim_result['is_low_dimensional']
        
        if verbose:
            print(f"Effective dimensionality: {effective_dim} / {self.space.n_continuous}")
            print(f"Low-dimensional structure: {'Yes' if is_low_dim else 'No'}")
        
        # If no low-dimensional structure, fall back to standard optimization
        if not is_low_dim or effective_dim >= self.space.n_continuous * 0.8:
            if verbose:
                print("Falling back to standard optimization...")
                print(f"{'='*70}\n")
            
            # Use standard path but pass through highdim detection info
            return self._optimize_standard(
                objective=objective,
                n_trials=n_trials,
                x0=x0,
                verbose=verbose,
                highdim_info={'detected': False, 'effective_dim': effective_dim},
                **kwargs
            )
        
        # Select reduction method
        if self.reduction_method == 'auto':
            if self.space.n_continuous > 500:
                method = 'random_projection'
            elif initial_samples < self.space.n_continuous:
                method = 'kernel_pca'
            else:
                method = 'incremental_pca'
        else:
            method = self.reduction_method
        
        if verbose:
            print(f"Using reduction method: {method}")
        
        # Fit dimensionality reducer
        n_components = min(effective_dim + 5, self.space.n_continuous // 2)
        
        if method == 'kernel_pca':
            reducer_state = highdim_core.fit_kernel_pca(X_samples, n_components=n_components)
        elif method == 'incremental_pca':
            reducer_state = highdim_core.fit_incremental_pca(X_samples, n_components=n_components)
        else:  # random_projection
            reducer_state = highdim_core.fit_random_projection(
                self.space.n_continuous, n_components,
                'gaussian', self.random_state or 42
            )
        
        # Stage 1: Optimize in reduced space
        stage1_trials = int(n_trials * 0.7)
        
        if verbose:
            print(f"\nStage 1: Optimizing in {n_components}D reduced space ({stage1_trials} trials)...")
        
        # Create reduced space
        reduced_space = [
            {'name': f'z{i}', 'type': 'continuous', 'bounds': [-3.0, 3.0]}
            for i in range(n_components)
        ]
        
        # Add categorical parameters unchanged
        for param in self.space.parameters:
            if param.type == 'categorical':
                reduced_space.append({
                    'name': param.name,
                    'type': 'categorical',
                    'values': list(param.values)
                })
        
        # Create objective in reduced space
        def reduced_objective(reduced_params):
            # Extract reduced coordinates
            z = np.array([reduced_params[f'z{i}'] for i in range(n_components)], dtype=np.float64)
            z = z.reshape(1, -1)
            
            # Inverse transform to full space
            if method == 'kernel_pca':
                x_full = highdim_core.inverse_transform_kernel_pca(reducer_state, z)
            elif method == 'incremental_pca':
                x_full = highdim_core.inverse_transform_incremental_pca(reducer_state, z)
            else:
                x_full = highdim_core.inverse_transform_random_projection(reducer_state, z)
            
            x_full = x_full.flatten()
            
            # Clip to bounds
            bounds = self.space.get_bounds_array()
            x_full = np.clip(x_full, bounds[:, 0], bounds[:, 1])
            
            # Build full params dict
            full_params = {}
            cont_idx = 0
            for param in self.space.parameters:
                if param.type == 'categorical':
                    full_params[param.name] = reduced_params[param.name]
                else:
                    full_params[param.name] = float(x_full[cont_idx])
                    cont_idx += 1
            
            return objective(full_params)
        
        # Create reduced optimizer
        reduced_optimizer = RAGDAOptimizer(
            reduced_space,
            direction=self.direction,
            n_workers=self.n_workers,
            random_state=self.random_state,
            highdim_threshold=1000000  # Disable recursive highdim
        )
        
        # Run Stage 1 optimization
        stage1_result = reduced_optimizer.optimize(
            reduced_objective,
            n_trials=stage1_trials,
            verbose=False,
            **{k: v for k, v in kwargs.items() if k not in ['x0']}
        )
        
        # Get best point from Stage 1
        best_reduced = stage1_result.best_params
        z_best = np.array([best_reduced[f'z{i}'] for i in range(n_components)], dtype=np.float64)
        z_best = z_best.reshape(1, -1)
        
        if method == 'kernel_pca':
            x_best_full = highdim_core.inverse_transform_kernel_pca(reducer_state, z_best)
        elif method == 'incremental_pca':
            x_best_full = highdim_core.inverse_transform_incremental_pca(reducer_state, z_best)
        else:
            x_best_full = highdim_core.inverse_transform_random_projection(reducer_state, z_best)
        
        x_best_full = x_best_full.flatten()
        bounds = self.space.get_bounds_array()
        x_best_full = np.clip(x_best_full, bounds[:, 0], bounds[:, 1])
        
        # Build Stage 1 best params
        stage1_best_params = {}
        cont_idx = 0
        for param in self.space.parameters:
            if param.type == 'categorical':
                stage1_best_params[param.name] = best_reduced[param.name]
            else:
                stage1_best_params[param.name] = float(x_best_full[cont_idx])
                cont_idx += 1
        
        if verbose:
            print(f"Stage 1 best: {stage1_result.best_value:.6f}")
        
        # Stage 2: Trust region refinement in full space
        stage2_trials = n_trials - stage1_trials
        
        if stage2_trials > 20:
            if verbose:
                print(f"\nStage 2: Trust region refinement ({stage2_trials} trials)...")
            
            # Create tight trust region around Stage 1 solution
            trust_fraction = 0.1
            trust_space = []
            
            for param in self.space.parameters:
                if param.type == 'categorical':
                    trust_space.append({
                        'name': param.name,
                        'type': 'categorical',
                        'values': list(param.values)
                    })
                else:
                    center = stage1_best_params[param.name]
                    full_range = param.bounds[1] - param.bounds[0]
                    half_width = full_range * trust_fraction / 2
                    
                    new_lower = max(param.bounds[0], center - half_width)
                    new_upper = min(param.bounds[1], center + half_width)
                    
                    trust_space.append({
                        'name': param.name,
                        'type': 'continuous',
                        'bounds': [new_lower, new_upper]
                    })
            
            # Run Stage 2 optimization
            trust_optimizer = RAGDAOptimizer(
                trust_space,
                direction=self.direction,
                n_workers=self.n_workers,
                random_state=(self.random_state + 1000) if self.random_state else None,
                highdim_threshold=1000000  # Disable recursive highdim
            )
            
            stage2_result = trust_optimizer.optimize(
                objective,
                n_trials=stage2_trials,
                x0=stage1_best_params,
                verbose=False,
                **{k: v for k, v in kwargs.items() if k not in ['x0']}
            )
            
            if verbose:
                print(f"Stage 2 best: {stage2_result.best_value:.6f}")
            
            # Use better of Stage 1 or Stage 2
            if self.direction == 'minimize':
                if stage2_result.best_value < stage1_result.best_value:
                    final_best_params = stage2_result.best_params
                    final_best_value = stage2_result.best_value
                else:
                    final_best_params = stage1_best_params
                    final_best_value = stage1_result.best_value
            else:
                if stage2_result.best_value > stage1_result.best_value:
                    final_best_params = stage2_result.best_params
                    final_best_value = stage2_result.best_value
                else:
                    final_best_params = stage1_best_params
                    final_best_value = stage1_result.best_value
        else:
            final_best_params = stage1_best_params
            final_best_value = stage1_result.best_value
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"High-Dim Optimization Complete")
            print(f"{'='*70}")
            print(f"Effective dim: {effective_dim} / {self.space.n_continuous}")
            print(f"Method: {method}")
            print(f"Final {self.direction}: {final_best_value:.6f}")
            print(f"{'='*70}")
        
        # Build result
        best_trial = Trial(
            trial_id=0,
            worker_id=0,
            iteration=n_trials - 1,
            params=final_best_params,
            value=final_best_value,
            batch_size=-1
        )
        
        result = OptimizationResult(
            best_params=final_best_params,
            best_value=final_best_value,
            best_trial=best_trial,
            best_worker_id=0,
            best_concentration=1.0,
            trials=[best_trial],
            n_trials=n_trials,
            n_workers=self.n_workers,
            direction=self.direction,
            space=self.space,
            optimization_params={
                'n_trials': n_trials,
                'highdim': True,
                'effective_dim': effective_dim,
                'reduction_method': method,
                'n_components': n_components,
            }
        )
        
        return result
    
    def _optimize_standard(
        self,
        objective: Callable,
        n_trials: int,
        x0: Optional[Union[Dict, List[Dict]]],
        verbose: bool,
        highdim_info: Optional[Dict] = None,
        **kwargs
    ) -> OptimizationResult:
        """Standard optimization path (extracted for reuse)."""
        # Get kwargs with defaults
        lambda_start = kwargs.get('lambda_start', 50)
        lambda_end = kwargs.get('lambda_end', 10)
        lambda_decay_rate = kwargs.get('lambda_decay_rate', 5.0)
        sigma_init = kwargs.get('sigma_init', 0.3)
        sigma_final_fraction = kwargs.get('sigma_final_fraction', 0.2)
        sigma_decay_schedule = kwargs.get('sigma_decay_schedule', 'exponential')
        shrink_factor = kwargs.get('shrink_factor', 0.9)
        shrink_patience = kwargs.get('shrink_patience', 10)
        shrink_threshold = kwargs.get('shrink_threshold', 1e-6)
        use_improvement_weights = kwargs.get('use_improvement_weights', True)
        top_n_min = kwargs.get('top_n_min', 0.2)
        top_n_max = kwargs.get('top_n_max', 1.0)
        weight_decay = kwargs.get('weight_decay', 0.95)
        sync_frequency = kwargs.get('sync_frequency', 100)
        use_minibatch = kwargs.get('use_minibatch', False)
        data_size = kwargs.get('data_size', None)
        minibatch_start = kwargs.get('minibatch_start', None)
        minibatch_end = kwargs.get('minibatch_end', None)
        minibatch_schedule = kwargs.get('minibatch_schedule', 'inverse_decay')
        adam_learning_rate = kwargs.get('adam_learning_rate', 0.001)
        adam_beta1 = kwargs.get('adam_beta1', 0.9)
        adam_beta2 = kwargs.get('adam_beta2', 0.999)
        adam_epsilon = kwargs.get('adam_epsilon', 1e-8)
        early_stop_threshold = kwargs.get('early_stop_threshold', 1e-12)
        early_stop_patience = kwargs.get('early_stop_patience', 50)
        
        # Dynamic worker strategy parameters
        worker_strategy = kwargs.get('worker_strategy', 'greedy')
        elite_fraction = kwargs.get('elite_fraction', 0.5)
        restart_mode = kwargs.get('restart_mode', 'adaptive')
        restart_elite_prob_start = kwargs.get('restart_elite_prob_start', 0.3)
        restart_elite_prob_end = kwargs.get('restart_elite_prob_end', 0.8)
        enable_worker_decay = kwargs.get('enable_worker_decay', False)
        min_workers = kwargs.get('min_workers', 2)
        worker_decay_rate = kwargs.get('worker_decay_rate', 0.5)
        
        # Auto-calculate mini-batch schedule
        if use_minibatch and data_size is not None:
            if minibatch_start is None:
                minibatch_start = max(32, data_size // 20)
            if minibatch_end is None:
                minibatch_end = int(data_size * 0.8)
        
        # Initialize
        x0_list = self._initialize_starting_points(x0)
        
        schedules = self._get_schedules(
            n_trials, lambda_start, lambda_end, lambda_decay_rate,
            sigma_init, sigma_final_fraction, sigma_decay_schedule,
            minibatch_start, minibatch_end, minibatch_schedule if use_minibatch else None
        )
        
        worker_strategies = self._generate_worker_strategies(self.n_workers, top_n_min, top_n_max)
        
        objective_wrapper = _ObjectiveWrapper(objective, self.space, self.direction)
        
        if use_minibatch and not objective_wrapper.supports_batch_size:
            warnings.warn("use_minibatch=True but objective lacks 'batch_size' parameter. Disabling.")
            use_minibatch = False
        
        # Calculate sync points for dynamic strategy
        if sync_frequency > 0:
            n_syncs = n_trials // sync_frequency
        else:
            n_syncs = 0
        
        # Print config
        if verbose:
            print(f"{'='*70}")
            print(f"RAGDA Optimization (Cython {core.get_version()})")
            print(f"{'='*70}")
            print(f"Search space: {self.space.n_params} parameters")
            print(f"  - Continuous/Ordinal: {self.space.n_continuous}")
            print(f"  - Categorical: {self.space.n_categorical}")
            print(f"Direction: {self.direction}")
            print(f"Workers: {self.n_workers} (max parallel: {self.max_parallel_workers})")
            print(f"  top_n: {top_n_max:.0%} -> {top_n_min:.0%}")
            print(f"Iterations per worker: {n_trials}")
            print(f"Total evaluations: ~{n_trials * self.n_workers * (lambda_start + lambda_end) // 2:,}")
            print(f"Batch size: lambda = {lambda_start} -> {lambda_end}")
            print(f"Sample space: sigma = {sigma_init:.3f} -> {sigma_init * sigma_final_fraction:.3f}")
            print(f"Shrinking: factor={shrink_factor}, patience={shrink_patience}")
            print(f"Weighting: only_improving={use_improvement_weights}, weight_decay={weight_decay}")
            print(f"Early stopping: threshold={early_stop_threshold}, patience={early_stop_patience}")
            print(f"Worker strategy: {worker_strategy}")
            if worker_strategy == 'dynamic':
                print(f"  Elite fraction: {elite_fraction:.0%}")
                print(f"  Restart mode: {restart_mode}")
                if restart_mode == 'adaptive':
                    print(f"  Elite restart prob: {restart_elite_prob_start:.0%} -> {restart_elite_prob_end:.0%}")
                if enable_worker_decay:
                    print(f"  Worker decay: enabled (min={min_workers}, rate={worker_decay_rate:.0%})")
            if sync_frequency > 0:
                if worker_strategy == 'greedy':
                    print(f"Sync: every {sync_frequency} iters (all workers reset to best)")
                else:
                    print(f"Sync: every {sync_frequency} iters (elite selection + adaptive restart)")
            if use_minibatch:
                print(f"Mini-batch: {minibatch_start} -> {minibatch_end} ({minibatch_schedule})")
            print(f"ADAM: lr={adam_learning_rate}, beta1={adam_beta1}, beta2={adam_beta2}")
            print(f"{'='*70}\n")
        
        # Run workers based on strategy
        if worker_strategy == 'greedy':
            # Original greedy strategy - run all workers in parallel
            worker_results = self._run_parallel(
                x0_list, objective_wrapper, schedules, worker_strategies,
                n_trials, shrink_factor, shrink_patience, shrink_threshold,
                use_improvement_weights, use_minibatch,
                adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
                weight_decay, early_stop_threshold, early_stop_patience, verbose
            )
        else:
            # Dynamic strategy with elite selection and adaptive restarts
            worker_results = self._run_dynamic_strategy(
                x0_list, objective_wrapper, schedules, worker_strategies,
                n_trials, shrink_factor, shrink_patience, shrink_threshold,
                use_improvement_weights, use_minibatch,
                adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
                weight_decay, early_stop_threshold, early_stop_patience,
                sync_frequency, elite_fraction, restart_mode,
                restart_elite_prob_start, restart_elite_prob_end,
                enable_worker_decay, min_workers, worker_decay_rate,
                sigma_init, verbose
            )
        
        # Process results (same as original optimize method)
        all_trials = []
        worker_summaries = []
        trial_id = 0
        
        for result in worker_results:
            x_best_cont, x_best_cat, f_best, history = result
            worker_id = history['worker_id']
            
            best_params = self.space.from_split_vectors(x_best_cont, x_best_cat)
            
            for iter_idx, params_data in enumerate(history['params_history']):
                params = self.space.from_split_vectors(params_data['x_cont'], params_data['x_cat'])
                
                trial = Trial(
                    trial_id=trial_id,
                    worker_id=worker_id,
                    iteration=iter_idx,
                    params=params,
                    value=params_data['f'],
                    batch_size=history['batch_sizes'][iter_idx]
                )
                all_trials.append(trial)
                trial_id += 1
            
            worker_summaries.append({
                'worker_id': worker_id,
                'f_best': f_best,
                'best_params': best_params,
                'top_n_fraction': worker_strategies[worker_id]['top_n_fraction'],
                'history': history,
                'converged': history.get('converged', False),
                'final_iteration': history.get('final_iteration', n_trials - 1)
            })
            
            if verbose:
                status = "CONVERGED" if history.get('converged', False) else ""
                print(f"Worker {worker_id:2d} [top_n={worker_strategies[worker_id]['top_n_fraction']:.2f}] "
                      f"f_best = {f_best:12.6f}  shrinks = {len(history['shrink_events']):2d}  "
                      f"iters = {history.get('final_iteration', n_trials-1)+1:4d} {status}")
        
        # Find global best
        best_worker = min(worker_summaries, key=lambda w: w['f_best'])
        x_best = best_worker['best_params']
        f_best = best_worker['f_best']
        
        # Re-evaluate final best with FULL sample for mini-batch mode
        if use_minibatch:
            try:
                sig = inspect.signature(objective)
                if 'batch_size' in sig.parameters:
                    f_best_full = objective(x_best, batch_size=-1)
                else:
                    f_best_full = objective(x_best)
                
                if self.direction == 'maximize':
                    f_best_full = -f_best_full
                else:
                    f_best_full = float(f_best_full)
                
                if verbose:
                    print(f"\nFinal re-evaluation (full sample): {f_best_full:.6f} (was {f_best:.6f} with minibatch)")
                
                f_best = f_best_full
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Could not re-evaluate with full sample: {e}")
        
        if self.direction == 'maximize':
            f_best = -f_best
        
        best_trial = Trial(
            trial_id=0,
            worker_id=best_worker['worker_id'],
            iteration=best_worker['final_iteration'],
            params=x_best,
            value=f_best,
            batch_size=-1
        )
        
        opt_params = {
            'n_trials': n_trials,
            'lambda_start': lambda_start,
            'lambda_end': lambda_end,
            'sigma_init': sigma_init,
            'use_minibatch': use_minibatch,
            'sync_frequency': sync_frequency,
            'adam_learning_rate': adam_learning_rate,
            'worker_strategy': worker_strategy,
        }
        if worker_strategy == 'dynamic':
            opt_params.update({
                'elite_fraction': elite_fraction,
                'restart_mode': restart_mode,
                'enable_worker_decay': enable_worker_decay,
            })
        if highdim_info:
            opt_params['highdim_info'] = highdim_info
        
        result = OptimizationResult(
            best_params=x_best,
            best_value=f_best,
            best_trial=best_trial,
            best_worker_id=best_worker['worker_id'],
            best_concentration=best_worker['top_n_fraction'],
            trials=all_trials,
            n_trials=len(all_trials),
            n_workers=self.n_workers,
            direction=self.direction,
            space=self.space,
            optimization_params=opt_params
        )
        
        result._worker_summaries = worker_summaries
        
        if verbose:
            total_shrinks = sum(len(w['history']['shrink_events']) for w in worker_summaries)
            converged_count = sum(1 for w in worker_summaries if w['converged'])
            
            print(f"\n{'='*70}")
            print(f"Optimization Complete")
            print(f"{'='*70}")
            print(f"Best worker: {best_worker['worker_id']} (top_n={best_worker['top_n_fraction']:.0%})")
            print(f"Final {self.direction}: {f_best:.6f}")
            print(f"Total shrink events: {total_shrinks}")
            print(f"Workers converged: {converged_count}/{self.n_workers}")
            print(f"{'='*70}")
        
        return result
    
    def _run_parallel(
        self,
        x0_list, objective_wrapper, schedules, worker_strategies,
        max_iter, shrink_factor, shrink_patience, shrink_threshold,
        use_improvement_weights, use_minibatch,
        adam_lr, adam_beta1, adam_beta2, adam_epsilon,
        weight_decay, early_stop_threshold, early_stop_patience, verbose
    ):
        """Run workers in parallel using loky (robust Windows multiprocessing).
        
        Supports wave-based execution when n_workers > max_parallel_workers.
        """
        
        # Build task arguments for each worker
        tasks = []
        for i in range(self.n_workers):
            x0_dict = x0_list[i]
            x0_cont, x0_cat, cat_n_values = self.space.to_split_vectors(x0_dict)
            
            task = _WorkerTask(
                x0_cont=x0_cont,
                x0_cat=x0_cat,
                cat_n_values=cat_n_values,
                bounds=self.space.get_bounds_array(),
                max_iter=max_iter,
                lambda_schedule=schedules['lambda'],
                mu_schedule=schedules['mu'],
                sigma_schedule=schedules['sigma'],
                minibatch_schedule=schedules['minibatch'],
                use_minibatch=use_minibatch,
                top_n_fraction=worker_strategies[i]['top_n_fraction'],
                adam_lr=adam_lr,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                adam_epsilon=adam_epsilon,
                shrink_factor=shrink_factor,
                shrink_patience=shrink_patience,
                shrink_threshold=shrink_threshold,
                use_improvement_weights=use_improvement_weights,
                random_seed=self.random_state + i if self.random_state else i,
                worker_id=i,
                weight_decay=weight_decay,
                early_stop_threshold=early_stop_threshold,
                early_stop_patience=early_stop_patience,
                objective=objective_wrapper.objective,
                space_params=[
                    {
                        'name': p.name,
                        'type': p.type,
                        'bounds': list(p.bounds) if p.bounds else None,
                        'values': list(p.values) if p.values else None,
                        'log': p.log
                    }
                    for p in self.space.parameters
                ],
                direction=self.direction
            )
            tasks.append(task)
        
        # Run in waves if n_workers > max_parallel_workers
        results = []
        n_waves = math.ceil(len(tasks) / self.max_parallel_workers)
        
        for wave_idx in range(n_waves):
            wave_start = wave_idx * self.max_parallel_workers
            wave_end = min(wave_start + self.max_parallel_workers, len(tasks))
            wave_tasks = tasks[wave_start:wave_end]
            
            # Use loky for robust parallel execution (handles cloudpickle)
            executor = get_reusable_executor(max_workers=len(wave_tasks))
            futures = [executor.submit(_run_worker_task_loky, task) for task in wave_tasks]
            
            # Collect results from this wave
            for future in futures:
                results.append(future.result(timeout=3600))  # 1 hour timeout
        
        return results
    
    def _run_dynamic_strategy(
        self,
        x0_list, objective_wrapper, schedules, worker_strategies,
        max_iter, shrink_factor, shrink_patience, shrink_threshold,
        use_improvement_weights, use_minibatch,
        adam_lr, adam_beta1, adam_beta2, adam_epsilon,
        weight_decay, early_stop_threshold, early_stop_patience,
        sync_frequency, elite_fraction, restart_mode,
        restart_elite_prob_start, restart_elite_prob_end,
        enable_worker_decay, min_workers, worker_decay_rate,
        sigma_init, verbose
    ):
        """Run workers with dynamic elite selection and adaptive restarts.
        
        This method runs optimization in phases, with synchronization points where:
        1. Top elite_fraction of workers survive unchanged
        2. Non-elite workers are restarted based on restart_mode
        3. Optionally, workers are decayed (dropped) over time
        """
        
        if sync_frequency <= 0:
            # No sync points, fall back to parallel run
            return self._run_parallel(
                x0_list, objective_wrapper, schedules, worker_strategies,
                max_iter, shrink_factor, shrink_patience, shrink_threshold,
                use_improvement_weights, use_minibatch,
                adam_lr, adam_beta1, adam_beta2, adam_epsilon,
                weight_decay, early_stop_threshold, early_stop_patience, verbose
            )
        
        # Calculate phase structure
        n_phases = max(1, max_iter // sync_frequency)
        iters_per_phase = sync_frequency
        
        # Track active workers and their states
        n_active_workers = self.n_workers
        active_worker_ids = list(range(self.n_workers))
        
        # Current positions for all workers
        current_positions = [x0.copy() for x0 in x0_list]
        
        # Accumulated results
        all_phase_results = []
        global_best_value = float('inf')
        global_best_params = None
        global_best_worker = 0
        
        # Track worker history for final result
        worker_histories = {i: {'phases': [], 'active': True, 'retired_at': None} 
                          for i in range(self.n_workers)}
        
        for phase_idx in range(n_phases):
            phase_progress = phase_idx / n_phases
            phase_start_iter = phase_idx * iters_per_phase
            phase_end_iter = min((phase_idx + 1) * iters_per_phase, max_iter)
            phase_iters = phase_end_iter - phase_start_iter
            
            if phase_iters <= 0:
                break
            
            # Calculate number of active workers for this phase (worker decay)
            if enable_worker_decay:
                # Linear decay from n_workers to min_workers
                decay_progress = phase_progress ** (1.0 / (1.0 - worker_decay_rate + 0.01))
                target_workers = int(self.n_workers - (self.n_workers - min_workers) * decay_progress)
                n_active_workers = max(min_workers, min(target_workers, len(active_worker_ids)))
            else:
                n_active_workers = len(active_worker_ids)
            
            # Slice schedules for this phase
            phase_schedules = {
                'lambda': schedules['lambda'][phase_start_iter:phase_end_iter].copy(),
                'mu': schedules['mu'][phase_start_iter:phase_end_iter].copy(),
                'sigma': schedules['sigma'][phase_start_iter:phase_end_iter].copy(),
                'minibatch': schedules['minibatch'][phase_start_iter:phase_end_iter].copy(),
            }
            
            # Build tasks for active workers only
            tasks = []
            task_worker_map = []  # Maps task index to worker_id
            
            for i, worker_id in enumerate(active_worker_ids[:n_active_workers]):
                x0_dict = current_positions[worker_id]
                x0_cont, x0_cat, cat_n_values = self.space.to_split_vectors(x0_dict)
                
                task = _WorkerTask(
                    x0_cont=x0_cont,
                    x0_cat=x0_cat,
                    cat_n_values=cat_n_values,
                    bounds=self.space.get_bounds_array(),
                    max_iter=phase_iters,
                    lambda_schedule=phase_schedules['lambda'],
                    mu_schedule=phase_schedules['mu'],
                    sigma_schedule=phase_schedules['sigma'],
                    minibatch_schedule=phase_schedules['minibatch'],
                    use_minibatch=use_minibatch,
                    top_n_fraction=worker_strategies[worker_id]['top_n_fraction'],
                    adam_lr=adam_lr,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    adam_epsilon=adam_epsilon,
                    shrink_factor=shrink_factor,
                    shrink_patience=shrink_patience,
                    shrink_threshold=shrink_threshold,
                    use_improvement_weights=use_improvement_weights,
                    random_seed=(self.random_state + worker_id + phase_idx * 1000) if self.random_state else (worker_id + phase_idx * 1000),
                    worker_id=worker_id,
                    weight_decay=weight_decay,
                    early_stop_threshold=early_stop_threshold,
                    early_stop_patience=early_stop_patience,
                    objective=objective_wrapper.objective,
                    space_params=[
                        {
                            'name': p.name,
                            'type': p.type,
                            'bounds': list(p.bounds) if p.bounds else None,
                            'values': list(p.values) if p.values else None,
                            'log': p.log
                        }
                        for p in self.space.parameters
                    ],
                    direction=self.direction
                )
                tasks.append(task)
                task_worker_map.append(worker_id)
            
            # Run this phase in waves if needed
            phase_results = []
            n_waves = math.ceil(len(tasks) / self.max_parallel_workers)
            
            for wave_idx in range(n_waves):
                wave_start = wave_idx * self.max_parallel_workers
                wave_end = min(wave_start + self.max_parallel_workers, len(tasks))
                wave_tasks = tasks[wave_start:wave_end]
                
                executor = get_reusable_executor(max_workers=len(wave_tasks))
                futures = [executor.submit(_run_worker_task_loky, task) for task in wave_tasks]
                
                for future in futures:
                    phase_results.append(future.result(timeout=3600))
            
            # Process phase results
            worker_results_this_phase = []
            for task_idx, result in enumerate(phase_results):
                x_best_cont, x_best_cat, f_best, history = result
                worker_id = task_worker_map[task_idx]
                
                best_params = self.space.from_split_vectors(x_best_cont, x_best_cat)
                
                worker_results_this_phase.append({
                    'worker_id': worker_id,
                    'f_best': f_best,
                    'best_params': best_params,
                    'x_best_cont': x_best_cont,
                    'x_best_cat': x_best_cat,
                    'history': history,
                })
                
                # Update current position to best found
                current_positions[worker_id] = best_params.copy()
                
                # Track in worker history
                worker_histories[worker_id]['phases'].append({
                    'phase': phase_idx,
                    'f_best': f_best,
                    'history': history,
                })
                
                # Update global best
                if f_best < global_best_value:
                    global_best_value = f_best
                    global_best_params = best_params.copy()
                    global_best_worker = worker_id
            
            all_phase_results.extend(worker_results_this_phase)
            
            # Print phase summary if verbose
            if verbose:
                phase_best = min(worker_results_this_phase, key=lambda x: x['f_best'])
                print(f"Phase {phase_idx + 1}/{n_phases}: "
                      f"workers={len(worker_results_this_phase)}, "
                      f"phase_best={phase_best['f_best']:.6f} (worker {phase_best['worker_id']}), "
                      f"global_best={global_best_value:.6f}")
            
            # Skip elite selection on last phase
            if phase_idx >= n_phases - 1:
                break
            
            # Elite selection and restart
            sorted_workers = sorted(worker_results_this_phase, key=lambda x: x['f_best'])
            n_elite = max(1, int(len(sorted_workers) * elite_fraction))
            
            elite_workers = sorted_workers[:n_elite]
            non_elite_workers = sorted_workers[n_elite:]
            
            # Determine restart probability based on mode and progress
            if restart_mode == 'elite':
                restart_from_elite_prob = 1.0
            elif restart_mode == 'random':
                restart_from_elite_prob = 0.0
            else:  # adaptive
                restart_from_elite_prob = (restart_elite_prob_start + 
                    (restart_elite_prob_end - restart_elite_prob_start) * phase_progress)
            
            # Get current sigma for perturbation
            current_sigma = schedules['sigma'][min(phase_end_iter, len(schedules['sigma']) - 1)]
            
            # Restart non-elite workers
            for worker_info in non_elite_workers:
                worker_id = worker_info['worker_id']
                
                if np.random.random() < restart_from_elite_prob:
                    # Restart from elite position with perturbation
                    donor = np.random.choice(elite_workers)
                    donor_params = donor['best_params'].copy()
                    
                    # Add perturbation
                    new_params = {}
                    for param in self.space.parameters:
                        if param.type == 'categorical':
                            # Occasionally mutate categorical
                            if np.random.random() < 0.1:
                                new_params[param.name] = np.random.choice(param.values)
                            else:
                                new_params[param.name] = donor_params[param.name]
                        else:
                            # Add Gaussian noise
                            unit_val = param.transform_to_unit(donor_params[param.name])
                            noise = np.random.randn() * current_sigma * 0.5
                            noisy_unit = np.clip(unit_val + noise, 0, 1)
                            new_params[param.name] = param.transform_from_unit(noisy_unit)
                    
                    current_positions[worker_id] = new_params
                else:
                    # Random restart
                    current_positions[worker_id] = self.space.sample(n=1, method='random')[0]
            
            # Handle worker decay - update active worker list
            if enable_worker_decay and n_active_workers < len(active_worker_ids):
                # Keep only the top n_active_workers
                active_worker_ids = [w['worker_id'] for w in sorted_workers[:n_active_workers]]
                
                # Mark retired workers
                for worker_info in sorted_workers[n_active_workers:]:
                    worker_id = worker_info['worker_id']
                    worker_histories[worker_id]['active'] = False
                    worker_histories[worker_id]['retired_at'] = phase_idx
        
        # Build final results in the format expected by the caller
        # Combine all phase results into worker-centric view
        final_results = []
        
        for worker_id in range(self.n_workers):
            if not worker_histories[worker_id]['phases']:
                continue
            
            # Get the last phase result for this worker
            last_phase = worker_histories[worker_id]['phases'][-1]
            
            # Combine histories from all phases
            combined_history = {
                'fitness': [],
                'sigma': [],
                'shrink_events': [],
                'sync_events': [],
                'batch_sizes': [],
                'params_history': [],
                'worker_id': worker_id,
                'converged': last_phase['history'].get('converged', False),
                'final_iteration': 0,
            }
            
            iter_offset = 0
            for phase_data in worker_histories[worker_id]['phases']:
                hist = phase_data['history']
                combined_history['fitness'].extend(hist.get('fitness', []))
                combined_history['sigma'].extend(hist.get('sigma', []))
                combined_history['shrink_events'].extend([e + iter_offset for e in hist.get('shrink_events', [])])
                combined_history['sync_events'].extend([e + iter_offset for e in hist.get('sync_events', [])])
                combined_history['batch_sizes'].extend(hist.get('batch_sizes', []))
                combined_history['params_history'].extend(hist.get('params_history', []))
                iter_offset += len(hist.get('fitness', []))
            
            combined_history['final_iteration'] = iter_offset - 1 if iter_offset > 0 else 0
            
            # Get best result for this worker across all phases
            best_phase = min(worker_histories[worker_id]['phases'], 
                           key=lambda p: p['f_best'])
            
            x_best_cont, x_best_cat, _ = self.space.to_split_vectors(
                current_positions[worker_id]
            )
            
            final_results.append((
                x_best_cont,
                x_best_cat,
                best_phase['f_best'],
                combined_history
            ))
        
        return final_results


# =============================================================================
# Convenience Function (scipy-style)
# =============================================================================
def ragda_optimize(
    objective: Callable,
    bounds: np.ndarray,
    x0: Optional[np.ndarray] = None,
    direction: Literal['minimize', 'maximize'] = 'minimize',
    n_trials: int = 1000,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Scipy-style convenience function for continuous-only problems.
    
    Parameters
    ----------
    objective : callable
        Function f(x) -> float where x is ndarray
    bounds : ndarray, shape (n_dims, 2)
        [[lower0, upper0], [lower1, upper1], ...]
    x0 : ndarray, optional
        Starting point
    direction : str
        'minimize' or 'maximize'
    n_trials : int
        Number of iterations
    random_state : int, optional
        Random seed
    **kwargs
        Additional arguments passed to optimize()
    
    Returns
    -------
    x_best : ndarray
        Best solution found
    f_best : float
        Best objective value
    info : dict
        Optimization info
    """
    if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be ndarray of shape (n_dims, 2)")
    
    space = [
        {'name': f'x{i}', 'type': 'continuous', 'bounds': [float(bounds[i, 0]), float(bounds[i, 1])]}
        for i in range(len(bounds))
    ]
    
    for i, (lower, upper) in enumerate(bounds):
        if lower >= upper:
            raise ValueError(f"Invalid bounds for dim {i}: lower ({lower}) >= upper ({upper})")
    
    def objective_dict(params):
        x = np.array([params[f'x{i}'] for i in range(len(bounds))])
        return objective(x)
    
    x0_dict = None
    if x0 is not None:
        if len(x0) != len(bounds):
            raise ValueError(f"x0 length ({len(x0)}) must match bounds length ({len(bounds)})")
        x0_dict = {f'x{i}': float(x0[i]) for i in range(len(x0))}
    
    optimizer = RAGDAOptimizer(space, direction, random_state=random_state)
    result = optimizer.optimize(objective_dict, n_trials, x0_dict, **kwargs)
    
    x_best = np.array([result.best_params[f'x{i}'] for i in range(len(bounds))])
    
    return x_best, result.best_value, result.optimization_params
