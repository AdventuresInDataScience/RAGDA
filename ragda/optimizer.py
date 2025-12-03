"""
RAGDA Optimizer - High-Performance Derivative-Free Optimization

Pure Cython/C implementation with:
- Parallel multi-worker exploration
- ADAM-based pseudo-gradient descent
- Mixed variable types (continuous, ordinal, categorical)
- Mini-batch evaluation for data-driven objectives
- Early stopping and convergence detection
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, List, Union, Tuple, Literal
from multiprocessing import cpu_count
import warnings
import inspect

# Use loky for robust Windows multiprocessing (handles cloudpickle)
from loky import get_reusable_executor

from .space import SearchSpace
from .result import OptimizationResult, Trial
from . import core  # Pure Cython - no fallback


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
    
    Algorithm:
    1. W workers initialized with different top_n fractions
    2. Each worker samples N points around current position
    3. Only IMPROVING samples (f < f_current) contribute to gradient
    4. Gradient = weighted sum of directions to improving samples
    5. Weight decay applied by rank (best = 1.0, rank i = decay^i)
    6. ADAM update determines step size
    7. Every sync_frequency iters, ALL workers reset to global best
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
    """
    
    __slots__ = ('space', 'direction', 'random_state', 'n_workers')
    
    def __init__(
        self,
        space: List[Dict[str, Any]],
        direction: Literal['minimize', 'maximize'] = 'minimize',
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None
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
            Number of parallel workers. Default: CPU_count // 2
        random_state : int, optional
            Random seed for reproducibility
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
            How often workers sync to global best (0 = never)
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
        
        if not callable(objective):
            raise ValueError("objective must be callable")
        
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
        
        # Print config
        if verbose:
            print(f"{'='*70}")
            print(f"RAGDA Optimization (Cython {core.get_version()})")
            print(f"{'='*70}")
            print(f"Search space: {self.space.n_params} parameters")
            print(f"  - Continuous/Ordinal: {self.space.n_continuous}")
            print(f"  - Categorical: {self.space.n_categorical}")
            print(f"Direction: {self.direction}")
            print(f"Workers: {self.n_workers} (top_n: {top_n_max:.0%} -> {top_n_min:.0%})")
            print(f"Iterations per worker: {n_trials}")
            print(f"Total evaluations: ~{n_trials * self.n_workers * (lambda_start + lambda_end) // 2:,}")
            print(f"Batch size: lambda = {lambda_start} -> {lambda_end}")
            print(f"Sample space: sigma = {sigma_init:.3f} -> {sigma_init * sigma_final_fraction:.3f}")
            print(f"Shrinking: factor={shrink_factor}, patience={shrink_patience}")
            print(f"Weighting: only_improving={use_improvement_weights}, weight_decay={weight_decay}")
            print(f"Early stopping: threshold={early_stop_threshold}, patience={early_stop_patience}")
            if sync_frequency > 0:
                print(f"Sync: every {sync_frequency} iters (all workers reset to best)")
            if use_minibatch:
                print(f"Mini-batch: {minibatch_start} -> {minibatch_end} ({minibatch_schedule})")
            print(f"ADAM: lr={adam_learning_rate}, beta1={adam_beta1}, beta2={adam_beta2}")
            print(f"{'='*70}\n")
        
        # Run workers
        worker_results = self._run_parallel(
            x0_list, objective_wrapper, schedules, worker_strategies,
            n_trials, shrink_factor, shrink_patience, shrink_threshold,
            use_improvement_weights, use_minibatch,
            adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
            weight_decay, early_stop_threshold, early_stop_patience, verbose
        )
        
        # Process results
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
        
        # Re-evaluate final best with FULL sample (no minibatch) for accurate final loss
        # This is critical for curriculum learning / minibatch scenarios
        if use_minibatch:
            try:
                # Check if objective supports batch_size parameter
                sig = inspect.signature(objective)
                if 'batch_size' in sig.parameters:
                    # Call with batch_size=-1 or None to use full sample
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
            optimization_params={
                'n_trials': n_trials,
                'lambda_start': lambda_start,
                'lambda_end': lambda_end,
                'sigma_init': sigma_init,
                'use_minibatch': use_minibatch,
                'sync_frequency': sync_frequency,
                'adam_learning_rate': adam_learning_rate,
            }
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
        """Run workers in parallel using loky (robust Windows multiprocessing)."""
        
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
        
        # Use loky for robust parallel execution (handles cloudpickle)
        executor = get_reusable_executor(max_workers=self.n_workers)
        futures = [executor.submit(_run_worker_task_loky, task) for task in tasks]
        
        # Collect results
        results = []
        for future in futures:
            results.append(future.result(timeout=3600))  # 1 hour timeout
        
        return results


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
