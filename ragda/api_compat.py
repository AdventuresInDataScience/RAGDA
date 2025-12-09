"""
API Compatibility Layer for RAGDA

Provides Optuna-style and Scipy-style high-level functions that wrap
RAGDAOptimizer with appropriate adapters.
"""

from typing import Callable, List, Tuple, Optional, Any, Dict
import warnings

from .optimizer import RAGDAOptimizer
from .api_adapters import OptunaAdapter, ScipyAdapter
from .result import OptimizationResult


# =============================================================================
# Optuna-Style API
# =============================================================================

class Study:
    """
    Optuna-compatible Study object for RAGDA.
    
    Usage:
        study = create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        print(study.best_value)
        print(study.best_params)
    """
    
    def __init__(
        self,
        direction: str = 'minimize',
        sampler: Optional[Any] = None,
        random_state: Optional[int] = None,
        **optimizer_kwargs
    ):
        """
        Create a study for hyperparameter optimization.
        
        Parameters
        ----------
        direction : str, default='minimize'
            'minimize' or 'maximize'
        sampler : optional
            Ignored (RAGDA uses its own sampling strategy)
        random_state : int, optional
            Random seed for reproducibility
        **optimizer_kwargs
            Additional arguments passed to RAGDAOptimizer
        """
        self.direction = direction
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        
        self._adapter = OptunaAdapter()
        self._optimizer = None
        self._result = None
        
        if sampler is not None:
            warnings.warn(
                "Custom samplers are not supported. RAGDA uses its own optimization strategy.",
                UserWarning
            )
    
    def optimize(
        self,
        objective: Callable,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = False
    ) -> None:
        """
        Optimize the objective function.
        
        Parameters
        ----------
        objective : callable
            Objective function that takes a Trial object:
            def objective(trial):
                x = trial.suggest_float('x', -5, 5)
                return x**2
        n_trials : int, default=100
            Number of trials
        n_jobs : int, default=1
            Number of parallel workers
        timeout : float, optional
            Time budget in seconds (not implemented)
        callbacks : list, optional
            Not implemented
        show_progress_bar : bool, default=False
            Not implemented
        """
        # Discover space by running objective once
        canonical_space = self._adapter.discover_space(objective)
        
        # Wrap objective
        canonical_objective = self._adapter.wrap_objective(objective)
        
        # Create optimizer
        self._optimizer = RAGDAOptimizer(
            space=canonical_space,
            direction=self.direction,
            random_state=self.random_state,
            n_workers=n_jobs,
            **self.optimizer_kwargs
        )
        
        # Run optimization
        self._result = self._optimizer.optimize(
            objective=canonical_objective,
            n_trials=n_trials
        )
    
    @property
    def best_value(self) -> float:
        """Best objective value found."""
        if self._result is None:
            raise RuntimeError("optimize() has not been called yet.")
        return self._result.best_value
    
    @property
    def best_params(self) -> Dict[str, Any]:
        """Best parameters found."""
        if self._result is None:
            raise RuntimeError("optimize() has not been called yet.")
        # Post-process to convert integer parameters back to int type
        return self._adapter.postprocess_params(self._result.best_params)
    
    @property
    def best_trial(self):
        """Best trial object (simplified)."""
        if self._result is None:
            raise RuntimeError("optimize() has not been called yet.")
        
        # Create a simple namespace to hold best trial info
        class BestTrial:
            def __init__(self, params, value):
                self.params = params
                self.value = value
                self.number = 0  # Not tracked
        
        # Use post-processed params to ensure correct types
        return BestTrial(self.best_params, self.best_value)
    
    @property
    def trials(self):
        """All trials (returns simplified list)."""
        if self._result is None:
            raise RuntimeError("optimize() has not been called yet.")
        
        # Return simplified trial-like objects
        class SimpleTrial:
            def __init__(self, params, value, number):
                self.params = params
                self.value = value
                self.number = number
        
        # Handle Trial objects or dicts
        result_trials = []
        for i, trial in enumerate(self._result.trials):
            if hasattr(trial, 'params'):
                # It's a Trial dataclass
                result_trials.append(SimpleTrial(
                    params=trial.params,
                    value=trial.value,
                    number=i
                ))
            else:
                # It's a dict (shouldn't happen but be defensive)
                result_trials.append(SimpleTrial(
                    params=trial.get('params', {}),
                    value=trial.get('value', 0.0),
                    number=i
                ))
        
        return result_trials


def create_study(
    direction: str = 'minimize',
    sampler: Optional[Any] = None,
    random_state: Optional[int] = None,
    **optimizer_kwargs
) -> Study:
    """
    Create a study for hyperparameter optimization (Optuna-compatible).
    
    Parameters
    ----------
    direction : str, default='minimize'
        'minimize' or 'maximize'
    sampler : optional
        Ignored (RAGDA uses its own sampling strategy)
    random_state : int, optional
        Random seed for reproducibility
    **optimizer_kwargs
        Additional arguments passed to RAGDAOptimizer
    
    Returns
    -------
    Study
        Study object for optimization
    
    Examples
    --------
    >>> def objective(trial):
    ...     x = trial.suggest_float('x', -5, 5)
    ...     y = trial.suggest_float('y', -5, 5)
    ...     return x**2 + y**2
    >>> 
    >>> study = create_study(direction='minimize', random_state=42)
    >>> study.optimize(objective, n_trials=100)
    >>> print(study.best_value)
    >>> print(study.best_params)
    """
    return Study(
        direction=direction,
        sampler=sampler,
        random_state=random_state,
        **optimizer_kwargs
    )


# =============================================================================
# Scipy-Style API
# =============================================================================

def minimize(
    fun: Callable,
    bounds: List[Tuple[float, float]],
    method: str = 'ragda',
    options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None
):
    """
    Minimize a function (scipy.optimize.minimize compatible).
    
    Parameters
    ----------
    fun : callable
        Objective function to minimize. Takes array x and returns scalar.
        def fun(x):
            return x[0]**2 + x[1]**2
    bounds : list of tuples
        Bounds for each parameter: [(low, high), (low, high), ...]
    method : str, default='ragda'
        Only 'ragda' is supported
    options : dict, optional
        Options for the optimizer:
        - 'maxiter': maximum iterations (default: 100)
        - 'random_state': random seed
        - 'n_workers': number of parallel workers
        - Other RAGDAOptimizer parameters
    callback : callable, optional
        Not implemented
    
    Returns
    -------
    OptimizeResult
        Scipy-compatible result object with attributes:
        - x: best parameters (array)
        - fun: best value (float)
        - success: True
        - nit: number of iterations
        - nfev: number of function evaluations
    
    Examples
    --------
    >>> def sphere(x):
    ...     return sum(x**2)
    >>> 
    >>> result = minimize(
    ...     sphere,
    ...     bounds=[(-5, 5), (-5, 5)],
    ...     options={'maxiter': 100, 'random_state': 42}
    ... )
    >>> print(result.x)
    >>> print(result.fun)
    """
    if method != 'ragda':
        raise ValueError(f"Only method='ragda' is supported, got: {method}")
    
    # Parse options
    if options is None:
        options = {}
    
    n_trials = options.pop('maxiter', 100)
    random_state = options.pop('random_state', None)
    n_workers = options.pop('n_workers', 1)
    
    # Create adapter
    adapter = ScipyAdapter()
    canonical_space = adapter.to_canonical_space(bounds)
    canonical_objective = adapter.wrap_objective(fun)
    
    # Create optimizer
    optimizer = RAGDAOptimizer(
        space=canonical_space,
        direction='minimize',
        random_state=random_state,
        n_workers=n_workers,
        **options
    )
    
    # Run optimization
    result = optimizer.optimize(
        objective=canonical_objective,
        n_trials=n_trials
    )
    
    # Convert to scipy format
    return adapter.wrap_result(result)


def maximize(
    fun: Callable,
    bounds: List[Tuple[float, float]],
    method: str = 'ragda',
    options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None
):
    """
    Maximize a function (scipy-compatible).
    
    Parameters
    ----------
    fun : callable
        Objective function to maximize. Takes array x and returns scalar.
    bounds : list of tuples
        Bounds for each parameter: [(low, high), (low, high), ...]
    method : str, default='ragda'
        Only 'ragda' is supported
    options : dict, optional
        Options for the optimizer (see minimize())
    callback : callable, optional
        Not implemented
    
    Returns
    -------
    OptimizeResult
        Scipy-compatible result object
    
    Examples
    --------
    >>> def neg_sphere(x):
    ...     return -sum(x**2)
    >>> 
    >>> result = maximize(
    ...     neg_sphere,
    ...     bounds=[(-5, 5), (-5, 5)],
    ...     options={'maxiter': 100}
    ... )
    """
    if method != 'ragda':
        raise ValueError(f"Only method='ragda' is supported, got: {method}")
    
    # Parse options
    if options is None:
        options = {}
    
    n_trials = options.pop('maxiter', 100)
    random_state = options.pop('random_state', None)
    n_workers = options.pop('n_workers', 1)
    
    # Create adapter
    adapter = ScipyAdapter()
    canonical_space = adapter.to_canonical_space(bounds)
    canonical_objective = adapter.wrap_objective(fun)
    
    # Create optimizer
    optimizer = RAGDAOptimizer(
        space=canonical_space,
        direction='maximize',  # Note: maximize
        random_state=random_state,
        n_workers=n_workers,
        **options
    )
    
    # Run optimization
    result = optimizer.optimize(
        objective=canonical_objective,
        n_trials=n_trials
    )
    
    # Convert to scipy format
    return adapter.wrap_result(result)
