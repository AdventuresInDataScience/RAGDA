"""
RAGDA High-Dimensional Optimizer

Two-stage optimization for ultra-high-dimensional problems:
1. Stage 1: Dimensionality reduction + optimization in reduced space
2. Stage 2: Local refinement in full space within tight trust region

Supports:
- Effective dimensionality testing (eigenvalue analysis)
- Kernel PCA for nonlinear dimensionality reduction
- Random projections (Johnson-Lindenstrauss) as fast alternative
- Incremental PCA for adaptive refinement
- Adaptive variance threshold based on optimization progress
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, List, Union, Tuple, Literal
from multiprocessing import cpu_count
import warnings

from .space import SearchSpace
from .result import OptimizationResult, Trial
from .optimizer import RAGDAOptimizer, _validate_parameters

# Import Cython high-dim core
try:
    from . import highdim_core
    HIGHDIM_AVAILABLE = True
except ImportError:
    HIGHDIM_AVAILABLE = False
    warnings.warn(
        "highdim_core not available. Run 'python setup.py build_ext --inplace' to build.",
        RuntimeWarning
    )


# =============================================================================
# Dimensionality Reduction Wrapper
# =============================================================================

class DimensionalityReducer:
    """
    Unified interface for dimensionality reduction methods.
    
    Supports:
    - Kernel PCA (for nonlinear structure)
    - Incremental PCA (linear, fast, updatable)
    - Random Projections (fastest, no fitting required)
    """
    
    __slots__ = (
        'method', 'n_components', 'variance_threshold', 'state',
        'original_dim', 'reduced_dim', '_fitted', 'adaptive_gamma',
        'random_seed', 'projection_type'
    )
    
    def __init__(
        self,
        method: Literal['kernel_pca', 'incremental_pca', 'random_projection'] = 'kernel_pca',
        n_components: int = 0,
        variance_threshold: float = 0.90,
        adaptive_gamma: bool = True,
        projection_type: str = 'gaussian',
        random_seed: Optional[int] = None
    ):
        """
        Initialize dimensionality reducer.
        
        Parameters
        ----------
        method : str
            'kernel_pca' - Kernel PCA with RBF kernel (best for nonlinear)
            'incremental_pca' - Linear PCA, can be updated online
            'random_projection' - Fastest, no fitting needed
        n_components : int
            Target dimensions. If 0, determined by variance_threshold.
        variance_threshold : float
            Fraction of variance to preserve (used if n_components=0)
        adaptive_gamma : bool
            For kernel_pca: use median distance heuristic for gamma
        projection_type : str
            For random_projection: 'gaussian', 'sparse', or 'rademacher'
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.adaptive_gamma = adaptive_gamma
        self.projection_type = projection_type
        self.random_seed = random_seed or 0
        
        self.state = None
        self.original_dim = 0
        self.reduced_dim = 0
        self._fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit the dimensionality reducer.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data
        
        Returns
        -------
        self
        """
        if not HIGHDIM_AVAILABLE:
            raise RuntimeError("highdim_core not available. Build with setup.py first.")
        
        self.original_dim = X.shape[1]
        
        if self.method == 'kernel_pca':
            self.state = highdim_core.fit_kernel_pca(
                X,
                n_components=self.n_components,
                variance_threshold=self.variance_threshold,
                gamma=0.0,  # Auto
                adaptive_gamma=self.adaptive_gamma
            )
            self.reduced_dim = self.state.n_components
        
        elif self.method == 'incremental_pca':
            self.state = highdim_core.fit_incremental_pca(
                X,
                n_components=self.n_components,
                variance_threshold=self.variance_threshold
            )
            self.reduced_dim = self.state.n_components
        
        elif self.method == 'random_projection':
            # For random projection, we need to know target dim
            if self.n_components > 0:
                n_comp = self.n_components
            else:
                # Estimate based on variance threshold
                # Use eigenvalue analysis on sample
                eigenvalues = highdim_core.compute_eigenvalues_fast(X)
                result = highdim_core.estimate_effective_dimensionality(
                    eigenvalues,
                    variance_threshold=self.variance_threshold
                )
                n_comp = result['effective_dim']
            
            self.state = highdim_core.fit_random_projection(
                self.original_dim,
                n_comp,
                projection_type=self.projection_type,
                random_seed=self.random_seed
            )
            self.reduced_dim = n_comp
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to reduced space.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data in original space
        
        Returns
        -------
        X_reduced : ndarray, shape (n_samples, n_components)
            Data in reduced space
        """
        if not self._fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        if self.method == 'kernel_pca':
            return highdim_core.transform_kernel_pca(self.state, X)
        elif self.method == 'incremental_pca':
            return highdim_core.transform_incremental_pca(self.state, X)
        elif self.method == 'random_projection':
            return highdim_core.transform_random_projection(self.state, X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Transform back from reduced space to original space.
        
        Note: For Kernel PCA and Random Projections, this is an approximation.
        
        Parameters
        ----------
        X_reduced : ndarray, shape (n_samples, n_components)
            Data in reduced space
        
        Returns
        -------
        X_original : ndarray, shape (n_samples, n_features)
            Approximate reconstruction in original space
        """
        if not self._fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        if self.method == 'kernel_pca':
            return highdim_core.inverse_transform_kernel_pca(self.state, X_reduced)
        elif self.method == 'incremental_pca':
            return highdim_core.inverse_transform_incremental_pca(self.state, X_reduced)
        elif self.method == 'random_projection':
            return highdim_core.inverse_transform_random_projection(self.state, X_reduced)
    
    def partial_fit(self, X: np.ndarray):
        """
        Update the reducer with new samples (only for incremental_pca).
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            New samples
        
        Returns
        -------
        self
        """
        if self.method != 'incremental_pca':
            raise ValueError("partial_fit only available for incremental_pca")
        
        if not self._fitted:
            return self.fit(X)
        
        self.state = highdim_core.partial_fit_incremental_pca(self.state, X)
        return self
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of reduced to original dimensions."""
        if self.original_dim == 0:
            return 1.0
        return self.reduced_dim / self.original_dim


# =============================================================================
# High-Dimensional Optimizer
# =============================================================================

class HighDimRAGDAOptimizer:
    """
    RAGDA Optimizer for ultra-high-dimensional problems.
    
    Uses a two-stage approach:
    1. Stage 1: Dimensionality reduction + optimization in reduced space
    2. Stage 2: Local refinement in full space within tight trust region
    
    Automatically determines effective dimensionality and selects
    appropriate reduction method.
    
    Example:
        >>> from ragda import HighDimRAGDAOptimizer
        >>> 
        >>> def objective(params):
        ...     x = np.array([params[f'x{i}'] for i in range(10000)])
        ...     return np.sum(x**2)  # 10000-dimensional sphere
        >>> 
        >>> space = [
        ...     {'name': f'x{i}', 'type': 'continuous', 'bounds': [-5, 5]}
        ...     for i in range(10000)
        ... ]
        >>> 
        >>> optimizer = HighDimRAGDAOptimizer(space, direction='minimize')
        >>> result = optimizer.optimize(objective, n_trials=500)
    """
    
    __slots__ = (
        'space', 'direction', 'random_state', 'n_workers',
        'dim_threshold', 'variance_threshold', 'reduction_method',
        'adaptive_threshold', 'min_variance_threshold', 'max_variance_threshold',
        'trust_region_fraction', 'stage2_trials_fraction',
        'initial_samples', 'refit_frequency', '_effective_dim_info'
    )
    
    def __init__(
        self,
        space: List[Dict[str, Any]],
        direction: Literal['minimize', 'maximize'] = 'minimize',
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        # High-dim specific parameters
        dim_threshold: int = 1000,
        variance_threshold: float = 0.90,
        reduction_method: Literal['kernel_pca', 'incremental_pca', 'random_projection', 'auto'] = 'auto',
        adaptive_threshold: bool = True,
        min_variance_threshold: float = 0.70,
        max_variance_threshold: float = 0.99,
        trust_region_fraction: float = 0.1,
        stage2_trials_fraction: float = 0.2,
        initial_samples: int = 500,
        refit_frequency: int = 100
    ):
        """
        Initialize high-dimensional RAGDA optimizer.
        
        Parameters
        ----------
        space : list of dict
            Search space definition (same as RAGDAOptimizer)
        direction : str
            'minimize' or 'maximize'
        n_workers : int, optional
            Number of parallel workers
        random_state : int, optional
            Random seed
        dim_threshold : int
            Dimensions above this trigger high-dim mode
        variance_threshold : float
            Base variance threshold for dimensionality reduction
        reduction_method : str
            'kernel_pca' - Best for nonlinear structure
            'incremental_pca' - Fast, linear, updatable
            'random_projection' - Fastest, random
            'auto' - Choose based on effective dimensionality test
        adaptive_threshold : bool
            Adjust variance threshold during optimization
        min_variance_threshold : float
            Minimum variance threshold (early optimization)
        max_variance_threshold : float
            Maximum variance threshold (late optimization)
        trust_region_fraction : float
            Fraction of space for stage 2 trust region
        stage2_trials_fraction : float
            Fraction of total trials for stage 2 refinement
        initial_samples : int
            Number of initial samples for dimensionality analysis
        refit_frequency : int
            How often to refit the reducer (0 = never refit)
        """
        if len(space) == 0:
            raise ValueError("Search space cannot be empty")
        
        self.space = SearchSpace(space)
        self.direction = direction
        self.random_state = random_state
        
        if n_workers is None:
            n_workers = max(1, cpu_count() // 2)
        self.n_workers = n_workers
        
        # High-dim parameters
        self.dim_threshold = dim_threshold
        self.variance_threshold = variance_threshold
        self.reduction_method = reduction_method
        self.adaptive_threshold = adaptive_threshold
        self.min_variance_threshold = min_variance_threshold
        self.max_variance_threshold = max_variance_threshold
        self.trust_region_fraction = trust_region_fraction
        self.stage2_trials_fraction = stage2_trials_fraction
        self.initial_samples = initial_samples
        self.refit_frequency = refit_frequency
        
        self._effective_dim_info = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def analyze_effective_dimensionality(
        self,
        objective: Optional[Callable] = None,
        n_samples: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze effective dimensionality of the search space.
        
        Parameters
        ----------
        objective : callable, optional
            If provided, samples are generated and objective is evaluated
            to better understand the function's effective dimensionality.
        n_samples : int
            Number of samples for analysis (0 = use self.initial_samples)
        
        Returns
        -------
        info : dict
            effective_dim : int
            variance_explained : ndarray
            participation_ratio : float
            is_low_dimensional : bool
            confidence : float
            recommended_method : str
            sample_points : ndarray (if objective provided)
            sample_values : ndarray (if objective provided)
        """
        if not HIGHDIM_AVAILABLE:
            raise RuntimeError("highdim_core not available")
        
        n_samples = n_samples or self.initial_samples
        n_cont = self.space.n_continuous
        
        if n_cont == 0:
            return {
                'effective_dim': self.space.n_categorical,
                'variance_explained': np.array([1.0]),
                'participation_ratio': 1.0,
                'is_low_dimensional': False,
                'confidence': 0.0,
                'recommended_method': 'none'
            }
        
        # Generate samples using LHS
        sample_dicts = self.space.sample(n=n_samples, method='lhs')
        
        # Convert to continuous array
        X = np.zeros((n_samples, n_cont), dtype=np.float64)
        for i, params in enumerate(sample_dicts):
            x_cont, _, _ = self.space.to_split_vectors(params)
            X[i] = x_cont
        
        # Compute eigenvalues
        eigenvalues = highdim_core.compute_eigenvalues_fast(X)
        
        # Estimate effective dimensionality
        info = highdim_core.estimate_effective_dimensionality(
            eigenvalues,
            variance_threshold=self.variance_threshold
        )
        
        # Recommend method based on analysis
        if info['is_low_dimensional']:
            if info['participation_ratio'] < 0.2:
                # Very concentrated - linear PCA sufficient
                info['recommended_method'] = 'incremental_pca'
            elif info['confidence'] > 0.7:
                # High confidence in low-dim structure - kernel PCA for nonlinear
                info['recommended_method'] = 'kernel_pca'
            else:
                # Moderate confidence - random projection safer
                info['recommended_method'] = 'random_projection'
        else:
            info['recommended_method'] = 'none'
        
        # Optionally evaluate objective to understand function structure
        if objective is not None:
            values = np.zeros(n_samples, dtype=np.float64)
            for i, params in enumerate(sample_dicts):
                try:
                    values[i] = float(objective(params))
                except:
                    values[i] = 1e10
            
            info['sample_points'] = X
            info['sample_values'] = values
            
            # Compute gradient-based effective dimensionality estimate
            # Using finite differences on a subset
            if n_samples > 10:
                info['gradient_analysis'] = self._analyze_gradient_structure(
                    objective, sample_dicts[:min(20, n_samples)], values[:min(20, n_samples)]
                )
        
        self._effective_dim_info = info
        return info
    
    def _analyze_gradient_structure(
        self,
        objective: Callable,
        sample_dicts: List[Dict],
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze gradient structure to understand sensitivity."""
        n_samples = len(sample_dicts)
        n_cont = self.space.n_continuous
        
        # Compute finite difference gradients at sample points
        gradients = np.zeros((n_samples, n_cont), dtype=np.float64)
        epsilon = 1e-4
        
        for i, params in enumerate(sample_dicts):
            x_cont, x_cat, _ = self.space.to_split_vectors(params)
            
            for j in range(n_cont):
                x_plus = x_cont.copy()
                x_minus = x_cont.copy()
                x_plus[j] = min(1.0, x_cont[j] + epsilon)
                x_minus[j] = max(0.0, x_cont[j] - epsilon)
                
                params_plus = self.space.from_split_vectors(x_plus, x_cat)
                params_minus = self.space.from_split_vectors(x_minus, x_cat)
                
                try:
                    f_plus = float(objective(params_plus))
                    f_minus = float(objective(params_minus))
                    gradients[i, j] = (f_plus - f_minus) / (x_plus[j] - x_minus[j] + 1e-10)
                except:
                    gradients[i, j] = 0.0
        
        # Analyze gradient covariance
        grad_cov = np.cov(gradients.T)
        if grad_cov.ndim == 0:
            grad_eigenvalues = np.array([grad_cov])
        else:
            grad_eigenvalues = np.linalg.eigvalsh(grad_cov)[::-1]
        
        return {
            'gradient_eigenvalues': grad_eigenvalues,
            'gradient_mean_magnitude': np.mean(np.abs(gradients)),
            'active_dimensions': int(np.sum(np.mean(np.abs(gradients), axis=0) > 1e-6))
        }
    
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: int = 1000,
        x0: Optional[Union[Dict, List[Dict]]] = None,
        
        # Standard RAGDA parameters (passed to internal optimizer)
        lambda_start: int = 50,
        lambda_end: int = 10,
        lambda_decay_rate: float = 5.0,
        sigma_init: float = 0.3,
        sigma_final_fraction: float = 0.2,
        sigma_decay_schedule: Literal['exponential', 'linear', 'cosine'] = 'exponential',
        shrink_factor: float = 0.9,
        shrink_patience: int = 10,
        shrink_threshold: float = 1e-6,
        use_improvement_weights: bool = True,
        top_n_min: float = 0.2,
        top_n_max: float = 1.0,
        weight_decay: float = 0.95,
        sync_frequency: int = 100,
        use_minibatch: bool = False,
        data_size: Optional[int] = None,
        minibatch_start: Optional[int] = None,
        minibatch_end: Optional[int] = None,
        minibatch_schedule: Literal['constant', 'linear', 'exponential', 'inverse_decay', 'step'] = 'inverse_decay',
        adam_learning_rate: float = 0.001,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        early_stop_threshold: float = 1e-12,
        early_stop_patience: int = 50,
        
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run high-dimensional RAGDA optimization.
        
        Uses two-stage optimization:
        1. Stage 1: Optimize in reduced space (fast exploration)
        2. Stage 2: Refine in full space within trust region (fine-tuning)
        
        Parameters
        ----------
        objective : callable
            Function to optimize. Takes dict of params, returns float.
        n_trials : int
            Total number of iterations
        x0 : dict or list of dict, optional
            Starting point(s)
        ... (other parameters same as RAGDAOptimizer.optimize())
        
        Returns
        -------
        OptimizationResult
            Contains best_params, best_value, all trials, etc.
        """
        n_continuous = self.space.n_continuous
        
        # Check if high-dimensional mode is needed
        if n_continuous < self.dim_threshold:
            if verbose:
                print(f"Dimensions ({n_continuous}) below threshold ({self.dim_threshold}). "
                      f"Using standard RAGDA.")
            
            return self._run_standard_optimization(
                objective, n_trials, x0,
                lambda_start, lambda_end, lambda_decay_rate,
                sigma_init, sigma_final_fraction, sigma_decay_schedule,
                shrink_factor, shrink_patience, shrink_threshold,
                use_improvement_weights, top_n_min, top_n_max, weight_decay,
                sync_frequency, use_minibatch, data_size,
                minibatch_start, minibatch_end, minibatch_schedule,
                adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
                early_stop_threshold, early_stop_patience, verbose
            )
        
        if verbose:
            print(f"{'='*70}")
            print(f"RAGDA High-Dimensional Optimizer")
            print(f"{'='*70}")
            print(f"Original dimensions: {n_continuous}")
            print(f"Analyzing effective dimensionality...")
        
        # Step 1: Analyze effective dimensionality
        dim_info = self.analyze_effective_dimensionality(objective)
        
        if verbose:
            print(f"Effective dimensionality: {dim_info['effective_dim']}")
            print(f"Participation ratio: {dim_info['participation_ratio']:.3f}")
            print(f"Is low-dimensional: {dim_info['is_low_dimensional']}")
            print(f"Confidence: {dim_info['confidence']:.3f}")
            print(f"Recommended method: {dim_info['recommended_method']}")
        
        # Decide whether to use dimensionality reduction
        if not dim_info['is_low_dimensional'] or dim_info['confidence'] < 0.3:
            if verbose:
                print(f"\nLow-dimensional structure not detected. Using standard RAGDA.")
            
            return self._run_standard_optimization(
                objective, n_trials, x0,
                lambda_start, lambda_end, lambda_decay_rate,
                sigma_init, sigma_final_fraction, sigma_decay_schedule,
                shrink_factor, shrink_patience, shrink_threshold,
                use_improvement_weights, top_n_min, top_n_max, weight_decay,
                sync_frequency, use_minibatch, data_size,
                minibatch_start, minibatch_end, minibatch_schedule,
                adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon,
                early_stop_threshold, early_stop_patience, verbose
            )
        
        # Step 2: Select reduction method
        if self.reduction_method == 'auto':
            method = dim_info['recommended_method']
            if method == 'none':
                method = 'incremental_pca'  # Default fallback
        else:
            method = self.reduction_method
        
        if verbose:
            print(f"\nUsing reduction method: {method}")
            print(f"Target reduced dimensions: {dim_info['effective_dim']}")
        
        # Step 3: Create reducer and fit on initial samples
        reducer = DimensionalityReducer(
            method=method,
            n_components=dim_info['effective_dim'],
            variance_threshold=self.variance_threshold,
            random_seed=self.random_state
        )
        
        # Generate initial samples
        initial_sample_dicts = self.space.sample(n=self.initial_samples, method='lhs')
        X_initial = np.zeros((self.initial_samples, n_continuous), dtype=np.float64)
        
        for i, params in enumerate(initial_sample_dicts):
            x_cont, _, _ = self.space.to_split_vectors(params)
            X_initial[i] = x_cont
        
        reducer.fit(X_initial)
        
        if verbose:
            print(f"Reduced dimensions: {reducer.reduced_dim}")
            print(f"Compression ratio: {reducer.compression_ratio:.2%}")
            print(f"{'='*70}\n")
        
        # Step 4: Create reduced space
        reduced_space = self._create_reduced_space(reducer.reduced_dim)
        
        # Step 5: Calculate trial allocation
        stage1_trials = int(n_trials * (1 - self.stage2_trials_fraction))
        stage2_trials = n_trials - stage1_trials
        
        if verbose:
            print(f"Stage 1 (reduced space): {stage1_trials} trials")
            print(f"Stage 2 (full space refinement): {stage2_trials} trials")
            print()
        
        # Step 6: Run Stage 1 - Optimization in reduced space
        if verbose:
            print(f"{'='*70}")
            print(f"STAGE 1: Reduced Space Optimization")
            print(f"{'='*70}")
        
        # Create wrapped objective that works in reduced space
        wrapped_objective = self._create_reduced_objective(
            objective, reducer, self.space
        )
        
        stage1_optimizer = RAGDAOptimizer(
            reduced_space,
            direction=self.direction,
            n_workers=self.n_workers,
            random_state=self.random_state
        )
        
        stage1_result = stage1_optimizer.optimize(
            wrapped_objective,
            n_trials=stage1_trials,
            x0=None,  # Start fresh in reduced space
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
        
        # Convert stage 1 result back to original space
        stage1_best_reduced = np.array([
            stage1_result.best_params[f'z{i}'] for i in range(reducer.reduced_dim)
        ]).reshape(1, -1)
        stage1_best_full = reducer.inverse_transform(stage1_best_reduced)[0]
        
        # Clip to [0, 1] and convert to params
        stage1_best_full = np.clip(stage1_best_full, 0.0, 1.0)
        x_cat = np.zeros(self.space.n_categorical, dtype=np.int32)  # Default categorical
        stage1_best_params = self.space.from_split_vectors(stage1_best_full, x_cat)
        
        if verbose:
            print(f"\nStage 1 best value: {stage1_result.best_value:.6f}")
            print()
        
        # Step 7: Run Stage 2 - Local refinement in full space
        if stage2_trials > 0:
            if verbose:
                print(f"{'='*70}")
                print(f"STAGE 2: Full Space Refinement (Trust Region)")
                print(f"{'='*70}")
            
            # Create tight trust region around stage 1 solution
            trust_space = self._create_trust_region_space(
                stage1_best_full,
                self.trust_region_fraction
            )
            
            stage2_optimizer = RAGDAOptimizer(
                trust_space,
                direction=self.direction,
                n_workers=self.n_workers,
                random_state=self.random_state
            )
            
            # Use smaller sigma for local refinement
            local_sigma = sigma_init * 0.5
            
            stage2_result = stage2_optimizer.optimize(
                objective,
                n_trials=stage2_trials,
                x0=stage1_best_params,  # Start from stage 1 best
                lambda_start=max(10, lambda_start // 2),
                lambda_end=max(5, lambda_end // 2),
                lambda_decay_rate=lambda_decay_rate,
                sigma_init=local_sigma,
                sigma_final_fraction=sigma_final_fraction,
                sigma_decay_schedule=sigma_decay_schedule,
                shrink_factor=shrink_factor,
                shrink_patience=shrink_patience // 2,
                shrink_threshold=shrink_threshold,
                use_improvement_weights=use_improvement_weights,
                top_n_min=top_n_min,
                top_n_max=top_n_max,
                weight_decay=weight_decay,
                sync_frequency=sync_frequency,
                use_minibatch=use_minibatch,
                data_size=data_size,
                minibatch_start=minibatch_start,
                minibatch_end=minibatch_end,
                minibatch_schedule=minibatch_schedule,
                adam_learning_rate=adam_learning_rate * 0.5,  # Smaller LR
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                adam_epsilon=adam_epsilon,
                early_stop_threshold=early_stop_threshold,
                early_stop_patience=early_stop_patience,
                verbose=verbose
            )
            
            # Determine final best
            if self.direction == 'minimize':
                if stage2_result.best_value < stage1_result.best_value:
                    final_result = stage2_result
                    best_stage = 2
                else:
                    final_result = self._convert_stage1_result(
                        stage1_result, stage1_best_params, objective
                    )
                    best_stage = 1
            else:
                if stage2_result.best_value > stage1_result.best_value:
                    final_result = stage2_result
                    best_stage = 2
                else:
                    final_result = self._convert_stage1_result(
                        stage1_result, stage1_best_params, objective
                    )
                    best_stage = 1
        else:
            final_result = self._convert_stage1_result(
                stage1_result, stage1_best_params, objective
            )
            best_stage = 1
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"High-Dimensional Optimization Complete")
            print(f"{'='*70}")
            print(f"Best found in Stage {best_stage}")
            print(f"Final {self.direction}: {final_result.best_value:.6f}")
            print(f"Original dimensions: {n_continuous}")
            print(f"Reduced dimensions used: {reducer.reduced_dim}")
            print(f"{'='*70}")
        
        # Add high-dim info to result
        final_result.optimization_params['high_dim'] = {
            'original_dim': n_continuous,
            'reduced_dim': reducer.reduced_dim,
            'reduction_method': method,
            'effective_dim_info': dim_info,
            'stage1_trials': stage1_trials,
            'stage2_trials': stage2_trials,
            'best_stage': best_stage,
            'compression_ratio': reducer.compression_ratio
        }
        
        return final_result
    
    def _run_standard_optimization(self, objective, n_trials, x0, *args, **kwargs):
        """Run standard RAGDA optimization without dimensionality reduction."""
        optimizer = RAGDAOptimizer(
            [{'name': p.name, 'type': p.type, 'bounds': p.bounds, 'values': p.values, 'log': p.log}
             for p in self.space.parameters],
            direction=self.direction,
            n_workers=self.n_workers,
            random_state=self.random_state
        )
        return optimizer.optimize(objective, n_trials, x0, *args, **kwargs)
    
    def _create_reduced_space(self, n_components: int) -> List[Dict[str, Any]]:
        """Create a search space for the reduced dimensions."""
        return [
            {'name': f'z{i}', 'type': 'continuous', 'bounds': [-3.0, 3.0]}
            for i in range(n_components)
        ]
    
    def _create_reduced_objective(
        self,
        objective: Callable,
        reducer: DimensionalityReducer,
        original_space: SearchSpace
    ) -> Callable:
        """Create objective function that works in reduced space."""
        
        def reduced_objective(reduced_params: Dict[str, Any]) -> float:
            # Extract reduced coordinates
            n_comp = reducer.reduced_dim
            z = np.array([reduced_params[f'z{i}'] for i in range(n_comp)]).reshape(1, -1)
            
            # Inverse transform to original space (unit space)
            x_full = reducer.inverse_transform(z)[0]
            
            # Clip to [0, 1]
            x_full = np.clip(x_full, 0.0, 1.0)
            
            # Convert to original parameter space
            x_cat = np.zeros(original_space.n_categorical, dtype=np.int32)
            params = original_space.from_split_vectors(x_full, x_cat)
            
            # Evaluate
            return objective(params)
        
        return reduced_objective
    
    def _create_trust_region_space(
        self,
        center: np.ndarray,
        fraction: float
    ) -> List[Dict[str, Any]]:
        """Create a trust region space around the center point."""
        trust_space = []
        
        cont_idx = 0
        for param in self.space.parameters:
            if param.type == 'continuous':
                # Get center value in unit space
                center_val = center[cont_idx] if cont_idx < len(center) else 0.5
                
                # Create tight bounds around center
                half_width = fraction / 2
                lower = max(0.0, center_val - half_width)
                upper = min(1.0, center_val + half_width)
                
                # Transform back to original scale
                original_lower = param.transform_from_unit(lower)
                original_upper = param.transform_from_unit(upper)
                
                trust_space.append({
                    'name': param.name,
                    'type': 'continuous',
                    'bounds': [original_lower, original_upper],
                    'log': param.log
                })
                cont_idx += 1
            
            elif param.type == 'ordinal':
                center_val = center[cont_idx] if cont_idx < len(center) else 0.5
                half_width = fraction / 2
                lower_idx = max(0, int((center_val - half_width) * (len(param.values) - 1)))
                upper_idx = min(len(param.values) - 1, int((center_val + half_width) * (len(param.values) - 1)) + 1)
                
                trust_space.append({
                    'name': param.name,
                    'type': 'ordinal',
                    'values': param.values[lower_idx:upper_idx + 1]
                })
                cont_idx += 1
            
            else:  # categorical - keep all values
                trust_space.append({
                    'name': param.name,
                    'type': 'categorical',
                    'values': param.values
                })
        
        return trust_space
    
    def _convert_stage1_result(
        self,
        stage1_result: OptimizationResult,
        best_params: Dict[str, Any],
        objective: Callable
    ) -> OptimizationResult:
        """Convert stage 1 result to have correct params in original space."""
        # Re-evaluate in original space
        try:
            best_value = float(objective(best_params))
            if self.direction == 'maximize':
                best_value = best_value  # Keep as-is for maximize
        except:
            best_value = stage1_result.best_value
        
        best_trial = Trial(
            trial_id=0,
            worker_id=stage1_result.best_worker_id,
            iteration=stage1_result.best_trial.iteration,
            params=best_params,
            value=best_value,
            batch_size=-1
        )
        
        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_trial=best_trial,
            best_worker_id=stage1_result.best_worker_id,
            best_concentration=stage1_result.best_concentration,
            trials=stage1_result.trials,
            n_trials=stage1_result.n_trials,
            n_workers=stage1_result.n_workers,
            direction=self.direction,
            space=self.space,
            optimization_params=stage1_result.optimization_params
        )


# =============================================================================
# Convenience Function
# =============================================================================

def highdim_ragda_optimize(
    objective: Callable,
    bounds: np.ndarray,
    x0: Optional[np.ndarray] = None,
    direction: Literal['minimize', 'maximize'] = 'minimize',
    n_trials: int = 1000,
    random_state: Optional[int] = None,
    dim_threshold: int = 1000,
    variance_threshold: float = 0.90,
    reduction_method: str = 'auto',
    **kwargs
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Scipy-style convenience function for high-dimensional continuous-only problems.
    
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
    dim_threshold : int
        Dimensions above this trigger high-dim mode
    variance_threshold : float
        Variance to preserve in dimensionality reduction
    reduction_method : str
        'auto', 'kernel_pca', 'incremental_pca', or 'random_projection'
    **kwargs
        Additional arguments passed to optimize()
    
    Returns
    -------
    x_best : ndarray
        Best solution found
    f_best : float
        Best objective value
    info : dict
        Optimization info including high-dim stats
    """
    if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be ndarray of shape (n_dims, 2)")
    
    n_dims = len(bounds)
    
    space = [
        {'name': f'x{i}', 'type': 'continuous', 'bounds': [float(bounds[i, 0]), float(bounds[i, 1])]}
        for i in range(n_dims)
    ]
    
    def objective_dict(params):
        x = np.array([params[f'x{i}'] for i in range(n_dims)])
        return objective(x)
    
    x0_dict = None
    if x0 is not None:
        x0_dict = {f'x{i}': float(x0[i]) for i in range(n_dims)}
    
    optimizer = HighDimRAGDAOptimizer(
        space,
        direction=direction,
        random_state=random_state,
        dim_threshold=dim_threshold,
        variance_threshold=variance_threshold,
        reduction_method=reduction_method
    )
    
    result = optimizer.optimize(objective_dict, n_trials, x0_dict, **kwargs)
    
    x_best = np.array([result.best_params[f'x{i}'] for i in range(n_dims)])
    
    info = result.optimization_params.copy()
    if 'high_dim' in info:
        info.update(info.pop('high_dim'))
    
    return x_best, result.best_value, info
