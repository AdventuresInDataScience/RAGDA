"""
API Adapters for RAGDA - Multi-API Support

This module provides thin adapter layers that translate different optimizer APIs
(Optuna, Scipy, Sklearn) into RAGDA's canonical internal format.

Architecture:
    User API → Adapter → Canonical Format → Shared Core → Result Adapter → User API

All adapters must implement:
    - to_canonical_space(): Convert API-specific space to canonical dict format
    - wrap_objective(): Wrap user's objective to work with canonical **kwargs
    - wrap_result(): Convert RAGDA result to API-specific format (optional)
"""

import numpy as np
from typing import Callable, Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# Base Adapter Interface
# =============================================================================

class BaseAPIAdapter(ABC):
    """Base class for all API adapters."""
    
    @abstractmethod
    def to_canonical_space(self, space_definition: Any) -> Dict[str, Dict[str, Any]]:
        """
        Convert API-specific space definition to canonical format.
        
        Canonical format:
        {
            'param_name': {
                'type': 'continuous' | 'ordinal' | 'categorical',
                'bounds': [low, high],  # for continuous/ordinal
                'values': [...],        # for categorical
                'log': bool             # optional, for continuous
            }
        }
        """
        pass
    
    @abstractmethod
    def wrap_objective(self, user_objective: Callable) -> Callable:
        """
        Wrap user's objective function to work with canonical format.
        
        Canonical objective signature: func(**params) -> float
        """
        pass
    
    def wrap_result(self, ragda_result):
        """
        Convert RAGDA OptimizationResult to API-specific format.
        Default: return as-is.
        """
        return ragda_result


# =============================================================================
# RAGDA Native Adapter (Pass-through)
# =============================================================================

class RAGDANativeAdapter(BaseAPIAdapter):
    """
    Adapter for RAGDA's native API.
    
    Space format: dict with parameter names as keys
    Objective format: func(**params) -> float
    
    This is essentially a pass-through since it's already in canonical format.
    """
    
    def to_canonical_space(self, space_definition: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Space is already in canonical format."""
        return space_definition
    
    def wrap_objective(self, user_objective: Callable) -> Callable:
        """Objective already expects **kwargs."""
        return user_objective


# =============================================================================
# Optuna-Style Adapter
# =============================================================================

class RAGDATrial:
    """
    Mock Optuna Trial object.
    
    Instead of sampling parameters, it reads from pre-sampled values.
    On first call, it records the space definition for later conversion.
    """
    
    def __init__(self, params: Dict[str, Any], space_definition: Dict[str, Dict[str, Any]]):
        """
        Parameters
        ----------
        params : dict
            Pre-sampled parameter values from SearchSpace
        space_definition : dict
            Mutable dict that gets populated with space specs on first suggest_* call
        """
        self.params = params
        self.space_definition = space_definition
        self._frozen = False  # Set to True after space is discovered
    
    def suggest_float(self, name: str, low: float, high: float, *, 
                     log: bool = False, step: Optional[float] = None) -> float:
        """Suggest a float parameter (mimics optuna.trial.Trial.suggest_float)."""
        
        # Record space definition on first encounter
        if name not in self.space_definition and not self._frozen:
            spec = {
                'type': 'float',
                'low': low,
                'high': high,
                'log': log
            }
            if step is not None:
                spec['step'] = step
            self.space_definition[name] = spec
        
        # Return pre-sampled value (or sample if first trial)
        if name not in self.params:
            # First trial - sample on the fly
            if log:
                self.params[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            elif step is not None:
                # Discrete float
                n_steps = int((high - low) / step) + 1
                self.params[name] = low + np.random.randint(0, n_steps) * step
            else:
                self.params[name] = np.random.uniform(low, high)
        
        return self.params[name]
    
    def suggest_int(self, name: str, low: int, high: int, *, 
                   log: bool = False, step: int = 1) -> int:
        """Suggest an int parameter (mimics optuna.trial.Trial.suggest_int)."""
        
        if name not in self.space_definition and not self._frozen:
            self.space_definition[name] = {
                'type': 'int',
                'low': low,
                'high': high,
                'log': log,
                'step': step
            }
        
        if name not in self.params:
            if log:
                log_range = np.log(high) - np.log(low)
                self.params[name] = int(np.exp(np.random.uniform(np.log(low), np.log(high))))
            else:
                n_steps = (high - low) // step + 1
                self.params[name] = low + np.random.randint(0, n_steps) * step
        
        return int(self.params[name])
    
    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        """Suggest a categorical parameter (mimics optuna.trial.Trial.suggest_categorical)."""
        
        if name not in self.space_definition and not self._frozen:
            self.space_definition[name] = {
                'type': 'categorical',
                'choices': list(choices)
            }
        
        if name not in self.params:
            self.params[name] = np.random.choice(choices)
        
        return self.params[name]
    
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        """Suggest a discrete uniform parameter (deprecated in Optuna, but supported)."""
        return self.suggest_float(name, low, high, step=q)


class OptunaAdapter(BaseAPIAdapter):
    """
    Adapter for Optuna-style API.
    
    Optuna's API uses a Trial object with suggest_* methods:
    
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_categorical('y', ['A', 'B'])
            return loss
    
    Strategy:
    1. On first trial, user's objective calls suggest_*, which builds space_definition
    2. Convert space_definition to canonical format
    3. For subsequent trials, pre-sample params and pass via trial object
    """
    
    def __init__(self):
        self.space_definition = {}
        self.space_discovered = False
        self.user_objective = None
    
    def to_canonical_space(self, space_definition: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
        """
        Convert Optuna-style space to canonical format.
        
        If space_definition is None, use the internally discovered space.
        """
        if space_definition is None:
            space_definition = self.space_definition
        
        if not space_definition:
            raise ValueError("Space not yet discovered. Run one trial first.")
        
        canonical = {}
        
        for name, spec in space_definition.items():
            if spec['type'] == 'float':
                if spec.get('step') is not None:
                    # Discrete float → ordinal
                    low, high, step = spec['low'], spec['high'], spec['step']
                    values = np.arange(low, high + step/2, step).tolist()
                    canonical[name] = {'type': 'ordinal', 'values': values}
                else:
                    canonical[name] = {
                        'type': 'continuous',
                        'bounds': [spec['low'], spec['high']],
                        'log': spec.get('log', False)
                    }
            
            elif spec['type'] == 'int':
                # Int with step > 1 or log → ordinal (discrete values)
                if spec.get('step', 1) > 1 or spec.get('log', False):
                    low, high, step = spec['low'], spec['high'], spec.get('step', 1)
                    values = list(range(low, high + 1, step))
                    canonical[name] = {'type': 'ordinal', 'values': values}
                else:
                    # Continuous int → treat as continuous then round
                    canonical[name] = {
                        'type': 'continuous',
                        'bounds': [float(spec['low']), float(spec['high'])],
                        'log': False
                    }
            
            elif spec['type'] == 'categorical':
                canonical[name] = {
                    'type': 'categorical',
                    'values': spec['choices']
                }
        
        return canonical
    
    def wrap_objective(self, user_objective: Callable) -> Callable:
        """
        Wrap Optuna-style objective to work with canonical **params.
        
        The wrapped function creates a Trial object and passes it to user's objective.
        """
        self.user_objective = user_objective
        
        def canonical_objective(**params):
            # Create trial object with pre-sampled params
            trial = RAGDATrial(params, self.space_definition)
            
            # If this is the first trial, mark space as discovered after objective runs
            was_empty = len(self.space_definition) == 0
            
            # User's objective calls trial.suggest_*
            result = user_objective(trial)
            
            if was_empty and len(self.space_definition) > 0:
                self.space_discovered = True
                trial._frozen = True  # Don't allow space changes after discovery
            
            return result
        
        return canonical_objective
    
    def discover_space(self, user_objective: Callable) -> Dict[str, Dict[str, Any]]:
        """
        Discover space by running objective once with empty params.
        
        This is called before optimization starts to build the SearchSpace.
        """
        dummy_params = {}
        trial = RAGDATrial(dummy_params, self.space_definition)
        
        try:
            user_objective(trial)
        except Exception as e:
            # It's okay if objective fails - we just need the space definition
            pass
        
        if not self.space_definition:
            raise ValueError(
                "Could not discover search space. Objective must call trial.suggest_* methods."
            )
        
        return self.to_canonical_space()
    
    def postprocess_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process parameters to convert them back to their original types.
        
        Needed because integer parameters are stored as floats in the optimizer
        but should be returned as integers to match Optuna's API.
        """
        processed = {}
        for name, value in params.items():
            if name in self.space_definition:
                spec = self.space_definition[name]
                if spec['type'] == 'int':
                    # Convert to integer and ensure it respects step
                    int_val = int(round(value))
                    low, high, step = spec['low'], spec['high'], spec.get('step', 1)
                    # Snap to nearest valid step
                    int_val = low + round((int_val - low) / step) * step
                    # Clamp to bounds
                    int_val = max(low, min(high, int_val))
                    processed[name] = int_val
                else:
                    processed[name] = value
            else:
                processed[name] = value
        
        return processed


# =============================================================================
# Scipy-Style Adapter
# =============================================================================

class ScipyAdapter(BaseAPIAdapter):
    """
    Adapter for scipy.optimize-style API.
    
    Scipy's API uses array-based objectives:
    
        def objective(x):  # x is numpy array
            return loss
        
        result = minimize(objective, bounds=[(-5, 5), (-5, 5)])
    
    Strategy:
    1. Convert bounds list to canonical space with auto-generated names (x0, x1, ...)
    2. Wrap objective to convert **params dict → array → user objective
    3. Convert result back to scipy's OptimizeResult format
    """
    
    def __init__(self):
        self.n_params = 0
        self.param_names = []
    
    def to_canonical_space(self, bounds: List[Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert scipy bounds to canonical format.
        
        Parameters
        ----------
        bounds : list of tuples
            [(low, high), (low, high), ...]
        """
        self.n_params = len(bounds)
        self.param_names = [f'x{i}' for i in range(self.n_params)]
        
        canonical = {}
        for i, (low, high) in enumerate(bounds):
            canonical[self.param_names[i]] = {
                'type': 'continuous',
                'bounds': [float(low), float(high)]
            }
        
        return canonical
    
    def wrap_objective(self, user_objective: Callable) -> Callable:
        """
        Wrap scipy-style objective (array input) to canonical (**kwargs).
        """
        def canonical_objective(**params):
            # Convert dict to array in correct order
            x = np.array([params[name] for name in self.param_names])
            return user_objective(x)
        
        return canonical_objective
    
    def wrap_result(self, ragda_result):
        """Convert RAGDA result to scipy OptimizeResult format."""
        from scipy.optimize import OptimizeResult
        
        # Extract array from dict
        x = np.array([ragda_result.best_params[name] for name in self.param_names])
        
        return OptimizeResult(
            x=x,
            fun=ragda_result.best_value,
            success=True,
            nit=len(ragda_result.trials),
            nfev=len(ragda_result.trials),
            message='Optimization terminated successfully.'
        )


# =============================================================================
# Legacy RAGDA Adapter (List of Dicts)
# =============================================================================

class RAGDALegacyAdapter(BaseAPIAdapter):
    """
    Adapter for legacy RAGDA API (list of dicts with 'name' keys).
    
    Old format:
        space = [
            {'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]},
            {'name': 'y', 'type': 'categorical', 'values': ['A', 'B']},
        ]
    
    New format (canonical):
        space = {
            'x': {'type': 'continuous', 'bounds': [-5, 5]},
            'y': {'type': 'categorical', 'values': ['A', 'B']},
        }
    """
    
    def to_canonical_space(self, space_definition: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convert list-of-dicts to dict-of-dicts."""
        canonical = {}
        
        for param_spec in space_definition:
            if 'name' not in param_spec:
                raise ValueError(f"Legacy space format requires 'name' key: {param_spec}")
            
            name = param_spec['name']
            spec = {k: v for k, v in param_spec.items() if k != 'name'}
            canonical[name] = spec
        
        return canonical
    
    def wrap_objective(self, user_objective: Callable) -> Callable:
        """
        Legacy objectives might expect dict or **kwargs.
        Try **kwargs first, fall back to dict.
        """
        import inspect
        sig = inspect.signature(user_objective)
        
        # Check if objective accepts **kwargs or has VAR_KEYWORD parameter
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD 
            for p in sig.parameters.values()
        )
        
        # Check if objective takes exactly one dict parameter
        params = list(sig.parameters.values())
        takes_single_dict = (
            len(params) == 1 and 
            params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        
        if has_var_keyword or not takes_single_dict:
            # Modern: accepts **kwargs
            return user_objective
        else:
            # Legacy: accepts single dict
            def canonical_objective(**params):
                return user_objective(params)
            return canonical_objective


# =============================================================================
# Adapter Factory
# =============================================================================

def get_adapter(api_mode: str, **kwargs) -> BaseAPIAdapter:
    """
    Factory function to get the appropriate adapter for an API mode.
    
    Parameters
    ----------
    api_mode : str
        One of: 'ragda', 'optuna', 'scipy', 'legacy'
    
    Returns
    -------
    BaseAPIAdapter
        Appropriate adapter instance
    """
    adapters = {
        'ragda': RAGDANativeAdapter,
        'optuna': OptunaAdapter,
        'scipy': ScipyAdapter,
        'legacy': RAGDALegacyAdapter,
    }
    
    if api_mode not in adapters:
        raise ValueError(
            f"Unknown api_mode: {api_mode}. "
            f"Choose from: {', '.join(adapters.keys())}"
        )
    
    return adapters[api_mode](**kwargs)
