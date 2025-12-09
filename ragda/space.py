"""
Search space definitions for RAGDA optimizer.

Supports continuous, ordinal, and categorical parameter types with
Latin Hypercube Sampling and proper transformations.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass


@dataclass
class Parameter:
    """
    A single parameter in the search space.
    
    Attributes
    ----------
    name : str
        Parameter name (used as key in params dict)
    type : str
        One of 'continuous', 'ordinal', 'categorical'
    bounds : tuple, optional
        (lower, upper) for continuous/ordinal
    values : list, optional
        Possible values for ordinal/categorical
    log : bool
        Whether to sample in log space (continuous only)
    """
    name: str
    type: Literal['continuous', 'ordinal', 'categorical']
    bounds: Optional[tuple] = None
    values: Optional[List] = None
    log: bool = False
    
    def __post_init__(self):
        """Validate parameter definition."""
        if self.type == 'continuous':
            if self.bounds is None:
                raise ValueError(f"Continuous parameter '{self.name}' requires bounds")
            if len(self.bounds) != 2:
                raise ValueError(f"Bounds for '{self.name}' must be (lower, upper)")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Invalid bounds for '{self.name}': lower >= upper")
            if self.log and self.bounds[0] <= 0:
                raise ValueError(f"Log-scale parameter '{self.name}' requires positive bounds")
        
        elif self.type == 'ordinal':
            if self.values is None:
                raise ValueError(f"Ordinal parameter '{self.name}' requires values list")
            if len(self.values) < 2:
                raise ValueError(f"Ordinal parameter '{self.name}' needs at least 2 values")
            # Set bounds based on index range for internal use
            self.bounds = (0.0, float(len(self.values) - 1))
        
        elif self.type == 'categorical':
            if self.values is None:
                raise ValueError(f"Categorical parameter '{self.name}' requires values list")
            if len(self.values) < 1:
                raise ValueError(f"Categorical parameter '{self.name}' needs at least 1 value")
        
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")
    
    def transform_to_unit(self, value: Any) -> float:
        """
        Transform a parameter value to unit interval [0, 1].
        
        For continuous: linear or log mapping
        For ordinal: index / (n_values - 1)
        For categorical: not applicable (returns 0)
        """
        if self.type == 'continuous':
            lower, upper = self.bounds
            if self.log:
                return (np.log(value) - np.log(lower)) / (np.log(upper) - np.log(lower))
            else:
                return (value - lower) / (upper - lower)
        
        elif self.type == 'ordinal':
            try:
                idx = self.values.index(value)
            except ValueError:
                # Find closest value
                idx = min(range(len(self.values)), 
                         key=lambda i: abs(self.values[i] - value))
            return idx / max(1, len(self.values) - 1)
        
        else:  # categorical
            return 0.0
    
    def transform_from_unit(self, unit_value: float) -> Any:
        """
        Transform from unit interval [0, 1] to parameter value.
        """
        unit_value = np.clip(unit_value, 0.0, 1.0)
        
        if self.type == 'continuous':
            lower, upper = self.bounds
            if self.log:
                log_val = np.log(lower) + unit_value * (np.log(upper) - np.log(lower))
                return np.exp(log_val)
            else:
                return lower + unit_value * (upper - lower)
        
        elif self.type == 'ordinal':
            idx = int(round(unit_value * (len(self.values) - 1)))
            idx = max(0, min(len(self.values) - 1, idx))
            return self.values[idx]
        
        else:  # categorical
            raise ValueError("Cannot transform categorical from unit")
    
    def sample_uniform(self) -> Any:
        """Sample uniformly from parameter domain."""
        if self.type == 'continuous':
            if self.log:
                log_val = np.random.uniform(np.log(self.bounds[0]), np.log(self.bounds[1]))
                return np.exp(log_val)
            else:
                return np.random.uniform(self.bounds[0], self.bounds[1])
        
        elif self.type == 'ordinal':
            return np.random.choice(self.values)
        
        else:  # categorical
            return np.random.choice(self.values)
    
    @property
    def n_values(self) -> int:
        """Number of discrete values (for categorical/ordinal)."""
        if self.type in ('ordinal', 'categorical'):
            return len(self.values)
        return 0


class SearchSpace:
    """
    Defines the search space for optimization.
    
    Handles mixed variable types and provides sampling/transformation utilities.
    
    Parameters
    ----------
    parameters : dict
        Search space definition where keys are parameter names and values are parameter definitions.
        
        Example
        -------
        >>> space = SearchSpace({
        ...     'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
        ...     'n_layers': {'type': 'ordinal', 'values': [1, 2, 4, 8, 16]},
        ...     'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
        ... })
    """
    
    def __init__(self, parameters: Dict[str, Dict[str, Any]]):
        if not parameters:
            raise ValueError("Search space cannot be empty")
        
        if not isinstance(parameters, dict):
            raise TypeError(
                f"parameters must be dict, got {type(parameters).__name__}. "
                "Use format: {'param1': {'type': ..., 'bounds': ...}, 'param2': {...}}"
            )
        
        # Convert to list format internally for processing
        parameters = self._dict_to_list(parameters)
        
        self.parameters = []
        self._param_names = []
        self._continuous_indices = []
        self._ordinal_indices = []
        self._categorical_indices = []
        
        for i, param_dict in enumerate(parameters):
            # Create Parameter object
            param = Parameter(
                name=param_dict['name'],
                type=param_dict['type'],
                bounds=param_dict.get('bounds'),
                values=param_dict.get('values'),
                log=param_dict.get('log', False)
            )
            self.parameters.append(param)
            self._param_names.append(param.name)
            
            # Track indices by type
            if param.type == 'continuous':
                self._continuous_indices.append(i)
            elif param.type == 'ordinal':
                self._ordinal_indices.append(i)
            else:  # categorical
                self._categorical_indices.append(i)
    
    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return len(self.parameters)
    
    @property
    def n_continuous(self) -> int:
        """Number of continuous + ordinal parameters (treated as continuous internally)."""
        return len(self._continuous_indices) + len(self._ordinal_indices)
    
    @property
    def n_categorical(self) -> int:
        """Number of categorical parameters."""
        return len(self._categorical_indices)
    
    @property
    def param_names(self) -> List[str]:
        """List of parameter names."""
        return self._param_names.copy()
    
    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Check if params dict is valid for this space.
        
        Returns True if valid, False otherwise.
        """
        try:
            for param in self.parameters:
                if param.name not in params:
                    return False
                
                value = params[param.name]
                
                if param.type == 'continuous':
                    if not isinstance(value, (int, float, np.number)):
                        return False
                    if not (param.bounds[0] <= value <= param.bounds[1]):
                        return False
                
                elif param.type == 'ordinal':
                    if value not in param.values:
                        return False
                
                elif param.type == 'categorical':
                    if value not in param.values:
                        return False
            
            return True
        except:
            return False
    
    def to_split_vectors(self, params: Dict[str, Any]) -> tuple:
        """
        Convert params dict to split continuous/categorical vectors.
        
        Returns
        -------
        x_cont : np.ndarray, shape (n_continuous,)
            Continuous values in unit space [0, 1]
        x_cat : np.ndarray, shape (n_categorical,), dtype=int32
            Categorical indices
        cat_n_values : np.ndarray, shape (n_categorical,), dtype=int32
            Number of values for each categorical
        """
        # Continuous + ordinal in unit space
        x_cont = np.zeros(self.n_continuous, dtype=np.float64)
        
        cont_idx = 0
        for i in self._continuous_indices:
            param = self.parameters[i]
            x_cont[cont_idx] = param.transform_to_unit(params[param.name])
            cont_idx += 1
        
        for i in self._ordinal_indices:
            param = self.parameters[i]
            x_cont[cont_idx] = param.transform_to_unit(params[param.name])
            cont_idx += 1
        
        # Categorical as indices
        x_cat = np.zeros(self.n_categorical, dtype=np.int32)
        cat_n_values = np.zeros(self.n_categorical, dtype=np.int32)
        
        cat_idx = 0
        for i in self._categorical_indices:
            param = self.parameters[i]
            value = params[param.name]
            x_cat[cat_idx] = param.values.index(value)
            cat_n_values[cat_idx] = len(param.values)
            cat_idx += 1
        
        return x_cont, x_cat, cat_n_values
    
    def from_split_vectors(self, x_cont: np.ndarray, x_cat: np.ndarray) -> Dict[str, Any]:
        """
        Convert split vectors back to params dict.
        
        Parameters
        ----------
        x_cont : np.ndarray
            Continuous values in unit space [0, 1]
        x_cat : np.ndarray
            Categorical indices
        
        Returns
        -------
        params : dict
            Parameter dictionary
        """
        params = {}
        
        # Continuous parameters
        cont_idx = 0
        for i in self._continuous_indices:
            param = self.parameters[i]
            unit_val = float(x_cont[cont_idx]) if cont_idx < len(x_cont) else 0.5
            params[param.name] = param.transform_from_unit(unit_val)
            cont_idx += 1
        
        # Ordinal parameters
        for i in self._ordinal_indices:
            param = self.parameters[i]
            unit_val = float(x_cont[cont_idx]) if cont_idx < len(x_cont) else 0.5
            params[param.name] = param.transform_from_unit(unit_val)
            cont_idx += 1
        
        # Categorical parameters
        cat_idx = 0
        for i in self._categorical_indices:
            param = self.parameters[i]
            if cat_idx < len(x_cat):
                idx = int(x_cat[cat_idx])
                idx = max(0, min(len(param.values) - 1, idx))
                params[param.name] = param.values[idx]
            else:
                params[param.name] = param.values[0]
            cat_idx += 1
        
        return params
    
    def get_bounds_array(self) -> np.ndarray:
        """
        Get bounds array for continuous parameters (in unit space).
        
        Returns
        -------
        bounds : np.ndarray, shape (n_continuous, 2)
            All bounds are [0, 1] since we work in unit space
        """
        n_cont = self.n_continuous
        if n_cont == 0:
            return np.empty((0, 2), dtype=np.float64)
        
        # All in unit space [0, 1]
        return np.tile(np.array([0.0, 1.0]), (n_cont, 1))
    
    def sample(self, n: int = 1, method: str = 'lhs') -> List[Dict[str, Any]]:
        """
        Sample n points from the search space.
        
        Parameters
        ----------
        n : int
            Number of samples
        method : str
            'lhs' for Latin Hypercube Sampling
            'random' for uniform random sampling
        
        Returns
        -------
        samples : list of dict
            n parameter dictionaries
        """
        if method == 'lhs':
            return self._sample_lhs(n)
        elif method == 'random':
            return self._sample_random(n)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _sample_lhs(self, n: int) -> List[Dict[str, Any]]:
        """Latin Hypercube Sampling."""
        try:
            from scipy.stats.qmc import LatinHypercube
            
            # LHS for continuous dimensions
            n_cont = self.n_continuous
            if n_cont > 0:
                lhs = LatinHypercube(d=n_cont, seed=None)
                lhs_samples = lhs.random(n)  # Shape: (n, n_cont)
            else:
                lhs_samples = np.zeros((n, 0))
            
        except ImportError:
            # Fallback if scipy doesn't have qmc
            n_cont = self.n_continuous
            if n_cont > 0:
                lhs_samples = self._basic_lhs(n, n_cont)
            else:
                lhs_samples = np.zeros((n, 0))
        
        samples = []
        for i in range(n):
            params = {}
            
            # Continuous parameters
            cont_idx = 0
            for param_idx in self._continuous_indices:
                param = self.parameters[param_idx]
                unit_val = lhs_samples[i, cont_idx]
                params[param.name] = param.transform_from_unit(unit_val)
                cont_idx += 1
            
            # Ordinal parameters
            for param_idx in self._ordinal_indices:
                param = self.parameters[param_idx]
                unit_val = lhs_samples[i, cont_idx]
                params[param.name] = param.transform_from_unit(unit_val)
                cont_idx += 1
            
            # Categorical parameters (random)
            for param_idx in self._categorical_indices:
                param = self.parameters[param_idx]
                params[param.name] = np.random.choice(param.values)
            
            samples.append(params)
        
        return samples
    
    def _basic_lhs(self, n: int, d: int) -> np.ndarray:
        """Basic LHS implementation without scipy.qmc."""
        samples = np.zeros((n, d))
        
        for j in range(d):
            # Create n intervals
            cut_points = np.linspace(0, 1, n + 1)
            
            # Sample from each interval
            for i in range(n):
                samples[i, j] = np.random.uniform(cut_points[i], cut_points[i + 1])
            
            # Shuffle
            np.random.shuffle(samples[:, j])
        
        return samples
    
    @staticmethod
    def _dict_to_list(space_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert dict-based space to list-based format (internal representation).
        
        Parameters
        ----------
        space_dict : dict
            Dict where keys are parameter names, values are parameter specs
        
        Returns
        -------
        space_list : list of dict
            List format with 'name' key added to each parameter
        """
        space_list = []
        for name, spec in space_dict.items():
            param_dict = {'name': name}
            param_dict.update(spec)
            space_list.append(param_dict)
        return space_list
    
    @staticmethod
    def _list_to_dict(space_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert list-based space to dict-based format.
        
        Parameters
        ----------
        space_list : list of dict
            List format with 'name' key in each parameter
        
        Returns
        -------
        space_dict : dict
            Dict where keys are parameter names
        """
        space_dict = {}
        for param in space_list:
            name = param['name']
            spec = {k: v for k, v in param.items() if k != 'name'}
            space_dict[name] = spec
        return space_dict
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Export search space as dict format.
        
        Returns
        -------
        space_dict : dict
            Dict-based space definition
        """
        space_dict = {}
        for param in self.parameters:
            spec = {'type': param.type}
            if param.bounds is not None:
                spec['bounds'] = list(param.bounds)
            if param.values is not None:
                spec['values'] = list(param.values)
            if param.log:
                spec['log'] = True
            space_dict[param.name] = spec
        return space_dict
    
    def _sample_random(self, n: int) -> List[Dict[str, Any]]:
        """Uniform random sampling."""
        samples = []
        for _ in range(n):
            params = {}
            for param in self.parameters:
                params[param.name] = param.sample_uniform()
            samples.append(params)
        return samples
    
    def __repr__(self):
        param_strs = []
        for p in self.parameters:
            if p.type == 'continuous':
                param_strs.append(f"  {p.name}: continuous{' (log)' if p.log else ''} {p.bounds}")
            elif p.type == 'ordinal':
                param_strs.append(f"  {p.name}: ordinal {p.values}")
            else:
                param_strs.append(f"  {p.name}: categorical {p.values}")
        
        return f"SearchSpace(\n{chr(10).join(param_strs)}\n)"
