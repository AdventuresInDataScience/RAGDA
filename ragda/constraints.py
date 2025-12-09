"""
Constraint parsing and evaluation for RAGDA optimizer.

Supports string-based constraints with numeric and categorical comparisons.
"""

import re
import ast
import warnings
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Set


class ConstraintParser:
    """
    Parse string-based constraints into executable functions.
    
    Supports:
    - Numeric comparisons: <, >, <=, >=, ==, !=
    - Categorical comparisons: optimizer == "adam"
    - Logical operators: and, or, not
    - Implication: A -> B (if A then B)
    - Safe mathematical functions: abs, min, max, sqrt, exp, log, sin, cos
    
    Examples
    --------
    >>> parser = ConstraintParser(['x', 'y', 'optimizer'])
    >>> fn = parser.parse('x + y <= 5')
    >>> fn(x=2, y=3, optimizer='adam')
    True
    >>> fn(x=3, y=3, optimizer='adam')
    False
    
    >>> fn = parser.parse('optimizer == "sgd" -> x <= 0.01')
    >>> fn(x=0.005, y=1, optimizer='sgd')
    True
    >>> fn(x=0.05, y=1, optimizer='sgd')
    False
    """
    
    # Allowed functions in constraints
    SAFE_FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
    }
    
    # Math functions (imported on demand)
    MATH_FUNCTIONS = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'log10']
    
    def __init__(self, param_names: List[str]):
        """
        Initialize constraint parser.
        
        Parameters
        ----------
        param_names : list of str
            Valid parameter names from the search space
        """
        self.param_names = set(param_names)
        self._safe_namespace = self._build_safe_namespace()
    
    def parse(self, constraint_str: str) -> Callable:
        """
        Parse a constraint string into a callable function.
        
        Supports complex boolean expressions with logical operators (and, or, not).
        
        Parameters
        ----------
        constraint_str : str
            Constraint expression (can contain multiple comparisons with and/or)
            
        Returns
        -------
        constraint_fn : callable
            Function that takes **kwargs and returns bool
            
        Raises
        ------
        ValueError
            If constraint cannot be parsed or contains invalid variables
        """
        constraint_str = constraint_str.strip()
        
        # Handle implication operator: A -> B becomes (not A) or B
        if '->' in constraint_str:
            return self._parse_implication(constraint_str)
        
        # For general expressions (including those with and/or), validate and compile directly
        self._validate_expression(constraint_str, constraint_str)
        
        # Build the constraint function
        param_list = ', '.join(sorted(self.param_names))
        
        # Create function string with safe namespace
        func_str = f"lambda {param_list}: {constraint_str}"
        
        try:
            # Compile and evaluate the lambda
            constraint_fn = eval(func_str, self._safe_namespace, {})
            
            # Test the function with dummy values to ensure it works
            test_params = {name: 0.0 for name in self.param_names}
            result = constraint_fn(**test_params)
            
            # Ensure result is boolean
            if not isinstance(result, (bool, np.bool_)):
                raise ValueError(f"Constraint must return a boolean value, got {type(result)}")
            
            return constraint_fn
            
        except Exception as e:
            raise ValueError(
                f"Failed to create constraint function from: '{constraint_str}'\n"
                f"Generated: {func_str}\n"
                f"Error: {e}"
            )
    
    def _parse_implication(self, constraint_str: str) -> Callable:
        """
        Parse implication constraint: A -> B.
        
        Converts to: (not A) or B
        """
        parts = constraint_str.split('->')
        if len(parts) != 2:
            raise ValueError(
                f"Implication must have exactly one '->' operator.\n"
                f"Got: '{constraint_str}'"
            )
        
        antecedent = parts[0].strip()
        consequent = parts[1].strip()
        
        # Validate both sides
        self._validate_expression(antecedent, constraint_str)
        self._validate_expression(consequent, constraint_str)
        
        # Build: (not antecedent) or consequent
        param_list = ', '.join(sorted(self.param_names))
        func_str = f"lambda {param_list}: not ({antecedent}) or ({consequent})"
        
        try:
            constraint_fn = eval(func_str, self._safe_namespace, {})
            
            # Test
            test_params = {name: 0.0 for name in self.param_names}
            _ = constraint_fn(**test_params)
            
            return constraint_fn
            
        except Exception as e:
            raise ValueError(
                f"Failed to create implication constraint from: '{constraint_str}'\n"
                f"Generated: {func_str}\n"
                f"Error: {e}"
            )
    
    def _validate_expression(self, expr: str, original: str):
        """
        Validate that expression only contains allowed parameters and operations.
        
        Raises ValueError if invalid.
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expr, mode='eval')
            
            # Find all variable names and function calls
            used_names = set()
            used_functions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        used_functions.add(node.func.id)
            
            # Check for invalid variable names (excluding function names)
            invalid = used_names - self.param_names - set(self.SAFE_FUNCTIONS.keys()) - set(self.MATH_FUNCTIONS)
            if invalid:
                raise ValueError(
                    f"Constraint contains undefined parameters: {invalid}\n"
                    f"Valid parameters: {sorted(self.param_names)}\n"
                    f"Constraint: '{original}'"
                )
            
            # Check for invalid functions
            all_allowed = set(self.SAFE_FUNCTIONS.keys()) | set(self.MATH_FUNCTIONS)
            invalid_funcs = used_functions - all_allowed
            if invalid_funcs:
                raise ValueError(
                    f"Function(s) '{invalid_funcs}' not allowed in constraints.\n"
                    f"Allowed: {sorted(all_allowed)}"
                )
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError(
                        f"Import statements not allowed in constraints.\n"
                        f"Constraint: '{original}'"
                    )
        
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python expression in constraint: '{expr}'\n"
                f"Original constraint: '{original}'\n"
                f"Error: {e}"
            )
    
    def _build_safe_namespace(self) -> Dict[str, Any]:
        """Build safe namespace for eval with allowed functions."""
        namespace = {"__builtins__": {}}
        namespace.update(self.SAFE_FUNCTIONS)
        
        # Import math functions
        import math
        for fname in self.MATH_FUNCTIONS:
            if hasattr(math, fname):
                namespace[fname] = getattr(math, fname)
        
        return namespace
    
    def parse_all(self, constraints: List[str]) -> List[Callable]:
        """
        Parse multiple constraint strings.
        
        Parameters
        ----------
        constraints : list of str
            List of constraint strings
            
        Returns
        -------
        constraint_fns : list of callable
            List of constraint functions
        """
        parsed = []
        for i, constraint_str in enumerate(constraints):
            try:
                fn = self.parse(constraint_str)
                parsed.append(fn)
            except ValueError as e:
                raise ValueError(f"Error parsing constraint {i+1}: {e}")
        
        return parsed


def parse_constraints(
    constraints: List[str],
    param_names: List[str]
) -> List[Callable]:
    """
    Convenience function to parse constraint strings.
    
    Parameters
    ----------
    constraints : list of str
        List of constraint strings
    param_names : list of str
        Valid parameter names from search space
        
    Returns
    -------
    constraint_fns : list of callable
        List of constraint functions that take **kwargs and return bool
    
    Examples
    --------
    >>> constraints = ['x + y <= 5', 'x >= 0']
    >>> fns = parse_constraints(constraints, ['x', 'y', 'z'])
    >>> fns[0](x=2, y=3, z=1)
    True
    >>> fns[0](x=3, y=3, z=1)
    False
    """
    parser = ConstraintParser(param_names)
    return parser.parse_all(constraints)


def create_constraint_wrapper(
    objective: Callable,
    constraints: List[Callable],
    penalty: float = 1e10
) -> Callable:
    """
    Wrap objective function with constraint checking.
    
    Parameters
    ----------
    objective : callable
        Original objective function
    constraints : list of callable
        List of constraint functions (return True if satisfied)
    penalty : float
        Penalty value returned when constraints are violated
        
    Returns
    -------
    wrapped_objective : callable
        Objective with constraint checking
    
    Examples
    --------
    >>> def obj(x, y):
    ...     return x**2 + y**2
    >>> 
    >>> constraints = [lambda x, y: x + y <= 5]
    >>> wrapped = create_constraint_wrapper(obj, constraints, penalty=1e10)
    >>> 
    >>> wrapped(x=2, y=2)  # Satisfied: 2+2 <= 5
    8
    >>> wrapped(x=3, y=3)  # Violated: 3+3 > 5
    1e10
    """
    def wrapped_objective(**params):
        # Check all constraints
        for constraint_fn in constraints:
            try:
                if not constraint_fn(**params):
                    return penalty
            except Exception as e:
                # If constraint evaluation fails, treat as violation
                warnings.warn(
                    f"Constraint evaluation failed with params {params}: {e}",
                    RuntimeWarning
                )
                return penalty
        
        # All constraints satisfied, evaluate objective
        try:
            return objective(**params)
        except Exception as e:
            warnings.warn(
                f"Objective evaluation failed with params {params}: {e}",
                RuntimeWarning
            )
            return penalty
    
    return wrapped_objective
