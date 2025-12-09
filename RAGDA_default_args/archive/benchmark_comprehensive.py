"""
Comprehensive RAGDA Benchmarking Suite for Research Paper

Systematic evaluation across:
- 3 dimensionality levels × 3 cost levels × 3 noise levels = 27 problem classes
- 5+ problems per class = 135+ benchmark problems
- Plus real-world applications with data batching
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Callable, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*80)
print("RAGDA Comprehensive Benchmarking Suite")
print("="*80)
print("\nImporting optimizers...")

# ============================================================================
# Import Optimizers
# ============================================================================

OPTIMIZERS_AVAILABLE = {}

try:
    from ragda import RAGDAOptimizer
    OPTIMIZERS_AVAILABLE['RAGDA'] = True
    print("✓ RAGDA")
except ImportError:
    OPTIMIZERS_AVAILABLE['RAGDA'] = False
    print("✗ RAGDA not available")

try:
    import cma
    OPTIMIZERS_AVAILABLE['CMA-ES'] = True
    print("✓ CMA-ES")
except ImportError:
    OPTIMIZERS_AVAILABLE['CMA-ES'] = False
    print("✗ CMA-ES (pip install cma)")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTIMIZERS_AVAILABLE['Optuna'] = True
    print("✓ Optuna")
except ImportError:
    OPTIMIZERS_AVAILABLE['Optuna'] = False
    print("✗ Optuna (pip install optuna)")

try:
    from scipy.optimize import differential_evolution, dual_annealing
    OPTIMIZERS_AVAILABLE['DiffEvo'] = True
    OPTIMIZERS_AVAILABLE['DualAnneal'] = True
    print("✓ Scipy optimizers")
except ImportError:
    OPTIMIZERS_AVAILABLE['DiffEvo'] = False
    OPTIMIZERS_AVAILABLE['DualAnneal'] = False
    print("✗ Scipy")

try:
    from bayes_opt import BayesianOptimization
    OPTIMIZERS_AVAILABLE['BayesOpt'] = True
    print("✓ Bayesian Optimization")
except ImportError:
    OPTIMIZERS_AVAILABLE['BayesOpt'] = False
    print("✗ BayesOpt (pip install bayesian-optimization)")

try:
    import hyperopt
    OPTIMIZERS_AVAILABLE['Hyperopt'] = True
    print("✓ Hyperopt")
except ImportError:
    OPTIMIZERS_AVAILABLE['Hyperopt'] = False
    print("✗ Hyperopt (pip install hyperopt)")

OPTIMIZERS_AVAILABLE['Random'] = True
print("✓ Random Search")


# ============================================================================
# Problem Taxonomy
# ============================================================================

@dataclass
class ProblemSpec:
    """Specification for benchmark problem."""
    name: str
    category: str  # 'synthetic', 'ml_hyperparams', 'portfolio', 'real_world'
    dimension_class: str  # 'small' (2-5), 'medium' (10-30), 'large' (50-100)
    cost_class: str  # 'low' (<0.01s), 'medium' (0.01-0.1s), 'high' (>0.1s)
    noise_class: str  # 'none', 'low', 'medium', 'high'
    dimension: int
    func: Callable
    bounds: np.ndarray
    space: List[Dict]
    optimum: Optional[float]
    description: str
    supports_batching: bool = False


# ============================================================================
# Synthetic Test Functions
# ============================================================================

class SyntheticFunctions:
    """Comprehensive synthetic test function library."""
    
    @staticmethod
    def create_function(
        base_func: str,
        n_dims: int,
        cost_class: str,
        noise_class: str
    ) -> ProblemSpec:
        """
        Create a test function with specified characteristics.
        
        Parameters
        ----------
        base_func : str
            Base function name
        n_dims : int
            Dimensionality
        cost_class : str
            'low', 'medium', 'high'
        noise_class : str
            'none', 'low', 'medium', 'high'
        """
        # Determine dimension class
        if n_dims <= 5:
            dim_class = 'small'
        elif n_dims <= 30:
            dim_class = 'medium'
        else:
            dim_class = 'large'
        
        # Noise mapping
        noise_std_map = {
            'none': 0.0,
            'low': 0.01,
            'medium': 0.1,
            'high': 0.5
        }
        
        # Cost delays (in seconds)
        cost_delay_map = {
            'low': 0.0,
            'medium': 0.01,
            'high': 0.1
        }
        
        noise_std = noise_std_map[noise_class]
        cost_delay = cost_delay_map[cost_class]
        
        # Define base functions
        if base_func == 'sphere':
            def func(**kwargs):
                x = np.array([kwargs[f'x{i}'] for i in range(n_dims)])
                
                if cost_delay > 0:
                    time.sleep(cost_delay)
                
                result = np.sum(x**2)
                
                if noise_std > 0:
                    result += np.random.randn() * noise_std * np.sqrt(result + 1)
                
                return result
            
            bounds = np.array([[-5.12, 5.12]] * n_dims)
            optimum = 0.0
            description = "Unimodal quadratic"
        
        elif base_func == 'rosenbrock':
            def func(**kwargs):
                x = np.array([kwargs[f'x{i}'] for i in range(n_dims)])
                
                if cost_delay > 0:
                    time.sleep(cost_delay)
                
                result = sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
                
                if noise_std > 0:
                    result += np.random.randn() * noise_std * np.sqrt(result + 1)
                
                return result
            
            bounds = np.array([[-5, 10]] * n_dims)
            optimum = 0.0
            description = "Narrow curved valley"
        
        elif base_func == 'rastrigin':
            def func(**kwargs):
                x = np.array([kwargs[f'x{i}'] for i in range(n_dims)])
                
                if cost_delay > 0:
                    time.sleep(cost_delay)
                
                A = 10
                result = A * n_dims + sum(x**2 - A * np.cos(2 * np.pi * x))
                
                if noise_std > 0:
                    result += np.random.randn() * noise_std * (1.0 + np.sqrt(result))
                
                return result
            
            bounds = np.array([[-5.12, 5.12]] * n_dims)
            optimum = 0.0
            description = "Highly multimodal"
        
        elif base_func == 'ackley':
            def func(**kwargs):
                x = np.array([kwargs[f'x{i}'] for i in range(n_dims)])
                
                if cost_delay > 0:
                    time.sleep(cost_delay)
                
                a, b, c = 20, 0.2, 2*np.pi
                sum1 = np.sum(x**2)
                sum2 = np.sum(np.cos(c * x))
                result = -a * np.exp(-b * np.sqrt(sum1/n_dims)) - np.exp(sum2/n_dims) + a + np.e
                
                if noise_std > 0:
                    result += np.random.randn() * noise_std
                
                return result
            
            bounds = np.array([[-32.768, 32.768]] * n_dims)
            optimum = 0.0
            description = "Multimodal with deep basin"
        
        elif base_func == 'griewank':
            def func(**kwargs):
                x = np.array([kwargs[f'x{i}'] for i in range(n_dims)])
                
                if cost_delay > 0:
                    time.sleep(cost_delay)
                
                sum_term = np.sum(x**2) / 4000
                prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n_dims + 1))))
                result = sum_term - prod_term + 1
                
                if noise_std > 0:
                    result += np.random.randn() * noise_std
                
                return result
            
            bounds = np.array([[-600, 600]] * n_dims)
            optimum = 0.0
            description = "Product term coupling"
        
        else:
            raise ValueError(f"Unknown base function: {base_func}")
        
        # Create space definition
        space = {
            f'x{i}': {'type': 'continuous', 'bounds': [float(bounds[i, 0]), float(bounds[i, 1])]}
            for i in range(n_dims)
        }
        
        name = f"{base_func.capitalize()}-{n_dims}D-{cost_class}-{noise_class}"
        
        return ProblemSpec(
            name=name,
            category='synthetic',
            dimension_class=dim_class,
            cost_class=cost_class,
            noise_class=noise_class,
            dimension=n_dims,
            func=func,
            bounds=bounds,
            space=space,
            optimum=optimum,
            description=f"{description} ({n_dims}D, {cost_class} cost, {noise_class} noise)",
            supports_batching=False
        )


# ============================================================================
# Real-World Applications with Data Batching
# ============================================================================

class RealWorldProblems:
    """Real-world optimization problems with data batching support."""
    
    @staticmethod
    def lightgbm_classification(dataset_name: str = 'breast_cancer'):
        """LightGBM hyperparameter tuning with mini-batch CV."""
        
        try:
            import lightgbm as lgb
            from sklearn.datasets import load_breast_cancer, load_digits, load_wine
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return None
        
        # Load dataset
        if dataset_name == 'breast_cancer':
            X, y = load_breast_cancer(return_X_y=True)
            n_folds_full = 5
        elif dataset_name == 'digits':
            X, y = load_digits(return_X_y=True)
            n_folds_full = 5
        elif dataset_name == 'wine':
            X, y = load_wine(return_X_y=True)
            n_folds_full = 3
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        eval_count = {'count': 0, 'batch_sizes': []}
        
        def func(num_leaves, learning_rate, feature_fraction, bagging_fraction, bagging_freq, min_child_samples, batch_size: int = None):
            """
            Objective with batch_size support.
            
            batch_size controls number of CV folds:
            - None or < 0: Full CV
            - Small: Fewer folds or subset of data
            """
            eval_count['count'] += 1
            
            lgb_params = {
                'objective': 'multiclass' if dataset_name == 'digits' else 'binary',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'min_child_samples': min_child_samples,
                'verbose': -1,
                'n_jobs': 1
            }
            
            # Determine CV strategy based on batch_size
            if batch_size is not None and batch_size > 0:
                # Progressive CV: fewer folds early, more folds later
                if batch_size < 500:
                    n_folds = 2
                elif batch_size < 2000:
                    n_folds = 3
                else:
                    n_folds = n_folds_full
                
                # Also subsample data for very small batches
                if batch_size < len(X):
                    indices = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
                    X_eval, y_eval = X[indices], y[indices]
                else:
                    X_eval, y_eval = X, y
                
                eval_count['batch_sizes'].append(batch_size)
            else:
                n_folds = n_folds_full
                X_eval, y_eval = X, y
                eval_count['batch_sizes'].append(len(X))
            
            # Train and evaluate
            model = lgb.LGBMClassifier(**lgb_params, n_estimators=50, random_state=42)
            
            try:
                scores = cross_val_score(
                    model, X_eval, y_eval, 
                    cv=min(n_folds, len(np.unique(y_eval))), 
                    scoring='neg_log_loss' if dataset_name != 'digits' else 'accuracy'
                )
                result = -scores.mean() if dataset_name != 'digits' else 1.0 - scores.mean()
            except:
                result = 1e10  # Penalty for invalid config
            
            return result
        
        space = {
            'num_leaves': {'type': 'ordinal', 'values': [15, 31, 63, 127, 255]},
            'learning_rate': {'type': 'continuous', 'bounds': [0.001, 0.3], 'log': True},
            'feature_fraction': {'type': 'continuous', 'bounds': [0.5, 1.0]},
            'bagging_fraction': {'type': 'continuous', 'bounds': [0.5, 1.0]},
            'bagging_freq': {'type': 'ordinal', 'values': [0, 1, 5, 10]},
            'min_child_samples': {'type': 'ordinal', 'values': [5, 10, 20, 50]},
        }
        
        bounds = np.array([
            [15, 255],
            [0.001, 0.3],
            [0.5, 1.0],
            [0.5, 1.0],
            [0, 10],
            [5, 50]
        ])
        
        return ProblemSpec(
            name=f"LightGBM-{dataset_name}",
            category='ml_hyperparams',
            dimension_class='small',
            cost_class='high',
            noise_class='medium',
            dimension=6,
            func=func,
            bounds=bounds,
            space=space,
            optimum=None,
            description=f"LightGBM tuning on {dataset_name} with progressive CV",
            supports_batching=True
        )
    
    @staticmethod
    def neural_network_tuning():
        """Neural network hyperparameter tuning with batch size support."""
        
        try:
            from sklearn.datasets import load_digits
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return None
        
        X, y = load_digits(return_X_y=True)
        
        def func(hidden_size, alpha, learning_rate_init, activation, batch_size: int = None):
            hidden_size = int(hidden_size)
            
            # Progressive evaluation
            if batch_size is not None and batch_size > 0:
                if batch_size < 500:
                    max_iter = 50
                    n_folds = 2
                elif batch_size < 1000:
                    max_iter = 100
                    n_folds = 3
                else:
                    max_iter = 200
                    n_folds = 5
                
                # Subsample data
                if batch_size < len(X):
                    indices = np.random.choice(len(X), batch_size, replace=False)
                    X_eval, y_eval = X[indices], y[indices]
                else:
                    X_eval, y_eval = X, y
            else:
                max_iter = 200
                n_folds = 5
                X_eval, y_eval = X, y
            
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                activation=activation,
                max_iter=max_iter,
                random_state=42
            )
            
            try:
                scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
                return 1.0 - scores.mean()  # Return error
            except:
                return 1e10
        
        space = {
            'hidden_size': {'type': 'ordinal', 'values': [32, 64, 128, 256]},
            'alpha': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
            'learning_rate_init': {'type': 'continuous', 'bounds': [1e-4, 1e-1], 'log': True},
            'activation': {'type': 'categorical', 'values': ['relu', 'tanh', 'logistic']},
        }
        
        bounds = np.array([
            [32, 256],
            [1e-5, 1e-1],
            [1e-4, 1e-1],
            [0, 2]
        ])
        
        return ProblemSpec(
            name="NeuralNet-Digits",
            category='ml_hyperparams',
            dimension_class='small',
            cost_class='high',
            noise_class='medium',
            dimension=4,
            func=func,
            bounds=bounds,
            space=space,
            optimum=None,
            description="Neural network tuning with progressive training",
            supports_batching=True
        )
    
    @staticmethod
    def portfolio_optimization(n_assets: int, n_periods: int = 252):
        """Portfolio optimization with data batching (different time periods)."""
        
        # Generate synthetic return data
        np.random.seed(42)
        returns = np.random.randn(n_periods, n_assets) * 0.01
        
        # Add some structure (correlations)
        factor = np.random.randn(n_periods, 3) * 0.015
        loadings = np.random.randn(n_assets, 3)
        returns = returns + factor @ loadings.T
        
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)
        
        def func(**kwargs):
            """
            Portfolio Sharpe ratio.
            
            batch_size: number of periods to use for estimation
            """
            batch_size = kwargs.pop('batch_size', None)
            weights = np.array([kwargs[f'w{i}'] for i in range(n_assets)])
            
            # Normalize
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            # Use subset of periods if batch_size specified
            if batch_size is not None and batch_size > 0 and batch_size < n_periods:
                indices = np.random.choice(n_periods, batch_size, replace=False)
                returns_batch = returns[indices]
                mean_ret = returns_batch.mean(axis=0)
                cov_mat = np.cov(returns_batch.T)
            else:
                mean_ret = mean_returns
                cov_mat = cov_matrix
            
            portfolio_return = weights @ mean_ret
            portfolio_risk = np.sqrt(weights @ cov_mat @ weights)
            
            if portfolio_risk < 1e-8:
                return 1e10
            
            sharpe = portfolio_return / portfolio_risk
            
            return -sharpe  # Minimize negative Sharpe
        
        space = {
            f'w{i}': {'type': 'continuous', 'bounds': [0.0, 1.0]}
            for i in range(n_assets)
        }
        
        bounds = np.array([[0.0, 1.0]] * n_assets)
        
        dim_class = 'small' if n_assets <= 5 else ('medium' if n_assets <= 30 else 'large')
        
        return ProblemSpec(
            name=f"Portfolio-{n_assets}assets",
            category='portfolio',
            dimension_class=dim_class,
            cost_class='medium',
            noise_class='low',
            dimension=n_assets,
            func=func,
            bounds=bounds,
            space=space,
            optimum=None,
            description=f"Portfolio optimization with {n_assets} assets, {n_periods} periods",
            supports_batching=True
        )
    
    @staticmethod
    def svm_tuning():
        """SVM hyperparameter tuning with progressive validation."""
        
        try:
            from sklearn.svm import SVC
            from sklearn.datasets import load_digits
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return None
        
        X, y = load_digits(return_X_y=True)
        
        def func(C, gamma, kernel, batch_size: int = None):
            
            # Progressive evaluation
            if batch_size is not None and batch_size > 0:
                if batch_size < 500:
                    n_folds = 2
                    max_iter = 100
                elif batch_size < 1000:
                    n_folds = 3
                    max_iter = 200
                else:
                    n_folds = 5
                    max_iter = 500
                
                if batch_size < len(X):
                    indices = np.random.choice(len(X), batch_size, replace=False)
                    X_eval, y_eval = X[indices], y[indices]
                else:
                    X_eval, y_eval = X, y
            else:
                n_folds = 5
                max_iter = 500
                X_eval, y_eval = X, y
            
            model = SVC(C=C, gamma=gamma, kernel=kernel, max_iter=max_iter, random_state=42)
            
            try:
                scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
                return 1.0 - scores.mean()
            except:
                return 1e10
        
        space = {
            'C': {'type': 'continuous', 'bounds': [0.1, 100.0], 'log': True},
            'gamma': {'type': 'continuous', 'bounds': [1e-4, 1.0], 'log': True},
            'kernel': {'type': 'categorical', 'values': ['rbf', 'poly', 'sigmoid']},
        }
        
        bounds = np.array([
            [0.1, 100.0],
            [1e-4, 1.0],
            [0, 2]
        ])
        
        return ProblemSpec(
            name=f"SVM-{dataset_name}",
            category='ml_hyperparams',
            dimension_class='small',
            cost_class='high',
            noise_class='medium',
            dimension=3,
            func=func,
            bounds=bounds,
            space=space,
            optimum=None,
            description=f"SVM tuning on {dataset_name} with progressive validation",
            supports_batching=True
        )
    
    @staticmethod
    def random_forest_tuning():
        """Random Forest hyperparameter tuning."""
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return None
        
        X, y = load_breast_cancer(return_X_y=True)
        
        def func(n_estimators, max_depth, min_samples_split, max_features, batch_size: int = None):
            n_estimators = int(n_estimators)
            max_depth = int(max_depth) if max_depth > 0 else None
            min_samples_split = int(min_samples_split)
            
            # Progressive evaluation
            if batch_size is not None and batch_size > 0:
                if batch_size < 500:
                    n_folds = 2
                    n_est_use = min(50, n_estimators)
                elif batch_size < 1000:
                    n_folds = 3
                    n_est_use = min(100, n_estimators)
                else:
                    n_folds = 5
                    n_est_use = n_estimators
                
                if batch_size < len(X):
                    indices = np.random.choice(len(X), batch_size, replace=False)
                    X_eval, y_eval = X[indices], y[indices]
                else:
                    X_eval, y_eval = X, y
            else:
                n_folds = 5
                n_est_use = n_estimators
                X_eval, y_eval = X, y
            
            model = RandomForestClassifier(
                n_estimators=n_est_use,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=42,
                n_jobs=1
            )
            
            try:
                scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
                return 1.0 - scores.mean()
            except:
                return 1e10
        
        space = {
            'n_estimators': {'type': 'ordinal', 'values': [10, 50, 100, 200, 500]},
            'max_depth': {'type': 'ordinal', 'values': [5, 10, 20, 30, -1]},  # -1 = None
            'min_samples_split': {'type': 'continuous', 'bounds': [2.0, 20.0]},
            'max_features': {'type': 'continuous', 'bounds': [0.1, 1.0]},
        }
        
        bounds = np.array([
            [10, 500],
            [5, 30],
            [2, 20],
            [0.1, 1.0]
        ])
        
        return ProblemSpec(
            name="RandomForest-BreastCancer",
            category='ml_hyperparams',
            dimension_class='small',
            cost_class='high',
            noise_class='low',
            dimension=4,
            func=func,
            bounds=bounds,
            space=space,
            optimum=None,
            description="Random Forest tuning with progressive n_estimators",
            supports_batching=True
        )


# ============================================================================
# Generate Complete Problem Suite
# ============================================================================

def generate_comprehensive_suite() -> List[ProblemSpec]:
    """
    Generate comprehensive benchmark suite.
    
    Systematic coverage of:
    - Dimension: small (2-5), medium (10-30), large (50-100)
    - Cost: low (<0.01s), medium (0.01-0.1s), high (>0.1s)
    - Noise: none, low, medium, high
    
    = 3 × 3 × 4 = 36 categories
    × 5 problems per category = 180 total problems
    + Real-world applications
    """
    problems = []
    
    print("\n" + "="*80)
    print("Generating Comprehensive Problem Suite")
    print("="*80)
    
    # Define test functions to use
    base_functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank']
    
    # Define systematic parameter grid
    dimension_configs = {
        'small': [2, 3, 5],
        'medium': [10, 15, 20, 30],
        'large': [50, 75, 100]
    }
    
    cost_classes = ['low', 'medium', 'high']
    noise_classes = ['none', 'low', 'medium', 'high']
    
    # Generate synthetic problems
    print("\nGenerating synthetic problems...")
    
    for dim_class, dims in dimension_configs.items():
        print(f"\n  {dim_class.upper()} dimension ({dims}):")
        
        for cost_class in cost_classes:
            for noise_class in noise_classes:
                print(f"    {cost_class} cost, {noise_class} noise:", end=' ')
                
                # Select subset of functions for this category
                if dim_class == 'large':
                    # Only use simple functions for large dimensions
                    funcs = ['sphere', 'rosenbrock', 'rastrigin']
                else:
                    funcs = base_functions
                
                # For expensive + high noise, reduce number of problems
                if cost_class == 'high' and noise_class == 'high':
                    funcs = funcs[:2]  # Only 2 functions
                    dims_subset = dims[:2]  # Only 2 dimensions
                elif cost_class == 'high':
                    dims_subset = dims[:3]  # 3 dimensions for expensive
                else:
                    dims_subset = dims
                
                count = 0
                for func_name in funcs:
                    for dim in dims_subset:
                        prob = SyntheticFunctions.create_function(
                            func_name, dim, cost_class, noise_class
                        )
                        problems.append(prob)
                        count += 1
                
                print(f"{count} problems")
    
    # Real-world problems with data batching
    print("\n  REAL-WORLD applications:")
    
    real_world = RealWorldProblems()
    
    # LightGBM on multiple datasets
    for dataset in ['breast_cancer', 'digits', 'wine']:
        lgb_prob = real_world.lightgbm_classification(dataset)
        if lgb_prob:
            problems.append(lgb_prob)
            print(f"    ✓ {lgb_prob.name}")
    
    # Neural network
    nn_prob = real_world.neural_network_tuning()
    if nn_prob:
        problems.append(nn_prob)
        print(f"    ✓ {nn_prob.name}")
    
    # SVM
    svm_prob = real_world.svm_tuning()
    if svm_prob:
        problems.append(svm_prob)
        print(f"    ✓ {svm_prob.name}")
    
    # Random Forest
    rf_prob = real_world.random_forest_tuning()
    if rf_prob:
        problems.append(rf_prob)
        print(f"    ✓ {rf_prob.name}")
    
    # Portfolio optimization (various sizes)
    for n_assets in [5, 10, 20, 50]:
        port_prob = real_world.portfolio_optimization(n_assets)
        if port_prob:
            problems.append(port_prob)
            print(f"    ✓ {port_prob.name}")
    
    print(f"\n{'='*80}")
    print(f"Total problems generated: {len(problems)}")
    
    # Print breakdown
    print("\nBreakdown by category:")
    breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for prob in problems:
        breakdown[prob.dimension_class][prob.cost_class][prob.noise_class] += 1
    
    for dim_class in ['small', 'medium', 'large']:
        print(f"\n  {dim_class.upper()}:")
        for cost_class in ['low', 'medium', 'high']:
            for noise_class in ['none', 'low', 'medium', 'high']:
                count = breakdown[dim_class][cost_class][noise_class]
                if count > 0:
                    print(f"    {cost_class:6s} cost, {noise_class:6s} noise: {count:3d} problems")
    
    print(f"\n  REAL-WORLD: {sum(1 for p in problems if p.category != 'synthetic')} problems")
    print(f"{'='*80}")
    
    return problems


# ============================================================================
# Optimizer Wrappers (Same as before, with batching support)
# ============================================================================

class OptimizerWrapper:
    """Base class for optimizer wrappers."""
    
    def __init__(self, name: str):
        self.name = name
        self.supports_batching = False
    
    def optimize(
        self, 
        problem: ProblemSpec, 
        budget: int, 
        random_state: int,
        use_batching: bool = False
    ) -> Dict:
        """Run optimization and return results."""
        raise NotImplementedError


class RAGDAWrapper(OptimizerWrapper):
    """RAGDA optimizer wrapper with batching support."""
    
    def __init__(self):
        super().__init__("RAGDA")
        self.supports_batching = True
    
    def optimize(self, problem: ProblemSpec, budget: int, random_state: int, 
                 use_batching: bool = False) -> Dict:
        
        optimizer = RAGDAOptimizer(
            space=problem.space,
            direction='minimize',
            n_workers=4,
            random_state=random_state
        )
        
        # Adaptive parameters based on problem
        lambda_start = min(50, max(20, 4 + 3*int(np.log(problem.dimension))))
        shrink_patience = max(10, problem.dimension // 3)
        
        # Mini-batch for problems that support it
        if use_batching and problem.supports_batching:
            use_mb = True
            # Estimate dataset size
            if 'LightGBM' in problem.name or 'SVM' in problem.name:
                mb_start = 200
                mb_end = 2000
            elif 'NeuralNet' in problem.name:
                mb_start = 300
                mb_end = 1800
            elif 'Portfolio' in problem.name:
                mb_start = 50
                mb_end = 252
            else:
                mb_start = 100
                mb_end = 1000
        else:
            use_mb = False
            mb_start = None
            mb_end = None
        
        start_time = time.time()
        
        result = optimizer.optimize(
            problem.func,
            n_trials=budget,
            lambda_start=lambda_start,
            lambda_end=10,
            shrink_patience=shrink_patience,
            use_minibatch=use_mb,
            minibatch_start=mb_start,
            minibatch_end=mb_end,
            minibatch_schedule='hyperband',
            verbose=False
        )
        
        elapsed = time.time() - start_time
        
        # Extract convergence history
        if len(result.trials) > 0:
            df = result.trials_df
            df_sorted = df.sort_values('trial_id')
            cummin = df_sorted['value'].cummin()
            history = list(zip(range(len(cummin)), cummin.values))
        else:
            history = [(0, result.best_value)]
        
        return {
            'best_value': result.best_value,
            'best_params': result.best_params,
            'history': history,
            'time': elapsed,
            'evaluations': result.n_trials,
            'used_batching': use_mb
        }


class CMAESWrapper(OptimizerWrapper):
    """CMA-ES wrapper."""
    
    def __init__(self):
        super().__init__("CMA-ES")
        self.supports_batching = False
    
    def optimize(self, problem: ProblemSpec, budget: int, random_state: int,
                 use_batching: bool = False) -> Dict:
        np.random.seed(random_state)
        
        # Convert dict-based objective to array-based
        def func_array(x):
            if problem.space:
                params = {problem.space[i]['name']: x[i] for i in range(len(x))}
                return problem.func(params)
            return problem.func(x)
        
        x0 = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        sigma0 = 0.25 * np.mean(problem.bounds[:, 1] - problem.bounds[:, 0])
        
        start_time = time.time()
        
        opts = {
            'bounds': [problem.bounds[:, 0].tolist(), problem.bounds[:, 1].tolist()],
            'maxfevals': budget * 30,  # CMA-ES budget is in function evaluations
            'seed': random_state,
            'verbose': -9
        }
        
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            history = []
            best_so_far = float('inf')
            iteration = 0
            
            while not es.stop() and iteration < budget:
                solutions = es.ask()
                fitness = [func_array(x) for x in solutions]
                es.tell(solutions, fitness)
                
                current_best = min(fitness)
                if current_best < best_so_far:
                    best_so_far = current_best
                
                history.append((iteration, best_so_far))
                iteration += 1
            
            elapsed = time.time() - start_time
            
            return {
                'best_value': es.result.fbest,
                'best_params': es.result.xbest,
                'history': history,
                'time': elapsed,
                'evaluations': es.result.evaluations,
                'used_batching': False
            }
        
        except Exception as e:
            return {
                'best_value': float('inf'),
                'best_params': None,
                'history': [],
                'time': 0.0,
                'evaluations': 0,
                'used_batching': False,
                'error': str(e)
            }


class OptunaWrapper(OptimizerWrapper):
    """Optuna TPE wrapper."""
    
    def __init__(self):
        super().__init__("Optuna-TPE")
        self.supports_batching = False
    
    def optimize(self, problem: ProblemSpec, budget: int, random_state: int,
                 use_batching: bool = False) -> Dict:
        
        def objective_optuna(trial):
            params = {}
            for param_def in problem.space:
                name = param_def['name']
                ptype = param_def['type']
                
                if ptype == 'continuous':
                    if param_def.get('log', False):
                        params[name] = trial.suggest_float(
                            name, param_def['bounds'][0], param_def['bounds'][1], log=True
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, param_def['bounds'][0], param_def['bounds'][1]
                        )
                elif ptype == 'ordinal':
                    params[name] = trial.suggest_categorical(name, param_def['values'])
                elif ptype == 'categorical':
                    params[name] = trial.suggest_categorical(name, param_def['values'])
            
            return problem.func(params)
        
        start_time = time.time()
        
        try:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=random_state)
            )
            
            study.optimize(objective_optuna, n_trials=budget, show_progress_bar=False)
            
            elapsed = time.time() - start_time
            
            # Extract history
            history = []
            best_so_far = float('inf')
            for i, trial in enumerate(study.trials):
                if trial.value and trial.value < best_so_far:
                    best_so_far = trial.value
                history.append((i, best_so_far))
            
            return {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'history': history,
                'time': elapsed,
                'evaluations': len(study.trials),
                'used_batching': False
            }
        
        except Exception as e:
            return {
                'best_value': float('inf'),
                'best_params': None,
                'history': [],
                'time': 0.0,
                'evaluations': 0,
                'used_batching': False,
                'error': str(e)
            }


class RandomSearchWrapper(OptimizerWrapper):
    """Random search baseline."""
    
    def __init__(self):
        super().__init__("Random")
        self.supports_batching = False
    
    def optimize(self, problem: ProblemSpec, budget: int, random_state: int,
                 use_batching: bool = False) -> Dict:
        np.random.seed(random_state)
        
        start_time = time.time()
        
        best_value = float('inf')
        best_params = None
        history = []
        
        for i in range(budget):
            # Generate random point
            if problem.space:
                params = {}
                for param_def in problem.space:
                    name = param_def['name']
                    ptype = param_def['type']
                    
                    if ptype == 'continuous':
                        if param_def.get('log', False):
                            log_val = np.random.uniform(
                                np.log(param_def['bounds'][0]),
                                np.log(param_def['bounds'][1])
                            )
                            params[name] = np.exp(log_val)
                        else:
                            params[name] = np.random.uniform(
                                param_def['bounds'][0],
                                param_def['bounds'][1]
                            )
                    elif ptype in ['ordinal', 'categorical']:
                        params[name] = np.random.choice(param_def['values'])
                
                value = problem.func(params)
            else:
                x = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
                value = problem.func(x)
                params = x
            
            if value < best_value:
                best_value = value
                best_params = params
            
            history.append((i, best_value))
        
        elapsed = time.time() - start_time
        
        return {
            'best_value': best_value,
            'best_params': best_params,
            'history': history,
            'time': elapsed,
            'evaluations': budget,
            'used_batching': False
        }


# ============================================================================
# Benchmarking Engine
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    problem_name: str
    problem_category: str
    dimension_class: str
    cost_class: str
    noise_class: str
    dimension: int
    optimizer_name: str
    run_id: int
    best_value: float
    optimality_gap: Optional[float]
    time_seconds: float
    n_evaluations: int
    success: bool
    used_batching: bool
    convergence_history: List[Tuple[int, float]]
    error_message: Optional[str] = None


def run_single_benchmark(
    problem: ProblemSpec,
    optimizer: OptimizerWrapper,
    budget: int,
    random_state: int,
    run_id: int,
    timeout: float = 600.0
) -> BenchmarkResult:
    """Run a single benchmark."""
    
    try:
        # Use batching if both support it
        use_batching = optimizer.supports_batching and problem.supports_batching
        
        result = optimizer.optimize(problem, budget, random_state, use_batching)
        
        # Check for errors
        if 'error' in result:
            success = False
            error_msg = result['error']
        elif result['time'] > timeout:
            success = False
            error_msg = f"Timeout: {result['time']:.1f}s"
        else:
            success = True
            error_msg = None
        
        # Compute optimality gap
        if problem.optimum is not None:
            if problem.optimum == 0:
                gap = abs(result['best_value'])
            else:
                gap = abs(result['best_value'] - problem.optimum) / abs(problem.optimum)
        else:
            gap = None
        
        return BenchmarkResult(
            problem_name=problem.name,
            problem_category=problem.category,
            dimension_class=problem.dimension_class,
            cost_class=problem.cost_class,
            noise_class=problem.noise_class,
            dimension=problem.dimension,
            optimizer_name=optimizer.name,
            run_id=run_id,
            best_value=result['best_value'],
            optimality_gap=gap,
            time_seconds=result['time'],
            n_evaluations=result['evaluations'],
            success=success,
            used_batching=result.get('used_batching', False),
            convergence_history=result['history'],
            error_message=error_msg
        )
    
    except Exception as e:
        return BenchmarkResult(
            problem_name=problem.name,
            problem_category=problem.category,
            dimension_class=problem.dimension_class,
            cost_class=problem.cost_class,
            noise_class=problem.noise_class,
            dimension=problem.dimension,
            optimizer_name=optimizer.name,
            run_id=run_id,
            best_value=float('inf'),
            optimality_gap=float('inf'),
            time_seconds=0.0,
            n_evaluations=0,
            success=False,
            used_batching=False,
            convergence_history=[],
            error_message=str(e)
        )


def run_comprehensive_benchmark(
    problems: List[ProblemSpec],
    n_runs: int = 5,
    budget: int = 100,
    output_dir: str = './benchmark_results'
) -> Tuple[pd.DataFrame, List[BenchmarkResult]]:
    """
    Run comprehensive benchmark suite.
    
    Parameters
    ----------
    problems : list
        List of benchmark problems
    n_runs : int
        Number of independent runs per (problem, optimizer) pair
    budget : int
        Iteration budget per run
    output_dir : str
        Directory to save results
    
    Returns
    -------
    df : DataFrame
        Summary results
    detailed_results : list
        Full results with convergence histories
    """
    # Initialize optimizers
    optimizers = []
    
    if OPTIMIZERS_AVAILABLE['RAGDA']:
        optimizers.append(RAGDAWrapper())
    
    if OPTIMIZERS_AVAILABLE['CMA-ES']:
        optimizers.append(CMAESWrapper())
    
    if OPTIMIZERS_AVAILABLE['Optuna']:
        optimizers.append(OptunaWrapper())
    
    optimizers.append(RandomSearchWrapper())
    
    print(f"\n{'='*80}")
    print("Running Comprehensive Benchmark Suite")
    print(f"{'='*80}")
    print(f"Problems: {len(problems)}")
    print(f"Optimizers: {len(optimizers)} ({', '.join(o.name for o in optimizers)})")
    print(f"Runs per pair: {n_runs}")
    print(f"Budget per run: {budget}")
    print(f"Total benchmarks: {len(problems) * len(optimizers) * n_runs}")
    
    # Estimate time
    avg_time_per_run = 2.0  # seconds (conservative)
    total_estimated = len(problems) * len(optimizers) * n_runs * avg_time_per_run
    print(f"Estimated time: {total_estimated/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # Track progress
    total_benchmarks = len(problems) * len(optimizers) * n_runs
    current = 0
    start_time = time.time()
    
    for prob_idx, problem in enumerate(problems):
        print(f"\n[{prob_idx+1}/{len(problems)}] {problem.name}")
        print(f"  Category: {problem.category}")
        print(f"  Specs: {problem.dimension}D, {problem.cost_class} cost, {problem.noise_class} noise")
        print(f"  Batching: {'Yes' if problem.supports_batching else 'No'}")
        
        for opt in optimizers:
            opt_start = time.time()
            
            for run in range(n_runs):
                current += 1
                elapsed = time.time() - start_time
                eta = (elapsed / current) * (total_benchmarks - current)
                
                print(f"    {opt.name:15s} run {run+1}/{n_runs} "
                      f"[{current}/{total_benchmarks}, ETA: {eta/60:.1f}min]...", 
                      end=' ', flush=True)
                
                random_state = 42 + prob_idx * 1000 + run
                
                result = run_single_benchmark(
                    problem, opt, budget, random_state, run
                )
                
                all_results.append(result)
                
                if result.success:
                    batch_str = " +batch" if result.used_batching else ""
                    print(f"✓ f={result.best_value:.6f} ({result.time_seconds:.2f}s{batch_str})")
                else:
                    print(f"✗ {result.error_message}")
            
            opt_elapsed = time.time() - opt_start
            print(f"    {opt.name} completed in {opt_elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Benchmarking Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful runs: {sum(1 for r in all_results if r.success)}/{len(all_results)}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'problem': r.problem_name,
        'category': r.problem_category,
        'dim_class': r.dimension_class,
        'cost_class': r.cost_class,
        'noise_class': r.noise_class,
        'dimension': r.dimension,
        'optimizer': r.optimizer_name,
        'run': r.run_id,
        'best_value': r.best_value,
        'optimality_gap': r.optimality_gap,
        'time': r.time_seconds,
        'evaluations': r.n_evaluations,
        'success': r.success,
        'used_batching': r.used_batching,
        'error': r.error_message
    } for r in all_results])
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    df.to_csv(f"{output_dir}/benchmark_comprehensive.csv", index=False)
    print(f"\nSaved: {output_dir}/benchmark_comprehensive.csv")
    
    # Save detailed results with convergence histories
    detailed_json = []
    for r in all_results:
        r_dict = {
            'problem': r.problem_name,
            'optimizer': r.optimizer_name,
            'run': r.run_id,
            'best_value': r.best_value,
            'time': r.time_seconds,
            'used_batching': r.used_batching,
            'convergence': r.convergence_history
        }
        detailed_json.append(r_dict)
    
    with open(f"{output_dir}/benchmark_detailed.json", 'w') as f:
        json.dump(detailed_json, f, indent=2)
    print(f"Saved: {output_dir}/benchmark_detailed.json")
    
    return df, all_results


# ============================================================================
# Analysis by Problem Class
# ============================================================================

def analyze_by_problem_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results systematically by problem characteristics.
    
    Returns comprehensive breakdown table.
    """
    print(f"\n{'='*80}")
    print("Analysis by Problem Class")
    print(f"{'='*80}\n")
    
    # Create comprehensive breakdown
    analysis_results = []
    
    for dim_class in ['small', 'medium', 'large']:
        for cost_class in ['low', 'medium', 'high']:
            for noise_class in ['none', 'low', 'medium', 'high']:
                
                # Filter to this class
                class_df = df[
                    (df['dim_class'] == dim_class) &
                    (df['cost_class'] == cost_class) &
                    (df['noise_class'] == noise_class)
                ]
                
                if len(class_df) == 0:
                    continue
                
                print(f"{dim_class.upper():6s} dim, {cost_class.upper():6s} cost, {noise_class.upper():6s} noise:")
                print(f"  Problems: {class_df['problem'].nunique()}, Total runs: {len(class_df)}")
                
                # Performance by optimizer
                for opt in sorted(class_df['optimizer'].unique()):
                    opt_df = class_df[class_df['optimizer'] == opt]
                    
                    mean_val = opt_df['best_value'].mean()
                    std_val = opt_df['best_value'].std()
                    median_val = opt_df['best_value'].median()
                    success_rate = opt_df['success'].sum() / len(opt_df) * 100
                    mean_time = opt_df['time'].mean()
                    
                    print(f"    {opt:15s}: mean={mean_val:10.6f} ±{std_val:8.6f}, "
                          f"median={median_val:10.6f}, success={success_rate:5.1f}%, "
                          f"time={mean_time:6.2f}s")
                    
                    analysis_results.append({
                        'dim_class': dim_class,
                        'cost_class': cost_class,
                        'noise_class': noise_class,
                        'optimizer': opt,
                        'mean_value': mean_val,
                        'std_value': std_val,
                        'median_value': median_val,
                        'success_rate': success_rate,
                        'mean_time': mean_time,
                        'n_problems': class_df['problem'].nunique(),
                        'n_runs': len(opt_df)
                    })
                
                # Find best optimizer for this class
                best_opt = class_df.groupby('optimizer')['best_value'].mean().idxmin()
                best_val = class_df.groupby('optimizer')['best_value'].mean().min()
                print(f"    → Best: {best_opt} (f={best_val:.6f})")
                print()
    
    analysis_df = pd.DataFrame(analysis_results)
    return analysis_df


def analyze_batching_benefit(df: pd.DataFrame, detailed_results: List[BenchmarkResult]):
    """Analyze benefit of mini-batch feature on applicable problems."""
    
    print(f"\n{'='*80}")
    print("Mini-batch Feature Analysis")
    print(f"{'='*80}\n")
    
    # Filter to problems that support batching
    batch_problems = df[df['problem'].str.contains('LightGBM|NeuralNet|SVM|Portfolio|RandomForest')]
    
    if len(batch_problems) == 0:
        print("No batching-capable problems found")
        return
    
    print(f"Problems with batching support: {batch_problems['problem'].nunique()}")
    print()
    
    # Compare RAGDA with vs without batching (if we ran both)
    ragda_batch = batch_problems[
        (batch_problems['optimizer'] == 'RAGDA') & 
        (batch_problems['used_batching'] == True)
    ]
    
    if len(ragda_batch) > 0:
        print("RAGDA with mini-batch:")
        for prob in ragda_batch['problem'].unique():
            prob_df = ragda_batch[ragda_batch['problem'] == prob]
            print(f"  {prob:30s}: mean={prob_df['best_value'].mean():.6f}, "
                  f"time={prob_df['time'].mean():.2f}s")
    
    # Compare to other optimizers on same problems
    print("\nComparison on batching-capable problems:")
    for prob in batch_problems['problem'].unique():
        prob_df = batch_problems[batch_problems['problem'] == prob]
        
        print(f"\n  {prob}:")
        perf = prob_df.groupby('optimizer').agg({
            'best_value': ['mean', 'std'],
            'time': 'mean'
        }).round(4)
        
        for opt in perf.index:
            mean_val = perf.loc[opt, ('best_value', 'mean')]
            std_val = perf.loc[opt, ('best_value', 'std')]
            mean_time = perf.loc[opt, ('time', 'mean')]
            
            batch_mark = ""
            if opt == 'RAGDA':
                ragda_prob = prob_df[prob_df['optimizer'] == 'RAGDA']
                if ragda_prob['used_batching'].any():
                    batch_mark = " [+batch]"
            
            print(f"    {opt:15s}{batch_mark}: {mean_val:.6f} ±{std_val:.6f}  ({mean_time:.2f}s)")


# ============================================================================
# Generate LaTeX Results Tables
# ============================================================================

def generate_comprehensive_tables(df: pd.DataFrame, analysis_df: pd.DataFrame, output_dir: str):
    """Generate publication tables broken down by problem class."""
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Generating LaTeX Tables")
    print(f"{'='*80}\n")
    
    # Table 1: Overall summary
    overall = df.groupby('optimizer').agg({
        'best_value': ['mean', 'std', 'median'],
        'time': ['mean', 'median'],
        'success': lambda x: f"{x.sum()}/{len(x)}"
    }).round(4)
    
    overall.columns = ['_'.join(col) for col in overall.columns]
    overall = overall.sort_values('best_value_mean')
    
    latex = overall.to_latex(
        caption="Overall performance across all 180+ benchmark problems",
        label="tab:overall"
    )
    
    with open(f"{output_dir}/table_overall.tex", 'w') as f:
        f.write(latex)
    print(f"  ✓ table_overall.tex")
    
    # Table 2: By dimension class
    dim_pivot = analysis_df.pivot_table(
        values='mean_value',
        index='dim_class',
        columns='optimizer',
        aggfunc='mean'
    ).round(4)
    
    latex = dim_pivot.to_latex(
        caption="Mean performance by dimensionality class",
        label="tab:by_dimension"
    )
    
    with open(f"{output_dir}/table_by_dimension.tex", 'w') as f:
        f.write(latex)
    print(f"  ✓ table_by_dimension.tex")
    
    # Table 3: By cost class
    cost_pivot = analysis_df.pivot_table(
        values='mean_value',
        index='cost_class',
        columns='optimizer'
    ).round(4)
    
    latex = cost_pivot.to_latex(
        caption="Mean performance by evaluation cost class",
        label="tab:by_cost"
    )
    
    with open(f"{output_dir}/table_by_cost.tex", 'w') as f:
        f.write(latex)
    print(f"  ✓ table_by_cost.tex")
    
    # Table 4: By noise class
    noise_pivot = analysis_df.pivot_table(
        values='mean_value',
        index='noise_class',
        columns='optimizer',
        aggfunc='mean'
    ).round(4)
    
    latex = noise_pivot.to_latex(
        caption="Mean performance by noise level class",
        label="tab:by_noise"
    )
    
    with open(f"{output_dir}/table_by_noise.tex", 'w') as f:
        f.write(latex)
    print(f"  ✓ table_by_noise.tex")
    
    # Table 5: 3D breakdown (dimension × cost × noise) - for RAGDA only
    if 'RAGDA' in df['optimizer'].values:
        ragda_df = df[df['optimizer'] == 'RAGDA']
        
        # Create multi-index table
        breakdown_3d = ragda_df.groupby(['dim_class', 'cost_class', 'noise_class']).agg({
            'best_value': ['mean', 'std', 'count'],
            'time': 'mean'
        }).round(4)
        
        latex = breakdown_3d.to_latex(
            caption="RAGDA performance breakdown by all three problem characteristics",
            label="tab:ragda_3d_breakdown"
        )
        
        with open(f"{output_dir}/table_ragda_breakdown.tex", 'w') as f:
            f.write(latex)
        print(f"  ✓ table_ragda_breakdown.tex")
    
    # Table 6: Win rate matrix (which optimizer wins in each category)
    win_matrix = []
    
    for dim_class in ['small', 'medium', 'large']:
        for cost_class in ['low', 'medium', 'high']:
            for noise_class in ['none', 'low', 'medium', 'high']:
                class_df = df[
                    (df['dim_class'] == dim_class) &
                    (df['cost_class'] == cost_class) &
                    (df['noise_class'] == noise_class)
                ]
                
                if len(class_df) == 0:
                    continue
                
                # Find best optimizer
                best_opt = class_df.groupby('optimizer')['best_value'].mean().idxmin()
                best_val = class_df.groupby('optimizer')['best_value'].mean().min()
                
                win_matrix.append({
                    'Dimension': dim_class,
                    'Cost': cost_class,
                    'Noise': noise_class,
                    'Winner': best_opt,
                    'Best Value': best_val,
                    'N Problems': class_df['problem'].nunique()
                })
    
    win_df = pd.DataFrame(win_matrix)
    
    latex = win_df.to_latex(
        index=False,
        caption="Best optimizer for each problem class combination",
        label="tab:win_matrix"
    )
    
    with open(f"{output_dir}/table_win_matrix.tex", 'w') as f:
        f.write(latex)
    print(f"  ✓ table_win_matrix.tex")
    
    # Table 7: Batching impact (for RAGDA on applicable problems)
    if 'RAGDA' in df['optimizer'].values:
        batch_df = df[
            (df['optimizer'] == 'RAGDA') & 
            df['problem'].str.contains('LightGBM|NeuralNet|SVM|Portfolio|RandomForest')
        ]
        
        if len(batch_df) > 0:
            batch_summary = batch_df.groupby(['problem', 'used_batching']).agg({
                'best_value': ['mean', 'std'],
                'time': 'mean'
            }).round(4)
            
            latex = batch_summary.to_latex(
                caption="RAGDA performance with and without mini-batch feature on applicable problems",
                label="tab:batching_impact"
            )
            
            with open(f"{output_dir}/table_batching_impact.tex", 'w') as f:
                f.write(latex)
            print(f"  ✓ table_batching_impact.tex")


# ============================================================================
# Generate Comprehensive Plots
# ============================================================================

def plot_3d_heatmaps(analysis_df: pd.DataFrame, output_dir: str):
    """Generate heatmaps for each dimension class."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/seaborn not available, skipping plots")
        return
    
    print(f"\n{'='*80}")
    print("Generating 3D Heatmaps (Dimension × Cost × Noise)")
    print(f"{'='*80}\n")
    
    for dim_class in ['small', 'medium', 'large']:
        dim_df = analysis_df[analysis_df['dim_class'] == dim_class]
        
        if len(dim_df) == 0:
            continue
        
        # Create separate heatmap for each optimizer
        optimizers = sorted(dim_df['optimizer'].unique())
        n_opts = len(optimizers)
        
        fig, axes = plt.subplots(1, n_opts, figsize=(5*n_opts, 5))
        if n_opts == 1:
            axes = [axes]
        
        for ax_idx, opt in enumerate(optimizers):
            opt_df = dim_df[dim_df['optimizer'] == opt]
            
            # Pivot: cost × noise
            pivot = opt_df.pivot_table(
                values='mean_value',
                index='cost_class',
                columns='noise_class',
                aggfunc='mean'
            )
            
            # Reorder
            cost_order = ['low', 'medium', 'high']
            noise_order = ['none', 'low', 'medium', 'high']
            pivot = pivot.reindex(index=cost_order, columns=noise_order)
            
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn_r',
                ax=axes[ax_idx],
                cbar_kws={'label': 'Mean Best Value'}
            )
            
            axes[ax_idx].set_title(f'{opt}', fontsize=12, fontweight='bold')
            axes[ax_idx].set_xlabel('Noise Level', fontsize=10)
            axes[ax_idx].set_ylabel('Cost Level', fontsize=10)
        
        plt.suptitle(f'{dim_class.upper()} Dimension Problems', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{output_dir}/heatmap_{dim_class}_dimension.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✓ Saved: heatmap_{dim_class}_dimension.pdf")


def plot_convergence_by_class(detailed_results: List[BenchmarkResult], output_dir: str):
    """Plot convergence curves grouped by problem class."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    print(f"\n{'='*80}")
    print("Generating Convergence Plots by Class")
    print(f"{'='*80}\n")
    
    # Group by problem class
    by_class = defaultdict(lambda: defaultdict(list))
    
    for result in detailed_results:
        key = f"{result.dimension_class}-{result.cost_class}-{result.noise_class}"
        by_class[key][result.optimizer_name].append(result.convergence_history)
    
    # Plot selected representative classes
    selected_classes = [
        ('small-low-none', 'Small Dim, Low Cost, No Noise'),
        ('medium-medium-low', 'Medium Dim, Medium Cost, Low Noise'),
        ('large-low-none', 'Large Dim, Low Cost, No Noise'),
        ('small-high-medium', 'Small Dim, High Cost, Medium Noise (ML tasks)'),
        ('medium-low-high', 'Medium Dim, Low Cost, High Noise'),
    ]
    
    for class_key, title in selected_classes:
        if class_key not in by_class:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for opt_name, histories in by_class[class_key].items():
            if not histories or not histories[0]:
                continue
            
            # Average across runs
            max_len = max(len(h) for h in histories if h)
            
            padded = []
            for h in histories:
                if not h:
                    continue
                iterations, values = zip(*h)
                
                # Pad with last value
                if len(values) < max_len:
                    last_val = values[-1]
                    values = list(values) + [last_val] * (max_len - len(values))
                
                padded.append(values)
            
            if not padded:
                continue
            
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            iterations = np.arange(len(mean_curve))
            
            ax.plot(iterations, mean_curve, label=opt_name, linewidth=2.5, alpha=0.8)
            ax.fill_between(iterations, 
                           mean_curve - std_curve, 
                           mean_curve + std_curve, 
                           alpha=0.2)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Best Value Found (log scale)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{output_dir}/convergence_{class_key}.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✓ Saved: convergence_{class_key}.pdf")


def plot_dimension_scaling(df: pd.DataFrame, output_dir: str):
    """Plot how optimizers scale with dimension."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    print(f"\n{'='*80}")
    print("Generating Dimension Scaling Analysis")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter to clean problems (low noise, low cost)
    clean_df = df[(df['noise_class'] == 'none') & (df['cost_class'] == 'low')]
    
    # Plot 1: Mean value vs dimension
    ax = axes[0, 0]
    for opt in sorted(clean_df['optimizer'].unique()):
        opt_df = clean_df[clean_df['optimizer'] == opt]
        dim_perf = opt_df.groupby('dimension')['best_value'].mean()
        
        ax.plot(dim_perf.index, dim_perf.values, marker='o', label=opt, linewidth=2)
    
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Mean Best Value', fontsize=11)
    ax.set_title('Scaling with Dimension', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time vs dimension
    ax = axes[0, 1]
    for opt in sorted(clean_df['optimizer'].unique()):
        opt_df = clean_df[clean_df['optimizer'] == opt]
        dim_time = opt_df.groupby('dimension')['time'].mean()
        
        ax.plot(dim_time.index, dim_time.values, marker='s', label=opt, linewidth=2)
    
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Mean Time (seconds)', fontsize=11)
    ax.set_title('Computational Cost vs Dimension', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success rate vs dimension
    ax = axes[1, 0]
    for opt in sorted(clean_df['optimizer'].unique()):
        opt_df = clean_df[clean_df['optimizer'] == opt]
        dim_success = opt_df.groupby('dimension')['success'].mean() * 100
        
        ax.plot(dim_success.index, dim_success.values, marker='^', label=opt, linewidth=2)
    
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate vs Dimension', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Optimality gap vs dimension (for problems with known optimum)
    ax = axes[1, 1]
    gap_df = clean_df[clean_df['optimality_gap'].notna()]
    
    if len(gap_df) > 0:
        for opt in sorted(gap_df['optimizer'].unique()):
            opt_df = gap_df[gap_df['optimizer'] == opt]
            dim_gap = opt_df.groupby('dimension')['optimality_gap'].median()
            
            ax.plot(dim_gap.index, dim_gap.values, marker='d', label=opt, linewidth=2)
        
        ax.set_xlabel('Dimension', fontsize=11)
        ax.set_ylabel('Median Optimality Gap', fontsize=11)
        ax.set_title('Solution Quality vs Dimension', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No problems with known optimum', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dimension_scaling.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: dimension_scaling.pdf")


def plot_batching_speedup(df: pd.DataFrame, output_dir: str):
    """Plot speedup from mini-batch feature."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    print(f"\n{'='*80}")
    print("Generating Mini-batch Speedup Analysis")
    print(f"{'='*80}\n")
    
    # Filter to RAGDA on batching problems
    ragda_batch = df[
        (df['optimizer'] == 'RAGDA') &
        df['problem'].str.contains('LightGBM|NeuralNet|SVM|Portfolio|RandomForest')
    ]
    
    if len(ragda_batch) == 0:
        print("  No batching data available")
        return
    
    # Check if we have both batched and non-batched runs
    has_both = ragda_batch.groupby('problem')['used_batching'].nunique().max() > 1
    
    if not has_both:
        print("  Only batched or non-batched runs available, skipping comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time comparison
    ax = axes[0]
    problems = sorted(ragda_batch['problem'].unique())
    
    batch_times = []
    no_batch_times = []
    labels = []
    
    for prob in problems:
        prob_df = ragda_batch[ragda_batch['problem'] == prob]
        
        batch_data = prob_df[prob_df['used_batching'] == True]
        no_batch_data = prob_df[prob_df['used_batching'] == False]
        
        if len(batch_data) > 0 and len(no_batch_data) > 0:
            batch_times.append(batch_data['time'].mean())
            no_batch_times.append(no_batch_data['time'].mean())
            labels.append(prob.split('-')[0])
    
    if batch_times:
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, no_batch_times, width, label='No batching', alpha=0.8)
        ax.bar(x + width/2, batch_times, width, label='With batching', alpha=0.8)
        
        ax.set_xlabel('Problem', fontsize=11)
        ax.set_ylabel('Mean Time (seconds)', fontsize=11)
        ax.set_title('Mini-batch Time Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Quality comparison
    ax = axes[1]
    
    batch_quality = []
    no_batch_quality = []
    
    for prob in problems:
        prob_df = ragda_batch[ragda_batch['problem'] == prob]
        
        batch_data = prob_df[prob_df['used_batching'] == True]
        no_batch_data = prob_df[prob_df['used_batching'] == False]
        
        if len(batch_data) > 0 and len(no_batch_data) > 0:
            batch_quality.append(batch_data['best_value'].mean())
            no_batch_quality.append(no_batch_data['best_value'].mean())
    
    if batch_quality:
        x = np.arange(len(labels))
        
        ax.bar(x - width/2, no_batch_quality, width, label='No batching', alpha=0.8)
        ax.bar(x + width/2, batch_quality, width, label='With batching', alpha=0.8)
        
        ax.set_xlabel('Problem', fontsize=11)
        ax.set_ylabel('Mean Best Value', fontsize=11)
        ax.set_title('Mini-batch Quality Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/batching_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: batching_comparison.pdf")


def plot_category_comparison(df: pd.DataFrame, output_dir: str):
    """Compare performance across different problem categories."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    
    print(f"\n{'='*80}")
    print("Generating Category Comparison")
    print(f"{'='*80}\n")
    
    categories = sorted(df['category'].unique())
    optimizers = sorted(df['optimizer'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.8 / len(optimizers)
    
    for i, opt in enumerate(optimizers):
        means = []
        stds = []
        
        for cat in categories:
            cat_df = df[(df['category'] == cat) & (df['optimizer'] == opt)]
            if len(cat_df) > 0:
                means.append(cat_df['best_value'].mean())
                stds.append(cat_df['best_value'].std())
            else:
                means.append(np.nan)
                stds.append(0)
        
        offset = (i - len(optimizers)/2) * width + width/2
        ax.bar(x + offset, means, width, label=opt, yerr=stds, 
               capsize=3, alpha=0.7)
    
    ax.set_xlabel('Problem Category', fontsize=11)
    ax.set_ylabel('Mean Best Value', fontsize=11)
    ax.set_title('Performance by Problem Category', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: category_comparison.pdf")


# ============================================================================
# Statistical Analysis
# ============================================================================

def statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform statistical tests for paper."""
    
    print(f"\n{'='*80}")
    print("Statistical Analysis")
    print(f"{'='*80}\n")
    
    from scipy.stats import wilcoxon, friedmanchisquare, rankdata
    
    analysis = {}
    
    # Friedman test across all problems
    print("Friedman Test (overall difference across optimizers):")
    
    # Pivot: problems × optimizers
    pivot = df.pivot_table(
        values='best_value',
        index='problem',
        columns='optimizer',
        aggfunc='mean'
    )
    
    # Drop NaNs
    pivot = pivot.dropna()
    
    if len(pivot) >= 3 and len(pivot.columns) >= 3:
        stat, pval = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        print(f"  χ² = {stat:.4f}, p-value = {pval:.6f}")
        
        if pval < 0.05:
            print("  → Significant difference detected among optimizers (p < 0.05)")
        else:
            print("  → No significant difference detected")
        
        analysis['friedman_statistic'] = stat
        analysis['friedman_pvalue'] = pval
    
    # Pairwise comparisons with RAGDA
    if 'RAGDA' in df['optimizer'].values:
        print("\nPairwise Comparisons vs RAGDA (Wilcoxon signed-rank test):")
        print("-" * 80)
        
        ragda_results = df[df['optimizer'] == 'RAGDA'].groupby('problem')['best_value'].mean()
        
        pairwise_results = []
        
        for opt in df['optimizer'].unique():
            if opt == 'RAGDA':
                continue
            
            opt_results = df[df['optimizer'] == opt].groupby('problem')['best_value'].mean()
            
            # Match problems
            common = sorted(set(ragda_results.index) & set(opt_results.index))
            
            if len(common) < 5:
                print(f"  RAGDA vs {opt:15s}: insufficient common problems ({len(common)})")
                continue
            
            ragda_vals = [ragda_results[p] for p in common]
            opt_vals = [opt_results[p] for p in common]
            
            try:
                stat, pval = wilcoxon(ragda_vals, opt_vals, alternative='less')
                
                ragda_better = sum(r < o for r, o in zip(ragda_vals, opt_vals))
                mean_diff = np.mean([r - o for r, o in zip(ragda_vals, opt_vals)])
                
                sig_mark = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                
                print(f"  RAGDA vs {opt:15s}: p={pval:.4f}{sig_mark:3s}  "
                      f"RAGDA better: {ragda_better}/{len(common)}  "
                      f"mean_diff: {mean_diff:+.6f}")
                
                pairwise_results.append({
                    'opponent': opt,
                    'p_value': pval,
                    'n_problems': len(common),
                    'ragda_wins': ragda_better,
                    'mean_difference': mean_diff
                })
            
            except Exception as e:
                print(f"  RAGDA vs {opt:15s}: {e}")
        
        analysis['pairwise_tests'] = pairwise_results
    
    # Effect size analysis
    print("\nEffect Size Analysis (Cohen's d):")
    print("-" * 80)
    
    if 'RAGDA' in df['optimizer'].values:
        ragda_all = df[df['optimizer'] == 'RAGDA']['best_value'].values
        
        for opt in df['optimizer'].unique():
            if opt == 'RAGDA':
                continue
            
            opt_all = df[df['optimizer'] == opt]['best_value'].values
            
            if len(opt_all) > 0:
                mean_diff = np.mean(ragda_all) - np.mean(opt_all)
                pooled_std = np.sqrt((np.std(ragda_all)**2 + np.std(opt_all)**2) / 2)
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    
                    magnitude = "negligible" if abs(cohens_d) < 0.2 else (
                        "small" if abs(cohens_d) < 0.5 else (
                        "medium" if abs(cohens_d) < 0.8 else "large"
                    ))
                    
                    print(f"  RAGDA vs {opt:15s}: d = {cohens_d:+.3f} ({magnitude})")
    
    return analysis


# ============================================================================
# Main Execution
# ============================================================================

def main(
    n_runs: int = 5,
    budget: int = 100,
    output_dir: str = './benchmark_results',
    skip_expensive: bool = False,
    quick_mode: bool = False
):
    """
    Run comprehensive benchmark suite.
    
    Parameters
    ----------
    n_runs : int
        Number of independent runs per (problem, optimizer) pair
    budget : int
        Iteration budget per run
    output_dir : str
        Output directory
    skip_expensive : bool
        Skip expensive problems (for faster testing)
    quick_mode : bool
        Quick mode: fewer runs, smaller budget
    """
    
    if quick_mode:
        n_runs = 3
        budget = 50
        print("\n⚡ QUICK MODE: 3 runs, 50 iterations")
    
    # Generate problem suite
    problems = generate_comprehensive_suite()
    
    # Filter expensive if requested
    if skip_expensive:
        problems = [p for p in problems if p.cost_class != 'high']
        print(f"\nSkipped expensive problems, {len(problems)} remaining")
    
    # Run benchmarks
    df, detailed_results = run_comprehensive_benchmark(
        problems, n_runs, budget, output_dir
    )
    
    # Analyze results
    analysis_df = analyze_by_problem_class(df)
    
    # Batching analysis
    analyze_batching_benefit(df, detailed_results)
    
    # Statistical tests
    stats = statistical_analysis(df)
    
    # Generate tables
    generate_comprehensive_tables(df, analysis_df, output_dir)
    
    # Generate plots
    plot_3d_heatmaps(analysis_df, output_dir)
    plot_convergence_by_class(detailed_results, output_dir)
    plot_dimension_scaling(df, output_dir)
    plot_batching_speedup(df, output_dir)
    plot_category_comparison(df, output_dir)
    
    # Save analysis results
    analysis_df.to_csv(f"{output_dir}/analysis_by_class.csv", index=False)
    
    with open(f"{output_dir}/statistical_analysis.json", 'w') as f:
        # Convert to JSON-serializable
        stats_json = {}
        for k, v in stats.items():
            if isinstance(v, (int, float, str)):
                stats_json[k] = v
            elif isinstance(v, list):
                stats_json[k] = v
        json.dump(stats_json, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BENCHMARKING COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  • benchmark_comprehensive.csv - Raw results")
    print(f"  • benchmark_detailed.json - With convergence histories")
    print(f"  • analysis_by_class.csv - Aggregated by problem class")
    print(f"  • statistical_analysis.json - Statistical test results")
    print(f"  • table_*.tex - LaTeX tables for paper")
    print(f"  • *.pdf - Publication-quality figures")
    print(f"{'='*80}")
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("-" * 80)
    
    if 'RAGDA' in df['optimizer'].values:
        ragda_df = df[df['optimizer'] == 'RAGDA']
        
        print(f"\nRAGDA Overall:")
        print(f"  Mean performance: {ragda_df['best_value'].mean():.6f} (±{ragda_df['best_value'].std():.6f})")
        print(f"  Success rate: {ragda_df['success'].sum() / len(ragda_df) * 100:.1f}%")
        print(f"  Mean time: {ragda_df['time'].mean():.2f}s")
        
        # Best categories
        print(f"\nRAGDA excels on:")
        ragda_analysis = analysis_df[analysis_df['optimizer'] == 'RAGDA'].copy()
        ragda_analysis['rank'] = ragda_analysis.groupby(['dim_class', 'cost_class', 'noise_class'])['mean_value'].rank()
        
        top_categories = ragda_analysis[ragda_analysis['rank'] == 1].head(5)
        for _, row in top_categories.iterrows():
            print(f"  ✓ {row['dim_class']} dim, {row['cost_class']} cost, {row['noise_class']} noise "
                  f"(f={row['mean_value']:.6f})")
        
        # Batching benefit
        if ragda_df['used_batching'].any():
            batch_df = ragda_df[ragda_df['used_batching'] == True]
            print(f"\nMini-batch feature:")
            print(f"  Used on: {batch_df['problem'].nunique()} problems")
            print(f"  Mean performance: {batch_df['best_value'].mean():.6f}")
            print(f"  Mean time: {batch_df['time'].mean():.2f}s")
    
    return df, analysis_df, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAGDA comprehensive benchmarks')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per pair')
    parser.add_argument('--budget', type=int, default=100, help='Iteration budget')
    parser.add_argument('--output', type=str, default='./benchmark_results', help='Output directory')
    parser.add_argument('--skip-expensive', action='store_true', help='Skip expensive problems')
    parser.add_argument('--quick', action='store_true', help='Quick mode for testing')
    
    args = parser.parse_args()
    
    df, analysis_df, stats = main(
        n_runs=args.runs,
        budget=args.budget,
        output_dir=args.output,
        skip_expensive=args.skip_expensive,
        quick_mode=args.quick
    )