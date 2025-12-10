"""
Machine Learning Hyperparameter Tuning Benchmark Problems

All problems use Optuna-style API for cross-optimizer compatibility.
These are real ML model tuning problems on sklearn datasets.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import ElasticNet
    from sklearn.datasets import (
        load_breast_cancer, load_digits, load_wine, 
        load_iris, load_diabetes
    )
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class BenchmarkProblem:
    """Container for a benchmark optimization problem."""
    name: str
    objective: Callable  # Optuna-style: trial -> float
    dimension: int
    bounds: list  # List of (low, high) tuples
    known_optimum: Optional[float]
    category: str
    description: str


def _make_optuna_objective(
    base_func: Callable,
    bounds: list,
    dimension: int
) -> Callable:
    """
    Wrap a function to work with Optuna trial API.
    
    Args:
        base_func: Function that takes np.ndarray -> float
        bounds: List of (low, high) tuples
        dimension: Problem dimension
    
    Returns:
        Function that takes trial -> float
    """
    def optuna_wrapper(trial):
        # Collect parameters from trial
        params = np.array([
            trial.suggest_float(f'x{i}', bounds[i][0], bounds[i][1])
            for i in range(dimension)
        ])
        return base_func(params)
    
    return optuna_wrapper


# =============================================================================
# ML Problem Implementations
# =============================================================================

def _lightgbm_breast_cancer_6d(params: np.ndarray) -> float:
    """LightGBM on breast cancer dataset (6D)."""
    if not LIGHTGBM_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_breast_cancer(return_X_y=True)
        
        lgb_params = {
            'objective': 'binary',
            'num_leaves': int(15 + params[0] * 240),  # 15-255
            'learning_rate': 10 ** (params[1] * 2.477 - 3),  # 0.001-0.3 log
            'feature_fraction': 0.5 + params[2] * 0.5,  # 0.5-1.0
            'bagging_fraction': 0.5 + params[3] * 0.5,  # 0.5-1.0
            'bagging_freq': int(params[4] * 10),  # 0-10
            'min_child_samples': int(5 + params[5] * 45),  # 5-50
            'verbose': -1,
            'n_jobs': 1,
            'random_state': 42
        }
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=50)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _lightgbm_digits_6d(params: np.ndarray) -> float:
    """LightGBM on digits dataset (6D)."""
    if not LIGHTGBM_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_digits(return_X_y=True)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 10,
            'num_leaves': int(15 + params[0] * 240),
            'learning_rate': 10 ** (params[1] * 2.477 - 3),
            'feature_fraction': 0.5 + params[2] * 0.5,
            'bagging_fraction': 0.5 + params[3] * 0.5,
            'bagging_freq': int(params[4] * 10),
            'min_child_samples': int(5 + params[5] * 45),
            'verbose': -1,
            'n_jobs': 1,
            'random_state': 42
        }
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=50)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _lightgbm_wine_6d(params: np.ndarray) -> float:
    """LightGBM on wine dataset (6D)."""
    if not LIGHTGBM_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_wine(return_X_y=True)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': int(15 + params[0] * 240),
            'learning_rate': 10 ** (params[1] * 2.477 - 3),
            'feature_fraction': 0.5 + params[2] * 0.5,
            'bagging_fraction': 0.5 + params[3] * 0.5,
            'bagging_freq': int(params[4] * 10),
            'min_child_samples': int(5 + params[5] * 45),
            'verbose': -1,
            'n_jobs': 1,
            'random_state': 42
        }
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=50)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _lightgbm_iris_6d(params: np.ndarray) -> float:
    """LightGBM on iris dataset (6D)."""
    if not LIGHTGBM_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_iris(return_X_y=True)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': int(15 + params[0] * 240),
            'learning_rate': 10 ** (params[1] * 2.477 - 3),
            'feature_fraction': 0.5 + params[2] * 0.5,
            'bagging_fraction': 0.5 + params[3] * 0.5,
            'bagging_freq': int(params[4] * 10),
            'min_child_samples': int(5 + params[5] * 45),
            'verbose': -1,
            'n_jobs': 1,
            'random_state': 42
        }
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=50)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _xgboost_breast_cancer_6d(params: np.ndarray) -> float:
    """XGBoost on breast cancer dataset (6D)."""
    if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_breast_cancer(return_X_y=True)
        
        xgb_params = {
            'max_depth': int(2 + params[0] * 8),  # 2-10
            'learning_rate': 10 ** (params[1] * 2.477 - 3),  # 0.001-0.3
            'n_estimators': int(50 + params[2] * 150),  # 50-200
            'min_child_weight': int(1 + params[3] * 9),  # 1-10
            'subsample': 0.5 + params[4] * 0.5,  # 0.5-1.0
            'colsample_bytree': 0.5 + params[5] * 0.5,  # 0.5-1.0
            'random_state': 42,
            'n_jobs': 1
        }
        
        model = xgb.XGBClassifier(**xgb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _xgboost_digits_6d(params: np.ndarray) -> float:
    """XGBoost on digits dataset (6D)."""
    if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
        return 1.0
    
    try:
        
        X, y = load_digits(return_X_y=True)
        
        xgb_params = {
            'max_depth': int(2 + params[0] * 8),
            'learning_rate': 10 ** (params[1] * 2.477 - 3),
            'n_estimators': int(50 + params[2] * 150),
            'min_child_weight': int(1 + params[3] * 9),
            'subsample': 0.5 + params[4] * 0.5,
            'colsample_bytree': 0.5 + params[5] * 0.5,
            'random_state': 42,
            'n_jobs': 1
        }
        
        model = xgb.XGBClassifier(**xgb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1.0 - scores.mean()
    except:
        return 1.0


def _random_forest_breast_cancer_4d(params: np.ndarray) -> float:
    """Random Forest on breast cancer dataset (4D)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    rf_params = {
        'n_estimators': int(50 + params[0] * 150),  # 50-200
        'max_depth': int(5 + params[1] * 25),  # 5-30
        'min_samples_split': int(2 + params[2] * 18),  # 2-20
        'min_samples_leaf': int(1 + params[3] * 9),  # 1-10
        'random_state': 42,
        'n_jobs': 1
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _random_forest_wine_4d(params: np.ndarray) -> float:
    """Random Forest on wine dataset (4D)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_wine(return_X_y=True)
    
    rf_params = {
        'n_estimators': int(50 + params[0] * 150),
        'max_depth': int(5 + params[1] * 25),
        'min_samples_split': int(2 + params[2] * 18),
        'min_samples_leaf': int(1 + params[3] * 9),
        'random_state': 42,
        'n_jobs': 1
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _svm_breast_cancer_2d(params: np.ndarray) -> float:
    """SVM on breast cancer dataset (2D: C, gamma)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    svm_params = {
        'C': 10 ** (params[0] * 4 - 2),  # 0.01-100 log scale
        'gamma': 10 ** (params[1] * 6 - 5),  # 1e-5 to 10 log scale
        'random_state': 42
    }
    
    model = SVC(**svm_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _svm_digits_2d(params: np.ndarray) -> float:
    """SVM on digits dataset (2D: C, gamma)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    
    svm_params = {
        'C': 10 ** (params[0] * 4 - 2),
        'gamma': 10 ** (params[1] * 6 - 5),
        'random_state': 42
    }
    
    model = SVC(**svm_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _mlp_digits_4d(params: np.ndarray) -> float:
    """MLP on digits dataset (4D: alpha, learning_rate_init, beta_1, beta_2)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    
    mlp_params = {
        'hidden_layer_sizes': (64, 32),
        'alpha': 10 ** (params[0] * 5 - 5),  # 1e-5 to 1 log scale
        'learning_rate_init': 10 ** (params[1] * 3 - 4),  # 1e-4 to 0.1 log scale
        'beta_1': 0.5 + params[2] * 0.4,  # 0.5-0.9
        'beta_2': 0.9 + params[3] * 0.099,  # 0.9-0.999
        'random_state': 42,
        'max_iter': 100
    }
    
    model = MLPClassifier(**mlp_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _mlp_breast_cancer_4d(params: np.ndarray) -> float:
    """MLP on breast cancer dataset (4D)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    mlp_params = {
        'hidden_layer_sizes': (32, 16),
        'alpha': 10 ** (params[0] * 5 - 5),
        'learning_rate_init': 10 ** (params[1] * 3 - 4),
        'beta_1': 0.5 + params[2] * 0.4,
        'beta_2': 0.9 + params[3] * 0.099,
        'random_state': 42,
        'max_iter': 100
    }
    
    model = MLPClassifier(**mlp_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return 1.0 - scores.mean()


def _gradient_boosting_regressor_diabetes_4d(params: np.ndarray) -> float:
    """Gradient Boosting Regressor on diabetes dataset (4D)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_diabetes(return_X_y=True)
    
    gbr_params = {
        'n_estimators': int(50 + params[0] * 150),  # 50-200
        'learning_rate': 10 ** (params[1] * 2.477 - 3),  # 0.001-0.3 log
        'max_depth': int(2 + params[2] * 8),  # 2-10
        'min_samples_split': int(2 + params[3] * 18),  # 2-20
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**gbr_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()  # Return positive MSE (minimize)


def _elastic_net_diabetes_2d(params: np.ndarray) -> float:
    """Elastic Net on diabetes dataset (2D: alpha, l1_ratio)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_diabetes(return_X_y=True)
    
    en_params = {
        'alpha': 10 ** (params[0] * 4 - 3),  # 0.001-10 log scale
        'l1_ratio': params[1],  # 0-1
        'random_state': 42,
        'max_iter': 1000
    }
    
    model = ElasticNet(**en_params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()


def _portfolio_optimization(params: np.ndarray) -> float:
    """
    Portfolio optimization problem.
    Params are portfolio weights (normalized to sum to 1).
    Minimize variance for given expected return.
    """
    n_assets = len(params)
    
    # Normalize weights to sum to 1
    weights = np.abs(params) / np.sum(np.abs(params))
    
    # Generate synthetic covariance matrix (positive definite)
    np.random.seed(42)
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = A.T @ A / n_assets + np.eye(n_assets) * 0.1
    
    # Generate synthetic expected returns
    expected_returns = 0.05 + 0.1 * np.random.rand(n_assets)
    
    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights
    
    # Portfolio return
    portfolio_return = weights @ expected_returns
    
    # Objective: minimize variance with penalty for low returns
    target_return = 0.10
    return_penalty = max(0, target_return - portfolio_return) ** 2 * 100
    
    return portfolio_variance + return_penalty


# =============================================================================
# CHUNK 3.4.1: EXPENSIVE HIGH-DIMENSIONAL ML PROBLEMS (15 problems)
# Purpose: Fill high_expensive_* categories (smooth, moderate, rugged)
# Strategy: Use real sklearn datasets with EXPENSIVE models (many trees, deep nets)
#          + many CV folds for natural expense (>100ms per evaluation)
# =============================================================================

# =============================================================================
# CHUNK 3.4.1: EXPENSIVE HIGH-DIMENSIONAL ML PROBLEMS (15 problems)
# Purpose: Fill high_expensive_* categories (smooth, moderate, rugged)
# Strategy: Real sklearn datasets + expensive models with MANY hyperparameters
#          High-dim via: tuning ensemble of multiple models (each with own params)
# =============================================================================

def _multi_rf_expensive_60d(params: np.ndarray) -> float:
    """Tune 10 RandomForests in parallel, each with 6 hyperparameters = 60D (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    # 60D = 10 models × 6 params each
    # Each RF has: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion weight
    ensemble_scores = []
    
    for i in range(10):
        rf_params = {
            'n_estimators': int(100 + params[i*6] * 400),  # 100-500 trees (expensive!)
            'max_depth': int(5 + params[i*6+1] * 20),  # 5-25
            'min_samples_split': int(2 + params[i*6+2] * 18),  # 2-20
            'min_samples_leaf': int(1 + params[i*6+3] * 9),  # 1-10
            'max_features': 0.3 + params[i*6+4] * 0.7,  # 0.3-1.0
            'random_state': 42 + i
        }
        
        model = RandomForestClassifier(**rf_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    # Return error of best model in ensemble
    return 1.0 - max(ensemble_scores)


def _multi_rf_expensive_80d(params: np.ndarray) -> float:
    """Tune 20 RandomForests, each with 4 hyperparameters = 80D (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(20):
        rf_params = {
            'n_estimators': int(100 + params[i*4] * 400),
            'max_depth': int(5 + params[i*4+1] * 20),
            'min_samples_split': int(2 + params[i*4+2] * 18),
            'max_features': 0.3 + params[i*4+3] * 0.7,
            'random_state': 42 + i
        }
        
        model = RandomForestClassifier(**rf_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_gbr_expensive_70d(params: np.ndarray) -> float:
    """Tune 10 GradientBoosting regressors, each with 7 hyperparameters = 70D (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_diabetes(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(10):
        gb_params = {
            'n_estimators': int(100 + params[i*7] * 300),  # 100-400 (expensive!)
            'learning_rate': 10 ** (params[i*7+1] * 2 - 2),  # 0.01-1.0 log scale
            'max_depth': int(3 + params[i*7+2] * 7),  # 3-10
            'subsample': 0.6 + params[i*7+3] * 0.4,  # 0.6-1.0
            'min_samples_split': int(2 + params[i*7+4] * 18),
            'min_samples_leaf': int(1 + params[i*7+5] * 9),
            'random_state': 42 + i
        }
        
        model = GradientBoostingRegressor(**gb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)  # Best (lowest) MSE


def _multi_gbr_expensive_60d(params: np.ndarray) -> float:
    """Tune 10 GradientBoosting regressors, 6 params each = 60D (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_diabetes(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(10):
        gb_params = {
            'n_estimators': int(100 + params[i*6] * 300),
            'learning_rate': 10 ** (params[i*6+1] * 2 - 2),
            'max_depth': int(3 + params[i*6+2] * 7),
            'subsample': 0.6 + params[i*6+3] * 0.4,
            'min_samples_split': int(2 + params[i*6+4] * 18),
            'random_state': 42 + i
        }
        
        model = GradientBoostingRegressor(**gb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)


def _multi_rf_expensive_100d(params: np.ndarray) -> float:
    """Tune 25 RandomForests, 4 params each = 100D (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(25):
        rf_params = {
            'n_estimators': int(100 + params[i*4] * 400),
            'max_depth': int(5 + params[i*4+1] * 20),
            'min_samples_split': int(2 + params[i*4+2] * 18),
            'max_features': 0.3 + params[i*4+3] * 0.7,
            'random_state': 42 + i
        }
        
        model = RandomForestClassifier(**rf_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


# MODERATE RUGGEDNESS: Nonlinear data or complex interactions

def _multi_svm_expensive_60d(params: np.ndarray) -> float:
    """Tune 30 SVMs with RBF kernel, 2 params each (C, gamma) = 60D (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    # Subsample for speed but still expensive
    X = X[:300]
    y = y[:300]
    
    ensemble_scores = []
    
    for i in range(30):
        svm_params = {
            'C': 10 ** (params[i*2] * 4 - 2),  # 0.01-100 log scale
            'gamma': 10 ** (params[i*2+1] * 4 - 4),  # 0.0001-1.0 log scale
            'kernel': 'rbf',
            'random_state': 42 + i
        }
        
        model = SVC(**svm_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_svm_expensive_80d(params: np.ndarray) -> float:
    """Tune 40 SVMs, 2 params each = 80D (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    X = X[:500]
    y = y[:500]
    
    ensemble_scores = []
    
    for i in range(40):
        svm_params = {
            'C': 10 ** (params[i*2] * 4 - 2),
            'gamma': 10 ** (params[i*2+1] * 4 - 4),
            'kernel': 'rbf',
            'random_state': 42 + i
        }
        
        model = SVC(**svm_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_gbr_nonlinear_70d(params: np.ndarray) -> float:
    """GradientBoosting on nonlinear target, 10 models × 7 params = 70D (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    # Create nonlinear regression problem
    np.random.seed(42)
    X = np.random.randn(400, 10)
    y = (X[:, 0] ** 2 + np.sin(X[:, 1]) * X[:, 2] + X[:, 3] ** 3) + np.random.randn(400) * 0.3
    
    ensemble_scores = []
    
    for i in range(10):
        gb_params = {
            'n_estimators': int(100 + params[i*7] * 300),
            'learning_rate': 10 ** (params[i*7+1] * 2 - 2),
            'max_depth': int(3 + params[i*7+2] * 7),
            'subsample': 0.6 + params[i*7+3] * 0.4,
            'min_samples_split': int(2 + params[i*7+4] * 18),
            'min_samples_leaf': int(1 + params[i*7+5] * 9),
            'random_state': 42 + i
        }
        
        model = GradientBoostingRegressor(**gb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)


def _multi_rf_multiclass_60d(params: np.ndarray) -> float:
    """RandomForest multiclass (10-class digits), 10 models × 6 params = 60D (moderate, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(10):
        rf_params = {
            'n_estimators': int(100 + params[i*6] * 400),
            'max_depth': int(5 + params[i*6+1] * 20),
            'min_samples_split': int(2 + params[i*6+2] * 18),
            'min_samples_leaf': int(1 + params[i*6+3] * 9),
            'max_features': 0.3 + params[i*6+4] * 0.7,
            'random_state': 42 + i
        }
        
        model = RandomForestClassifier(**rf_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_gbr_complex_80d(params: np.ndarray) -> float:
    """GradientBoosting on complex target, 10 models × 8 params = 80D (moderate, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(43)
    X = np.random.randn(400, 15)
    y = (X[:, :5] ** 2).sum(axis=1) + np.sin(X[:, 5:10].sum(axis=1)) + np.random.randn(400) * 0.5
    
    ensemble_scores = []
    
    for i in range(10):
        gb_params = {
            'n_estimators': int(100 + params[i*8] * 300),
            'learning_rate': 10 ** (params[i*8+1] * 2 - 2),
            'max_depth': int(3 + params[i*8+2] * 7),
            'subsample': 0.6 + params[i*8+3] * 0.4,
            'min_samples_split': int(2 + params[i*8+4] * 18),
            'min_samples_leaf': int(1 + params[i*8+5] * 9),
            'max_features': 0.3 + params[i*8+6] * 0.7,
            'random_state': 42 + i
        }
        
        model = GradientBoostingRegressor(**gb_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)


# RUGGED: Neural networks with many hyperparameters

def _multi_mlp_expensive_60d(params: np.ndarray) -> float:
    """Tune 10 MLPs, 6 params each = 60D (rugged - NN landscape, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(10):
        mlp_params = {
            'hidden_layer_sizes': (
                int(20 + params[i*6] * 80),  # 20-100 neurons layer 1
                int(10 + params[i*6+1] * 40),  # 10-50 neurons layer 2
            ),
            'alpha': 10 ** (params[i*6+2] * 4 - 4),  # 0.0001-1.0
            'learning_rate_init': 10 ** (params[i*6+3] * 2 - 3),  # 0.001-0.1
            'beta_1': 0.85 + params[i*6+4] * 0.14,  # 0.85-0.99
            'beta_2': 0.9 + params[i*6+5] * 0.099,  # 0.9-0.999
            'max_iter': 200,
            'random_state': 42 + i
        }
        
        model = MLPClassifier(**mlp_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_mlp_expensive_80d(params: np.ndarray) -> float:
    """Tune 20 MLPs, 4 params each = 80D (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_digits(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(20):
        mlp_params = {
            'hidden_layer_sizes': (
                int(30 + params[i*4] * 120),  # 30-150 neurons
            ),
            'alpha': 10 ** (params[i*4+1] * 4 - 4),
            'learning_rate_init': 10 ** (params[i*4+2] * 2 - 3),
            'beta_1': 0.85 + params[i*4+3] * 0.14,
            'max_iter': 200,
            'random_state': 42 + i
        }
        
        model = MLPClassifier(**mlp_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


def _multi_mlp_regression_70d(params: np.ndarray) -> float:
    """Tune 10 MLP regressors, 7 params each = 70D (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    from sklearn.neural_network import MLPRegressor
    
    X, y = load_diabetes(return_X_y=True)
    
    ensemble_scores = []
    
    for i in range(10):
        mlp_params = {
            'hidden_layer_sizes': (
                int(20 + params[i*7] * 80),
                int(10 + params[i*7+1] * 40),
            ),
            'alpha': 10 ** (params[i*7+2] * 4 - 4),
            'learning_rate_init': 10 ** (params[i*7+3] * 2 - 3),
            'beta_1': 0.85 + params[i*7+4] * 0.14,
            'beta_2': 0.9 + params[i*7+5] * 0.099,
            'max_iter': 200,
            'random_state': 42 + i
        }
        
        model = MLPRegressor(**mlp_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)


def _multi_mlp_complex_90d(params: np.ndarray) -> float:
    """Tune 15 MLPs on complex data, 6 params each = 90D (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    from sklearn.neural_network import MLPRegressor
    
    np.random.seed(44)
    X = np.random.randn(400, 20)
    y = (X[:, :5] ** 2).sum(axis=1) + np.sin(X[:, 5:10].sum(axis=1)) + np.random.randn(400) * 0.5
    
    ensemble_scores = []
    
    for i in range(15):
        mlp_params = {
            'hidden_layer_sizes': (
                int(20 + params[i*6] * 80),
                int(10 + params[i*6+1] * 40),
            ),
            'alpha': 10 ** (params[i*6+2] * 4 - 4),
            'learning_rate_init': 10 ** (params[i*6+3] * 2 - 3),
            'beta_1': 0.85 + params[i*6+4] * 0.14,
            'max_iter': 200,
            'random_state': 42 + i
        }
        
        model = MLPRegressor(**mlp_params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        ensemble_scores.append(-scores.mean())
    
    return min(ensemble_scores)


def _multi_svm_grid_100d(params: np.ndarray) -> float:
    """Tune 50 SVMs, 2 params each (C, gamma) = 100D (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:300]
    y = y[:300]
    
    ensemble_scores = []
    
    for i in range(50):
        svm_params = {
            'C': 10 ** (params[i*2] * 4 - 2),
            'gamma': 10 ** (params[i*2+1] * 4 - 4),
            'kernel': 'rbf',
            'random_state': 42 + i
        }
        
        model = SVC(**svm_params)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        ensemble_scores.append(scores.mean())
    
    return 1.0 - max(ensemble_scores)


# =============================================================================
# ML Problems Registry
# =============================================================================
    """RandomForest ensemble with 60 independent tree hyperparameters (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(60)
    X = np.random.randn(400, 30)
    y = (X[:, :5].sum(axis=1) > 0).astype(int)
    
    # 60D: 10 trees × 6 params each (max_depth, min_samples_split, min_samples_leaf, etc.)
    n_estimators = 10
    max_depths = [int(3 + params[i*6] * 17) for i in range(n_estimators)]  # 3-20 per tree
    min_samples_splits = [int(2 + params[i*6+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*6+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*6+3] * 0.5 + 0.5 for i in range(n_estimators)]  # 0.5-1.0
    
    # Average parameters for the ensemble
    rf_params = {
        'n_estimators': n_estimators * 20,  # Many trees for expense
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _rf_ensemble_expensive_80d(params: np.ndarray) -> float:
    """RandomForest ensemble with 80D hyperparameters (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(61)
    X = np.random.randn(400, 40)
    y = (X[:, :8].sum(axis=1) > 0).astype(int)
    
    # 80D: Configure 20 aspects × 4 params
    n_estimators = 20
    max_depths = [int(3 + params[i*4] * 17) for i in range(n_estimators)]
    min_samples_splits = [int(2 + params[i*4+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*4+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*4+3] * 0.5 + 0.5 for i in range(n_estimators)]
    
    rf_params = {
        'n_estimators': n_estimators * 15,
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _gbr_ensemble_expensive_70d(params: np.ndarray) -> float:
    """Gradient Boosting with 70D hyperparameters (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(62)
    X = np.random.randn(400, 35)
    y = X[:, :7].sum(axis=1) + np.random.randn(400) * 0.1
    
    # 70D: learning rates + depths + subsample rates for staged boosting
    n_stages = 10
    learning_rates = [10 ** (params[i*7] * 2 - 2) for i in range(n_stages)]  # 0.01-1.0
    max_depths = [int(3 + params[i*7+1] * 7) for i in range(n_stages)]
    subsamples = [0.6 + params[i*7+2] * 0.4 for i in range(n_stages)]
    
    gb_params = {
        'n_estimators': 300,  # Many estimators for expense
        'learning_rate': float(np.mean(learning_rates)),
        'max_depth': int(np.mean(max_depths)),
        'subsample': float(np.mean(subsamples)),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**gb_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _gbr_ensemble_expensive_60d(params: np.ndarray) -> float:
    """Gradient Boosting with 60D hyperparameters (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(63)
    X = np.random.randn(400, 30)
    y = X[:, :6].sum(axis=1) + np.random.randn(400) * 0.1
    
    n_stages = 10
    learning_rates = [10 ** (params[i*6] * 2 - 2) for i in range(n_stages)]
    max_depths = [int(3 + params[i*6+1] * 7) for i in range(n_stages)]
    subsamples = [0.6 + params[i*6+2] * 0.4 for i in range(n_stages)]
    
    gb_params = {
        'n_estimators': 300,
        'learning_rate': float(np.mean(learning_rates)),
        'max_depth': int(np.mean(max_depths)),
        'subsample': float(np.mean(subsamples)),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**gb_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _rf_ensemble_expensive_100d(params: np.ndarray) -> float:
    """RandomForest with 100D hyperparameters (smooth, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(64)
    X = np.random.randn(400, 50)
    y = (X[:, :10].sum(axis=1) > 0).astype(int)
    
    # 100D: 25 trees × 4 params
    n_estimators = 25
    max_depths = [int(3 + params[i*4] * 17) for i in range(n_estimators)]
    min_samples_splits = [int(2 + params[i*4+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*4+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*4+3] * 0.5 + 0.5 for i in range(n_estimators)]
    
    rf_params = {
        'n_estimators': n_estimators * 10,
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _rf_nonlinear_expensive_60d(params: np.ndarray) -> float:
    """RandomForest 60D on nonlinear data (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(65)
    X = np.random.randn(400, 30)
    y = ((X[:, 0] ** 2 + X[:, 1] * X[:, 2] - X[:, 3] ** 2) > 0).astype(int)
    
    n_estimators = 10
    max_depths = [int(3 + params[i*6] * 17) for i in range(n_estimators)]
    min_samples_splits = [int(2 + params[i*6+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*6+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*6+3] * 0.5 + 0.5 for i in range(n_estimators)]
    
    rf_params = {
        'n_estimators': n_estimators * 20,
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _rf_nonlinear_expensive_80d(params: np.ndarray) -> float:
    """RandomForest 80D on nonlinear data (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(66)
    X = np.random.randn(400, 40)
    y = ((X[:, 0] ** 2 + X[:, 2] * X[:, 4] - X[:, 6] ** 2) > 0).astype(int)
    
    n_estimators = 20
    max_depths = [int(3 + params[i*4] * 17) for i in range(n_estimators)]
    min_samples_splits = [int(2 + params[i*4+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*4+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*4+3] * 0.5 + 0.5 for i in range(n_estimators)]
    
    rf_params = {
        'n_estimators': n_estimators * 15,
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _gbr_nonlinear_expensive_70d(params: np.ndarray) -> float:
    """Gradient Boosting 70D on nonlinear data (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(67)
    X = np.random.randn(400, 35)
    y = (X[:, 0] ** 2 + np.sin(X[:, 1]) * X[:, 2] + X[:, 3] ** 3) + np.random.randn(400) * 0.3
    
    n_stages = 10
    learning_rates = [10 ** (params[i*7] * 2 - 2) for i in range(n_stages)]
    max_depths = [int(3 + params[i*7+1] * 7) for i in range(n_stages)]
    subsamples = [0.6 + params[i*7+2] * 0.4 for i in range(n_stages)]
    
    gb_params = {
        'n_estimators': 300,
        'learning_rate': float(np.mean(learning_rates)),
        'max_depth': int(np.mean(max_depths)),
        'subsample': float(np.mean(subsamples)),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**gb_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _rf_multiclass_expensive_60d(params: np.ndarray) -> float:
    """RandomForest 60D multiclass (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(68)
    X = np.random.randn(400, 30)
    y = ((X[:, 0] + X[:, 1] * 2 + X[:, 2] ** 2) % 5).astype(int)
    
    n_estimators = 10
    max_depths = [int(3 + params[i*6] * 17) for i in range(n_estimators)]
    min_samples_splits = [int(2 + params[i*6+1] * 18) for i in range(n_estimators)]
    min_samples_leafs = [int(1 + params[i*6+2] * 9) for i in range(n_estimators)]
    max_features_list = [params[i*6+3] * 0.5 + 0.5 for i in range(n_estimators)]
    
    rf_params = {
        'n_estimators': n_estimators * 20,
        'max_depth': int(np.mean(max_depths)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'min_samples_leaf': int(np.mean(min_samples_leafs)),
        'max_features': float(np.mean(max_features_list)),
        'random_state': 42
    }
    
    model = RandomForestClassifier(**rf_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _gbr_multiaspect_expensive_80d(params: np.ndarray) -> float:
    """Gradient Boosting 80D multi-aspect tuning (moderate ruggedness, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(69)
    X = np.random.randn(400, 40)
    y = (X[:, :8] ** 2).sum(axis=1) + np.sin(X[:, 8:16].sum(axis=1)) + np.random.randn(400) * 0.5
    
    # 80D: 10 stages × 8 params
    n_stages = 10
    learning_rates = [10 ** (params[i*8] * 2 - 2) for i in range(n_stages)]
    max_depths = [int(3 + params[i*8+1] * 7) for i in range(n_stages)]
    subsamples = [0.6 + params[i*8+2] * 0.4 for i in range(n_stages)]
    min_samples_splits = [int(2 + params[i*8+3] * 18) for i in range(n_stages)]
    
    gb_params = {
        'n_estimators': 300,
        'learning_rate': float(np.mean(learning_rates)),
        'max_depth': int(np.mean(max_depths)),
        'subsample': float(np.mean(subsamples)),
        'min_samples_split': int(np.mean(min_samples_splits)),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**gb_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _mlp_deep_expensive_60d(params: np.ndarray) -> float:
    """Deep MLP with 60D hyperparameters (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(70)
    X = np.random.randn(400, 30)
    y = ((X[:, :5] ** 2).sum(axis=1) + np.sin(X[:, 5:10].sum(axis=1)) > 0).astype(int)
    
    # 60D: 6 layers × 10 params (size + regularization per layer)
    layer_sizes = tuple([int(20 + params[i*10] * 180) for i in range(6)])  # 20-200 per layer
    alphas = [10 ** (params[i*10+1] * 4 - 4) for i in range(6)]
    
    mlp_params = {
        'hidden_layer_sizes': layer_sizes,
        'alpha': float(np.mean(alphas)),
        'learning_rate_init': 10 ** (params[5] * 2 - 3),
        'max_iter': 200,
        'random_state': 42
    }
    
    model = MLPClassifier(**mlp_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _mlp_deep_expensive_80d(params: np.ndarray) -> float:
    """Deep MLP with 80D hyperparameters (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(71)
    X = np.random.randn(400, 40)
    y = ((X[:, :8] ** 2).sum(axis=1) + np.sin(X[:, 8:16].sum(axis=1)) > 0).astype(int)
    
    # 80D: 10 layers × 8 params
    layer_sizes = tuple([int(20 + params[i*8] * 180) for i in range(10)])
    alphas = [10 ** (params[i*8+1] * 4 - 4) for i in range(10)]
    
    mlp_params = {
        'hidden_layer_sizes': layer_sizes,
        'alpha': float(np.mean(alphas)),
        'learning_rate_init': 10 ** (params[8] * 2 - 3),
        'max_iter': 200,
        'random_state': 42
    }
    
    model = MLPClassifier(**mlp_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


def _mlp_regression_expensive_70d(params: np.ndarray) -> float:
    """Deep MLP regression with 70D hyperparameters (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    from sklearn.neural_network import MLPRegressor
    
    np.random.seed(72)
    X = np.random.randn(400, 35)
    y = (X[:, :7] ** 2).sum(axis=1) + np.sin(X[:, 7:14].sum(axis=1)) + np.random.randn(400) * 0.5
    
    # 70D: 7 layers × 10 params
    layer_sizes = tuple([int(20 + params[i*10] * 180) for i in range(7)])
    alphas = [10 ** (params[i*10+1] * 4 - 4) for i in range(7)]
    
    mlp_params = {
        'hidden_layer_sizes': layer_sizes,
        'alpha': float(np.mean(alphas)),
        'learning_rate_init': 10 ** (params[6] * 2 - 3),
        'max_iter': 200,
        'random_state': 42
    }
    
    model = MLPRegressor(**mlp_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _mlp_complex_expensive_90d(params: np.ndarray) -> float:
    """Complex MLP with 90D hyperparameters (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    from sklearn.neural_network import MLPRegressor
    
    np.random.seed(73)
    X = np.random.randn(400, 45)
    y = (X[:, :9] ** 2).sum(axis=1) + np.sin(X[:, 9:18].sum(axis=1)) + np.random.randn(400) * 0.5
    
    # 90D: 9 layers × 10 params
    layer_sizes = tuple([int(20 + params[i*10] * 180) for i in range(9)])
    alphas = [10 ** (params[i*10+1] * 4 - 4) for i in range(9)]
    
    mlp_params = {
        'hidden_layer_sizes': layer_sizes,
        'alpha': float(np.mean(alphas)),
        'learning_rate_init': 10 ** (params[8] * 2 - 3),
        'max_iter': 200,
        'random_state': 42
    }
    
    model = MLPRegressor(**mlp_params)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
    return -scores.mean()


def _svm_grid_expensive_100d(params: np.ndarray) -> float:
    """SVM with 100D hyperparameter grid (rugged, expensive)."""
    if not SKLEARN_AVAILABLE:
        return 1.0
    
    np.random.seed(74)
    X = np.random.randn(400, 50)
    y = ((X[:, :10] ** 2).sum(axis=1) > 5).astype(int)
    
    # 100D: 50 different C/gamma pairs for ensemble voting
    C_values = [10 ** (params[i*2] * 4 - 2) for i in range(50)]
    gamma_values = [10 ** (params[i*2+1] * 4 - 4) for i in range(50)]
    
    svm_params = {
        'C': float(np.median(C_values)),
        'gamma': float(np.median(gamma_values)),
        'kernel': 'rbf',
        'random_state': 42
    }
    
    model = SVC(**svm_params)
    scores = cross_val_score(model, X, y, cv=10, n_jobs=1)
    return 1.0 - scores.mean()


# =============================================================================
# ML Problems Registry
# =============================================================================

ALL_ML_PROBLEMS: Dict[str, BenchmarkProblem] = {
    'LightGBM-BreastCancer-6D': BenchmarkProblem(
        name='LightGBM-BreastCancer-6D',
        objective=_make_optuna_objective(_lightgbm_breast_cancer_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='LightGBM hyperparameter tuning on breast cancer dataset (6D)'
    ),
    'LightGBM-Digits-6D': BenchmarkProblem(
        name='LightGBM-Digits-6D',
        objective=_make_optuna_objective(_lightgbm_digits_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='LightGBM hyperparameter tuning on digits dataset (6D)'
    ),
    'LightGBM-Wine-6D': BenchmarkProblem(
        name='LightGBM-Wine-6D',
        objective=_make_optuna_objective(_lightgbm_wine_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='LightGBM hyperparameter tuning on wine dataset (6D)'
    ),
    'LightGBM-Iris-6D': BenchmarkProblem(
        name='LightGBM-Iris-6D',
        objective=_make_optuna_objective(_lightgbm_iris_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='LightGBM hyperparameter tuning on iris dataset (6D)'
    ),
    'XGBoost-BreastCancer-6D': BenchmarkProblem(
        name='XGBoost-BreastCancer-6D',
        objective=_make_optuna_objective(_xgboost_breast_cancer_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='XGBoost hyperparameter tuning on breast cancer dataset (6D)'
    ),
    'XGBoost-Digits-6D': BenchmarkProblem(
        name='XGBoost-Digits-6D',
        objective=_make_optuna_objective(_xgboost_digits_6d, [(0, 1)] * 6, 6),
        dimension=6,
        bounds=[(0, 1)] * 6,
        known_optimum=None,
        category='ml_tuning',
        description='XGBoost hyperparameter tuning on digits dataset (6D)'
    ),
    'RandomForest-BreastCancer-4D': BenchmarkProblem(
        name='RandomForest-BreastCancer-4D',
        objective=_make_optuna_objective(_random_forest_breast_cancer_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_tuning',
        description='Random Forest hyperparameter tuning on breast cancer dataset (4D)'
    ),
    'RandomForest-Wine-4D': BenchmarkProblem(
        name='RandomForest-Wine-4D',
        objective=_make_optuna_objective(_random_forest_wine_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_tuning',
        description='Random Forest hyperparameter tuning on wine dataset (4D)'
    ),
    'SVM-BreastCancer-2D': BenchmarkProblem(
        name='SVM-BreastCancer-2D',
        objective=_make_optuna_objective(_svm_breast_cancer_2d, [(0, 1)] * 2, 2),
        dimension=2,
        bounds=[(0, 1)] * 2,
        known_optimum=None,
        category='ml_tuning',
        description='SVM hyperparameter tuning on breast cancer dataset (2D: C, gamma)'
    ),
    'SVM-Digits-2D': BenchmarkProblem(
        name='SVM-Digits-2D',
        objective=_make_optuna_objective(_svm_digits_2d, [(0, 1)] * 2, 2),
        dimension=2,
        bounds=[(0, 1)] * 2,
        known_optimum=None,
        category='ml_tuning',
        description='SVM hyperparameter tuning on digits dataset (2D: C, gamma)'
    ),
    'MLP-Digits-4D': BenchmarkProblem(
        name='MLP-Digits-4D',
        objective=_make_optuna_objective(_mlp_digits_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_tuning',
        description='MLP hyperparameter tuning on digits dataset (4D: alpha, lr, beta1, beta2)'
    ),
    'MLP-BreastCancer-4D': BenchmarkProblem(
        name='MLP-BreastCancer-4D',
        objective=_make_optuna_objective(_mlp_breast_cancer_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_tuning',
        description='MLP hyperparameter tuning on breast cancer dataset (4D)'
    ),
    'GBR-Diabetes-4D': BenchmarkProblem(
        name='GBR-Diabetes-4D',
        objective=_make_optuna_objective(_gradient_boosting_regressor_diabetes_4d, [(0, 1)] * 4, 4),
        dimension=4,
        bounds=[(0, 1)] * 4,
        known_optimum=None,
        category='ml_tuning',
        description='Gradient Boosting Regressor on diabetes dataset (4D)'
    ),
    'ElasticNet-Diabetes-2D': BenchmarkProblem(
        name='ElasticNet-Diabetes-2D',
        objective=_make_optuna_objective(_elastic_net_diabetes_2d, [(0, 1)] * 2, 2),
        dimension=2,
        bounds=[(0, 1)] * 2,
        known_optimum=None,
        category='ml_tuning',
        description='Elastic Net hyperparameter tuning on diabetes dataset (2D: alpha, l1_ratio)'
    ),
    'Portfolio-5D': BenchmarkProblem(
        name='Portfolio-5D',
        objective=_make_optuna_objective(_portfolio_optimization, [(0, 1)] * 5, 5),
        dimension=5,
        bounds=[(0, 1)] * 5,
        known_optimum=None,
        category='finance',
        description='Portfolio optimization with 5 assets (minimize variance)'
    ),
    'Portfolio-10D': BenchmarkProblem(
        name='Portfolio-10D',
        objective=_make_optuna_objective(_portfolio_optimization, [(0, 1)] * 10, 10),
        dimension=10,
        bounds=[(0, 1)] * 10,
        known_optimum=None,
        category='finance',
        description='Portfolio optimization with 10 assets (minimize variance)'
    ),
    'Portfolio-20D': BenchmarkProblem(
        name='Portfolio-20D',
        objective=_make_optuna_objective(_portfolio_optimization, [(0, 1)] * 20, 20),
        dimension=20,
        bounds=[(0, 1)] * 20,
        known_optimum=None,
        category='finance',
        description='Portfolio optimization with 20 assets (minimize variance)'
    ),
    'Portfolio-30D': BenchmarkProblem(
        name='Portfolio-30D',
        objective=_make_optuna_objective(_portfolio_optimization, [(0, 1)] * 30, 30),
        dimension=30,
        bounds=[(0, 1)] * 30,
        known_optimum=None,
        category='finance',
        description='Portfolio optimization with 30 assets (minimize variance)'
    ),
    'Portfolio-50D': BenchmarkProblem(
        name='Portfolio-50D',
        objective=_make_optuna_objective(_portfolio_optimization, [(0, 1)] * 50, 50),
        dimension=50,
        bounds=[(0, 1)] * 50,
        known_optimum=None,
        category='finance',
        description='Portfolio optimization with 50 assets (minimize variance)'
    ),
    # CHUNK 3.4.1: Expensive High-Dimensional ML Problems (15 problems)
    # Strategy: Tune multiple models in ensemble (each with own hyperparameters)
    # = Many total hyperparameters (51D+) + expensive evaluation via CV
    
    # SMOOTH landscapes (5 problems): RandomForest and GradientBoosting ensembles
    'Multi-RF-Expensive-60D': BenchmarkProblem(
        name='Multi-RF-Expensive-60D',
        objective=_make_optuna_objective(_multi_rf_expensive_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 RandomForests on breast cancer (60D hyperparams, smooth, expensive)'
    ),
    'Multi-RF-Expensive-80D': BenchmarkProblem(
        name='Multi-RF-Expensive-80D',
        objective=_make_optuna_objective(_multi_rf_expensive_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 20 RandomForests on digits (80D hyperparams, smooth, expensive)'
    ),
    'Multi-GBR-Expensive-70D': BenchmarkProblem(
        name='Multi-GBR-Expensive-70D',
        objective=_make_optuna_objective(_multi_gbr_expensive_70d, [(0, 1)] * 70, 70),
        dimension=70,
        bounds=[(0, 1)] * 70,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 GradientBoosting regressors on diabetes (70D, smooth, expensive)'
    ),
    'Multi-GBR-Expensive-60D': BenchmarkProblem(
        name='Multi-GBR-Expensive-60D',
        objective=_make_optuna_objective(_multi_gbr_expensive_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 GradientBoosting regressors on diabetes (60D, smooth, expensive)'
    ),
    'Multi-RF-Expensive-100D': BenchmarkProblem(
        name='Multi-RF-Expensive-100D',
        objective=_make_optuna_objective(_multi_rf_expensive_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 25 RandomForests on breast cancer (100D, smooth, expensive)'
    ),
    
    # MODERATE ruggedness (5 problems): SVM ensembles and nonlinear targets
    'Multi-SVM-Expensive-60D': BenchmarkProblem(
        name='Multi-SVM-Expensive-60D',
        objective=_make_optuna_objective(_multi_svm_expensive_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 30 SVMs on breast cancer (60D, moderate ruggedness, expensive)'
    ),
    'Multi-SVM-Expensive-80D': BenchmarkProblem(
        name='Multi-SVM-Expensive-80D',
        objective=_make_optuna_objective(_multi_svm_expensive_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 40 SVMs on digits (80D, moderate ruggedness, expensive)'
    ),
    'Multi-GBR-Nonlinear-70D': BenchmarkProblem(
        name='Multi-GBR-Nonlinear-70D',
        objective=_make_optuna_objective(_multi_gbr_nonlinear_70d, [(0, 1)] * 70, 70),
        dimension=70,
        bounds=[(0, 1)] * 70,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 GradientBoosting on nonlinear target (70D, moderate, expensive)'
    ),
    'Multi-RF-Multiclass-60D': BenchmarkProblem(
        name='Multi-RF-Multiclass-60D',
        objective=_make_optuna_objective(_multi_rf_multiclass_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 RandomForests on 10-class digits (60D, moderate, expensive)'
    ),
    'Multi-GBR-Complex-80D': BenchmarkProblem(
        name='Multi-GBR-Complex-80D',
        objective=_make_optuna_objective(_multi_gbr_complex_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 GradientBoosting on complex target (80D, moderate, expensive)'
    ),
    
    # RUGGED landscapes (5 problems): Neural network ensembles
    'Multi-MLP-Expensive-60D': BenchmarkProblem(
        name='Multi-MLP-Expensive-60D',
        objective=_make_optuna_objective(_multi_mlp_expensive_60d, [(0, 1)] * 60, 60),
        dimension=60,
        bounds=[(0, 1)] * 60,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 MLPs on breast cancer (60D, rugged NN landscape, expensive)'
    ),
    'Multi-MLP-Expensive-80D': BenchmarkProblem(
        name='Multi-MLP-Expensive-80D',
        objective=_make_optuna_objective(_multi_mlp_expensive_80d, [(0, 1)] * 80, 80),
        dimension=80,
        bounds=[(0, 1)] * 80,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 20 MLPs on digits (80D, rugged NN landscape, expensive)'
    ),
    'Multi-MLP-Regression-70D': BenchmarkProblem(
        name='Multi-MLP-Regression-70D',
        objective=_make_optuna_objective(_multi_mlp_regression_70d, [(0, 1)] * 70, 70),
        dimension=70,
        bounds=[(0, 1)] * 70,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 10 MLP regressors on diabetes (70D, rugged, expensive)'
    ),
    'Multi-MLP-Complex-90D': BenchmarkProblem(
        name='Multi-MLP-Complex-90D',
        objective=_make_optuna_objective(_multi_mlp_complex_90d, [(0, 1)] * 90, 90),
        dimension=90,
        bounds=[(0, 1)] * 90,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 15 MLPs on complex target (90D, rugged, expensive)'
    ),
    'Multi-SVM-Grid-100D': BenchmarkProblem(
        name='Multi-SVM-Grid-100D',
        objective=_make_optuna_objective(_multi_svm_grid_100d, [(0, 1)] * 100, 100),
        dimension=100,
        bounds=[(0, 1)] * 100,
        known_optimum=None,
        category='ml_tuning_expensive',
        description='Ensemble of 50 SVMs on breast cancer (100D, rugged SVM landscape, expensive)'
    ),
}


def get_problem(name: str) -> BenchmarkProblem:
    """Get an ML problem by name."""
    if name not in ALL_ML_PROBLEMS:
        raise KeyError(f"Unknown problem: {name}. Available: {list(ALL_ML_PROBLEMS.keys())}")
    return ALL_ML_PROBLEMS[name]


def list_all_problems() -> list:
    """List all ML problem names."""
    return sorted(ALL_ML_PROBLEMS.keys())


def get_problems_by_category(category: str) -> Dict[str, BenchmarkProblem]:
    """Get all problems in a category."""
    return {
        name: prob for name, prob in ALL_ML_PROBLEMS.items()
        if prob.category == category
    }
