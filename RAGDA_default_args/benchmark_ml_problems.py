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
