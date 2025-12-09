"""
Comprehensive Benchmark Functions Library

Part 2: Machine Learning Hyperparameter Tuning Problems
These showcase RAGDA's mini-batch feature for data-driven objectives.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')


@dataclass 
class MLProblem:
    """Container for ML hyperparameter tuning problem."""
    name: str
    func: Callable
    space: list
    dim: int
    data_size: int
    supports_batching: bool
    category: str  # 'classification', 'regression'
    description: str


# =============================================================================
# SKLEARN CLASSIFIERS
# =============================================================================

def create_lightgbm_problem(dataset: str = 'breast_cancer') -> Optional[MLProblem]:
    """LightGBM classifier tuning."""
    try:
        import lightgbm as lgb
        from sklearn.datasets import load_breast_cancer, load_digits, load_wine, load_iris
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None
    
    datasets = {
        'breast_cancer': load_breast_cancer,
        'digits': load_digits,
        'wine': load_wine,
        'iris': load_iris
    }
    
    X, y = datasets[dataset](return_X_y=True)
    n_classes = len(np.unique(y))
    
    def func(num_leaves, learning_rate, feature_fraction, bagging_fraction, bagging_freq, min_child_samples, batch_size: int = -1):
        lgb_params = {
            'objective': 'multiclass' if n_classes > 2 else 'binary',
            'num_class': n_classes if n_classes > 2 else None,
            'num_leaves': int(num_leaves),
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': int(bagging_freq),
            'min_child_samples': int(min_child_samples),
            'verbose': -1, 'n_jobs': 1
        }
        if lgb_params['num_class'] is None:
            del lgb_params['num_class']
        
        # Batch sampling
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = min(3, len(np.unique(y_eval)))
        else:
            X_eval, y_eval = X, y
            n_folds = 5
        
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=50, random_state=42)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
            return 1.0 - scores.mean()
        except:
            return 1.0
    
    space = {
        'num_leaves': {'type': 'ordinal', 'values': [15, 31, 63, 127, 255]},
        'learning_rate': {'type': 'continuous', 'bounds': [0.001, 0.3], 'log': True},
        'feature_fraction': {'type': 'continuous', 'bounds': [0.5, 1.0]},
        'bagging_fraction': {'type': 'continuous', 'bounds': [0.5, 1.0]},
        'bagging_freq': {'type': 'ordinal', 'values': [0, 1, 5, 10]},
        'min_child_samples': {'type': 'ordinal', 'values': [5, 10, 20, 50]},
    }
    
    return MLProblem(
        name=f'LightGBM-{dataset}', func=func, space=space, dim=6,
        data_size=len(X), supports_batching=True,
        category='classification', description=f'LightGBM on {dataset}'
    )


def create_xgboost_problem(dataset: str = 'breast_cancer') -> Optional[MLProblem]:
    """XGBoost classifier tuning."""
    try:
        import xgboost as xgb
        from sklearn.datasets import load_breast_cancer, load_digits
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None
    
    datasets = {'breast_cancer': load_breast_cancer, 'digits': load_digits}
    X, y = datasets.get(dataset, load_breast_cancer)(return_X_y=True)
    
    def func(max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, batch_size: int = -1):
        xgb_params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'verbosity': 0, 'n_jobs': 1, 'random_state': 42
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
        else:
            X_eval, y_eval = X, y
            n_folds = 5
        
        model = xgb.XGBClassifier(**xgb_params, n_estimators=50, use_label_encoder=False, eval_metric='logloss')
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
            return 1.0 - scores.mean()
        except:
            return 1.0
    
    space = {
        'max_depth': {'type': 'ordinal', 'values': [3, 5, 7, 9, 11]},
        'learning_rate': {'type': 'continuous', 'bounds': [0.001, 0.3], 'log': True},
        'subsample': {'type': 'continuous', 'bounds': [0.5, 1.0]},
        'colsample_bytree': {'type': 'continuous', 'bounds': [0.5, 1.0]},
        'min_child_weight': {'type': 'continuous', 'bounds': [1, 10]},
        'gamma': {'type': 'continuous', 'bounds': [0, 5]},
    }
    
    return MLProblem(
        name=f'XGBoost-{dataset}', func=func, space=space, dim=6,
        data_size=len(X), supports_batching=True,
        category='classification', description=f'XGBoost on {dataset}'
    )


def create_random_forest_problem(dataset: str = 'breast_cancer') -> Optional[MLProblem]:
    """Random Forest classifier tuning - optimized for speed."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_breast_cancer, load_wine
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None
    
    # Only use small datasets - digits is too slow
    datasets = {'breast_cancer': load_breast_cancer, 'wine': load_wine}
    if dataset not in datasets:
        dataset = 'breast_cancer'
    X, y = datasets[dataset](return_X_y=True)
    
    def func(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, batch_size: int = -1):
        rf_params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth) if max_depth > 0 else None,
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': max_features,
            'random_state': 42, 'n_jobs': 1
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
            rf_params['n_estimators'] = min(30, rf_params['n_estimators'])
        else:
            X_eval, y_eval = X, y
            n_folds = 3  # Reduced from 5
        
        model = RandomForestClassifier(**rf_params)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
            return 1.0 - scores.mean()
        except:
            return 1.0
    
    space = {
        'n_estimators': {'type': 'ordinal', 'values': [20, 50, 100]},  # Reduced
        'max_depth': {'type': 'ordinal', 'values': [5, 10, 20, -1]},  # -1 = None
        'min_samples_split': {'type': 'ordinal', 'values': [2, 5, 10]},
        'min_samples_leaf': {'type': 'ordinal', 'values': [1, 2, 4]},
        'max_features': {'type': 'continuous', 'bounds': [0.3, 1.0]},
    }
    
    return MLProblem(
        name=f'RandomForest-{dataset}', func=func, space=space, dim=5,
        data_size=len(X), supports_batching=True,
        category='classification', description=f'RF on {dataset}'
    )


def create_svm_problem(dataset: str = 'breast_cancer') -> Optional[MLProblem]:
    """SVM classifier tuning."""
    try:
        from sklearn.svm import SVC
        from sklearn.datasets import load_breast_cancer, load_digits
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    
    datasets = {'breast_cancer': load_breast_cancer, 'digits': load_digits}
    X, y = datasets.get(dataset, load_breast_cancer)(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    def func(C, gamma, kernel, batch_size: int = -1):
        svm_params = {
            'C': C,
            'gamma': gamma,
            'kernel': kernel,
            'random_state': 42
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
            max_iter = 200
        else:
            X_eval, y_eval = X, y
            n_folds = 5
            max_iter = 500
        
        model = SVC(**svm_params, max_iter=max_iter)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
            return 1.0 - scores.mean()
        except:
            return 1.0
    
    space = {
        'C': {'type': 'continuous', 'bounds': [0.01, 100], 'log': True},
        'gamma': {'type': 'continuous', 'bounds': [1e-4, 10], 'log': True},
        'kernel': {'type': 'categorical', 'values': ['rbf', 'poly', 'sigmoid']},
    }
    
    return MLProblem(
        name=f'SVM-{dataset}', func=func, space=space, dim=3,
        data_size=len(X), supports_batching=True,
        category='classification', description=f'SVM on {dataset}'
    )


def create_mlp_problem(dataset: str = 'digits') -> Optional[MLProblem]:
    """MLP Neural Network tuning."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.datasets import load_digits, load_breast_cancer
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    
    datasets = {'digits': load_digits, 'breast_cancer': load_breast_cancer}
    X, y = datasets.get(dataset, load_digits)(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    def func(hidden_size, alpha, learning_rate_init, activation, batch_size: int = -1):
        hidden_size = int(hidden_size)
        
        mlp_params = {
            'hidden_layer_sizes': (hidden_size, hidden_size // 2) if hidden_size > 32 else (hidden_size,),
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'activation': activation,
            'random_state': 42
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
            max_iter = 100
        else:
            X_eval, y_eval = X, y
            n_folds = 5
            max_iter = 300
        
        model = MLPClassifier(**mlp_params, max_iter=max_iter)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='accuracy')
            return 1.0 - scores.mean()
        except:
            return 1.0
    
    space = {
        'hidden_size': {'type': 'ordinal', 'values': [32, 64, 128, 256, 512]},
        'alpha': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
        'learning_rate_init': {'type': 'continuous', 'bounds': [1e-4, 1e-1], 'log': True},
        'activation': {'type': 'categorical', 'values': ['relu', 'tanh', 'logistic']},
    }
    
    return MLProblem(
        name=f'MLP-{dataset}', func=func, space=space, dim=4,
        data_size=len(X), supports_batching=True,
        category='classification', description=f'MLP on {dataset}'
    )


# =============================================================================
# SKLEARN REGRESSORS
# =============================================================================

def create_gradient_boosting_regressor(dataset: str = 'diabetes') -> Optional[MLProblem]:
    """Gradient Boosting Regressor tuning."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.datasets import load_diabetes, fetch_california_housing
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None
    
    if dataset == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    else:
        X, y = fetch_california_housing(return_X_y=True)
        # Subsample for speed
        idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
        X, y = X[idx], y[idx]
    
    def func(n_estimators, max_depth, learning_rate, subsample, min_samples_split, batch_size: int = -1):
        gbr_params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'min_samples_split': int(min_samples_split),
            'random_state': 42
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
            gbr_params['n_estimators'] = min(50, gbr_params['n_estimators'])
        else:
            X_eval, y_eval = X, y
            n_folds = 5
        
        model = GradientBoostingRegressor(**gbr_params)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='neg_mean_squared_error')
            return -scores.mean()  # Return positive MSE (to minimize)
        except:
            return 1e10
    
    space = {
        'n_estimators': {'type': 'ordinal', 'values': [50, 100, 200, 300]},
        'max_depth': {'type': 'ordinal', 'values': [3, 5, 7, 9]},
        'learning_rate': {'type': 'continuous', 'bounds': [0.01, 0.3], 'log': True},
        'subsample': {'type': 'continuous', 'bounds': [0.5, 1.0]},
        'min_samples_split': {'type': 'ordinal', 'values': [2, 5, 10]},
    }
    
    return MLProblem(
        name=f'GBRegressor-{dataset}', func=func, space=space, dim=5,
        data_size=len(X), supports_batching=True,
        category='regression', description=f'GBR on {dataset}'
    )


def create_elastic_net_problem(dataset: str = 'diabetes') -> Optional[MLProblem]:
    """ElasticNet regression tuning."""
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    
    X, y = load_diabetes(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    def func(alpha, l1_ratio, batch_size: int = -1):
        en_params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'max_iter': 1000, 'random_state': 42
        }
        
        if batch_size > 0 and batch_size < len(X):
            idx = np.random.choice(len(X), batch_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
            n_folds = 3
        else:
            X_eval, y_eval = X, y
            n_folds = 5
        
        model = ElasticNet(**en_params)
        try:
            scores = cross_val_score(model, X_eval, y_eval, cv=n_folds, scoring='neg_mean_squared_error')
            return -scores.mean()
        except:
            return 1e10
    
    space = {
        'alpha': {'type': 'continuous', 'bounds': [1e-5, 10], 'log': True},
        'l1_ratio': {'type': 'continuous', 'bounds': [0.0, 1.0]},
    }
    
    return MLProblem(
        name=f'ElasticNet-{dataset}', func=func, space=space, dim=2,
        data_size=len(X), supports_batching=True,
        category='regression', description=f'ElasticNet on {dataset}'
    )


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def create_portfolio_problem(n_assets: int = 10, n_periods: int = 252) -> MLProblem:
    """Portfolio optimization with data batching (time periods)."""
    np.random.seed(42)
    
    # Generate synthetic returns with structure
    returns = np.random.randn(n_periods, n_assets) * 0.01
    factor = np.random.randn(n_periods, 3) * 0.015
    loadings = np.random.randn(n_assets, 3)
    returns = returns + factor @ loadings.T
    
    mean_returns = returns.mean(axis=0)
    cov_matrix = np.cov(returns.T)
    
    def func(**kwargs):
        batch_size = kwargs.pop('batch_size', -1)
        weights = np.array([kwargs[f'w{i}'] for i in range(n_assets)])
        weights = np.abs(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        if batch_size > 0 and batch_size < n_periods:
            idx = np.random.choice(n_periods, batch_size, replace=False)
            returns_batch = returns[idx]
            mean_ret = returns_batch.mean(axis=0)
            cov_mat = np.cov(returns_batch.T) if batch_size > n_assets else cov_matrix
        else:
            mean_ret = mean_returns
            cov_mat = cov_matrix
        
        portfolio_return = weights @ mean_ret
        portfolio_risk = np.sqrt(weights @ cov_mat @ weights)
        
        if portfolio_risk < 1e-8:
            return 1e10
        
        sharpe = portfolio_return / portfolio_risk
        return -sharpe  # Maximize Sharpe
    
    space = {f'w{i}': {'type': 'continuous', 'bounds': [0.0, 1.0]} for i in range(n_assets)}
    
    return MLProblem(
        name=f'Portfolio-{n_assets}assets', func=func, space=space, dim=n_assets,
        data_size=n_periods, supports_batching=True,
        category='finance', description=f'Portfolio opt with {n_assets} assets'
    )


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

def get_all_ml_problems() -> Dict[str, MLProblem]:
    """Get dictionary of all ML tuning problems."""
    problems = {}
    
    # LightGBM on multiple datasets (fast enough even on digits)
    for ds in ['breast_cancer', 'digits', 'wine', 'iris']:
        prob = create_lightgbm_problem(ds)
        if prob:
            problems[f'lightgbm_{ds}'] = prob
    
    # XGBoost (moderate speed, good for expensive category)
    for ds in ['breast_cancer', 'digits']:
        prob = create_xgboost_problem(ds)
        if prob:
            problems[f'xgboost_{ds}'] = prob
    
    # Random Forest - only small datasets (RF on digits is too slow)
    for ds in ['breast_cancer', 'wine']:
        prob = create_random_forest_problem(ds)
        if prob:
            problems[f'rf_{ds}'] = prob
    
    # SVM - breast_cancer is fast, digits is moderate/expensive
    for ds in ['breast_cancer', 'digits']:
        prob = create_svm_problem(ds)
        if prob:
            problems[f'svm_{ds}'] = prob
    
    # MLP - both are moderate speed
    for ds in ['digits', 'breast_cancer']:
        prob = create_mlp_problem(ds)
        if prob:
            problems[f'mlp_{ds}'] = prob
    
    # Regressors
    for ds in ['diabetes']:
        prob = create_gradient_boosting_regressor(ds)
        if prob:
            problems[f'gbr_{ds}'] = prob
        prob = create_elastic_net_problem(ds)
        if prob:
            problems[f'elasticnet_{ds}'] = prob
    
    # Portfolio (various sizes)
    for n_assets in [5, 10, 20, 30, 50]:
        problems[f'portfolio_{n_assets}'] = create_portfolio_problem(n_assets)
    
    return problems


if __name__ == '__main__':
    problems = get_all_ml_problems()
    print(f"Total ML problems: {len(problems)}")
    
    categories = {}
    for p in problems.values():
        categories[p.category] = categories.get(p.category, 0) + 1
    
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nProblems:")
    for name, p in sorted(problems.items()):
        print(f"  {name}: dim={p.dim}, data_size={p.data_size}, batch={p.supports_batching}")
