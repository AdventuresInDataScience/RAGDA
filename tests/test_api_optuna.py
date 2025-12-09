"""
Comprehensive unit tests for Optuna-style API adapter.

Tests the OptunaAdapter, RAGDATrial, Study, and create_study functionality.
"""

import pytest
import numpy as np
from ragda import create_study, Study
from ragda.api_adapters import OptunaAdapter, RAGDATrial


class TestRAGDATrial:
    """Tests for the RAGDATrial mock object."""
    
    def test_suggest_float_basic(self):
        """Test basic float suggestion."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_float('x', -5.0, 5.0)
        
        assert 'x' in space_def
        assert space_def['x']['type'] == 'float'
        assert space_def['x']['low'] == -5.0
        assert space_def['x']['high'] == 5.0
        assert space_def['x']['log'] is False
        assert -5.0 <= value <= 5.0
    
    def test_suggest_float_log_scale(self):
        """Test float suggestion with log scale."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        
        assert space_def['lr']['log'] is True
        assert 1e-5 <= value <= 1e-1
    
    def test_suggest_float_with_step(self):
        """Test float suggestion with discrete steps."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_float('x', 0.0, 1.0, step=0.1)
        
        assert 'step' in space_def['x']
        assert space_def['x']['step'] == 0.1
        assert 0.0 <= value <= 1.0
        # Check it's a multiple of step (within floating point precision)
        assert abs(value - round(value / 0.1) * 0.1) < 1e-9
    
    def test_suggest_int_basic(self):
        """Test basic int suggestion."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_int('n', 1, 10)
        
        assert space_def['n']['type'] == 'int'
        assert space_def['n']['low'] == 1
        assert space_def['n']['high'] == 10
        assert space_def['n']['step'] == 1
        assert 1 <= value <= 10
        assert isinstance(value, int)
    
    def test_suggest_int_with_step(self):
        """Test int suggestion with custom step."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_int('n', 0, 100, step=10)
        
        assert space_def['n']['step'] == 10
        assert value % 10 == 0
        assert 0 <= value <= 100
    
    def test_suggest_int_log_scale(self):
        """Test int suggestion with log scale."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_int('size', 1, 1000, log=True)
        
        assert space_def['size']['log'] is True
        assert 1 <= value <= 1000
        assert isinstance(value, int)
    
    def test_suggest_categorical(self):
        """Test categorical suggestion."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        choices = ['adam', 'sgd', 'rmsprop']
        value = trial.suggest_categorical('optimizer', choices)
        
        assert space_def['optimizer']['type'] == 'categorical'
        assert space_def['optimizer']['choices'] == choices
        assert value in choices
    
    def test_suggest_discrete_uniform(self):
        """Test deprecated discrete_uniform (should work like float with step)."""
        space_def = {}
        params = {}
        trial = RAGDATrial(params, space_def)
        
        value = trial.suggest_discrete_uniform('x', 0.0, 1.0, 0.2)
        
        assert 'x' in space_def
        assert 0.0 <= value <= 1.0
    
    def test_pre_sampled_params(self):
        """Test that pre-sampled params are used."""
        space_def = {}
        params = {'x': 3.14, 'y': 2.71}
        trial = RAGDATrial(params, space_def)
        
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        
        assert x == 3.14
        assert y == 2.71
    
    def test_frozen_space(self):
        """Test that space doesn't change after being frozen."""
        space_def = {'x': {'type': 'float', 'low': -5, 'high': 5, 'log': False}}
        params = {'x': 1.0}
        trial = RAGDATrial(params, space_def)
        trial._frozen = True
        
        # Should not add new parameter to frozen space
        y = trial.suggest_float('y', 0, 10)
        
        assert 'y' not in space_def  # Frozen, so y not added


class TestOptunaAdapter:
    """Tests for the OptunaAdapter class."""
    
    def test_space_discovery_continuous(self):
        """Test space discovery with continuous parameters."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', 0, 10, log=False)
            return x**2 + y**2
        
        canonical = adapter.discover_space(objective)
        
        assert 'x' in canonical
        assert 'y' in canonical
        assert canonical['x']['type'] == 'continuous'
        assert canonical['x']['bounds'] == [-5.0, 5.0]
        assert canonical['y']['type'] == 'continuous'
        assert canonical['y']['bounds'] == [0.0, 10.0]
    
    def test_space_discovery_log_scale(self):
        """Test space discovery with log-scale parameter."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            return lr
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['learning_rate']['type'] == 'continuous'
        assert canonical['learning_rate']['log'] is True
    
    def test_space_discovery_discrete_float(self):
        """Test that discrete float becomes ordinal."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            x = trial.suggest_float('x', 0.0, 1.0, step=0.1)
            return x
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['x']['type'] == 'ordinal'
        assert len(canonical['x']['values']) == 11  # 0.0, 0.1, ..., 1.0
    
    def test_space_discovery_int_continuous(self):
        """Test that simple int becomes continuous."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            n = trial.suggest_int('n', 1, 100)
            return n
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['n']['type'] == 'continuous'
        assert canonical['n']['bounds'] == [1.0, 100.0]
    
    def test_space_discovery_int_discrete(self):
        """Test that int with step > 1 becomes ordinal."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            n = trial.suggest_int('n', 0, 100, step=10)
            return n
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['n']['type'] == 'ordinal'
        assert canonical['n']['values'] == list(range(0, 110, 10))
    
    def test_space_discovery_int_log(self):
        """Test that log-scale int becomes ordinal."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            size = trial.suggest_int('size', 1, 1024, log=True)
            return size
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['size']['type'] == 'ordinal'
        assert 1 in canonical['size']['values']
        assert 1024 in canonical['size']['values']
    
    def test_space_discovery_categorical(self):
        """Test space discovery with categorical parameter."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            opt = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
            return 0
        
        canonical = adapter.discover_space(objective)
        
        assert canonical['optimizer']['type'] == 'categorical'
        assert canonical['optimizer']['values'] == ['adam', 'sgd', 'rmsprop']
    
    def test_space_discovery_mixed(self):
        """Test space discovery with mixed parameter types."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            n = trial.suggest_int('n', 1, 10)
            method = trial.suggest_categorical('method', ['A', 'B'])
            return x + n
        
        canonical = adapter.discover_space(objective)
        
        assert len(canonical) == 3
        assert canonical['x']['type'] == 'continuous'
        assert canonical['n']['type'] == 'continuous'
        assert canonical['method']['type'] == 'categorical'
    
    def test_wrap_objective(self):
        """Test objective wrapping."""
        adapter = OptunaAdapter()
        
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            return x**2 + y**2
        
        # Discover space first
        adapter.discover_space(objective)
        
        # Wrap objective
        wrapped = adapter.wrap_objective(objective)
        
        # Call with kwargs
        result = wrapped(x=2.0, y=3.0)
        assert result == 13.0  # 2^2 + 3^2
    
    def test_space_not_discovered_error(self):
        """Test error when trying to use adapter before space discovery."""
        adapter = OptunaAdapter()
        
        with pytest.raises(ValueError, match="Space not yet discovered"):
            adapter.to_canonical_space()


class TestStudyClass:
    """Tests for the Study class (Optuna-compatible API)."""
    
    def test_study_creation(self):
        """Test basic study creation."""
        study = Study(direction='minimize', random_state=42)
        
        assert study.direction == 'minimize'
        assert study.random_state == 42
    
    def test_study_optimize_minimize(self):
        """Test study optimization (minimize)."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            return x**2
        
        study = Study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value < 0.5
        assert abs(study.best_params['x']) < 1.0
    
    def test_study_optimize_maximize(self):
        """Test study optimization (maximize)."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            return -x**2
        
        study = Study(direction='maximize', random_state=42)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value > -0.5
        assert abs(study.best_params['x']) < 1.0
    
    def test_study_best_properties(self):
        """Test study best_value, best_params, best_trial."""
        def objective(trial):
            x = trial.suggest_float('x', 0, 10)
            return x
        
        study = Study(random_state=42)
        study.optimize(objective, n_trials=20)
        
        assert isinstance(study.best_value, float)
        assert isinstance(study.best_params, dict)
        assert 'x' in study.best_params
        assert hasattr(study.best_trial, 'value')
        assert hasattr(study.best_trial, 'params')
        assert study.best_trial.value == study.best_value
        assert study.best_trial.params == study.best_params
    
    def test_study_trials_list(self):
        """Test study.trials list."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            return x**2 + y**2
        
        study = Study(random_state=42)
        study.optimize(objective, n_trials=30)
        
        assert len(study.trials) > 0
        assert len(study.trials) <= 30  # May converge early
        
        for trial in study.trials:
            assert hasattr(trial, 'value')
            assert hasattr(trial, 'params')
            assert hasattr(trial, 'number')
            assert isinstance(trial.params, dict)
    
    def test_study_not_optimized_error(self):
        """Test error when accessing results before optimization."""
        study = Study()
        
        with pytest.raises(RuntimeError, match="optimize.*not been called"):
            _ = study.best_value
        
        with pytest.raises(RuntimeError, match="optimize.*not been called"):
            _ = study.best_params
        
        with pytest.raises(RuntimeError, match="optimize.*not been called"):
            _ = study.best_trial
    
    def test_study_n_jobs(self):
        """Test study with multiple workers."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            return x**2
        
        study = Study(random_state=42)
        study.optimize(objective, n_trials=50, n_jobs=2)
        
        assert study.best_value < 0.5


class TestCreateStudyFunction:
    """Tests for the create_study() convenience function."""
    
    def test_create_study_basic(self):
        """Test basic create_study usage."""
        study = create_study(direction='minimize', random_state=42)
        
        assert isinstance(study, Study)
        assert study.direction == 'minimize'
        assert study.random_state == 42
    
    def test_create_study_with_kwargs(self):
        """Test create_study with optimizer kwargs."""
        study = create_study(
            direction='maximize',
            random_state=123,
            top_n_percent=0.3
        )
        
        assert study.direction == 'maximize'
        assert 'top_n_percent' in study.optimizer_kwargs
    
    def test_create_study_end_to_end(self):
        """Test end-to-end workflow with create_study."""
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return (x - 3)**2 + (y + 2)**2
        
        study = create_study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=100)
        
        # Should find near (3, -2)
        assert study.best_value < 1.0
        assert abs(study.best_params['x'] - 3) < 1.5
        assert abs(study.best_params['y'] + 2) < 1.5


class TestOptunaComplexScenarios:
    """Tests for complex scenarios with Optuna-style API."""
    
    def test_mixed_parameter_types(self):
        """Test optimization with all parameter types."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            n = trial.suggest_int('n', 1, 10)
            method = trial.suggest_categorical('method', ['A', 'B', 'C'])
            
            base = x**2 + n * 0.1
            if method == 'A':
                base *= 1.0
            elif method == 'B':
                base *= 0.5
            else:
                base *= 2.0
            
            return base
        
        study = create_study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=100)
        
        assert study.best_value < 1.5
        assert 'x' in study.best_params
        assert 'n' in study.best_params
        assert 'method' in study.best_params
    
    def test_log_scale_parameters(self):
        """Test optimization with log-scale parameters."""
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            reg = trial.suggest_float('regularization', 1e-6, 1e-2, log=True)
            
            # Optimal around lr=1e-3, reg=1e-4
            return (np.log10(lr) + 3)**2 + (np.log10(reg) + 4)**2
        
        study = create_study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=100)
        
        assert study.best_value < 2.0
        assert 1e-4 < study.best_params['learning_rate'] < 1e-2
        assert 1e-5 < study.best_params['regularization'] < 1e-3
    
    def test_discrete_parameters(self):
        """Test optimization with discrete float parameters."""
        def objective(trial):
            # Discrete values: 0.0, 0.1, 0.2, ..., 1.0
            dropout = trial.suggest_float('dropout', 0.0, 1.0, step=0.1)
            # Optimal around 0.3
            return (dropout - 0.3)**2
        
        study = create_study(direction='minimize', random_state=42)
        study.optimize(objective, n_trials=50)
        
        assert study.best_value < 0.05
        assert abs(study.best_params['dropout'] - 0.3) <= 0.1
    
    def test_reproducibility(self):
        """Test that same random_state gives similar results."""
        def objective(trial):
            x = trial.suggest_float('x', -5, 5)
            y = trial.suggest_float('y', -5, 5)
            return x**2 + y**2
        
        study1 = create_study(direction='minimize', random_state=42)
        study1.optimize(objective, n_trials=50)
        
        study2 = create_study(direction='minimize', random_state=42)
        study2.optimize(objective, n_trials=50)
        
        # Should give similar (not necessarily identical) results due to stochastic nature
        assert study1.best_value < 0.5
        assert study2.best_value < 0.5
        # Both should find good solutions, but exact values may differ slightly
        assert abs(study1.best_value - study2.best_value) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
