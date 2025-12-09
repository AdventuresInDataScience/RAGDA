"""
Tests for new dict-based SearchSpace API
"""
import pytest
import numpy as np
from ragda.space import SearchSpace, Parameter


def test_dict_based_space_creation():
    """Test creating SearchSpace with dict format."""
    space_dict = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]},
        'lr': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
    }
    
    space = SearchSpace(space_dict)
    
    assert space.n_params == 3
    assert space.n_continuous == 3
    assert space.n_categorical == 0
    assert set(space.param_names) == {'x', 'y', 'lr'}


def test_dict_with_mixed_types():
    """Test dict format with continuous, ordinal, and categorical."""
    space_dict = {
        'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
        'n_layers': {'type': 'ordinal', 'values': [1, 2, 4, 8, 16]},
        'batch_size': {'type': 'ordinal', 'values': [16, 32, 64, 128]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
        'activation': {'type': 'categorical', 'values': ['relu', 'tanh']},
    }
    
    space = SearchSpace(space_dict)
    
    assert space.n_params == 5
    assert space.n_continuous == 3  # continuous + ordinal
    assert space.n_categorical == 2
    assert 'learning_rate' in space.param_names
    assert 'optimizer' in space.param_names


def test_conversion_list_to_dict():
    """Test conversion from list to dict format."""
    space_list = [
        {'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]},
        {'name': 'y', 'type': 'continuous', 'bounds': [0, 10]},
    ]
    
    space_dict = SearchSpace._list_to_dict(space_list)
    
    assert len(space_dict) == 2
    assert 'x' in space_dict
    assert 'y' in space_dict
    assert 'name' not in space_dict['x']
    assert space_dict['x']['type'] == 'continuous'
    assert space_dict['x']['bounds'] == [-5, 5]


def test_to_dict_export():
    """Test exporting SearchSpace to dict format."""
    space_dict = {
        'learning_rate': {'type': 'continuous', 'bounds': [1e-5, 1e-1], 'log': True},
        'n_layers': {'type': 'ordinal', 'values': [1, 2, 4, 8]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
    }
    
    space = SearchSpace(space_dict)
    exported = space.to_dict()
    
    assert exported.keys() == space_dict.keys()
    assert exported['learning_rate']['type'] == 'continuous'
    assert exported['learning_rate']['log'] == True
    assert exported['n_layers']['values'] == [1, 2, 4, 8]
    assert exported['optimizer']['values'] == ['adam', 'sgd']


def test_sampling_with_dict_format():
    """Test that sampling works with dict-based space."""
    space_dict = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [-5, 5]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
    }
    
    space = SearchSpace(space_dict)
    samples = space.sample(n=10, method='lhs')
    
    assert len(samples) == 10
    for sample in samples:
        assert 'x' in sample
        assert 'y' in sample
        assert 'optimizer' in sample
        assert -5 <= sample['x'] <= 5
        assert -5 <= sample['y'] <= 5
        assert sample['optimizer'] in ['adam', 'sgd']


def test_validate_with_dict_format():
    """Test parameter validation with dict-based space."""
    space_dict = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [0, 10]},
    }
    
    space = SearchSpace(space_dict)
    
    # Valid params
    assert space.validate({'x': 0.0, 'y': 5.0}) == True
    assert space.validate({'x': -5.0, 'y': 10.0}) == True
    
    # Invalid params
    assert space.validate({'x': 10.0, 'y': 5.0}) == False  # x out of bounds
    assert space.validate({'x': 0.0}) == False  # missing y
    assert space.validate({'x': 0.0, 'y': 5.0, 'z': 1.0}) == True  # extra param ignored


def test_empty_space_raises_error():
    """Test that empty space raises ValueError."""
    with pytest.raises(ValueError, match="Search space cannot be empty"):
        SearchSpace({})


def test_invalid_space_type_raises_error():
    """Test that invalid space type raises TypeError."""
    with pytest.raises(TypeError, match="parameters must be dict"):
        SearchSpace("invalid")
    
    with pytest.raises(TypeError, match="parameters must be dict"):
        SearchSpace(42)
    
    # List format (even empty) raises TypeError, not ValueError
    with pytest.raises(TypeError, match="parameters must be dict"):
        SearchSpace([{'name': 'x', 'type': 'continuous', 'bounds': [-5, 5]}])


def test_split_vectors_conversion():
    """Test conversion to/from split vectors works with dict format."""
    space_dict = {
        'x': {'type': 'continuous', 'bounds': [-5, 5]},
        'y': {'type': 'continuous', 'bounds': [0, 10]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd', 'rmsprop']},
    }
    
    space = SearchSpace(space_dict)
    
    params = {'x': 2.5, 'y': 7.0, 'optimizer': 'sgd'}
    x_cont, x_cat, cat_n_values = space.to_split_vectors(params)
    
    assert len(x_cont) == 2
    assert len(x_cat) == 1
    assert x_cat[0] == 1  # 'sgd' is index 1
    assert cat_n_values[0] == 3
    
    # Convert back
    params_restored = space.from_split_vectors(x_cont, x_cat)
    assert params_restored['x'] == pytest.approx(2.5, abs=0.01)
    assert params_restored['y'] == pytest.approx(7.0, abs=0.01)
    assert params_restored['optimizer'] == 'sgd'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
