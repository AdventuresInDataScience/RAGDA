"""
Tests for constraint parsing and evaluation.
"""
import pytest
import warnings
from ragda.constraints import ConstraintParser, parse_constraints, create_constraint_wrapper


class TestConstraintParser:
    """Test ConstraintParser functionality."""
    
    def test_simple_numeric_constraint(self):
        """Test parsing simple numeric constraint."""
        parser = ConstraintParser(['x', 'y'])
        fn = parser.parse('x + y <= 5')
        
        assert fn(x=2, y=2) == True
        assert fn(x=3, y=3) == False
        assert fn(x=2, y=3) == True
        assert fn(x=5, y=0) == True
    
    def test_complex_numeric_constraint(self):
        """Test parsing complex numeric expressions."""
        parser = ConstraintParser(['x', 'y'])
        fn = parser.parse('x**2 + y**2 <= 100')
        
        assert fn(x=0, y=0) == True
        assert fn(x=5, y=5) == True
        assert fn(x=10, y=10) == False
        assert fn(x=6, y=8) == True  # 36 + 64 = 100
    
    def test_comparison_operators(self):
        """Test all comparison operators."""
        parser = ConstraintParser(['x'])
        
        # Less than
        fn_lt = parser.parse('x < 5')
        assert fn_lt(x=4) == True
        assert fn_lt(x=5) == False
        
        # Less than or equal
        fn_le = parser.parse('x <= 5')
        assert fn_le(x=5) == True
        assert fn_le(x=6) == False
        
        # Greater than
        fn_gt = parser.parse('x > 5')
        assert fn_gt(x=6) == True
        assert fn_gt(x=5) == False
        
        # Greater than or equal
        fn_ge = parser.parse('x >= 5')
        assert fn_ge(x=5) == True
        assert fn_ge(x=4) == False
        
        # Equality
        fn_eq = parser.parse('x == 5')
        assert fn_eq(x=5) == True
        assert fn_eq(x=4) == False
        
        # Inequality
        fn_ne = parser.parse('x != 5')
        assert fn_ne(x=4) == True
        assert fn_ne(x=5) == False
    
    def test_categorical_constraint(self):
        """Test categorical comparisons with strings."""
        parser = ConstraintParser(['optimizer', 'x'])
        fn = parser.parse('optimizer == "adam"')
        
        assert fn(optimizer='adam', x=1) == True
        assert fn(optimizer='sgd', x=1) == False
    
    def test_categorical_inequality(self):
        """Test categorical inequality."""
        parser = ConstraintParser(['activation', 'x'])
        fn = parser.parse('activation != "sigmoid"')
        
        assert fn(activation='relu', x=1) == True
        assert fn(activation='sigmoid', x=1) == False
    
    def test_implication_operator(self):
        """Test implication operator: A -> B."""
        parser = ConstraintParser(['x', 'y'])
        fn = parser.parse('x > 5 -> y > 10')
        
        # If x > 5, then y must be > 10
        assert fn(x=6, y=11) == True   # x > 5, y > 10: satisfied
        assert fn(x=6, y=5) == False   # x > 5, y <= 10: violated
        assert fn(x=4, y=5) == True    # x <= 5: antecedent false, always true
        assert fn(x=4, y=15) == True   # x <= 5: antecedent false, always true
    
    def test_categorical_implication(self):
        """Test implication with categorical variable."""
        parser = ConstraintParser(['optimizer', 'learning_rate'])
        fn = parser.parse('optimizer == "sgd" -> learning_rate <= 0.01')
        
        assert fn(optimizer='sgd', learning_rate=0.005) == True
        assert fn(optimizer='sgd', learning_rate=0.05) == False
        assert fn(optimizer='adam', learning_rate=0.05) == True
    
    def test_math_functions(self):
        """Test allowed math functions."""
        parser = ConstraintParser(['x', 'y'])
        
        # abs
        fn_abs = parser.parse('abs(x - y) <= 2')
        assert fn_abs(x=5, y=6) == True
        assert fn_abs(x=5, y=8) == False
        
        # sqrt
        fn_sqrt = parser.parse('sqrt(x**2 + y**2) <= 10')
        assert fn_sqrt(x=3, y=4) == True  # sqrt(25) = 5
        assert fn_sqrt(x=8, y=8) == False  # sqrt(128) > 10
    
    def test_logical_and(self):
        """Test logical AND in constraints."""
        parser = ConstraintParser(['x', 'y', 'z'])
        fn = parser.parse('x + y <= 5 and z >= 0')
        
        assert fn(x=2, y=2, z=1) == True
        assert fn(x=3, y=3, z=1) == False  # x + y > 5
        assert fn(x=2, y=2, z=-1) == False  # z < 0
    
    def test_logical_or(self):
        """Test logical OR in constraints."""
        parser = ConstraintParser(['x', 'y'])
        fn = parser.parse('x >= 5 or y >= 5')
        
        assert fn(x=6, y=0) == True
        assert fn(x=0, y=6) == True
        assert fn(x=6, y=6) == True
        assert fn(x=4, y=4) == False
    
    def test_complex_mixed_constraint(self):
        """Test complex constraint with categorical and numeric."""
        parser = ConstraintParser(['optimizer', 'n_layers', 'dropout'])
        fn = parser.parse('optimizer == "sgd" and n_layers > 4 -> dropout >= 0.3')
        
        # SGD + deep network -> needs dropout
        assert fn(optimizer='sgd', n_layers=8, dropout=0.4) == True
        assert fn(optimizer='sgd', n_layers=8, dropout=0.2) == False
        
        # SGD + shallow network -> no constraint on dropout
        assert fn(optimizer='sgd', n_layers=2, dropout=0.1) == True
        
        # Not SGD -> no constraint
        assert fn(optimizer='adam', n_layers=8, dropout=0.1) == True
    
    def test_undefined_parameter_error(self):
        """Test error when using undefined parameter."""
        parser = ConstraintParser(['x', 'y'])
        
        with pytest.raises(ValueError, match="undefined parameters"):
            parser.parse('x + z <= 5')
    
    def test_invalid_expression_error(self):
        """Test error on expressions that don't return boolean."""
        parser = ConstraintParser(['x', 'y'])
        
        # Expression without comparison - would return a number, not boolean
        with pytest.raises(ValueError, match="return a boolean"):
            parser.parse('x + y')
    
    def test_syntax_error(self):
        """Test error on invalid Python syntax."""
        parser = ConstraintParser(['x'])
        
        with pytest.raises(ValueError, match="Invalid Python expression"):
            parser.parse('x +* 5 <= 10')
    
    def test_dangerous_function_error(self):
        """Test that dangerous functions are detected as undefined parameters."""
        parser = ConstraintParser(['x'])
        
        # eval is not a valid parameter or allowed function
        with pytest.raises(ValueError, match="undefined parameters"):
            parser.parse('eval("x") <= 5')
    
    def test_parse_all(self):
        """Test parsing multiple constraints."""
        parser = ConstraintParser(['x', 'y', 'z'])
        constraints = [
            'x + y <= 10',
            'z >= 0',
            'x**2 + y**2 <= 100'
        ]
        
        fns = parser.parse_all(constraints)
        
        assert len(fns) == 3
        assert fns[0](x=5, y=5, z=0) == True
        assert fns[1](x=0, y=0, z=1) == True
        assert fns[2](x=6, y=8, z=0) == True


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_parse_constraints_function(self):
        """Test parse_constraints convenience function."""
        constraints = ['x + y <= 5', 'x >= 0', 'y >= 0']
        fns = parse_constraints(constraints, ['x', 'y'])
        
        assert len(fns) == 3
        assert all(fn(x=2, y=2) for fn in fns)
    
    def test_create_constraint_wrapper(self):
        """Test constraint wrapper creation."""
        def objective(x, y):
            return x**2 + y**2
        
        constraints = [
            lambda x, y: x + y <= 5,
            lambda x, y: x >= 0,
        ]
        
        wrapped = create_constraint_wrapper(objective, constraints, penalty=1000)
        
        # Satisfied constraints
        assert wrapped(x=2, y=2) == 8  # 2^2 + 2^2
        
        # Violated constraint: x + y > 5
        assert wrapped(x=3, y=3) == 1000
        
        # Violated constraint: x < 0
        assert wrapped(x=-1, y=2) == 1000
    
    def test_wrapper_with_no_constraints(self):
        """Test wrapper with empty constraints list."""
        def objective(x, y):
            return x + y
        
        wrapped = create_constraint_wrapper(objective, [], penalty=1000)
        
        # Should just call objective
        assert wrapped(x=5, y=3) == 8
    
    def test_wrapper_handles_evaluation_errors(self):
        """Test that wrapper handles constraint evaluation errors gracefully."""
        def objective(x, y):
            return x + y
        
        def bad_constraint(x, y):
            raise ValueError("Constraint error")
        
        wrapped = create_constraint_wrapper(objective, [bad_constraint], penalty=999)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped(x=1, y=2)
            assert result == 999
            assert len(w) == 1
            assert "Constraint evaluation failed" in str(w[0].message)


class TestRealWorldScenarios:
    """Test real-world constraint scenarios."""
    
    def test_ml_hyperparameter_constraints(self):
        """Test typical ML hyperparameter constraints."""
        parser = ConstraintParser(['learning_rate', 'n_layers', 'dropout', 'optimizer', 'batch_size'])
        
        constraints = [
            'n_layers <= 4 or dropout >= 0.2',  # Deep nets need dropout
            'optimizer == "sgd" -> learning_rate <= 0.01',  # SGD needs low LR
            'learning_rate * batch_size <= 0.5',  # Effective LR bound (relaxed for test)
        ]
        
        fns = [parser.parse(c) for c in constraints]
        
        # Valid config
        params = {'learning_rate': 0.001, 'n_layers': 8, 'dropout': 0.3, 'optimizer': 'adam', 'batch_size': 128}
        # lr * bs = 0.001 * 128 = 0.128 <= 0.5 ✓
        assert all(fn(**params) for fn in fns)
        
        # Invalid: deep network without dropout
        params = {'learning_rate': 0.001, 'n_layers': 8, 'dropout': 0.1, 'optimizer': 'adam', 'batch_size': 128}
        assert not fns[0](**params)  # n_layers > 4 and dropout < 0.2
        
        # Invalid: SGD with high LR
        params = {'learning_rate': 0.05, 'n_layers': 4, 'dropout': 0.2, 'optimizer': 'sgd', 'batch_size': 128}
        assert not fns[1](**params)
    
    def test_portfolio_constraints(self):
        """Test portfolio optimization constraints."""
        parser = ConstraintParser(['stock_a', 'stock_b', 'stock_c', 'bonds'])
        
        constraints = [
            'stock_a + stock_b + stock_c + bonds == 1.0',  # Budget
            'stock_a >= 0.1',  # Minimum allocation
            'bonds >= 0.2',  # Minimum safe allocation
            'stock_a + stock_b + stock_c <= 0.8',  # Max in stocks
        ]
        
        fns = [parser.parse(c) for c in constraints]
        
        # Valid allocation
        params = {'stock_a': 0.3, 'stock_b': 0.2, 'stock_c': 0.2, 'bonds': 0.3}
        assert all(fn(**params) for fn in fns)
    
    def test_incompatible_categorical_combinations(self):
        """Test excluding specific categorical combinations."""
        parser = ConstraintParser(['param1', 'param2', 'x'])
        
        # If param1 == "a", then param2 cannot be "b"
        # Equivalent to: not(param1 == "a" and param2 == "b")
        # Which is: param1 != "a" or param2 != "b"
        fn = parser.parse('param1 != "a" or param2 != "b"')
        
        assert fn(param1='a', param2='c', x=1) == True
        assert fn(param1='a', param2='b', x=1) == False
        assert fn(param1='c', param2='b', x=1) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
