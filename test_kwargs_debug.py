"""Debug kwargs issue with constraint wrapper."""
import inspect
from ragda.constraints import create_constraint_wrapper

# Original objective with new API
def objective(x, y):
    print(f"objective called with x={x}, y={y}")
    return x**2 + y**2

# Create constraint wrapper
constraints_fns = [lambda x, y: x + y <= 5]
wrapped = create_constraint_wrapper(objective, constraints_fns, penalty=1e10)

# Check signature
sig = inspect.signature(wrapped)
print(f"Wrapped signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")

# Test calling it
params_dict = {'x': 2.0, 'y': 2.0}
print(f"\nCalling wrapped(**params_dict) where params_dict = {params_dict}")
result = wrapped(**params_dict)
print(f"Result: {result}")

# Test with violation
params_dict2 = {'x': 3.0, 'y': 3.0}
print(f"\nCalling wrapped(**params_dict) where params_dict = {params_dict2}")
result2 = wrapped(**params_dict2)
print(f"Result: {result2}")
