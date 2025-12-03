## Installation

### Using pip
```bash
pip install ragda
```

That's it! RAGDA will automatically compile the Cython extensions during installation.

### From source
```bash
git clone https://github.com/yourusername/ragda.git
cd ragda
pip install .
```

### Requirements

RAGDA automatically installs these dependencies:
- `numpy>=1.20.0` - Core numerical operations
- `pandas>=1.3.0` - Results DataFrames and analysis
- `scipy>=1.7.0` - Latin Hypercube Sampling
- `cython>=0.29.0` - Build-time dependency for compilation

**Platform Support:**
- Linux ✓
- macOS ✓
- Windows ✓

**Python Support:**
- Python 3.8+ ✓

**No GPU required** - Speed comes from Cython compilation and multi-core CPU parallelization.

### Verifying Installation
```python
import ragda
print(f"RAGDA version: {ragda.__version__}")

# Run quick test
from ragda import ragda_optimize
import numpy as np

def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-5, 5]] * 3)
x_best, f_best, info = ragda_optimize(sphere, bounds, n_trials=50)
print(f"Test optimization: f_best = {f_best:.6f} (should be near 0)")
```

If this runs without errors, RAGDA is installed correctly!