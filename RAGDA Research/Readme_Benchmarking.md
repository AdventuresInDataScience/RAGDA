# RAGDA Benchmarking Suite

Complete benchmarking infrastructure for research paper.

## Quick Start
```bash
# Install all dependencies
pip install -r requirements_benchmark.txt

# Quick test run (3 runs, 50 iterations)
python benchmark_comprehensive.py --quick

# Full benchmark suite (5 runs, 100 iterations, ~2-3 hours)
python benchmark_comprehensive.py --runs 5 --budget 100

# Full paper benchmarks (10 runs, 200 iterations, ~8-10 hours)
python benchmark_comprehensive.py --runs 10 --budget 200

# Skip expensive problems for faster testing
python benchmark_comprehensive.py --runs 5 --budget 100 --skip-expensive
```

## Problem Coverage

### Systematic Synthetic Problems

**3 Dimension Classes:**
- Small: 2, 3, 5D
- Medium: 10, 15, 20, 30D
- Large: 50, 75, 100D

**3 Cost Classes:**
- Low: <0.01s per evaluation
- Medium: 0.01-0.1s per evaluation
- High: >0.1s per evaluation

**4 Noise Classes:**
- None: Deterministic
- Low: 1% noise
- Medium: 10% noise  
- High: 50% noise

**5 Base Functions:**
- Sphere (unimodal)
- Rosenbrock (narrow valley)
- Rastrigin (highly multimodal)
- Ackley (deep basin multimodal)
- Griewank (product coupling)

**Total Synthetic: ~150 problems**

### Real-World Applications (with Data Batching)

**ML Hyperparameter Tuning:**
- LightGBM on 3 datasets (breast_cancer, digits, wine)
- Neural Network (MLPClassifier) on digits
- SVM on multiple datasets
- Random Forest on breast_cancer

**Portfolio Optimization:**
- 5, 10, 20, 50 asset portfolios
- Progressive time-period evaluation

**Total Real-World: ~15 problems**

**Grand Total: 165+ benchmark problems**

## Optimizers Compared

1. **RAGDA** (ours) - with and without mini-batch
2. **CMA-ES** - State-of-the-art evolution strategy
3. **Optuna (TPE)** - Popular BO framework
4. **Bayesian Optimization** - GP-based (if installed)
5. **Differential Evolution** - Scipy genetic algorithm
6. **Dual Annealing** - Scipy simulated annealing
7. **Hyperopt** - Tree-structured Parzen estimator (if installed)
8. **Random Search** - Baseline

## Output Files

### Data Files
- `benchmark_comprehensive.csv` - All results (one row per run)
- `benchmark_detailed.json` - With convergence histories
- `analysis_by_class.csv` - Aggregated by (dimension, cost, noise)
- `statistical_analysis.json` - Statistical test results

### LaTeX Tables
- `table_overall.tex` - Overall performance summary
- `table_by_dimension.tex` - Performance by dimensionality
- `table_by_cost.tex` - Performance by evaluation cost
- `table_by_noise.tex` - Performance by noise level
- `table_ragda_breakdown.tex` - RAGDA 3D breakdown
- `table_win_matrix.tex` - Best optimizer per class
- `table_batching_impact.tex` - Mini-batch feature impact

### Figures
- `heatmap_{dim}_dimension.pdf` - 3D heatmaps per dimension class
- `convergence_{class}.pdf` - Representative convergence curves
- `dimension_scaling.pdf` - Scaling analysis
- `batching_comparison.pdf` - Mini-batch speedup
- `category_comparison.pdf` - Performance by category

## Analysis Scripts
```bash
# Generate all paper materials
python generate_paper_results.py

# Create specific analyses
python -c "
from benchmark_comprehensive import *
df = pd.read_csv('./benchmark_results/benchmark_comprehensive.csv')
analysis_df = analyze_by_problem_class(df)
"
```

## Customization

### Add Custom Problems
```python
# In benchmark_comprehensive.py, add to generate_comprehensive_suite():

def my_custom_problem():
    def func(params):
        # Your objective
        return value
    
    space = [
        {'name': 'x', 'type': 'continuous', 'bounds': [0, 1]},
        # ...
    ]
    
    return ProblemSpec(
        name="MyProblem",
        category='custom',
        dimension_class='medium',
        cost_class='high',
        noise_class='low',
        dimension=10,
        func=func,
        bounds=bounds,
        space=space,
        optimum=None,
        description="My custom problem",
        supports_batching=True  # if applicable
    )

problems.append(my_custom_problem())
```

### Run Subset of Benchmarks
```python
from benchmark_comprehensive import *

# Generate all problems
all_problems = generate_comprehensive_suite()

# Filter to specific class
small_clean = [p for p in all_problems 
               if p.dimension_class == 'small' 
               and p.noise_class == 'none'
               and p.cost_class == 'low']

# Run subset
df, results = run_comprehensive_benchmark(small_clean, n_runs=5, budget=100)
```

## Expected Runtime

With default settings (5 runs, 100 iterations):

- **Small, low-cost problems**: ~0.5s per run
- **Medium problems**: ~1-2s per run
- **Large problems**: ~5-10s per run
- **Expensive problems**: ~10-60s per run
- **ML hyperparameter problems**: ~30-120s per run

**Total estimated time**: 2-4 hours for complete suite

With quick mode (3 runs, 50 iterations): ~30-60 minutes

## Tips for Paper Writing

### Citing Results

All statistics are in `summary_statistics.json`:
```python
import json
with open('./paper_results/summary_statistics.json') as f:
    stats = json.load(f)

print(f"RAGDA achieved {stats['ragda']['success_rate_pct']:.1f}% success rate")
```

### Reproducibility

All random seeds are deterministic:
- seed = 42 + problem_index * 1000 + run_id

Re-running with same parameters gives identical results.

### Statistical Significance

p-values from Wilcoxon tests are in `statistical_analysis.json`.

Convention:
- p < 0.001: ***
- p < 0.01: **
- p < 0.05: *

## Troubleshooting

**Out of memory?**
- Use `--skip-expensive`
- Reduce `--runs`
- Run subsets separately

**Taking too long?**
- Use `--quick` mode
- Reduce `--budget`
- Use fewer optimizers (comment out in code)

**Missing dependencies?**
```bash
pip install cma optuna hyperopt bayesian-optimization lightgbm scikit-learn matplotlib seaborn scipy
```