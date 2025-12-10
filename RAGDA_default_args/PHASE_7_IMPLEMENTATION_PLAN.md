# Phase 7: Meta-Optimizer Implementation Plan

## Overview

**Goal**: Optimize RAGDA's default parameters using MARsOpt across 234 benchmark problems, classified into 28 categories (27 specific + 1 general).

**Environment**: UV virtual environment (`.venv\Scripts\activate`)

**Core Principle**: 
- All benchmark problems use **Optuna-style API** for cross-optimizer compatibility
- All tests in single file (`tests/test_phase7_meta_optimizer.py`)
- Each step validated before proceeding to next
- Archive old implementation, build clean from scratch

---

## Step 0: Archive Current Implementation

**STATUS**: ✅ COMPLETE

### Actions

1. Create archive directory:
   ```powershell
   New-Item -Path "RAGDA_default_args\archive" -ItemType Directory -Force
   ```

2. Move all current files to archive:
   ```powershell
   Get-ChildItem -Path "RAGDA_default_args\*" -Exclude "archive","PHASE_7_IMPLEMENTATION_PLAN.md" | Move-Item -Destination "RAGDA_default_args\archive\"
   ```

3. Verify archive contents:
   - audit_problems.py
   - benchmark_comprehensive.py
   - benchmark_functions.py
   - benchmark_ml_problems.py
   - benchmark_realworld_problems.py
   - debug_marsopt.py
   - debug_problems.py
   - meta_optimizer.py
   - META_OPTIMIZER_DEBUG_SUMMARY.md
   - parameter_audit.py
   - problem_classifications_cache.json
   - problem_classifier.py
   - ragda_parameter_space.py
   - test_marsopt_debug.py
   - test_meta_simple.py

**Validation**: Confirm all files moved, only archive/ and this plan remain.

---

## Step 1: AUC Metric Implementation & Verification

**STATUS**: ✅ COMPLETE

**Purpose**: Verify/extract the AUC (Area Under Curve) metric used to evaluate optimizer convergence speed.

### 1.1 Extract AUC Implementation from Archive

**Actions**:

1. Search archive for AUC calculation:
   - Search `archive/meta_optimizer.py` for "auc", "area_under", "convergence"
   - Search `archive/benchmark_*.py` files
   - Identify the exact metric calculation

2. Document the AUC metric:
   - What is being measured? (loss vs iterations, loss vs evaluations, etc.)
   - How is normalization done? (0-1 range)
   - Lower is better (faster convergence)
   - Does it handle cases where true minimum is unknown?

### 1.2 Create AUC Utilities Module

**File**: `RAGDA_default_args/auc_metric.py`

**Content**:
```python
"""
AUC Metric for Optimizer Convergence Evaluation

Measures the area under the convergence curve (normalized 0-1).
Lower AUC = faster convergence = better optimizer performance.
"""

def calculate_auc(
    convergence_history: List[float],
    normalize: bool = True,
    best_known_value: Optional[float] = None
) -> float:
    """
    Calculate AUC from convergence history.
    
    Args:
        convergence_history: List of best-so-far objective values
        normalize: If True, normalize to [0, 1] range
        best_known_value: Known optimal value (if available)
    
    Returns:
        AUC value (0-1, lower is better)
    """
    pass  # Implementation extracted from archive

def evaluate_optimizer_on_problem(
    optimizer_factory: Callable,
    problem: Callable,
    n_evaluations: int,
    **optimizer_kwargs
) -> Tuple[float, List[float]]:
    """
    Run optimizer on problem and return AUC + convergence history.
    """
    pass
```

### 1.3 Tests for AUC Metric

**File**: `tests/test_phase7_meta_optimizer.py` (Section 1)

**Tests**:
```python
def test_auc_perfect_convergence():
    """Immediate convergence should give near-zero AUC."""
    pass

def test_auc_no_convergence():
    """No improvement should give AUC near 1.0."""
    pass

def test_auc_linear_convergence():
    """Linear improvement should give AUC around 0.5."""
    pass

def test_auc_normalization():
    """Test normalization with known bounds."""
    pass

def test_evaluate_optimizer_on_simple_problem():
    """Test wrapper that runs optimizer and computes AUC."""
    pass
```

**Validation**: All AUC metric tests pass before proceeding.

---

## Step 2: Benchmark Functions - Optuna API

**STATUS**: ✅ COMPLETE (230/234 = 98.3%)

**Implementation**:
- Mathematical: 78/78 ✅
- Real-world: 133/133 ✅ 
- ML: 19/19 ✅
- **Total: 230/234**

**Missing**: 4 problems (archive duplicates discovered during implementation)

**Tests**: 133/133 passing (58s)

---

### 2.1 Mathematical Benchmark Functions (78 total)

**File**: `RAGDA_default_args/benchmark_mathematical_problems.py`

**Source**: `archive/benchmark_functions.py`

**Chunking Strategy**:

#### Chunk 2.1.1: Core Structure + Unimodal (12 functions) ✅
- Create dataclass, helper functions, registry structure
- **Note**: `known_optimum` field is **Optional** - not used by AUC metric (which is scale-invariant). Kept as reference metadata for research/validation. Real-world problems use `None`.
- **Functions from archive**:
  - `sphere_2d, sphere_5d, sphere_10d, sphere_20d, sphere_50d, sphere_100d` (6)
  - `sum_squares_2d, sum_squares_5d, sum_squares_10d, sum_squares_20d, sum_squares_50d, sum_squares_100d` (6)
- **Test**: Import, count, test sphere_2d with mock trial

#### Chunk 2.1.2: Multimodal - Core Functions (30 functions)
- **Functions from archive** (6 base functions, varying dimensions):
  - `ackley_2d, ackley_5d, ackley_10d, ackley_20d, ackley_50d` (5 - archive verified)
  - `rastrigin_2d, rastrigin_5d, rastrigin_10d, rastrigin_20d, rastrigin_50d` (5)
  - `schwefel_2d, schwefel_5d, schwefel_10d, schwefel_20d, schwefel_50d` (5)
  - `griewank_2d, griewank_5d, griewank_10d, griewank_20d, griewank_50d` (5)
  - `levy_2d, levy_5d, levy_10d, levy_20d, levy_50d` (5)
  - `styblinski_tang_2d, styblinski_tang_5d, styblinski_tang_10d, styblinski_tang_20d, styblinski_tang_50d, styblinski_tang_100d` (6)
- **Test**: Count = 42 total, test ackley_10d, test rastrigin_5d

#### Chunk 2.1.3: Multimodal - Special 2D Functions (6 functions) ✅ DONE
- **Functions from archive** (2D only):
  - `beale_2d, branin_2d, goldstein_price_2d` (3)
  - `eggholder_2d, holder_table_2d` (2)
  - `hartmann_3d` (1, special 3D)
- **Expected Count After**: 50 total (44 + 6)
- **Tests Added**: test_special_2d_count, test_eggholder_2d_optuna, test_hartmann_3d_dimension
- **Status**: ✅ All 12 benchmark tests passing (50 functions total)

#### Chunk 2.1.4: Multimodal - Fixed Dimension (2 functions) ✅ DONE
- **Functions from archive**:
  - `shekel_4d` (4D only, bounds (0,10), optimum -10.5364)
  - `hartmann_6d` (6D only, bounds (0,1), optimum -3.32237)
- **Expected Count After**: 52 total (50 + 2)
- **Tests Added**: test_fixed_dim_count, test_shekel_4d_fixed, test_hartmann_6d_fixed
- **Status**: ✅ All 15 benchmark tests passing (52 functions: 12 unimodal + 40 multimodal)

#### Chunk 2.1.5: Valley Functions (18 functions) ✅ DONE
- **Functions from archive**:
  - `rosenbrock_2d, rosenbrock_5d, rosenbrock_10d, rosenbrock_20d, rosenbrock_50d, rosenbrock_100d` (6)
  - `dixon_price_2d, dixon_price_5d, dixon_price_10d, dixon_price_20d, dixon_price_50d, dixon_price_100d` (6)
  - `six_hump_camel_2d` (1)
  - `powell_4d, powell_8d, powell_12d, powell_20d, powell_40d` (5)
- **Expected Count After**: 70 total (52 + 18)
- **Tests Added**: test_valley_count, test_valley_category_count, test_rosenbrock_10d_valley, test_dixon_price_50d, test_six_hump_camel_2d, test_powell_20d
- **Status**: ✅ All 21 benchmark tests passing (70 functions: 12 unimodal + 40 multimodal + 18 valley)

#### Chunk 2.1.6: Plate Functions (7 functions) ✅ DONE
- **Functions from archive**:
  - `zakharov_2d, zakharov_5d, zakharov_10d, zakharov_20d, zakharov_50d, zakharov_100d` (6)
  - `booth_2d` (1)
- **Expected Count After**: 77 total (70 + 7)
- **Tests Added**: test_plate_count, test_plate_category_count, test_zakharov_20d_plate, test_booth_2d_plate
- **Status**: ✅ All 24 benchmark tests passing (77 functions: 12 unimodal + 39 multimodal + 19 valley + 7 plate)
- **Note**: colville_4d categorized as VALLEY in archive, not plate

#### Chunk 2.1.7: Steep Functions (1 function) ✅ DONE - COMPLETES ALL MATHEMATICAL FUNCTIONS
- **Functions from archive**:
  - `easom_2d` (1) - Only steep function in archive
- **Expected Count After**: 78 total (77 + 1)
- **Tests Added**: test_steep_count, test_steep_category_count, test_easom_2d_steep, test_mathematical_functions_complete
- **Status**: ✅ All 30 benchmark tests passing (78 functions: 12 unimodal + 39 multimodal + 19 valley + 7 plate + 1 steep)
- **Corrections Applied**:
  - Moved colville_4d from plate to valley (per archive)
  - Removed ackley_100d (archive only has ackley up to 50D)
  - **ARCHIVE VERIFIED**: All 78 mathematical functions now match archive exactly

**Structure**: See chunks above for exact implementation order.

---

### 2.2 Real-World Benchmark Problems (137 total)

**File**: `RAGDA_default_args/benchmark_realworld_problems.py`

**Source**: `archive/benchmark_realworld_problems.py`

**Chunking Strategy**:

#### Chunk 2.2.1: Core Structure + Chaotic Systems (16 functions) ✅ DONE
- Created dataclass, helper functions (Optuna wrapper for callable functions)
- **Functions from archive** (16 chaotic systems):
  - Low-dim: `LogisticMap-1D, Henon-2D, RabinovichFabrikant-2D, Lorenz-3D, Rossler-3D` (5)
  - Med-dim: `MackeyGlass-4D, DoublePendulum-4D, Duffing-5D, CoupledLogistic-10D` (4)
  - High-dim: `Lorenz96-20D, HenonExtended-20D, StandardMapChain-30D, Lorenz96Extended-60D, SpatiotemporalChaos-60D, CoupledMapLattice-64D, CoupledLogisticMaps-100D` (7)
- **Expected Count After**: 94 total (78 + 16)
- **Tests Added**: test_chaotic_count, test_chaotic_category_count, test_mackey_glass_4d, test_lorenz_3d, test_henon_2d, test_logistic_map_1d, test_high_dimensional_chaotic, test_all_chaotic_callable
- **Status**: ✅ All 50 tests passing (12 AUC + 30 mathematical + 8 chaotic)
- **Note**: Archive had 16 chaotic problems (not 13). File renamed: `benchmark_functions.py` → `benchmark_mathematical_problems.py`

#### Chunk 2.2.2: Dynamical Systems (6 functions) ✅ DONE
- **Functions from archive** (6 dynamical systems):
  - `LotkaVolterra-4D, LotkaVolterra4Species-8D, VanDerPol-1D` (3)
  - `CoupledOscillators-15D, KuramotoOscillators-20D` (2)
  - `NeuralField-70D` (1)
- **Expected Count After**: 100 total (78 + 16 + 6)
- **Tests Added**: test_dynamical_count, test_dynamical_category_count, test_lotka_volterra_4d, test_lotka_volterra_4species_8d, test_van_der_pol_1d, test_kuramoto_20d, test_neural_field_70d
- **Status**: ✅ All 58 tests passing (12 AUC + 30 mathematical + 8 chaotic + 7 dynamical + 1 count fix)
- **Note**: Archive had 6 dynamical problems (not 9). Some problems were duplicates or in other categories.

#### Chunk 2.2.3: Neural Network Weights (16 functions) ✅ DONE
- **Functions from archive** (16 NN weight optimization problems):
  - `NN-XOR-17D, NN-Regression-89D, NN-MNIST-1074D, NN-Large-1377D` (4)
  - `NN-Medium-20D, NN-Deep-100D` (2)
  - `Autoencoder-80D, RBM-60D, WordEmbedding-75D, Hopfield-64D` (4)
  - `SparseAutoencoder-100D, VAE-90D, DenoisingAE-80D, ContrastiveLearning-70D` (4)
  - `NeuralHessian-80D, NeuralHessian-100D` (2)
- **Expected Count After**: 116 total (78 + 16 + 6 + 16)
- **Tests Added**: test_nn_weights_count, test_nn_weights_category_count, test_nn_xor_17d, test_nn_regression_89d, test_nn_mnist_1074d, test_nn_large_1377d, test_sparse_autoencoder_100d, test_neural_hessian_80d, test_all_nn_weights_callable
- **Status**: ✅ All 66 tests passing (12 AUC + 30 mathematical + 8 chaotic + 7 dynamical + 9 NN weights)
- **Fixes Applied**:
  - Fixed MockTrial parsing: `name[1:]` instead of `name.split('_')[1]`
  - Fixed NN-Regression-89D: 1→8→8→1 without b2 (89 params exactly)
  - Fixed NN-Large-1377D: 10→32→31→1 with 2-element b2 (1377 params exactly)
  - Fixed SparseAutoencoder-100D: 10→5→9 without decoder bias (100 params exactly)
- **Progress**: **49.6% complete (116/234 functions)** - PASSED HALFWAY MARK! 🎉

#### Chunk 2.2.4: ML Training Problems Part 1 (15 functions) ✅
- **Functions implemented**:
  - `SVM-CV-2D, RF-CV-4D, Ridge-CV-1D, Lasso-CV-1D, ElasticNet-CV-2D` (5)
  - `LogisticReg-CV-1D, KNN-CV-3D, DecisionTree-CV-3D, AdaBoost-CV-2D` (4)
  - `SVM-Large-CV-2D, Bagging-CV-3D, GradientBoost-CV-3D, MLP-Regressor-CV-3D` (4)
  - `NestedCV-5D, BayesianAcquisition-6D` (2)
- **Tests Added**: test_ml_training_count, test_ml_training_category_count, test_svm_cv_2d, test_ridge_cv_1d, test_elasticnet_cv_2d, test_rf_cv_4d, test_gradientboost_cv_3d, test_nested_cv_5d, test_bayesian_acquisition_6d, test_all_ml_training_callable
- **Status**: ✅ All 76 tests passing (12 AUC + 30 mathematical + 8 chaotic + 7 dynamical + 9 NN weights + 10 ML training)
- **Category**: ml_training (new category)
- **Progress**: **56.0% complete (131/234 functions)** 🎯

#### Chunk 2.2.5: ML Training Problems Part 2 (18 functions) ✅
- **Functions implemented**:
  - Ensemble methods: `NeuralNet-Dropout-20D, LightGBM-CV-4D, XGBoost-CV-4D, CatBoost-CV-4D, ExtraTrees-CV-4D` (5)
  - Advanced ensembles: `StackingEnsemble-15D, VotingEnsemble-10D` (2)
  - Feature selection: `FeatureSelection-RFE-20D, FeatureSelection-MI-15D` (2)
  - Feature engineering: `FeatureEngineering-Poly-12D, FeatureScaling-Robust-8D, ClassWeights-Imbalanced-10D` (3)
  - Dimensionality reduction: `PCA-Components-1D, SVD-Components-1D, TSNE-Hyperparams-3D, UMAP-Hyperparams-4D` (4)
  - Anomaly & reconstruction: `IsolationForest-CV-3D, AutoEncoder-Hyperparams-5D` (2)
- **Tests Added**: test_ml_training_count_updated, test_ml_training_category_count_updated, test_neuralnet_dropout_20d, test_lightgbm_cv_4d, test_stacking_ensemble_15d, test_feature_selection_rfe_20d, test_pca_components_1d, test_tsne_hyperparams_3d, test_isolation_forest_cv_3d, test_autoencoder_hyperparams_5d (10 tests)
- **Status**: ✅ All 86 tests passing (20 ML training tests)
- **Category**: ml_training (expanded from 15→33 problems)
- **Key fixes**: Fixed TSNE max_iter parameter (was n_iter in old sklearn), updated old test assertions
- **Progress**: **63.7% complete (149/234 functions)** 🎯 Approaching 2/3 milestone!

#### Chunk 2.2.6: ML Training Problems Part 3 (10 functions) ✅ COMPLETE

**STATUS**: ✅ DONE (159/234 = 67.9% complete)

**Problems Implemented**:
1. `SparseCoding-30D` - Dictionary learning with sparse representation
2. `NMF-Factorization-25D` - Non-negative matrix factorization with regularization
3. `KernelApproximation-20D` - RBF kernel approximation with SGD
4. `CalibrationTuning-15D` - Probability calibration (sigmoid vs isotonic)
5. `MultiOutputRegression-40D` - Chained regressors for multi-target problems
6. `QuantileRegression-12D` - Gradient boosting with quantile loss
7. `SemiSupervised-30D` - Label spreading with unlabeled data
8. `OrdinalRegression-18D` - OvR logistic regression for ordered categories
9. `CostSensitiveLearning-22D` - Sample weight tuning for imbalanced costs
10. `TransferLearning-35D` - Pre-training on source, fine-tuning on target

**Registry**: Added 10 entries to `_ML_TRAINING_REGISTRY` (total: 43 ml_training problems)

**Tests**: Added 7 new tests + updated 8 count tests = 27 ML tests passing

**Bugs Fixed**:
- NMF: Changed `alpha` parameter to `alpha_W` and `alpha_H` (sklearn API)
- CostSensitive: Replaced deprecated `fit_params` with manual KFold loop

**Validation**: All 27 ML training tests passing ✅



#### Chunk 2.2.7: PDE Problems (18 functions) ✅ COMPLETE

**STATUS**: ✅ DONE (177/234 = 75.6% complete) 🎯 **Passed 3/4 milestone!**

**Problems Implemented**:
1. `Burgers-9D` - 1D Burgers equation (nonlinear PDE, viscosity + Fourier coefficients)
2. `PDE-HeatEq-50D` - 1D heat equation with Fourier mode initial conditions
3. `HeatDiffusion-30D` - Heat diffusion equation optimization
4. `WaveEquation-30D` - Wave equation initial condition optimization
5. `AdvectionDiffusion-30D` - Advection-diffusion equation (velocity + diffusion + IC)
6. `ReactionDiffusion-30D` - Reaction-diffusion pattern formation (Turing patterns)
7. `Heat2D-60D` - 2D heat equation on grid
8. `Poisson-60D` - Poisson equation source term optimization
9. `Laplace-64D` - Laplace equation boundary optimization
10. `Helmholtz-60D` - Helmholtz equation parameter optimization
11. `Biharmonic-56D` - Biharmonic equation / plate bending problem
12. `SpectralMethod-64D` - Spectral method coefficients optimization
13. `FiniteElement-70D` - Finite element node values optimization
14. `Multigrid-60D` - Multigrid solver parameter optimization
15. `DomainDecomposition-72D` - Domain decomposition interface optimization
16. `AdaptiveMesh-65D` - Adaptive mesh refinement optimization
17. `GinzburgLandau-56D` - Complex Ginzburg-Landau equation (chaotic PDE)
18. `WaveEquation-120D` - 1D wave equation boundary reflection minimization

**Registry**: Added 18 entries to `_PDE_REGISTRY` (total: 99 real-world problems)

**Tests**: Added 11 new tests (9 specific problems + 2 count tests) = 104 total tests passing

**Bug Fixed**: Helmholtz equation had incorrect array slicing (x[1:] has 59 elements, not 60)

**Validation**: All 104 tests passing ✅

#### Chunk 2.2.8: Meta-Optimization & Control (18 functions) ✅ COMPLETE
- **Status**: ✅ Implemented and tested (112 tests passing)
- **Progress**: 195/234 (83.3%) - **Passed 5/6 milestone!** 🎯
- **Functions**:
  - **Meta-Optimization (10)**:
    - `GeneticAlgorithm-25D`: GA hyperparameter tuning on Rastrigin
    - `ParticleSwarm-30D`: PSO parameter tuning (inertia, cognitive, social)
    - `DifferentialEvolution-30D`: DE strategy tuning (F, CR, population)
    - `CMA-ES-25D`: CMA-ES hyperparameter optimization
    - `Hyperband-60D`: Multi-fidelity successive halving
    - `BayesianOpt-60D`: Gaussian process surrogate tuning
    - `NAS-70D`: Neural architecture search (layers, activations, connections)
    - `EvolutionStrategy-65D`: Natural evolution strategies
    - `SimulatedAnnealing-55D`: SA temperature schedule and cooling rate
    - `CovarianceAdaptation-60D`: Explicit covariance matrix adaptation
  - **Control Theory (8)**:
    - `PIDTuning-6D`: PID controller gains (Kp, Ki, Kd in [-10, 10])
    - `LQRControl-8D`: Linear-quadratic regulator synthesis
    - `TrajectoryOpt-100D`: Optimal control sequence (100 timesteps)
    - `TrajectoryOpt-120D`: Extended trajectory optimization (120 timesteps)
    - `InverseKinematics-80D`: Robot arm joint angles (80 joints)
    - `InverseKinematics-100D`: Extended IK (100 joints)
    - `InverseKinematicsLong-80D`: Long-horizon IK with smoothness (80 joints)
    - `InverseKinematicsLong-100D`: Long-horizon IK with smoothness (100 joints)
- **Registry**: Added 18 entries to `_META_CONTROL_REGISTRY` (10 meta_optimization + 8 control)
- **Tests**: Added 10 new tests (8 specific problems + 2 category tests) = 112 total tests passing
- **Bug Fixed**: PID tuning used wrong bounds - corrected to map [0,1] → [-10,10] per original spec
- **Validation**: All 112 tests passing ✅

#### Chunk 2.2.9: Remaining Problems (16 unique, 18 with variants) ✅ COMPLETE
- **Status**: ✅ DONE (133/234 = 56.8% complete) - **ALL REAL-WORLD PROBLEMS IMPLEMENTED**
- **Problems Implemented**:
  - Meta-optimization: `SA-Schedule-3D` (1)
  - Simulation: `CellularAutomata-25D, CellularAutomaton-120D, EpidemicControl-25D/40D` (4)
  - Physics: `SpinGlass-150D` (1)
  - Statistics: `CovarianceEstimation-120D, RegressionCoeffs-5D` (2)
  - Control: `LinearSystemID-144D` (1)
  - Dynamical: `CoupledPendulums-100D` (1)
  - Optimization: `SupplyChain-35D/50D, GraphPartition-25D/40D` (4)
  - Finance: `RiskParity-30D` (1)
  - Chemistry: `ChemicalKinetics-5D` (1)
- **Duplicates Removed**: `CoupledLogisticMaps-100D, StandardMapChain-30D` (already in _CHAOTIC_REGISTRY)
- **Registry**: Added 16 entries to `_REMAINING_REGISTRY` (total: 133 real-world problems)
- **Tests**: Added 10 new tests + updated 4 count assertions = 124 tests passing ✅
- **Categories Added**: simulation, physics, statistics, optimization, finance, chemistry
- **Validation**: All 124 tests passing (12 AUC + 30 mathematical + 82 real-world) ✅
- **Note**: Original plan expected 215/234 but actual is 133/234 due to duplicates discovered in earlier chunks

---

### 2.3 ML Benchmark Problems (19 total)

**File**: `RAGDA_default_args/benchmark_ml_problems.py`

**Source**: `archive/benchmark_ml_problems.py`

**Chunking Strategy**:

#### Chunk 2.3.1: All ML Problems (19 functions in one batch)
- Small enough to do in single batch
- **Functions from archive**: All 19 from `get_all_ml_problems()`
- **Test**: Count = 19 total ✅

---

### 2.4 Master Problem Registry

**File**: `RAGDA_default_args/problem_registry.py`

**Purpose**: Single point to access all 234 problems.

### 2.4 Master Problem Registry

**File**: `RAGDA_default_args/problem_registry.py`

**Purpose**: Single point to access all 234 problems.

**Content**:
```python
"""
Master Registry of All Benchmark Problems

Aggregates all benchmark problems from:
- benchmark_functions.py (78)
- benchmark_realworld_problems.py (137)
- benchmark_ml_problems.py (19)
Total: 234 problems
"""

from typing import Dict, List
from benchmark_functions import ALL_BENCHMARK_FUNCTIONS
from benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
from benchmark_ml_problems import ALL_ML_PROBLEMS

# Combine all registries
ALL_PROBLEMS: Dict[str, BenchmarkProblem] = {
    **ALL_BENCHMARK_FUNCTIONS,
    **ALL_REALWORLD_PROBLEMS,
    **ALL_ML_PROBLEMS,
}

def get_problem(name: str) -> BenchmarkProblem:
    """Get any problem by name."""
    if name not in ALL_PROBLEMS:
        raise KeyError(f"Unknown problem: {name}")
    return ALL_PROBLEMS[name]

def list_all_problems() -> List[str]:
    """List all 234 problem names."""
    return sorted(ALL_PROBLEMS.keys())

def get_problems_by_category(category: str) -> List[BenchmarkProblem]:
    """Get all problems in a category."""
    return [p for p in ALL_PROBLEMS.values() if p.category == category]
```

---

### 2.5 Tests for Benchmark Problems

**File**: `tests/test_phase7_meta_optimizer.py` (Section 2)

**Test Strategy**: Add tests incrementally as chunks are completed.

**Tests**:
```python
class TestBenchmarkFunctions:
    def test_problem_count(self):
        """Verify we have 234 total problems."""
        from RAGDA_default_args.problem_registry import ALL_PROBLEMS
        assert len(ALL_PROBLEMS) == 234
    
    def test_benchmark_functions_count(self):
        """Verify we have 78 mathematical functions."""
        from RAGDA_default_args.benchmark_functions import ALL_BENCHMARK_FUNCTIONS
        assert len(ALL_BENCHMARK_FUNCTIONS) == 78
    
    def test_realworld_problems_count(self):
        """Verify we have 137 real-world problems."""
        from RAGDA_default_args.benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
        assert len(ALL_REALWORLD_PROBLEMS) == 137
    
    def test_ml_problems_count(self):
        """Verify we have 19 ML problems."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        assert len(ALL_ML_PROBLEMS) == 19
    
    def test_sphere_2d_optuna(self):
        """Test simple sphere function works with Optuna trial mock."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        # Mock trial object
        class MockTrial:
            def __init__(self):
                self.params = {}
            def suggest_float(self, name, low, high):
                self.params[name] = 0.0  # Always suggest 0
                return 0.0
        
        problem = get_benchmark_function('sphere_2d')
        trial = MockTrial()
        result = problem.objective(trial)
        
        assert abs(result - 0.0) < 1e-10  # Should be 0 at origin
        assert problem.dimension == 2
        assert problem.known_optimum == 0.0
    
    def test_ackley_10d_optuna(self):
        """Test multimodal function (after Chunk 2.1.2)."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        class MockTrial:
            def suggest_float(self, name, low, high):
                return 0.0  # At optimum
        
        problem = get_benchmark_function('ackley_10d')
        result = problem.objective(MockTrial())
        
        assert abs(result - 0.0) < 1e-6  # Near 0 at optimum
        assert problem.dimension == 10
        assert problem.category == 'multimodal'
    
    def test_rosenbrock_10d_optuna(self):
        """Test valley function (after Chunk 2.1.5)."""
        from RAGDA_default_args.benchmark_functions import get_benchmark_function
        
        problem = get_benchmark_function('rosenbrock_10d')
        assert problem.dimension == 10
        assert problem.category == 'valley'
    
    def test_all_functions_callable(self):
        """Verify all registered functions are callable."""
        from RAGDA_default_args.problem_registry import ALL_PROBLEMS
        
        for name, problem in ALL_PROBLEMS.items():
            assert callable(problem.objective), f"{name} objective not callable"
    
    def test_problem_metadata_complete(self):
        """Verify all problems have required metadata."""
        from RAGDA_default_args.problem_registry import ALL_PROBLEMS
        
        for name, problem in ALL_PROBLEMS.items():
            assert problem.name == name
            assert problem.dimension > 0
            assert len(problem.bounds) == problem.dimension
            assert problem.category is not None
            assert problem.description
    
    def test_lorenz_3d_problem(self):
        """Test one chaotic system problem (after Chunk 2.2.1)."""
        from RAGDA_default_args.benchmark_realworld_problems import get_problem
        
        problem = get_problem('Lorenz-3D')
        assert problem.dimension == 3
        assert problem.category == 'chaotic'
    
    def test_ml_problem(self):
        """Test one ML hyperparameter problem (after Chunk 2.3.1)."""
        from RAGDA_default_args.benchmark_ml_problems import ALL_ML_PROBLEMS
        
        assert len(ALL_ML_PROBLEMS) == 19
        # Pick any problem
        problem = list(ALL_ML_PROBLEMS.values())[0]
        assert callable(problem.objective)
```

---

### Step 2 Completion Summary ✅

**Implemented**: 230/234 (98.3%) - COMPLETE
- Mathematical: 78 ✅
- Real-world: 133 ✅ 
- ML: 19 ✅

**Tests**: 133/133 passing (58s)

**Files**:
- `benchmark_mathematical_problems.py`
- `benchmark_realworld_problems.py`
- `benchmark_ml_problems.py`

**Categories**: 15 total (unimodal, multimodal, valley, plate, steep, chaotic, dynamical, nn_weights, ml_training, pde, meta_optimization, control, simulation, physics, statistics, optimization, finance, chemistry, ml_tuning)

**Ready for Step 3**: ✅

---

## Step 3: Problem Classification System

**STATUS**: ✅ IMPLEMENTED (Classification reveals gaps - see Step 4)

**Purpose**: Classify all 230 problems into 27 categories (3×3×3 dimensions).

**COMPLETED WORK**:
1. ✅ Created `problem_classifier.py` with full classification system
2. ✅ Created `classify_all_problems.py` with caching and resume support
3. ✅ Classified all 230 existing problems
4. ✅ Generated `problem_classifications_cache.json` with measurements
5. ✅ Identified distribution gaps across 27 categories

**CLASSIFICATION RESULTS**:
- **230 problems classified** into 27 categories (3×3×3):
  - **Dimensionality**: low (1-10D), medium (11-50D), high (51D+)
  - **Cost**: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
  - **Ruggedness**: smooth, moderate, rugged (based on local sensitivity)

**CURRENT DISTRIBUTION**:
```
Categories with ≥5 problems: 8/27
  ✅ high_cheap_moderate      : 40 problems
  ✅ low_cheap_moderate       : 50 problems
  ✅ low_cheap_rugged         :  7 problems
  ✅ low_cheap_smooth         : 48 problems
  ✅ medium_cheap_moderate    : 30 problems
  ✅ medium_cheap_smooth      : 27 problems
  ✅ high_cheap_smooth        :  6 problems
  ✅ medium_moderate_moderate :  6 problems

Categories with <5 problems: 19/27 (NEED 79 MORE PROBLEMS)
  ❌ high_cheap_rugged        :  4 (need 1 more)
  ❌ high_moderate_moderate   :  2 (need 3 more)
  ❌ high_moderate_smooth     :  2 (need 3 more)
  ❌ low_expensive_moderate   :  1 (need 4 more)
  ❌ low_moderate_moderate    :  3 (need 2 more)
  ❌ medium_cheap_rugged      :  1 (need 4 more)
  ❌ medium_expensive_smooth  :  2 (need 3 more)
  ❌ medium_moderate_rugged   :  1 (need 4 more)
  
  ZERO problems in 11 categories:
  ❌ high_expensive_moderate   : 0 (need 5)
  ❌ high_expensive_rugged     : 0 (need 5)
  ❌ high_expensive_smooth     : 0 (need 5)
  ❌ high_moderate_rugged      : 0 (need 5)
  ❌ low_expensive_rugged      : 0 (need 5)
  ❌ low_expensive_smooth      : 0 (need 5)
  ❌ low_moderate_rugged       : 0 (need 5)
  ❌ low_moderate_smooth       : 0 (need 5)
  ❌ medium_expensive_moderate : 0 (need 5)
  ❌ medium_expensive_rugged   : 0 (need 5)
  ❌ medium_moderate_smooth    : 0 (need 5)
```

**KEY INSIGHTS**:
- Most problems are **CHEAP** (fast execution <10ms)
- Very few **EXPENSIVE** problems (only 3 total, need 15 per expense level)
- Very few **MODERATE** cost problems (need more 10-100ms range)
- High-dimensional problems are mostly cheap (need expensive/moderate variants)

**REQUIREMENT**: Each of 27 categories needs **≥5 problems** → Need **135 minimum total** → Currently have **230** but poorly distributed → **Need 79 more problems** strategically placed

### 3.1 Problem Classifier Module

**File**: `RAGDA_default_args/problem_classifier.py`

**Structure**:
```python
"""
Problem Classifier for RAGDA Benchmarking

Classifies problems by THREE dimensions (3 levels each = 27 total categories):
1. DIMENSIONALITY: low (1-10D), medium (11-50D), high (51D+)
2. COST: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
3. RUGGEDNESS: smooth, moderate, rugged

All functions are DETERMINISTIC. "Ruggedness" = landscape complexity, not noise.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Tuple, Dict
import time
import numpy as np

class DimensionLevel(Enum):
    LOW = 'low'       # 1-10D
    MEDIUM = 'medium' # 11-50D
    HIGH = 'high'     # 51D+

class CostLevel(Enum):
    CHEAP = 'cheap'         # <10ms
    MODERATE = 'moderate'   # 10-100ms
    EXPENSIVE = 'expensive' # 100ms+

class RuggednessLevel(Enum):
    SMOOTH = 'smooth'       # Low sensitivity, consistent gradients
    MODERATE = 'moderate'   # Some local structure
    RUGGED = 'rugged'       # Many local features / chaotic

@dataclass
class ProblemClassification:
    """Classification result for a problem."""
    problem_name: str
    dimension_level: DimensionLevel
    cost_level: CostLevel
    ruggedness_level: RuggednessLevel
    category_key: str  # e.g., "low_cheap_smooth"
    
    # Measurements
    actual_dimension: int
    avg_eval_time_ms: float
    ruggedness_score: float

def classify_dimension(dim: int) -> DimensionLevel:
    """Classify dimensionality (trivial)."""
    if dim <= 10:
        return DimensionLevel.LOW
    elif dim <= 50:
        return DimensionLevel.MEDIUM
    else:
        return DimensionLevel.HIGH

def measure_cost(problem: Callable, n_samples: int = 10) -> Tuple[float, CostLevel]:
    """
    Measure evaluation time by running problem multiple times.
    Returns (avg_time_ms, cost_level).
    """
    pass

def measure_ruggedness(problem: Callable, n_samples: int = 50) -> Tuple[float, RuggednessLevel]:
    """
    Measure landscape ruggedness via:
    - Local sensitivity: output change for small input perturbations
    - Gradient variability: how much improvement direction changes
    
    Returns (ruggedness_score, ruggedness_level).
    """
    pass

def classify_problem(problem: BenchmarkProblem) -> ProblemClassification:
    """
    Fully classify a problem into one of 27 categories.
    """
    pass

def get_category_key(
    dim_level: DimensionLevel,
    cost_level: CostLevel,
    ruggedness_level: RuggednessLevel
) -> str:
    """Generate category key like 'low_cheap_smooth'."""
    return f"{dim_level.value}_{cost_level.value}_{ruggedness_level.value}"
```

### 3.2 Classification Script

**File**: `RAGDA_default_args/classify_all_problems.py`

**Purpose**: Classify all 234 problems and save to JSON cache.

```python
"""
Classify All Benchmark Problems

Runs classification on all 234 problems and saves to cache file.
Supports resume (skips already-classified problems).
"""

import json
from pathlib import Path
from tqdm import tqdm
from problem_registry import ALL_PROBLEMS
from problem_classifier import classify_problem

CACHE_FILE = Path(__file__).parent / "problem_classifications.json"

def load_cache() -> Dict[str, Dict]:
    """Load existing classifications."""
    pass

def save_cache(classifications: Dict[str, Dict]):
    """Save classifications to cache."""
    pass

def classify_all_problems(resume: bool = True):
    """
    Classify all problems.
    
    Args:
        resume: If True, skip already-classified problems
    """
    cache = load_cache() if resume else {}
    
    for name, problem in tqdm(ALL_PROBLEMS.items(), desc="Classifying"):
        if resume and name in cache:
            continue
        
        try:
            classification = classify_problem(problem)
            cache[name] = {
                'dimension_level': classification.dimension_level.value,
                'cost_level': classification.cost_level.value,
                'ruggedness_level': classification.ruggedness_level.value,
                'category_key': classification.category_key,
                'actual_dimension': classification.actual_dimension,
                'avg_eval_time_ms': classification.avg_eval_time_ms,
                'ruggedness_score': classification.ruggedness_score,
            }
            
            # Save after each problem (crash-resistant)
            save_cache(cache)
            
        except Exception as e:
            print(f"Error classifying {name}: {e}")
            continue
    
    return cache

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore cache)')
    args = parser.parse_args()
    
    classifications = classify_all_problems(resume=not args.fresh)
    print(f"\nClassified {len(classifications)} problems")
    
    # Print category distribution
    category_counts = {}
    for data in classifications.values():
        cat = data['category_key']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory Distribution:")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]} problems")
```

### 3.3 Tests for Classification

**File**: `tests/test_phase7_meta_optimizer.py` (Section 3)

**Tests**:
```python
class TestProblemClassification:
    def test_dimension_classification(self):
        """Test dimension level assignment."""
        assert classify_dimension(5) == DimensionLevel.LOW
        assert classify_dimension(25) == DimensionLevel.MEDIUM
        assert classify_dimension(100) == DimensionLevel.HIGH
    
    def test_cost_measurement(self):
        """Test cost measurement on simple function."""
        pass
    
    def test_ruggedness_measurement(self):
        """Test ruggedness on known smooth vs rugged functions."""
        pass
    
    def test_classify_sphere_2d(self):
        """Sphere should be: low, cheap, smooth."""
        pass
    
    def test_classify_ackley_10d(self):
        """Ackley 10D should be: low/medium, cheap/moderate, rugged."""
        pass
    
    def test_category_key_generation(self):
        """Test category key string generation."""
        pass
    
    def test_all_27_categories_possible(self):
        """Verify we can generate all 27 category keys."""
        assert len(set([
            get_category_key(d, c, r)
            for d in DimensionLevel
            for c in CostLevel
            for r in RuggednessLevel
        ])) == 27
```

**Validation**: All classification tests pass (7 tests).

**Manual Validation**: Run `python classify_all_problems.py` and verify:
- All 234 problems classified
- All 27 categories have at least 1 problem
- Distribution looks reasonable (not all in one category)

### 3.4 Fill Category Gaps (Add 79 Strategic Problems)

**STATUS**: ⏳ NEXT - START HERE

**Purpose**: Add 79 real benchmark problems to ensure all 27 categories have ≥5 problems.

**CURRENT STATE** (After Step 3.1-3.3 classification):
- ✅ 230 problems implemented (Step 2)
- ✅ Classification system implemented (Step 3.1-3.3)
- ✅ All 230 problems classified into 3x3x3 categories
- ❌ Only 8/27 categories have ≥5 problems
- ❌ 19/27 categories under threshold (need 79 more problems)

**STRATEGY**: Focus on **REAL** problems, not artificial dimension inflation:
1. **ML Hyperparameter Tuning** → Naturally EXPENSIVE/MODERATE cost (cross-validation)
2. **High-Dimensional ML** → Real high-feature datasets (gene expression, text, images)
3. **Model Selection** → Control cost via model choice (RandomForest=expensive, LightGBM=cheap)
4. **Expensive Mathematical Functions** → Complex PDEs, integrals, differential equations
5. **Moderate Cost Problems** → Medium-complexity simulations, dynamical systems

**DESIGN PRINCIPLES**:
- ✅ Use REAL benchmark problems from literature/datasets
- ✅ ML tuning is naturally expensive (CV loops)
- ✅ High dimensions from real data (not artificial padding)
- ✅ Model selection controls cost (RF vs LightGBM vs XGBoost)
- ❌ NO artificial dimension inflation (e.g., Sphere-1000D)
- ❌ NO fake problems just to fill categories

**IMPLEMENTATION CHUNKS**: Add 79 problems across 5 chunks

#### Chunk 3.4.1: Expensive High-Dim ML (15 problems) ⏳ NEXT
Target: `high_expensive_*` categories (0→5 each)
Files: `benchmark_ml_problems.py`

Problems (RandomForest/SVM/DeepNN on real datasets):
- 5 for high_expensive_smooth (GradientBoosting on large datasets)
- 5 for high_expensive_moderate (RandomForest tuning)
- 5 for high_expensive_rugged (Deep NN tuning)

#### Chunk 3.4.2: Expensive Low/Med ML (20 problems)
Target: `*_expensive_*` for low/medium dimensions
Files: `benchmark_ml_problems.py`

#### Chunk 3.4.3: Moderate High-Dim (11 problems)  
Target: `high_moderate_*` categories
Files: `benchmark_realworld_problems.py`, `benchmark_ml_problems.py`

#### Chunk 3.4.4: Moderate Low/Med (29 problems)
Target: Remaining `*_moderate_*` categories
Files: Multiple files

#### Chunk 3.4.5: Cheap Rugged (5 problems)
Target: Fill `*_cheap_rugged` gaps
Files: `benchmark_mathematical_problems.py`

**VALIDATION AFTER EACH CHUNK**:
```bash
python classify_all_problems.py --no-resume
python check_categories.py
```

**SUCCESS CRITERIA**:
- ✅ All 27 categories have ≥5 problems
- ✅ Total: 309 problems (230 + 79)
- ✅ All problems are REAL (no artificial inflation)
- ✅ All tests passing

---

## Step 4: RAGDA Parameter Space Definition

**STATUS**: ⏳ PENDING (After Step 3.4 complete)

**Purpose**: Define all 34 tunable RAGDA parameters for meta-optimization.

### 4.1 Parameter Space Module

**File**: `RAGDA_default_args/ragda_parameter_space.py`

**Content**: Extract from archive, verify compatibility with RAGDA v2.0 API.

**Structure**:
```python
"""
RAGDA Parameter Space for Meta-Optimization

Defines all 34 tunable parameters:
- 7 __init__ parameters (optimizer-level)
- 27 optimize() parameters (run-level)

Includes:
- Parameter bounds, types, defaults
- Constraint definitions
- Penalty functions for invalid configs
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

@dataclass
class ParameterDef:
    """Definition of a single RAGDA parameter."""
    name: str
    param_type: str  # 'int', 'float', 'bool', 'categorical'
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    choices: Optional[List[Any]] = None
    description: str = ""
    location: str = "optimize"  # 'init' or 'optimize'
    constraint_notes: str = ""

# All 34 parameters defined
RAGDA_PARAMETERS: Dict[str, ParameterDef] = {
    # __init__ parameters (7)
    'n_workers': ParameterDef(...),
    'random_state': ParameterDef(...),
    # ... etc
    
    # optimize() parameters (27)
    'maxiter': ParameterDef(...),
    'lambda_start': ParameterDef(...),
    'lambda_end': ParameterDef(...),
    # ... etc
}

def check_constraints(params: Dict[str, Any]) -> List[str]:
    """
    Check if parameter configuration violates constraints.
    Returns list of constraint violations (empty if valid).
    
    Constraints:
    - lambda_end <= lambda_start
    - min_workers <= n_workers
    - sigma_min < sigma_init
    - etc.
    """
    pass

def compute_constraint_penalty(params: Dict[str, Any]) -> float:
    """
    Compute penalty for constraint violations.
    Returns 0.0 if valid, large penalty (e.g., 1000.0 * num_violations) otherwise.
    """
    violations = check_constraints(params)
    return 1000.0 * len(violations)

def get_default_params() -> Dict[str, Any]:
    """Get RAGDA's current default parameters."""
    pass

def split_params_by_location(params: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Split parameters into init_params and optimize_params.
    Returns (init_params, optimize_params).
    """
    pass
```

### 4.2 Tests for Parameter Space

**File**: `tests/test_phase7_meta_optimizer.py` (Section 4)

**Tests**:
```python
class TestRAGDAParameterSpace:
    def test_parameter_count(self):
        """Verify we have 34 parameters defined."""
        assert len(RAGDA_PARAMETERS) == 34
    
    def test_parameter_locations(self):
        """Verify 7 init params, 27 optimize params."""
        init_params = [p for p in RAGDA_PARAMETERS.values() if p.location == 'init']
        opt_params = [p for p in RAGDA_PARAMETERS.values() if p.location == 'optimize']
        assert len(init_params) == 7
        assert len(opt_params) == 27
    
    def test_valid_config_no_penalty(self):
        """Valid config should have zero penalty."""
        valid_params = get_default_params()
        assert compute_constraint_penalty(valid_params) == 0.0
    
    def test_lambda_constraint_violation(self):
        """lambda_end > lambda_start should be penalized."""
        params = get_default_params()
        params['lambda_start'] = 10
        params['lambda_end'] = 20
        assert compute_constraint_penalty(params) > 0
    
    def test_split_params_by_location(self):
        """Test splitting params into init vs optimize."""
        pass
    
    def test_all_params_have_bounds(self):
        """Verify all numeric params have bounds."""
        pass
```

**Validation**: All parameter space tests pass (6 tests).

---

## Step 5: Meta-Optimizer Core Implementation

**STATUS**: ⏳ PENDING

**Purpose**: Implement the meta-optimizer that uses MARsOpt to optimize RAGDA parameters.

### 5.1 Meta-Optimizer Module

**File**: `RAGDA_default_args/meta_optimizer.py`

**Structure**:
```python
"""
RAGDA Meta-Optimizer

Uses MARsOpt to find optimal RAGDA parameters for each of 28 categories:
- 27 specific categories (3×3×3: dimension × cost × ruggedness)
- 1 general category (all problems)

For each category:
1. Select problems in that category
2. Create MARsOpt study
3. Optimize RAGDA params to minimize average AUC across problems
4. Save best parameters

Output: ragda_optimal_defaults.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time

# MARsOpt (external optimizer)
try:
    from marsopt import create_study
except ImportError:
    raise ImportError("MARsOpt required: uv pip install marsopt")

# RAGDA
from ragda import create_study as ragda_create_study

# Internal modules
from problem_registry import ALL_PROBLEMS, get_problem
from problem_classifier import ProblemClassification
from ragda_parameter_space import (
    RAGDA_PARAMETERS,
    compute_constraint_penalty,
    split_params_by_location,
    get_default_params,
)
from auc_metric import calculate_auc

@dataclass
class CategoryOptimizationResult:
    """Result of optimizing RAGDA params for one category."""
    category_key: str
    best_params: Dict[str, Any]
    best_auc: float
    n_problems: int
    n_trials: int
    optimization_time_seconds: float

def load_problem_classifications() -> Dict[str, Dict]:
    """Load problem classifications from cache."""
    cache_file = Path(__file__).parent / "problem_classifications.json"
    with open(cache_file, 'r') as f:
        return json.load(f)

def get_problems_for_category(
    category_key: str,
    classifications: Dict[str, Dict]
) -> List[str]:
    """
    Get list of problem names for a given category.
    
    Args:
        category_key: e.g., 'low_cheap_smooth' or 'general'
        classifications: Problem classification dict
    
    Returns:
        List of problem names in this category
    """
    if category_key == 'general':
        return list(classifications.keys())
    
    return [
        name for name, data in classifications.items()
        if data['category_key'] == category_key
    ]

def evaluate_ragda_on_problem(
    problem_name: str,
    ragda_params: Dict[str, Any],
    n_evaluations: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Evaluate RAGDA with given params on a problem.
    Returns AUC (lower is better).
    """
    problem = get_problem(problem_name)
    
    # Split params into init and optimize
    init_params, optimize_params = split_params_by_location(ragda_params)
    
    # Override seed if provided
    if seed is not None:
        init_params['random_state'] = seed
    
    # Override maxiter to use n_evaluations
    optimize_params['maxiter'] = n_evaluations
    
    try:
        # Run RAGDA on problem using Optuna API
        study = ragda_create_study(direction='minimize', **init_params)
        study.optimize(problem.objective, n_trials=n_evaluations)
        
        # Extract convergence history
        convergence_history = [trial.value for trial in study.trials]
        
        # Calculate AUC
        auc = calculate_auc(
            convergence_history,
            normalize=True,
            best_known_value=problem.known_optimum
        )
        
        return auc
        
    except Exception as e:
        print(f"Error evaluating {problem_name}: {e}")
        return 1.0  # Worst possible AUC

def create_marsopt_objective(
    problem_names: List[str],
    n_evaluations_per_problem: int = 100,
) -> Callable:
    """
    Create MARsOpt objective function for a category.
    
    The objective:
    - Takes MARsOpt trial
    - Extracts RAGDA params from trial
    - Evaluates RAGDA on all problems in category
    - Returns average AUC (lower is better)
    """
    def objective(trial):
        # Extract RAGDA parameters from MARsOpt trial
        ragda_params = {}
        
        for param_name, param_def in RAGDA_PARAMETERS.items():
            if param_def.param_type == 'float':
                if param_def.log_scale:
                    value = trial.suggest_float(
                        param_name,
                        param_def.low,
                        param_def.high,
                        log=True
                    )
                else:
                    value = trial.suggest_float(
                        param_name,
                        param_def.low,
                        param_def.high
                    )
            elif param_def.param_type == 'int':
                value = trial.suggest_int(
                    param_name,
                    int(param_def.low),
                    int(param_def.high)
                )
            elif param_def.param_type == 'bool':
                # MARsOpt doesn't handle bools, use categorical
                str_val = trial.suggest_categorical(
                    param_name,
                    ['True', 'False']
                )
                value = (str_val == 'True')
            elif param_def.param_type == 'categorical':
                value = trial.suggest_categorical(
                    param_name,
                    param_def.choices
                )
            else:
                raise ValueError(f"Unknown param type: {param_def.param_type}")
            
            ragda_params[param_name] = value
        
        # Check constraints
        penalty = compute_constraint_penalty(ragda_params)
        if penalty > 0:
            return 1000.0 + penalty  # Large penalty for invalid configs
        
        # Evaluate RAGDA on all problems in category
        aucs = []
        for problem_name in problem_names:
            auc = evaluate_ragda_on_problem(
                problem_name,
                ragda_params,
                n_evaluations=n_evaluations_per_problem
            )
            aucs.append(auc)
        
        # Return average AUC
        avg_auc = np.mean(aucs)
        return avg_auc
    
    return objective

def optimize_category(
    category_key: str,
    problem_names: List[str],
    n_trials: int = 50,
    n_evaluations_per_problem: int = 100,
    n_jobs: int = 1,
) -> CategoryOptimizationResult:
    """
    Optimize RAGDA parameters for one category.
    
    Args:
        category_key: Category name (e.g., 'low_cheap_smooth')
        problem_names: List of problems in this category
        n_trials: Number of MARsOpt trials
        n_evaluations_per_problem: RAGDA evaluations per problem
        n_jobs: Parallel workers for MARsOpt
    
    Returns:
        CategoryOptimizationResult with best params and AUC
    """
    print(f"\n{'='*80}")
    print(f"Optimizing category: {category_key}")
    print(f"Problems: {len(problem_names)}")
    print(f"Trials: {n_trials}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create MARsOpt objective
    objective = create_marsopt_objective(
        problem_names,
        n_evaluations_per_problem=n_evaluations_per_problem
    )
    
    # Create MARsOpt study
    study = create_study(direction='minimize')
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    
    # Extract best parameters
    best_params = study.best_params
    best_auc = study.best_value
    
    elapsed = time.time() - start_time
    
    print(f"\nBest AUC: {best_auc:.6f}")
    print(f"Time: {elapsed:.1f}s")
    
    return CategoryOptimizationResult(
        category_key=category_key,
        best_params=best_params,
        best_auc=best_auc,
        n_problems=len(problem_names),
        n_trials=n_trials,
        optimization_time_seconds=elapsed,
    )

def optimize_all_categories(
    n_trials: int = 50,
    n_evaluations_per_problem: int = 100,
    n_jobs: int = 1,
    categories: Optional[List[str]] = None,
    output_file: str = "ragda_optimal_defaults.json",
) -> Dict[str, CategoryOptimizationResult]:
    """
    Optimize RAGDA parameters for all 28 categories.
    
    Args:
        n_trials: MARsOpt trials per category
        n_evaluations_per_problem: RAGDA evaluations per problem
        n_jobs: Parallel workers
        categories: List of categories to optimize (None = all 28)
        output_file: Where to save results
    
    Returns:
        Dict mapping category_key -> CategoryOptimizationResult
    """
    # Load classifications
    classifications = load_problem_classifications()
    
    # Determine categories to optimize
    if categories is None:
        # All 27 specific + 1 general
        all_category_keys = set(
            data['category_key'] 
            for data in classifications.values()
        )
        categories = sorted(all_category_keys) + ['general']
    
    print(f"\n{'='*80}")
    print(f"RAGDA Meta-Optimizer")
    print(f"{'='*80}")
    print(f"Categories: {len(categories)}")
    print(f"Total problems: {len(classifications)}")
    print(f"Trials per category: {n_trials}")
    print(f"Evaluations per problem: {n_evaluations_per_problem}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"{'='*80}\n")
    
    # Optimize each category
    results = {}
    for i, category_key in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Category: {category_key}")
        
        problem_names = get_problems_for_category(category_key, classifications)
        
        if len(problem_names) == 0:
            print(f"WARNING: No problems for category {category_key}, skipping")
            continue
        
        result = optimize_category(
            category_key,
            problem_names,
            n_trials=n_trials,
            n_evaluations_per_problem=n_evaluations_per_problem,
            n_jobs=n_jobs,
        )
        
        results[category_key] = result
    
    # Save results
    output_path = Path(__file__).parent.parent / output_file
    with open(output_path, 'w') as f:
        json.dump(
            {
                key: asdict(result)
                for key, result in results.items()
            },
            f,
            indent=2
        )
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return results
```

### 5.2 Entry Point Script

**File**: `RAGDA_default_args/run_meta_optimizer.py`

**Purpose**: CLI interface to run meta-optimizer.

```python
"""
Run RAGDA Meta-Optimizer

Usage:
    python run_meta_optimizer.py                    # Full run (all 28 categories)
    python run_meta_optimizer.py --test_mode        # Quick test (1 category, 5 trials)
    python run_meta_optimizer.py --categories low_cheap_smooth medium_moderate_rugged
"""

import argparse
from meta_optimizer import optimize_all_categories

def main():
    parser = argparse.ArgumentParser(description="RAGDA Meta-Optimizer")
    
    parser.add_argument(
        '--categories',
        nargs='*',
        default=None,
        help='Specific categories to optimize (default: all 28)'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='MARsOpt trials per category (default: 50)'
    )
    
    parser.add_argument(
        '--n_evaluations_per_problem',
        type=int,
        default=100,
        help='RAGDA evaluations per problem (default: 100)'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='ragda_optimal_defaults.json',
        help='Output file (default: ragda_optimal_defaults.json)'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='Quick test: 1 category, 5 trials, 20 evaluations'
    )
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test_mode:
        print("\n*** TEST MODE ***\n")
        categories = ['low_cheap_smooth']  # Just one category
        n_trials = 5
        n_evaluations = 20
    else:
        categories = args.categories
        n_trials = args.n_trials
        n_evaluations = args.n_evaluations_per_problem
    
    # Run optimization
    results = optimize_all_categories(
        n_trials=n_trials,
        n_evaluations_per_problem=n_evaluations,
        n_jobs=args.n_jobs,
        categories=categories,
        output_file=args.output,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for category_key, result in results.items():
        print(f"{category_key:30s} | AUC: {result.best_auc:.6f} | Problems: {result.n_problems:3d}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
```

### 5.3 Tests for Meta-Optimizer

**File**: `tests/test_phase7_meta_optimizer.py` (Section 5)

**Tests**:
```python
class TestMetaOptimizer:
    def test_load_classifications(self):
        """Test loading problem classifications."""
        pass
    
    def test_get_problems_for_category(self):
        """Test filtering problems by category."""
        pass
    
    def test_evaluate_ragda_on_simple_problem(self):
        """Test RAGDA evaluation on sphere function."""
        pass
    
    def test_create_marsopt_objective(self):
        """Test MARsOpt objective creation."""
        pass
    
    def test_constraint_penalty_in_objective(self):
        """Test that invalid configs are penalized."""
        pass
    
    def test_optimize_single_category_minimal(self):
        """
        Minimal test: optimize 1 category with 2 problems, 3 trials.
        Should complete without errors.
        """
        pass
    
    def test_general_category_includes_all_problems(self):
        """Test that 'general' category includes all 234 problems."""
        pass
```

**Validation**: All meta-optimizer tests pass (7 tests).

---

## Step 6: Integration Testing & Validation

**STATUS**: ⏳ PENDING

**Purpose**: End-to-end validation before full run.

### 6.1 Integration Tests

**File**: `tests/test_phase7_meta_optimizer.py` (Section 6)

**Tests**:
```python
class TestMetaOptimizerIntegration:
    def test_end_to_end_single_category(self):
        """
        Full end-to-end test with single category:
        1. Load 2-3 problems in category
        2. Run MARsOpt with 3 trials
        3. Verify output format
        4. Check that AUC improves
        """
        pass
    
    def test_test_mode_execution(self):
        """
        Test that test mode runs successfully:
        python run_meta_optimizer.py --test_mode
        """
        pass
    
    def test_output_file_format(self):
        """
        Verify output JSON has correct structure:
        {
            "category_key": {
                "best_params": {...},
                "best_auc": 0.123,
                "n_problems": 10,
                ...
            }
        }
        """
        pass
    
    def test_all_28_categories_have_problems(self):
        """
        Verify that all 27 specific categories have at least 1 problem,
        and general category has all 234.
        """
        pass
```

**Validation**: All integration tests pass (4 tests).

### 6.2 Manual Validation

**Actions**:

1. **Test Mode Run**:
   ```powershell
   uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
   ```
   - Should complete in ~2-5 minutes
   - Verify no errors
   - Check output file created

2. **Single Category Run**:
   ```powershell
   uv run python RAGDA_default_args/run_meta_optimizer.py --categories low_cheap_smooth --n_trials 10
   ```
   - Should complete in ~10-20 minutes
   - Verify convergence
   - Check AUC values reasonable

3. **Classification Validation**:
   - Verify `problem_classifications.json` has 234 entries
   - Check distribution across 27 categories
   - Ensure no category is empty

---

## Step 7: Documentation

**STATUS**: ⏳ PENDING

**Purpose**: Document the meta-optimizer system.

### 7.1 README for RAGDA_default_args

**File**: `RAGDA_default_args/README.md`

**Content**:
```markdown
# RAGDA Meta-Optimizer

This directory contains the meta-optimization system for finding optimal RAGDA default parameters.

## Overview

- **Goal**: Find universally good default parameters for RAGDA
- **Method**: Use MARsOpt to optimize RAGDA params across 234 benchmark problems
- **Categories**: 28 total (27 specific + 1 general)
  - 3 dimensionality levels: low (1-10D), medium (11-50D), high (51D+)
  - 3 cost levels: cheap (<10ms), moderate (10-100ms), expensive (100ms+)
  - 3 ruggedness levels: smooth, moderate, rugged
- **Metric**: AUC (area under convergence curve, 0-1, lower is better)

## File Structure

- `benchmark_functions.py` - 50 mathematical test functions (Optuna API)
- `benchmark_realworld_problems.py` - 137 real-world problems (Optuna API)
- `benchmark_ml_problems.py` - 19 ML hyperparameter problems (Optuna API)
- `problem_registry.py` - Master registry of all 234 problems
- `problem_classifier.py` - Problem classification logic
- `classify_all_problems.py` - Script to classify all problems
- `problem_classifications.json` - Classification cache
- `ragda_parameter_space.py` - RAGDA parameter definitions & constraints
- `auc_metric.py` - AUC calculation for optimizer evaluation
- `meta_optimizer.py` - Core meta-optimization logic
- `run_meta_optimizer.py` - CLI interface
- `archive/` - Old implementation (reference only)

## Usage

### 1. Classify Problems (one-time)

```powershell
uv run python RAGDA_default_args/classify_all_problems.py
```

This measures dimension, cost, and ruggedness for all 234 problems.
Results cached in `problem_classifications.json`.

### 2. Run Meta-Optimizer

**Test mode** (quick validation):
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
```

**Single category**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --categories low_cheap_smooth --n_trials 50
```

**Full run** (all 28 categories):
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --n_trials 50 --n_jobs 4
```

### 3. Results

Output saved to `ragda_optimal_defaults.json`:
```json
{
  "low_cheap_smooth": {
    "best_params": {
      "n_workers": 8,
      "lambda_start": 100,
      ...
    },
    "best_auc": 0.234,
    "n_problems": 15,
    "n_trials": 50,
    "optimization_time_seconds": 1234.5
  },
  ...
}
```

## Development

Tests located in `../tests/test_phase7_meta_optimizer.py`.

Run all tests:
```powershell
uv run pytest tests/test_phase7_meta_optimizer.py -v
```

## Future: Cross-Optimizer Benchmarking

All problems use Optuna API, enabling easy comparison with other optimizers:
- Optuna's TPE, NSGAII, etc.
- Scipy optimizers
- Hyperopt, Ax, etc.

Same problems, same metric (AUC), fair comparison.
```

### 7.2 Update Main Implementation Plan

**File**: `API_REDESIGN_IMPLEMENTATION_PLAN.md`

**Action**: Replace Phase 7 section with:
```markdown
## Phase 7: Meta-Optimizer & Default Parameter Optimization (Days 14-18)

**STATUS**: ✅ COMPLETE

Phase 7 involved building a comprehensive meta-optimization system to find optimal default parameters for RAGDA.

See `RAGDA_default_args/PHASE_7_IMPLEMENTATION_PLAN.md` for detailed plan and progress.

**Summary**:
- Archived old implementation
- Created 234 benchmark problems (Optuna API)
- Implemented problem classification (28 categories)
- Built meta-optimizer using MARsOpt
- Optimized RAGDA parameters for each category
- Generated `ragda_optimal_defaults.json`

**Test Results**: X/X tests passing in `tests/test_phase7_meta_optimizer.py`

**Key Deliverables**:
- Clean benchmark problem library (Optuna API)
- Problem classifier (3×3×3 = 27 categories + general)
- Meta-optimizer (MARsOpt-based)
- Optimal default parameters for 28 categories
```

---

## Step 8: Full Execution

**STATUS**: ⏳ PENDING

**Purpose**: Run the full meta-optimization.

### 8.1 Problem Classification

**Command**:
```powershell
uv run python RAGDA_default_args/classify_all_problems.py
```

**Expected**:
- Runtime: ~30-60 minutes (234 problems, measuring cost & ruggedness)
- Output: `problem_classifications.json` with 234 entries
- Distribution across 27 categories

**Validation**:
- Check all problems classified
- Verify no category is empty
- Review distribution manually

### 8.2 Meta-Optimizer Test Run

**Command**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --test_mode
```

**Expected**:
- Runtime: ~2-5 minutes
- Tests single category with 5 trials
- Validates end-to-end flow

**Validation**:
- No errors
- Output file created
- AUC values reasonable (0-1 range)

### 8.3 Full Meta-Optimization

**Command**:
```powershell
uv run python RAGDA_default_args/run_meta_optimizer.py --n_trials 50 --n_jobs 4
```

**Expected**:
- Runtime: Hours to days (28 categories × 50 trials × ~8 problems/category × 100 evals)
- Progress printed for each category
- Resumable if interrupted (can run specific categories)

**Alternative (sequential categories)**:
```powershell
# Run each category separately (resumable)
foreach ($cat in @('low_cheap_smooth', 'low_cheap_moderate', ...)) {
    uv run python RAGDA_default_args/run_meta_optimizer.py --categories $cat --n_trials 50
}
```

**Validation**:
- Check `ragda_optimal_defaults.json` has 28 entries
- Review AUC values (should be < 1.0, ideally < 0.5)
- Compare with default RAGDA params (should be improvements)

---

## Success Criteria

### Must Have:
- [ ] All files archived to `archive/` subdirectory
- [ ] AUC metric implemented and tested
- [ ] All 234 benchmark problems defined (Optuna API)
- [ ] Problem classifier working (27 categories + general)
- [ ] All problems classified and cached
- [ ] RAGDA parameter space defined (34 parameters)
- [ ] Meta-optimizer implemented and tested
- [ ] Test mode runs successfully
- [ ] All tests pass (`tests/test_phase7_meta_optimizer.py`)
- [ ] Documentation complete
- [ ] Full meta-optimization run completed
- [ ] `ragda_optimal_defaults.json` generated

### Testing Milestones:
- [ ] Step 1: 5 AUC metric tests passing
- [ ] Step 2: 8 benchmark problem tests passing
- [ ] Step 3: 7 classification tests passing
- [ ] Step 4: 6 parameter space tests passing
- [ ] Step 5: 7 meta-optimizer unit tests passing
- [ ] Step 6: 4 integration tests passing
- [ ] **Total: ~37 tests passing before full run**

### Quality Checks:
- [ ] No old API usage (all Optuna-style)
- [ ] All problems have metadata (bounds, dimension, category)
- [ ] All 27 specific categories have ≥1 problem
- [ ] General category has all 234 problems
- [ ] Constraint violations properly penalized
- [ ] AUC values in valid range (0-1)
- [ ] Results reproducible with fixed seed

---


## Notes

- **UV Environment**: Always activate with `.venv\Scripts\activate`
- **MARsOpt**: Already installed in UV env
- **Tests**: All in single file `tests/test_phase7_meta_optimizer.py`
- **Resumability**: Classification and meta-optimizer support resume
- **Parallel**: Use `--n_jobs` for parallel MARsOpt trials
- **Test Mode**: Use `--test_mode` for quick validation
- **Archive**: Reference only, do not modify

---

## Future Work (Post-Phase 7)

1. **Integration into RAGDA**: Load optimal defaults based on problem characteristics
2. **Cross-optimizer benchmarking**: Compare RAGDA vs Optuna, Scipy, etc.
3. **Research paper**: Publish benchmark suite and results
4. **Adaptive defaults**: Real-time problem classification and parameter selection
5. **Continuous optimization**: Re-run meta-optimizer as RAGDA evolves
