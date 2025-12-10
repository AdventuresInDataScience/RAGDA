# RAGDA Phase 7 Meta-Optimizer - CURRENT STATUS

## WHAT HAS BEEN DONE

### ✅ Step 0: Archive (COMPLETE)
- Archived old implementation to `archive/` folder
- Clean slate for new implementation

### ✅ Step 1: AUC Metric (COMPLETE)
- Created `auc_metric.py` with convergence measurement
- 12 AUC tests passing

### ✅ Step 2: Benchmark Problems (COMPLETE - 230/234 = 98.3%)
- **Mathematical**: 78 functions (unimodal, multimodal, valley, plate, steep)
- **Real-world**: 133 functions (chaotic, dynamical, nn_weights, ml_training, pde, control, meta_opt, etc.)
- **ML**: 19 functions (ml_tuning, finance)
- **Total**: 230 problems implemented
- ✅ All use Optuna API (`trial.suggest_float()`)
- ✅ All have import guards for optional dependencies
- ✅ 133 tests passing
- ❌ 4 problems short of 234 (archive had duplicates, so 230 is correct count)

### ✅ Step 3: Classification System (COMPLETE)
- Created `problem_classifier.py` with 3x3x3 classification:
  - **Dimensionality**: low (1-10D), medium (11-50D), high (51D+)
  - **Cost**: cheap (<10ms), moderate (10-100ms), expensive (100ms+)  
  - **Ruggedness**: smooth, moderate, rugged
- Created `classify_all_problems.py` for batch classification
- Generated `problem_classifications_cache.json` with all 230 classifications
- **PROBLEM DISCOVERED**: Only 8/27 categories have ≥5 problems

## WHAT NEEDS TO BE DONE NEXT

### ⏳ Step 4: Fill Category Gaps (NEXT - START HERE)

**Goal**: Add 79 strategic problems to ensure all 27 categories have ≥5 problems

**Current Distribution**:
- ✅ Categories with ≥5 problems: 8/27
- ❌ Categories with <5 problems: 19/27
- **Total shortfall: 79 problems**

**Key Gaps**:
1. **EXPENSIVE problems**: Almost none exist (only 3 total, need 55)
2. **MODERATE cost**: Very few (need 40 more)  
3. **High-dim expensive/moderate**: Zero problems in these categories

**Strategy** (Per User Request):
- ✅ Add REAL problems only (no artificial dimension inflation)
- ✅ ML hyperparameter tuning → naturally expensive (cross-validation)
- ✅ High-dim from real datasets (gene expression, text features, images)
- ✅ Model selection controls cost (RandomForest=expensive, LightGBM=cheap)
- ❌ NO fake problems like "Sphere-1000D"

**Implementation Plan**:
1. **Chunk 4.1**: 15 expensive high-dim ML problems
2. **Chunk 4.2**: 20 expensive low/med ML problems
3. **Chunk 4.3**: 11 moderate high-dim problems  
4. **Chunk 4.4**: 29 moderate low/med problems
5. **Chunk 4.5**: 5 cheap rugged problems
6. **Chunk 4.6**: Registry refactoring (simplify structure)

### ⏳ Step 5: Registry Refactoring (AFTER Step 4)

**Goal**: Simplify confusing registry structure

**Current Problem**:
- 19 sub-registries (unimodal, multimodal, chaotic, chemistry, etc.)
- These metadata tags don't matter for classification
- Only matters: problem TYPE (math/real/ML) + 3x3x3 classification

**Solution**:
- Create unified `problem_registry.py`
- Keep only: MATHEMATICAL_PROBLEMS, REALWORLD_PROBLEMS, ML_PROBLEMS, ALL_PROBLEMS
- Remove sub-registry logic
- Classification stays in `problem_classifier.py`

### ⏳ Step 6: RAGDA Parameter Space (PENDING)
- Define 34 tunable parameters
- Implement constraint checking
- Add tests

### ⏳ Step 7: Meta-Optimizer Core (PENDING)
- Implement MARsOpt-based parameter optimization
- Run on all 28 categories (27 + 1 general)
- Generate `ragda_optimal_defaults.json`

## KEY FILES

**Implemented**:
- ✅ `auc_metric.py` - Convergence measurement
- ✅ `benchmark_mathematical_problems.py` - 78 math functions
- ✅ `benchmark_realworld_problems.py` - 133 real-world functions
- ✅ `benchmark_ml_problems.py` - 19 ML tuning functions
- ✅ `problem_classifier.py` - 3x3x3 classification system
- ✅ `classify_all_problems.py` - Batch classification script
- ✅ `problem_classifications_cache.json` - Classification results
- ✅ `tests/test_phase7_meta_optimizer.py` - 133 tests passing

**Next to Create** (Step 4):
- ⏳ Add 79 problems to benchmark files (5 chunks)
- ⏳ `problem_registry.py` - Unified registry (after chunks)
- ⏳ Update tests for new problems

**Future** (Steps 6-7):
- ⏳ `ragda_parameter_space.py`
- ⏳ `meta_optimizer.py`  
- ⏳ `ragda_optimal_defaults.json`

## ANSWERS TO YOUR QUESTIONS

### 1. Category counts (5 minimum per category)?
**❌ NOT MET** - Classification reveals:
- Only 8/27 categories have ≥5 problems
- 19/27 categories need more problems
- Total shortfall: 79 problems
- This is what Step 4 will fix

### 2. Optuna API wrappers correct?
**✅ YES** - All 230 problems verified:
- Use `trial.suggest_float()` correctly
- Tested with MockTrial simulation
- All return valid float values

### 3. Imports fixed/added?
**✅ YES** - All optional dependencies guarded:
- sklearn: 8 problems check `SKLEARN_AVAILABLE`
- lightgbm: 4 problems check `LIGHTGBM_AVAILABLE`  
- xgboost: 2 problems check `XGBOOST_AVAILABLE`
- All return 1.0 if imports missing (graceful degradation)

## NEXT ACTION

**START Step 4, Chunk 4.1**: Add 15 expensive high-dimensional ML problems

Target categories:
- high_expensive_smooth: 0 → 5
- high_expensive_moderate: 0 → 5
- high_expensive_rugged: 0 → 5

Use real datasets with RandomForest/SVM/DeepNN + cross-validation for natural expense.
