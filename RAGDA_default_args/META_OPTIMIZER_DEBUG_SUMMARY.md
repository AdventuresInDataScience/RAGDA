# RAGDA Meta-Optimizer Debug Summary

## Objective

Find optimal **default parameters** for RAGDA across 27 problem categories (3 dimension levels × 3 cost levels × 3 ruggedness levels) using MARSOpt as the meta-optimizer.

---

## What We Built

### Files Created

1. **`ragda_parameter_space.py`** - Defines all 34 tunable RAGDA parameters with:
   - 7 `__init__` parameters
   - 27 `optimize()` parameters
   - Constraint definitions (e.g., `lambda_end <= lambda_start`, `min_workers <= n_workers`)
   - Penalty functions for constraint violations (return 1000.0 per violation)

2. **`problem_classifier.py`** - Classifies problems by:
   - Dimension: low (<10D), medium (10-50D), high (>50D)
   - Cost: cheap (<10ms), moderate (10-100ms), expensive (>100ms)
   - Ruggedness: smooth, moderate, rugged

3. **`benchmark_realworld_problems.py`** - 100 real-world inspired problems

4. **`meta_optimizer.py`** - Main orchestration script that:
   - Loads 197 benchmark problems (78 synthetic + 19 ML + 100 realworld)
   - Classifies each into one of 27 categories
   - Uses MARSOpt to find optimal RAGDA params for each category
   - Saves results to `ragda_optimal_defaults.json`

5. **`problem_classifications_cache.json`** - Cache of 197 problem classifications

---

## Issues Encountered

### Issue 1: Classification Crashes (RESOLVED)

**Problem:** Script crashed silently after classifying ~140-180 problems.

**Root Cause:** 12 high-dimensional sklearn-based problems caused ARPACK eigenvalue decomposition errors during classification.

**Solution:** Manually added these 12 problems to the cache with estimated classifications:
```
KernelPCA-SVM-75D, NeuralNet-Dropout-80D, XGBoost-HighDim-60D,
LightGBM-HighDim-65D, EnsembleStacking-70D, FeatureSelection-75D,
Hyperband-60D, BayesianOpt-60D, NAS-70D, EvolutionStrategy-65D,
SimulatedAnnealing-55D, CovarianceAdaptation-60D
```

Also added per-problem cache saving to avoid losing progress.

---

### Issue 2: MARSOpt Boolean Handling (RESOLVED)

**Problem:** `TypeError` when MARSOpt tried to handle boolean parameters.

**Root Cause:** MARSOpt's `suggest_categorical()` requires string values, not Python booleans.

**Solution:** Convert booleans to strings for MARSOpt, then convert back:
```python
# For MARSOpt
str_val = trial.suggest_categorical(param_name, ["True", "False"])
params[param_name] = (str_val == "True")
```

---

### Issue 3: Optimization Returns Only Penalties (CURRENT - UNRESOLVED)

**Problem:** All MARSOpt trials return constraint penalties (1000) or RAGDA failures (1e10), never actual optimization results.

**Symptoms observed:**
```
Trial 5:  loss = 1000.0    (constraint: min_workers_bound violated)
Trial 6:  loss = 1000.0    (constraint: top_n_min > top_n_max)  
Trial 9:  loss = 1000.0    (constraint: lambda_end > lambda_start)
Trial 1:  loss = 5e9       (RAGDA failed: Python int too large)
Trial 3:  loss = 1e10      (RAGDA failed: Unknown minibatch_schedule: None)
```

**Constraint violations seen:**
- `min_workers_bound`: MARSOpt picked `n_workers=3, min_workers=7`
- `lambda_order`: MARSOpt picked `lambda_start=17, lambda_end=72`
- `top_n_order`: MARSOpt picked `top_n_min=0.42, top_n_max=0.31`

**RAGDA failures seen:**
- `"Python int too large to convert to C long"` - On high-dim problems
- `"n_cont 1074 exceeds MAX_DIMS 1000"` - Problems with >1000 dimensions
- `"Unknown minibatch_schedule: None"` - When `use_minibatch=False`

---

### Issue 4: High-Dimensional Problem Failures (CURRENT - UNRESOLVED)

**Problem:** RAGDA fails on high-dimensional problems with errors like:
- `"Python int too large to convert to C long"`
- `"n_cont 1074 exceeds MAX_DIMS 1000"`

**Observed dimensions causing failures:**
- 1074D, 1377D (from realworld problems)
- 60D-100D sklearn problems

**Note from user:** RAGDA is supposed to automatically use dimensionality reduction for high-dim problems. The integration tests show RAGDA works correctly on high-dim problems. So this failure mode suggests either:
1. The meta-optimizer is not invoking RAGDA correctly
2. The parameter combinations being tested are invalid
3. The benchmark problems themselves are malformed

---

## What Works

1. ✅ Classification caching (197 problems cached)
2. ✅ MARSOpt boolean handling (strings converted properly)
3. ✅ Script runs through all 23 active categories
4. ✅ RAGDA integration tests pass (including high-dim tests)

---

## What Doesn't Work

1. ❌ No MARSOpt trial returns a valid loss (all penalties or failures)
2. ❌ Constraint violations happening too frequently
3. ❌ RAGDA failing on high-dim problems despite working in integration tests
4. ❌ `minibatch_schedule: None` being passed when `use_minibatch=False`

---

## Key Questions to Investigate

1. **Why is RAGDA failing on high-dim problems here but not in integration tests?**
   - What parameters do integration tests use?
   - Is dimensionality reduction being triggered?

2. **Are the constraint ranges too wide?**
   - `n_workers`: 1-16, `min_workers`: 1-8 → easy to violate `min_workers <= n_workers`
   - `lambda_start`: 10-200, `lambda_end`: 5-100 → easy to violate `lambda_end <= lambda_start`

3. **Is the parameter passing correct?**
   - Are we passing parameters RAGDA doesn't expect?
   - Is `minibatch_schedule` being passed when it shouldn't be?

4. **What do the benchmark problems actually look like?**
   - Why do some have 1074 or 1377 dimensions?
   - Are these valid for RAGDA?

---

## Files to Examine

| File | Purpose |
|------|---------|
| `ragda/optimizer.py` | RAGDAOptimizer class - check `optimize()` signature |
| `ragda/highdim.py` | High-dim handling - when is dimensionality reduction triggered? |
| `tests/test_highdim.py` | Integration tests that work on high-dim |
| `tests/test_integration.py` | General integration tests |
| `RAGDA_default_args/meta_optimizer.py` | Main script - check how RAGDA is invoked |
| `RAGDA_default_args/ragda_parameter_space.py` | Parameter definitions and constraints |

---

## Recommended Next Steps

1. **Compare meta-optimizer RAGDA invocation vs integration tests**
   - What parameters do tests pass?
   - How do tests handle high-dim?

2. **Run a minimal RAGDA test inside meta_optimizer.py**
   - Pick one simple problem (e.g., Sphere-5D)
   - Use RAGDA's default parameters
   - Verify it works before adding MARSOpt

3. **Check if high-dim problems are valid**
   - Why do we have 1074D and 1377D problems?
   - Should these be excluded or handled differently?

4. **Simplify the parameter space**
   - Start with just 5-10 core parameters
   - Use tighter ranges that can't violate constraints
   - Get ONE successful optimization before scaling up

---

## Session Statistics

- Total problems: 197
- Categories with problems: 23 (of 27 possible)
- MARSOpt iterations attempted: 40 (for category 1)
- Successful RAGDA runs: 0
- Time spent: ~2 hours debugging
