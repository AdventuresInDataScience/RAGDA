# RAGDA Meta-Optimizer Debug Summary

## Objective

Find optimal **default parameters** for RAGDA across 27 problem categories (3 dimension levels × 3 cost levels × 3 ruggedness levels) using MARSOpt as the meta-optimizer.

---

## Current Status: ✅ READY FOR FULL RUN

All major issues have been resolved. The meta-optimizer is ready for a full optimization run.

---

## What We Built

### Files Created/Modified

1. **`ragda_parameter_space.py`** - Defines all 34 tunable RAGDA parameters with:
   - 7 `__init__` parameters
   - 27 `optimize()` parameters
   - Constraint definitions (e.g., `lambda_end <= lambda_start`, `min_workers <= n_workers`)
   - Penalty functions for constraint violations (return 1000.0 per violation)

2. **`problem_classifier.py`** - Classifies problems by:
   - Dimension: low (<10D), medium (10-50D), high (>50D)
   - Cost: cheap (<10ms), moderate (10-100ms), expensive (>100ms)
   - Ruggedness: smooth, moderate, rugged

3. **`benchmark_realworld_problems.py`** - **137 genuine real-world problems** including:
   - Chaotic systems (Lorenz, Rössler, coupled maps)
   - Neural network training problems
   - Control systems (PID tuning, LQR, trajectory optimization)
   - Physics simulations (coupled pendulums, wave equations, spin glass)
   - ML problems (hyperparameter tuning, acquisition functions)
   - Operations research (supply chain, epidemic control)

4. **`meta_optimizer.py`** - Main orchestration script that:
   - Loads **234 benchmark problems** (78 synthetic + 19 ML + 137 realworld)
   - Classifies each into one of 23 categories
   - Uses MARSOpt to find optimal RAGDA params for each category
   - Saves results to `ragda_optimal_defaults.json`

5. **`problem_classifications_cache.json`** - Cache of 234 problem classifications

---

## Issues Encountered and Resolved

### Issue 1: Classification Crashes ✅ RESOLVED

**Problem:** Script crashed silently after classifying ~140-180 problems.

**Root Cause:** High-dimensional sklearn-based problems caused ARPACK eigenvalue decomposition errors.

**Solution:** Added per-problem cache saving and pre-classified problematic problems.

---

### Issue 2: MARSOpt Boolean Handling ✅ RESOLVED

**Problem:** `TypeError` when MARSOpt tried to handle boolean parameters.

**Solution:** Convert booleans to strings for MARSOpt, then convert back:
```python
str_val = trial.suggest_categorical(param_name, ["True", "False"])
params[param_name] = (str_val == "True")
```

---

### Issue 3: minibatch_schedule None Bug ✅ RESOLVED

**Problem:** RAGDA crashed with `"Unknown minibatch_schedule: None"` when `use_minibatch=False`.

**Root Cause:** In `ragda/optimizer.py`, the code checked `if minibatch_schedule:` but None is falsy, so it fell through to the final `else` which raised an error.

**Solution:** Fixed in `ragda/optimizer.py` line ~460:
```python
# Before (buggy):
if minibatch_schedule:
    ...
else:
    raise ValueError(f"Unknown minibatch_schedule: {minibatch_schedule}")

# After (fixed):
if minibatch_schedule is None:
    # use_minibatch=False, no schedule needed
    pass
elif minibatch_schedule == 'linear':
    ...
```

---

### Issue 4: MARSOpt Seed Overflow ✅ RESOLVED

**Problem:** `"Python int too large to convert to C long"` error in random seed handling.

**Root Cause:** MARSOpt's internal seed generation could produce values > 2^32.

**Solution:** Added seed clamping in `meta_optimizer.py`:
```python
seed = trial.suggest_int("seed", 0, 999999) % (2**32)
```

---

### Issue 5: High-Dimensional (>1000D) Crashes ✅ RESOLVED

**Problem:** RAGDA crashed with `"n_cont 1074 exceeds MAX_DIMS 1000"` on problems with >1000 dimensions.

**Root Cause:** Cython core has `DEF MAX_DIMS = 1000` compile-time constant. When high-dim problems didn't trigger dimensionality reduction (because no low-dim structure was detected), they fell back to standard optimization which crashed.

**Solution:** Modified `ragda/optimizer.py` `_optimize_highdim()` to force dimensionality reduction when dims > 1000:
```python
CYTHON_MAX_DIMS = 1000

def _optimize_highdim(self, ...):
    # Force reduction if we exceed Cython's compiled limit
    force_reduction = self.space.n_continuous > CYTHON_MAX_DIMS
    
    if force_reduction and n_components is None:
        # Cap at 80% of Cython limit
        n_components = min(effective_dim, int(CYTHON_MAX_DIMS * 0.8))
    
    if intrinsic_dim <= threshold or force_reduction:
        # Use dimensionality reduction
        ...
```

**Verified:** 1074D, 1377D, and 2000D problems now work correctly.

---

### Issue 6: Insufficient Problems Per Category ✅ RESOLVED

**Problem:** 12 of 27 categories had fewer than 5 problems, making meta-optimization unreliable.

**Solution:** Added 37 **genuine** optimization problems to `benchmark_realworld_problems.py`:
- Coupled pendulums, wave equations, spin glass (physics)
- Matrix factorization, covariance estimation (linear algebra)
- Coupled logistic maps, Kuramoto oscillators (chaotic systems)
- Inverse kinematics, trajectory optimization (robotics/control)
- Epidemic control, supply chain optimization (operations research)
- Chemical kinetics, PID tuning, LQR control (engineering)

**Result:** All 23 active categories now have 5+ problems.

---

## Current Problem Distribution

| Category | Count |
|----------|-------|
| high_cheap_moderate | 5 |
| high_cheap_smooth | 29 |
| high_expensive_moderate | 5 |
| high_expensive_rugged | 12 |
| high_expensive_smooth | 5 |
| high_moderate_moderate | 5 |
| high_moderate_rugged | 5 |
| high_moderate_smooth | 5 |
| low_cheap_moderate | 14 |
| low_cheap_rugged | 41 |
| low_cheap_smooth | 21 |
| low_expensive_rugged | 5 |
| low_expensive_smooth | 5 |
| low_moderate_rugged | 5 |
| low_moderate_smooth | 5 |
| medium_cheap_moderate | 7 |
| medium_cheap_rugged | 5 |
| medium_cheap_smooth | 29 |
| medium_expensive_rugged | 5 |
| medium_expensive_smooth | 6 |
| medium_moderate_moderate | 5 |
| medium_moderate_rugged | 5 |
| medium_moderate_smooth | 5 |
| **TOTAL** | **234** |

---

## Mini Test Results ✅

Ran a simplified 20-trial MARSOpt test on the `low_cheap_smooth` category:

```
Trial  1: loss=0.0057 (baseline defaults)
Trial  8: loss=0.0020 (65% improvement)
Trial 15: loss=0.0003 (95% improvement!)
Trial 20: loss=0.0003

Best parameters found:
  n_workers: 8
  aggressive: True
  n_iterations: 300
  n_initial: 25
  ...
```

**Key finding:** MARSOpt successfully finds parameters that significantly outperform RAGDA's defaults.

---

## What Works

1. ✅ Classification caching (234 problems cached)
2. ✅ MARSOpt boolean handling
3. ✅ minibatch_schedule None handling
4. ✅ Seed overflow prevention
5. ✅ High-dimensional (>1000D) problem handling
6. ✅ All 23 categories have 5+ problems
7. ✅ All 319 RAGDA unit tests pass
8. ✅ Mini meta-optimization test shows 95% improvement

---

## Ready for Full Run

The meta-optimizer is ready. To run:

```bash
cd RAGDA_default_args
uv run python meta_optimizer.py
```

Expected runtime: Several hours (234 problems × 23 categories × multiple trials)

Output: `ragda_optimal_defaults.json` with optimal parameters per category

---

## Files Modified

| File | Changes |
|------|---------|
| `ragda/optimizer.py` | Fixed minibatch_schedule None bug, added >1000D force reduction |
| `benchmark_realworld_problems.py` | Added 37 genuine gap-filling problems + `get_gap_filling_problems()` |
| `meta_optimizer.py` | Removed fill_benchmark_gaps import, uses integrated problems |
| `problem_classifications_cache.json` | Updated with 234 problem classifications |

---

## Session Statistics

- Total problems: 234
- Categories with 5+ problems: 23 (of 23 active)
- All RAGDA tests passing: 319/319
- Mini-optimizer improvement: 95% over defaults
