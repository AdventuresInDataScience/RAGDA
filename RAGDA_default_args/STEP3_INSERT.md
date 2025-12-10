# STEP 4 INSERTION CONTENT

Insert this after line "❌ 19/27 categories under threshold (need 79 more problems)" in PHASE_7_IMPLEMENTATION_PLAN.md:

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

### Chunk 4.1: Expensive High-Dim ML (15 problems) ⏳ NEXT
Target: `high_expensive_*` categories (0→5 each)
Files: `benchmark_ml_problems.py`

Problems (RandomForest/SVM/DeepNN on real datasets):
1-5: high_expensive_smooth (GradientBoosting on large datasets)
6-10: high_expensive_moderate (RandomForest tuning)
11-15: high_expensive_rugged (Deep NN tuning)

### Chunk 4.2: Expensive Low/Med ML (20 problems)
Target: `*_expensive_*` for low/medium dimensions
Files: `benchmark_ml_problems.py`

### Chunk 4.3: Moderate High-Dim (11 problems)  
Target: `high_moderate_*` categories
Files: `benchmark_realworld_problems.py`, `benchmark_ml_problems.py`

### Chunk 4.4: Moderate Low/Med (29 problems)
Target: Remaining `*_moderate_*` categories
Files: Multiple files

### Chunk 4.5: Cheap Rugged (5 problems)
Target: Fill `*_cheap_rugged` gaps
Files: `benchmark_mathematical_problems.py`

### Chunk 4.6: Registry Refactoring
After all problems added, simplify registry structure:
- Remove confusing sub-registries (unimodal/multimodal/etc)
- Create unified `problem_registry.py`
- Keep only: MATHEMATICAL_PROBLEMS, REALWORLD_PROBLEMS, ML_PROBLEMS, ALL_PROBLEMS
- Classification is in `problem_classifier.py` (3x3x3 only)

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

Then continue with original Step 5 content (was Step 4)...
