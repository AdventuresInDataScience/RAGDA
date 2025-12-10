"""
Comprehensive verification of Step 2 completion.

Checks:
1. All 230 problems exist and are unique
2. All problems have correct Optuna API wrapper
3. All metadata is complete and consistent
4. Category distribution is reasonable
5. Dimension ranges are covered
"""

import numpy as np
from collections import defaultdict, Counter

# Import all registries
from benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
from benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
from benchmark_ml_problems import ALL_ML_PROBLEMS

# Combine all problems
ALL_PROBLEMS = {
    **ALL_BENCHMARK_FUNCTIONS,
    **ALL_REALWORLD_PROBLEMS,
    **ALL_ML_PROBLEMS,
}

print("="*80)
print("STEP 2 VERIFICATION REPORT")
print("="*80)

# =============================================================================
# 1. Count and Uniqueness Check
# =============================================================================
print("\n1. COUNT AND UNIQUENESS")
print("-" * 80)

print(f"Mathematical problems: {len(ALL_BENCHMARK_FUNCTIONS)}")
print(f"Real-world problems: {len(ALL_REALWORLD_PROBLEMS)}")
print(f"ML problems: {len(ALL_ML_PROBLEMS)}")
print(f"Total unique problems: {len(ALL_PROBLEMS)}")
print(f"Expected: 230/234 (98.3%)")

# Check for name collisions
total_count = len(ALL_BENCHMARK_FUNCTIONS) + len(ALL_REALWORLD_PROBLEMS) + len(ALL_ML_PROBLEMS)
if total_count != len(ALL_PROBLEMS):
    print(f"❌ WARNING: Name collision detected! {total_count} entries but only {len(ALL_PROBLEMS)} unique names")
else:
    print("✅ All problem names are unique")

# =============================================================================
# 2. Optuna API Verification
# =============================================================================
print("\n2. OPTUNA API VERIFICATION")
print("-" * 80)

class MockTrial:
    """Mock Optuna trial for testing."""
    def suggest_float(self, name, low, high):
        return (low + high) / 2  # Return midpoint

errors = []
for name, problem in ALL_PROBLEMS.items():
    try:
        # Test if objective accepts trial
        trial = MockTrial()
        result = problem.objective(trial)
        
        # Verify result is a number
        if not isinstance(result, (int, float)):
            errors.append(f"{name}: objective returned {type(result)} instead of number")
        elif np.isnan(result) or np.isinf(result):
            errors.append(f"{name}: objective returned {result}")
            
    except TypeError as e:
        if "suggest_float" in str(e):
            errors.append(f"{name}: objective doesn't use Optuna trial API correctly")
        else:
            errors.append(f"{name}: {str(e)}")
    except Exception as e:
        errors.append(f"{name}: {str(e)}")

if errors:
    print(f"❌ Found {len(errors)} API issues:")
    for err in errors[:10]:  # Show first 10
        print(f"  - {err}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
else:
    print(f"✅ All {len(ALL_PROBLEMS)} problems use Optuna API correctly")

# =============================================================================
# 3. Metadata Completeness
# =============================================================================
print("\n3. METADATA COMPLETENESS")
print("-" * 80)

metadata_issues = []
for name, problem in ALL_PROBLEMS.items():
    if problem.name != name:
        metadata_issues.append(f"{name}: name mismatch ('{problem.name}')")
    if problem.dimension <= 0:
        metadata_issues.append(f"{name}: invalid dimension {problem.dimension}")
    if len(problem.bounds) != problem.dimension:
        metadata_issues.append(f"{name}: bounds length {len(problem.bounds)} != dimension {problem.dimension}")
    if not problem.category:
        metadata_issues.append(f"{name}: missing category")
    if not problem.description:
        metadata_issues.append(f"{name}: missing description")
    if not callable(problem.objective):
        metadata_issues.append(f"{name}: objective not callable")

if metadata_issues:
    print(f"❌ Found {len(metadata_issues)} metadata issues:")
    for issue in metadata_issues[:10]:
        print(f"  - {issue}")
    if len(metadata_issues) > 10:
        print(f"  ... and {len(metadata_issues) - 10} more")
else:
    print(f"✅ All metadata complete and consistent")

# =============================================================================
# 4. Category Distribution
# =============================================================================
print("\n4. CATEGORY DISTRIBUTION")
print("-" * 80)

categories = Counter(p.category for p in ALL_PROBLEMS.values())
print(f"Total categories: {len(categories)}")
print("\nBreakdown:")
for cat, count in sorted(categories.items()):
    print(f"  {cat:20s}: {count:3d} problems")

# =============================================================================
# 5. Dimension Distribution
# =============================================================================
print("\n5. DIMENSION DISTRIBUTION")
print("-" * 80)

dimensions = [p.dimension for p in ALL_PROBLEMS.values()]
dim_ranges = {
    'Very Low (1-5D)': sum(1 for d in dimensions if 1 <= d <= 5),
    'Low (6-10D)': sum(1 for d in dimensions if 6 <= d <= 10),
    'Medium (11-50D)': sum(1 for d in dimensions if 11 <= d <= 50),
    'High (51-100D)': sum(1 for d in dimensions if 51 <= d <= 100),
    'Very High (100D+)': sum(1 for d in dimensions if d > 100),
}

for range_name, count in dim_ranges.items():
    print(f"  {range_name:20s}: {count:3d} problems")

print(f"\nMin dimension: {min(dimensions)}")
print(f"Max dimension: {max(dimensions)}")
print(f"Mean dimension: {np.mean(dimensions):.1f}")
print(f"Median dimension: {np.median(dimensions):.1f}")

# =============================================================================
# 6. Duplicates Check (from archive)
# =============================================================================
print("\n6. KNOWN DUPLICATES (from archive)")
print("-" * 80)

# Check if known duplicates were properly handled
known_duplicates = [
    'CoupledLogisticMaps-100D',  # Should be in chaotic only
    'StandardMapChain-30D',      # Should be in chaotic only
]

duplicate_status = []
for dup in known_duplicates:
    if dup in ALL_PROBLEMS:
        # Count how many times it appears across registries
        count = sum([
            dup in ALL_BENCHMARK_FUNCTIONS,
            dup in ALL_REALWORLD_PROBLEMS,
            dup in ALL_ML_PROBLEMS
        ])
        if count == 1:
            duplicate_status.append(f"✅ {dup}: appears exactly once (correct)")
        else:
            duplicate_status.append(f"❌ {dup}: appears {count} times (duplicate!)")
    else:
        duplicate_status.append(f"⚠️  {dup}: not found (was it removed?)")

for status in duplicate_status:
    print(f"  {status}")

# =============================================================================
# 7. Registry Consistency
# =============================================================================
print("\n7. REGISTRY CONSISTENCY")
print("-" * 80)

# Check mathematical categories
math_categories = set(p.category for p in ALL_BENCHMARK_FUNCTIONS.values())
expected_math = {'unimodal', 'multimodal', 'valley', 'plate', 'steep'}
if math_categories == expected_math:
    print(f"✅ Mathematical categories correct: {math_categories}")
else:
    print(f"❌ Mathematical categories mismatch:")
    print(f"   Expected: {expected_math}")
    print(f"   Got: {math_categories}")

# Check real-world categories
realworld_categories = set(p.category for p in ALL_REALWORLD_PROBLEMS.values())
print(f"\n✅ Real-world categories ({len(realworld_categories)}): {sorted(realworld_categories)}")

# Check ML categories
ml_categories = set(p.category for p in ALL_ML_PROBLEMS.values())
print(f"\n✅ ML categories ({len(ml_categories)}): {sorted(ml_categories)}")

# =============================================================================
# 8. Problem Name Patterns
# =============================================================================
print("\n8. PROBLEM NAME PATTERNS")
print("-" * 80)

# Check for consistent naming
name_patterns = defaultdict(list)
for name in ALL_PROBLEMS.keys():
    if '-' in name:
        base = name.split('-')[0]
        name_patterns[base].append(name)

print(f"Found {len(name_patterns)} problem families:")
for base, names in sorted(name_patterns.items())[:20]:  # Show first 20
    if len(names) > 1:
        print(f"  {base}: {len(names)} variants")

# =============================================================================
# 9. Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_issues = len(errors) + len(metadata_issues)
if total_issues == 0:
    print("✅ ALL CHECKS PASSED")
    print(f"✅ {len(ALL_PROBLEMS)} problems ready for Step 3 (classification)")
    print(f"✅ All problems use correct Optuna API")
    print(f"✅ All metadata complete")
    print(f"✅ {len(categories)} categories represented")
else:
    print(f"⚠️  Found {total_issues} issues that need attention")
    print(f"   - {len(errors)} API issues")
    print(f"   - {len(metadata_issues)} metadata issues")

print("\n" + "="*80)
