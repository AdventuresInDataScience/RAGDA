"""Check category counts and requirements."""
from benchmark_mathematical_problems import ALL_BENCHMARK_FUNCTIONS
from benchmark_realworld_problems import ALL_REALWORLD_PROBLEMS
from benchmark_ml_problems import ALL_ML_PROBLEMS
from collections import Counter

# Collect all problems
all_probs = (
    list(ALL_BENCHMARK_FUNCTIONS.values()) + 
    list(ALL_REALWORLD_PROBLEMS.values()) + 
    list(ALL_ML_PROBLEMS.values())
)

# Count by category
cats = Counter(p.category for p in all_probs)

print('CATEGORY COUNTS:')
print('=' * 60)
for cat in sorted(cats.keys()):
    status = '✅' if cats[cat] >= 5 else '❌'
    print(f'{status} {cat:25s}: {cats[cat]:3d} problems')

print('\n' + '=' * 60)
under_5 = [(cat, count) for cat, count in cats.items() if count < 5]

if under_5:
    print(f'\n❌ CATEGORIES WITH < 5 PROBLEMS: {len(under_5)}')
    for cat, count in sorted(under_5):
        print(f'  - {cat}: {count}')
else:
    print('\n✅ ALL CATEGORIES HAVE >= 5 PROBLEMS')

print(f'\nTotal categories: {len(cats)}')
print(f'Total problems: {len(all_probs)}')
