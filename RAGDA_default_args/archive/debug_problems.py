"""Quick debug script to identify which problem is #141"""
from benchmark_functions import get_all_functions
from benchmark_ml_problems import get_all_ml_problems
from benchmark_realworld_problems import get_all_genuine_problems

syn = list(get_all_functions().items())
ml = list(get_all_ml_problems().items())
rw = get_all_genuine_problems()

print(f"Synthetic: {len(syn)}")
print(f"ML: {len(ml)}")
print(f"Realworld: {len(rw)}")
print(f"Total: {len(syn)+len(ml)+len(rw)}")

# Find problems around #141
print("\nProblems 135-150:")
for idx in range(134, min(150, len(syn)+len(ml)+len(rw))):
    if idx < len(syn):
        name = f"synthetic - {syn[idx][1].name}"
    elif idx < len(syn)+len(ml):
        name = f"ML - {ml[idx-len(syn)][1].name}"
    else:
        name = f"realworld - {rw[idx-len(syn)-len(ml)].name}"
    print(f"  #{idx+1}: {name}")
