"""
Classify All Benchmark Problems

Runs classification on all 230 problems and saves to cache file.
Supports resume (skips already-classified problems).
"""

import json
from pathlib import Path
from problem_classifier import (
    get_all_problems, 
    classify_problem,
    ProblemClassification
)

# Try to import tqdm, fall back to basic iteration
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        """Fallback when tqdm not available."""
        if desc:
            print(f"{desc}...")
        return iterable

CACHE_FILE = Path(__file__).parent / "problem_classifications_cache.json"


def load_cache() -> dict:
    """Load existing classifications."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(classifications: dict):
    """Save classifications to cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(classifications, f, indent=2)


def classify_all_problems(resume: bool = True):
    """
    Classify all 230 problems and save to cache.
    
    Args:
        resume: If True, skip problems already in cache
    """
    # Load existing cache
    cache = load_cache() if resume else {}
    print(f"Loaded cache with {len(cache)} existing classifications")
    
    # Get all problems
    all_problems = get_all_problems()
    print(f"Found {len(all_problems)} total problems")
    
    # Filter out already-classified if resuming
    if resume:
        to_classify = {name: prob for name, prob in all_problems.items() if name not in cache}
        print(f"Will classify {len(to_classify)} new problems")
    else:
        to_classify = all_problems
    
    # Classify each problem with progress bar
    for name, problem in tqdm(to_classify.items(), desc="Classifying problems"):
        try:
            classification = classify_problem(problem)
            cache[name] = classification.to_dict()
            
            # Save after each classification (in case of interruption)
            if len(cache) % 10 == 0:
                save_cache(cache)
        
        except Exception as e:
            print(f"\n❌ Failed to classify {name}: {e}")
            continue
    
    # Final save
    save_cache(cache)
    print(f"\n✅ Classification complete! {len(cache)} problems classified")
    print(f"Cache saved to: {CACHE_FILE}")
    
    return cache


def print_category_summary(cache: dict):
    """Print summary of problem distribution across 27 categories."""
    from collections import Counter
    
    # Count by category
    categories = Counter(c['category_key'] for c in cache.values())
    
    print("\n" + "="*70)
    print("CATEGORY DISTRIBUTION (3x3x3 = 27 categories)")
    print("="*70)
    
    # Expected 27 categories
    all_categories = []
    for dim in ['low', 'medium', 'high']:
        for cost in ['cheap', 'moderate', 'expensive']:
            for rug in ['smooth', 'moderate', 'rugged']:
                all_categories.append(f"{dim}_{cost}_{rug}")
    
    # Print results
    under_5 = []
    for cat in sorted(all_categories):
        count = categories.get(cat, 0)
        status = '✅' if count >= 5 else '❌'
        print(f"{status} {cat:30s}: {count:3d} problems")
        
        if count < 5:
            under_5.append((cat, count))
    
    print("="*70)
    
    if under_5:
        print(f"\n❌ CATEGORIES WITH < 5 PROBLEMS: {len(under_5)}")
        for cat, count in under_5:
            print(f"  - {cat}: {count} (need {5-count} more)")
        print(f"\nTotal shortfall: {sum(5-count for _, count in under_5)} problems")
    else:
        print("\n✅ ALL 27 CATEGORIES HAVE >= 5 PROBLEMS!")
    
    print(f"\nTotal problems: {len(cache)}")
    print(f"Categories with problems: {len([c for c in categories.values() if c > 0])}/27")


if __name__ == '__main__':
    import sys
    
    # Allow --no-resume flag
    resume = '--no-resume' not in sys.argv
    
    print("RAGDA Problem Classification")
    print("="*70)
    
    # Run classification
    cache = classify_all_problems(resume=resume)
    
    # Print summary
    print_category_summary(cache)
