"""
RAGDA Parameter Audit & Meta-Optimization Framework

Complete audit of ALL tunable parameters in RAGDA, organized by category.
This file serves as the source of truth for parameter optimization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Literal
import numpy as np

# =============================================================================
# COMPLETE PARAMETER AUDIT
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a single tunable parameter."""
    name: str
    default: Any
    bounds: Tuple[Any, Any]  # (min, max) for continuous/ordinal, or list of values for categorical
    param_type: Literal['continuous', 'ordinal', 'categorical', 'boolean']
    category: str  # Grouping for analysis
    description: str
    dependencies: List[str] = field(default_factory=list)  # Other params this depends on
    scale: Literal['linear', 'log'] = 'linear'
    expected_sensitivity: Literal['high', 'medium', 'low'] = 'medium'
    notes: str = ""


# -----------------------------------------------------------------------------
# CONSTRUCTOR PARAMETERS (RAGDAOptimizer.__init__)
# -----------------------------------------------------------------------------
CONSTRUCTOR_PARAMS = [
    ParameterSpec(
        name='n_workers',
        default='cpu_count // 2',
        bounds=(1, 32),
        param_type='ordinal',
        category='parallelization',
        description='Number of parallel workers with different exploration strategies',
        expected_sensitivity='high',
        notes='More workers = more diverse exploration. Diminishing returns past ~8-16.'
    ),
    ParameterSpec(
        name='max_parallel_workers',
        default='cpu_count',
        bounds=(1, 32),
        param_type='ordinal',
        category='parallelization',
        description='Maximum workers to run simultaneously (wave-based execution)',
        expected_sensitivity='low',
        notes='Only relevant when n_workers > CPU cores.'
    ),
    ParameterSpec(
        name='highdim_threshold',
        default=100,
        bounds=(20, 500),
        param_type='ordinal',
        category='high_dimensional',
        description='Trigger high-dim mode at this many continuous dimensions',
        expected_sensitivity='medium',
        notes='Trade-off: lower = more dim reduction overhead, higher = may miss structure.'
    ),
    ParameterSpec(
        name='variance_threshold',
        default=0.95,
        bounds=(0.70, 0.99),
        param_type='continuous',
        category='high_dimensional',
        description='Fraction of variance to capture in dimensionality reduction',
        expected_sensitivity='medium',
        notes='Lower = faster but loses info. Higher = slower but more accurate.'
    ),
    ParameterSpec(
        name='reduction_method',
        default='auto',
        bounds=['auto', 'kernel_pca', 'incremental_pca', 'random_projection'],
        param_type='categorical',
        category='high_dimensional',
        description='Dimensionality reduction method for high-dim problems',
        expected_sensitivity='medium',
        notes='kernel_pca best for smooth, random_projection fastest.'
    ),
]

# -----------------------------------------------------------------------------
# POPULATION & SAMPLING PARAMETERS
# -----------------------------------------------------------------------------
SAMPLING_PARAMS = [
    ParameterSpec(
        name='lambda_start',
        default=50,
        bounds=(10, 200),
        param_type='ordinal',
        category='population',
        description='Initial number of samples per iteration',
        expected_sensitivity='high',
        notes='Higher = better gradient estimate but more evals. Scale with dimension.',
        scale='linear'
    ),
    ParameterSpec(
        name='lambda_end',
        default=10,
        bounds=(5, 50),
        param_type='ordinal',
        category='population',
        description='Final number of samples per iteration',
        expected_sensitivity='medium',
        notes='Lower = faster convergence phase. Should be << lambda_start.'
    ),
    ParameterSpec(
        name='lambda_decay_rate',
        default=5.0,
        bounds=(0.5, 20.0),
        param_type='continuous',
        category='population',
        description='Exponential decay rate for sample count',
        expected_sensitivity='medium',
        notes='Higher = faster transition to exploitation. Lower = longer exploration.',
        scale='log'
    ),
]

# -----------------------------------------------------------------------------
# SIGMA (STEP SIZE) PARAMETERS
# -----------------------------------------------------------------------------
SIGMA_PARAMS = [
    ParameterSpec(
        name='sigma_init',
        default=0.3,
        bounds=(0.05, 0.8),
        param_type='continuous',
        category='sigma',
        description='Initial sampling radius in [0,1] unit space',
        expected_sensitivity='high',
        notes='Critical parameter. Too high = random search. Too low = local trap.'
    ),
    ParameterSpec(
        name='sigma_final_fraction',
        default=0.2,
        bounds=(0.01, 0.5),
        param_type='continuous',
        category='sigma',
        description='Final sigma as fraction of initial (sigma_final = sigma_init * this)',
        expected_sensitivity='medium',
        notes='Lower = tighter final search. ~0.1-0.3 typical.',
        scale='log'
    ),
    ParameterSpec(
        name='sigma_decay_schedule',
        default='exponential',
        bounds=['exponential', 'linear', 'cosine'],
        param_type='categorical',
        category='sigma',
        description='Schedule for sigma decay over iterations',
        expected_sensitivity='low',
        notes='Cosine has gentler early decay. Exponential fastest convergence.'
    ),
]

# -----------------------------------------------------------------------------
# ADAPTIVE SHRINKING PARAMETERS
# -----------------------------------------------------------------------------
SHRINK_PARAMS = [
    ParameterSpec(
        name='shrink_factor',
        default=0.9,
        bounds=(0.5, 0.99),
        param_type='continuous',
        category='shrinking',
        description='Multiply sigma by this when stuck (stagnation)',
        expected_sensitivity='medium',
        notes='Lower = more aggressive shrinking on plateau.'
    ),
    ParameterSpec(
        name='shrink_patience',
        default=10,
        bounds=(3, 50),
        param_type='ordinal',
        category='shrinking',
        description='Iterations without improvement before shrinking',
        expected_sensitivity='high',
        notes='Critical for noisy objectives. Higher patience = more robust to noise.'
    ),
    ParameterSpec(
        name='shrink_threshold',
        default=1e-6,
        bounds=(1e-10, 1e-3),
        param_type='continuous',
        category='shrinking',
        description='Minimum relative improvement to count as progress',
        expected_sensitivity='medium',
        notes='Higher = more shrinking. Increase for noisy objectives.',
        scale='log'
    ),
]

# -----------------------------------------------------------------------------
# TOP-N WEIGHTING / GRADIENT ESTIMATION PARAMETERS  
# -----------------------------------------------------------------------------
WEIGHTING_PARAMS = [
    ParameterSpec(
        name='use_improvement_weights',
        default=True,
        bounds=[True, False],
        param_type='boolean',
        category='weighting',
        description='Use only improving samples for gradient estimation',
        expected_sensitivity='high',
        notes='True = RAGDA signature feature. False = standard ES-style.'
    ),
    ParameterSpec(
        name='top_n_min',
        default=0.2,
        bounds=(0.05, 0.5),
        param_type='continuous',
        category='weighting',
        description='Minimum top-n fraction (most greedy worker)',
        expected_sensitivity='medium',
        notes='Lower = more exploitation-focused workers.'
    ),
    ParameterSpec(
        name='top_n_max',
        default=1.0,
        bounds=(0.5, 1.0),
        param_type='continuous',
        category='weighting',
        description='Maximum top-n fraction (most exploratory worker)',
        expected_sensitivity='medium',
        notes='1.0 = use all samples. Lower reduces exploration.'
    ),
    ParameterSpec(
        name='weight_decay',
        default=0.95,
        bounds=(0.5, 0.999),
        param_type='continuous',
        category='weighting',
        description='Exponential decay for rank-based weights (best=1.0, rank i=decay^i)',
        expected_sensitivity='medium',
        notes='Lower = more emphasis on best samples. Higher = more uniform.'
    ),
]

# -----------------------------------------------------------------------------
# WORKER SYNCHRONIZATION PARAMETERS
# -----------------------------------------------------------------------------
SYNC_PARAMS = [
    ParameterSpec(
        name='sync_frequency',
        default=100,
        bounds=(0, 500),
        param_type='ordinal',
        category='synchronization',
        description='How often workers synchronize (0 = never)',
        expected_sensitivity='medium',
        notes='Lower = faster info sharing but less diversity. 0 = independent runs.'
    ),
]

# -----------------------------------------------------------------------------
# DYNAMIC WORKER STRATEGY PARAMETERS
# -----------------------------------------------------------------------------
DYNAMIC_WORKER_PARAMS = [
    ParameterSpec(
        name='worker_strategy',
        default='greedy',
        bounds=['greedy', 'dynamic'],
        param_type='categorical',
        category='worker_strategy',
        description="'greedy': all reset to best. 'dynamic': elite selection.",
        expected_sensitivity='high',
        notes="'dynamic' better for multimodal. 'greedy' faster for unimodal."
    ),
    ParameterSpec(
        name='elite_fraction',
        default=0.5,
        bounds=(0.1, 0.9),
        param_type='continuous',
        category='worker_strategy',
        description='Fraction of top workers to keep (only for dynamic strategy)',
        expected_sensitivity='medium',
        dependencies=['worker_strategy'],
        notes='Higher = more exploitation. Lower = more exploration.'
    ),
    ParameterSpec(
        name='restart_mode',
        default='adaptive',
        bounds=['elite', 'random', 'adaptive'],
        param_type='categorical',
        category='worker_strategy',
        description='How non-elite workers restart',
        expected_sensitivity='medium',
        dependencies=['worker_strategy'],
        notes="'adaptive' transitions from exploration to exploitation."
    ),
    ParameterSpec(
        name='restart_elite_prob_start',
        default=0.3,
        bounds=(0.0, 0.8),
        param_type='continuous',
        category='worker_strategy',
        description='Initial probability of restarting from elite (for adaptive mode)',
        expected_sensitivity='low',
        dependencies=['worker_strategy', 'restart_mode'],
        notes='Lower = more random restarts early on.'
    ),
    ParameterSpec(
        name='restart_elite_prob_end',
        default=0.8,
        bounds=(0.2, 1.0),
        param_type='continuous',
        category='worker_strategy',
        description='Final probability of restarting from elite (for adaptive mode)',
        expected_sensitivity='low',
        dependencies=['worker_strategy', 'restart_mode'],
        notes='Higher = more elite restarts late in optimization.'
    ),
    ParameterSpec(
        name='enable_worker_decay',
        default=False,
        bounds=[True, False],
        param_type='boolean',
        category='worker_strategy',
        description='Gradually reduce number of active workers',
        expected_sensitivity='low',
        dependencies=['worker_strategy'],
        notes='Can speed up late-stage convergence.'
    ),
    ParameterSpec(
        name='min_workers',
        default=2,
        bounds=(1, 8),
        param_type='ordinal',
        category='worker_strategy',
        description='Minimum workers to keep when decay enabled',
        expected_sensitivity='low',
        dependencies=['worker_strategy', 'enable_worker_decay'],
    ),
    ParameterSpec(
        name='worker_decay_rate',
        default=0.5,
        bounds=(0.1, 0.9),
        param_type='continuous',
        category='worker_strategy',
        description='Rate of worker decay (0.5 = reduce to ~50%)',
        expected_sensitivity='low',
        dependencies=['worker_strategy', 'enable_worker_decay'],
    ),
]

# -----------------------------------------------------------------------------
# MINI-BATCH PARAMETERS (Data-Driven Objectives)
# -----------------------------------------------------------------------------
MINIBATCH_PARAMS = [
    ParameterSpec(
        name='use_minibatch',
        default=False,
        bounds=[True, False],
        param_type='boolean',
        category='minibatch',
        description='Enable mini-batch evaluation for data-driven objectives',
        expected_sensitivity='high',
        notes='Major feature for ML/CV objectives. Enables curriculum learning.'
    ),
    ParameterSpec(
        name='minibatch_start',
        default=None,  # Auto-calculated
        bounds=(32, 1000),
        param_type='ordinal',
        category='minibatch',
        description='Starting batch size (if data_size specified, default=data_size//20)',
        expected_sensitivity='medium',
        dependencies=['use_minibatch'],
        notes='Smaller = faster early iterations, noisier gradient.'
    ),
    ParameterSpec(
        name='minibatch_end',
        default=None,  # Auto-calculated
        bounds=(500, 50000),
        param_type='ordinal',
        category='minibatch',
        description='Final batch size (if data_size specified, default=data_size*0.8)',
        expected_sensitivity='medium',
        dependencies=['use_minibatch'],
        notes='Larger = more accurate final evaluation.'
    ),
    ParameterSpec(
        name='minibatch_schedule',
        default='inverse_decay',
        bounds=['constant', 'linear', 'exponential', 'inverse_decay', 'step'],
        param_type='categorical',
        category='minibatch',
        description='Schedule for growing batch size',
        expected_sensitivity='medium',
        dependencies=['use_minibatch'],
        notes="'inverse_decay' = slow start, fast end. 'step' = discrete jumps."
    ),
]

# -----------------------------------------------------------------------------
# ADAM OPTIMIZER PARAMETERS
# -----------------------------------------------------------------------------
ADAM_PARAMS = [
    ParameterSpec(
        name='adam_learning_rate',
        default=0.001,
        bounds=(1e-5, 0.1),
        param_type='continuous',
        category='adam',
        description='ADAM learning rate for pseudo-gradient updates',
        expected_sensitivity='high',
        notes='Critical parameter. May need scaling with dimension.',
        scale='log'
    ),
    ParameterSpec(
        name='adam_beta1',
        default=0.9,
        bounds=(0.5, 0.99),
        param_type='continuous',
        category='adam',
        description='ADAM momentum decay (first moment)',
        expected_sensitivity='low',
        notes='Higher = more momentum. 0.9 typically good.'
    ),
    ParameterSpec(
        name='adam_beta2',
        default=0.999,
        bounds=(0.9, 0.9999),
        param_type='continuous',
        category='adam',
        description='ADAM variance decay (second moment)',
        expected_sensitivity='low',
        notes='Higher = longer memory of past gradients.'
    ),
    ParameterSpec(
        name='adam_epsilon',
        default=1e-8,
        bounds=(1e-12, 1e-4),
        param_type='continuous',
        category='adam',
        description='ADAM numerical stability constant',
        expected_sensitivity='low',
        scale='log',
        notes='Rarely needs tuning. Increase for noisy gradients.'
    ),
]

# -----------------------------------------------------------------------------
# EARLY STOPPING PARAMETERS
# -----------------------------------------------------------------------------
EARLY_STOP_PARAMS = [
    ParameterSpec(
        name='early_stop_threshold',
        default=1e-12,
        bounds=(1e-15, 1e-6),
        param_type='continuous',
        category='early_stopping',
        description='Stop if best value below this (for known optimum = 0)',
        expected_sensitivity='low',
        scale='log',
        notes='Only relevant for problems with optimum near 0.'
    ),
    ParameterSpec(
        name='early_stop_patience',
        default=50,
        bounds=(10, 200),
        param_type='ordinal',
        category='early_stopping',
        description='Stop if no improvement for this many iterations',
        expected_sensitivity='medium',
        notes='Higher = more persistent. Lower = faster termination on convergence.'
    ),
]


# =============================================================================
# COMPLETE PARAMETER LIST
# =============================================================================
ALL_PARAMETERS = (
    CONSTRUCTOR_PARAMS +
    SAMPLING_PARAMS +
    SIGMA_PARAMS +
    SHRINK_PARAMS +
    WEIGHTING_PARAMS +
    SYNC_PARAMS +
    DYNAMIC_WORKER_PARAMS +
    MINIBATCH_PARAMS +
    ADAM_PARAMS +
    EARLY_STOP_PARAMS
)

# Parameters to include in meta-optimization (exclude dependencies and low-impact)
PRIMARY_TUNABLE_PARAMS = [
    # Critical parameters (always tune)
    'n_workers',
    'lambda_start',
    'lambda_end', 
    'lambda_decay_rate',
    'sigma_init',
    'sigma_final_fraction',
    'shrink_factor',
    'shrink_patience',
    'shrink_threshold',
    'top_n_min',
    'top_n_max',
    'weight_decay',
    'sync_frequency',
    'adam_learning_rate',
    'adam_beta1',
    'adam_beta2',
    'early_stop_patience',
    
    # Categorical choices
    'sigma_decay_schedule',
    'worker_strategy',
    'use_improvement_weights',
]

# Additional params for specific problem types
MINIBATCH_TUNABLE_PARAMS = [
    'minibatch_start',
    'minibatch_end', 
    'minibatch_schedule',
]

DYNAMIC_STRATEGY_TUNABLE_PARAMS = [
    'elite_fraction',
    'restart_mode',
    'restart_elite_prob_start',
    'restart_elite_prob_end',
]

HIGHDIM_TUNABLE_PARAMS = [
    'highdim_threshold',
    'variance_threshold',
    'reduction_method',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_param_spec(name: str) -> Optional[ParameterSpec]:
    """Get parameter specification by name."""
    for param in ALL_PARAMETERS:
        if param.name == name:
            return param
    return None


def get_params_by_category(category: str) -> List[ParameterSpec]:
    """Get all parameters in a category."""
    return [p for p in ALL_PARAMETERS if p.category == category]


def get_high_sensitivity_params() -> List[ParameterSpec]:
    """Get parameters expected to have high sensitivity."""
    return [p for p in ALL_PARAMETERS if p.expected_sensitivity == 'high']


def build_search_space_for_tuning(
    include_params: List[str] = None,
    exclude_params: List[str] = None,
) -> list:
    """
    Build RAGDA search space for meta-optimization.
    
    Returns list of dicts suitable for RAGDAOptimizer.
    """
    if include_params is None:
        include_params = PRIMARY_TUNABLE_PARAMS
    
    if exclude_params is None:
        exclude_params = []
    
    space = []
    
    for param in ALL_PARAMETERS:
        if param.name not in include_params:
            continue
        if param.name in exclude_params:
            continue
        
        if param.param_type == 'continuous':
            space.append({
                'name': param.name,
                'type': 'continuous',
                'bounds': list(param.bounds),
                'log': param.scale == 'log'
            })
        elif param.param_type == 'ordinal':
            # Convert bounds to discrete values
            if isinstance(param.bounds, tuple):
                low, high = param.bounds
                if high - low <= 20:
                    values = list(range(low, high + 1))
                else:
                    # Sample logarithmically for large ranges
                    values = sorted(set([
                        int(np.round(v)) for v in 
                        np.logspace(np.log10(max(1, low)), np.log10(high), 10)
                    ]))
            else:
                values = list(param.bounds)
            space.append({
                'name': param.name,
                'type': 'ordinal',
                'values': values
            })
        elif param.param_type == 'categorical':
            space.append({
                'name': param.name,
                'type': 'categorical',
                'values': list(param.bounds)
            })
        elif param.param_type == 'boolean':
            space.append({
                'name': param.name,
                'type': 'categorical',
                'values': [True, False]
            })
    
    return space


def print_parameter_summary():
    """Print formatted summary of all parameters."""
    categories = {}
    for param in ALL_PARAMETERS:
        if param.category not in categories:
            categories[param.category] = []
        categories[param.category].append(param)
    
    print("=" * 80)
    print("RAGDA COMPLETE PARAMETER AUDIT")
    print("=" * 80)
    print(f"\nTotal parameters: {len(ALL_PARAMETERS)}")
    print(f"Primary tunable: {len(PRIMARY_TUNABLE_PARAMS)}")
    print(f"High sensitivity: {len(get_high_sensitivity_params())}")
    
    for cat_name, params in sorted(categories.items()):
        print(f"\n{'─' * 40}")
        print(f"{cat_name.upper().replace('_', ' ')} ({len(params)} params)")
        print(f"{'─' * 40}")
        
        for param in params:
            sens_marker = {'high': '⚠️ ', 'medium': '  ', 'low': '  '}[param.expected_sensitivity]
            print(f"{sens_marker}{param.name}: {param.default}")
            print(f"      bounds: {param.bounds}, type: {param.param_type}")
            if param.notes:
                print(f"      note: {param.notes}")


if __name__ == '__main__':
    print_parameter_summary()
    
    print("\n" + "=" * 80)
    print("GENERATED SEARCH SPACE FOR META-OPTIMIZATION")
    print("=" * 80)
    
    space = build_search_space_for_tuning()
    for s in space:
        print(f"  {s['name']}: {s}")
