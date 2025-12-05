"""Quick diagnostic to test MARSOpt behavior with 34 parameters."""

import sys
import time
from marsopt import Study as MARSOptStudy

def test_marsopt_34_params():
    """Test if MARSOpt is slow with 34 parameters."""
    
    trial_count = 0
    
    def objective(trial):
        nonlocal trial_count
        trial_count += 1
        start = time.time()
        
        # Suggest 34 parameters (similar to meta_optimizer)
        params = {}
        for i in range(10):
            params[f'int_param_{i}'] = trial.suggest_int(f'int_param_{i}', 1, 100)
        for i in range(10):
            params[f'float_param_{i}'] = trial.suggest_float(f'float_param_{i}', 0.0, 1.0)
        for i in range(10):
            params[f'log_param_{i}'] = trial.suggest_float(f'log_param_{i}', -10.0, 0.0)
        for i in range(4):
            params[f'cat_param_{i}'] = trial.suggest_categorical(f'cat_param_{i}', ['a', 'b', 'c'])
        
        suggest_time = time.time() - start
        
        # Simulate objective evaluation (very fast)
        eval_start = time.time()
        loss = sum(params[f'float_param_{i}'] for i in range(10))
        eval_time = time.time() - eval_start
        
        print(f"  Trial {trial_count}: suggest_time={suggest_time:.3f}s, eval_time={eval_time:.6f}s, loss={loss:.4f}")
        sys.stdout.flush()
        
        return loss
    
    print("Testing MARSOpt with 34 parameters, 5 trials...")
    print("=" * 60)
    sys.stdout.flush()
    
    study = MARSOptStudy(direction='minimize', random_state=42, verbose=True)
    
    start = time.time()
    study.optimize(objective, n_trials=5)
    total = time.time() - start
    
    print("=" * 60)
    print(f"Total time: {total:.2f}s for 5 trials")
    print(f"Average per trial: {total/5:.2f}s")

if __name__ == "__main__":
    test_marsopt_34_params()
