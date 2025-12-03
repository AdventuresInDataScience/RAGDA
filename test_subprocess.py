"""Test subprocess import of Cython core."""
import multiprocessing as mp
import numpy as np


def worker_import_only():
    """Just import the core."""
    from ragda import core
    print(f'Worker imported core: {core.get_version()}')
    return True


def worker_run_core():
    """Run the core optimizer."""
    import sys
    import faulthandler
    faulthandler.enable()
    
    print('Worker: importing core...', flush=True)
    from ragda import core
    print(f'Worker: core imported: {core.get_version()}', flush=True)
    
    def obj(x_cont, x_cat, batch_size):
        return float(np.sum(x_cont**2))
    
    x0 = np.array([1.0, 1.0])
    b = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    n_iter = 3
    
    print('Worker: calling optimize_worker_core...', flush=True)
    sys.stdout.flush()
    
    try:
        result = core.optimize_worker_core(
            x0, 
            np.array([], dtype=np.int32), 
            np.array([], dtype=np.int32), 
            b, 
            obj, 
            n_iter, 
            np.full(n_iter, 10, dtype=np.int32),
            np.full(n_iter, 5, dtype=np.int32),
            np.full(n_iter, 0.5),
            np.full(n_iter, -1, dtype=np.int32),
            False, 0.5, 0.01, 0.9, 0.999, 1e-8, 
            0.5, 5, 0.001, True, 42, 0, 
            None, None, 0, 0.001, 1e-8, 10
        )
        print(f'Worker result: {result[2]}', flush=True)
        return result[2]
    except Exception as e:
        print(f'Worker exception: {e}', flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print('Testing subprocess import...')
    ctx = mp.get_context('spawn')
    
    # Test 1: Just import
    p = ctx.Process(target=worker_import_only)
    p.start()
    p.join(timeout=10)
    print(f'Import test exitcode: {p.exitcode}')
    
    if p.exitcode == 0:
        # Test 2: Run core
        print('\nTesting subprocess run...')
        p2 = ctx.Process(target=worker_run_core)
        p2.start()
        p2.join(timeout=30)
        print(f'Run test exitcode: {p2.exitcode}')
    
    print('\nDone!')
