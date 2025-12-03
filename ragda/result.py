import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class Trial:
    """Single optimization trial."""
    trial_id: int
    worker_id: int
    iteration: int
    params: Dict[str, Any]
    value: float
    batch_size: int = -1
    
    def __repr__(self):
        return f"Trial(id={self.trial_id}, value={self.value:.6f}, params={self.params})"


@dataclass
class OptimizationResult:
    """
    Result from RAGDA optimization.
    
    Attributes
    ----------
    best_params : dict
        Best parameters found
    best_value : float
        Best objective value (in original direction)
    best_trial : Trial
        Full information about best trial
    best_worker_id : int
        ID of worker that found best solution
    best_concentration : float
        Concentration parameter of best worker
    trials : list of Trial
        All trials evaluated (populated from params_history)
    n_trials : int
        Total number of trials
    n_workers : int
        Number of parallel workers
    direction : str
        'minimize' or 'maximize'
    space : SearchSpace
        Search space definition
    optimization_params : dict
        All optimization parameters used
    """
    best_params: Dict[str, Any]
    best_value: float
    best_trial: Trial
    best_worker_id: int
    best_concentration: float
    trials: List[Trial] = field(default_factory=list)
    n_trials: int = 0
    n_workers: int = 1
    direction: str = 'minimize'
    space: Any = None
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def trials_df(self) -> pd.DataFrame:
        """
        Get all trials as a pandas DataFrame.
        
        Returns
        -------
        df : DataFrame
            Columns: trial_id, worker_id, iteration, value, batch_size, + all parameter names
        """
        if len(self.trials) == 0:
            # Return empty DataFrame with correct schema
            columns = ['trial_id', 'worker_id', 'iteration', 'value', 'batch_size']
            if self.space is not None:
                columns.extend(self.space.param_names)
            return pd.DataFrame(columns=columns)
        
        data = []
        for trial in self.trials:
            row = {
                'trial_id': trial.trial_id,
                'worker_id': trial.worker_id,
                'iteration': trial.iteration,
                'value': trial.value,
                'batch_size': trial.batch_size,
                **trial.params
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_worker_history(self, worker_id: int) -> pd.DataFrame:
        """Get history for specific worker."""
        df = self.trials_df
        if len(df) == 0:
            return df
        return df[df['worker_id'] == worker_id].reset_index(drop=True)
    
    def get_best_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n trials."""
        df = self.trials_df
        if len(df) == 0:
            return df
        
        if self.direction == 'minimize':
            return df.nsmallest(n, 'value')
        else:
            return df.nlargest(n, 'value')
    
    def plot_optimization_history(self, figsize=(14, 10), **kwargs):
        """
        Plot optimization history.
        
        Parameters
        ----------
        figsize : tuple, default=(14, 10)
            Figure size
        **kwargs
            Additional arguments passed to matplotlib
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. Install with: pip install matplotlib"
            )
        
        df = self.trials_df
        
        if len(df) == 0:
            raise ValueError("No trials to plot. Result has no trial history.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Best value over iterations
        df_sorted = df.sort_values('trial_id')
        if self.direction == 'minimize':
            best_so_far = df_sorted['value'].cummin()
        else:
            best_so_far = df_sorted['value'].cummax()
        
        axes[0, 0].plot(df_sorted['trial_id'], best_so_far, linewidth=2, color='#2E86AB')
        axes[0, 0].set_xlabel('Trial', fontsize=11)
        axes[0, 0].set_ylabel(f'Best Value ({self.direction})', fontsize=11)
        axes[0, 0].set_title('Optimization Progress', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-worker progress
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_workers))
        for worker_id in sorted(df['worker_id'].unique()):
            worker_df = df[df['worker_id'] == worker_id]
            axes[0, 1].scatter(
                worker_df['iteration'], 
                worker_df['value'], 
                label=f'Worker {worker_id}', 
                alpha=0.6, 
                s=20,
                color=colors[worker_id % len(colors)]
            )
        
        axes[0, 1].axhline(
            self.best_value, 
            color='red', 
            linestyle='--', 
            label='Global Best', 
            linewidth=2,
            alpha=0.8
        )
        axes[0, 1].set_xlabel('Iteration', fontsize=11)
        axes[0, 1].set_ylabel('Value', fontsize=11)
        axes[0, 1].set_title('Per-Worker Progress', fontsize=12, fontweight='bold')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mini-batch size over time (if applicable)
        if 'batch_size' in df.columns and df['batch_size'].max() > 0:
            for worker_id in sorted(df['worker_id'].unique()):
                worker_df = df[df['worker_id'] == worker_id]
                axes[1, 0].plot(
                    worker_df['iteration'], 
                    worker_df['batch_size'], 
                    label=f'Worker {worker_id}', 
                    alpha=0.7,
                    color=colors[worker_id % len(colors)]
                )
            axes[1, 0].set_xlabel('Iteration', fontsize=11)
            axes[1, 0].set_ylabel('Batch Size', fontsize=11)
            axes[1, 0].set_title('Mini-batch Schedule', fontsize=12, fontweight='bold')
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5, 0.5, 
                'No mini-batch data', 
                ha='center', 
                va='center', 
                transform=axes[1, 0].transAxes,
                fontsize=12,
                color='gray'
            )
            axes[1, 0].set_title('Mini-batch Schedule', fontsize=12, fontweight='bold')
        
        # Plot 4: Value distribution per worker
        worker_ids = sorted(df['worker_id'].unique())
        worker_values = [df[df['worker_id'] == wid]['value'].values for wid in worker_ids]
        
        bp = axes[1, 1].boxplot(
            worker_values, 
            labels=[f'W{i}' for i in worker_ids],
            patch_artist=True
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors[:len(worker_ids)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1, 1].set_xlabel('Worker', fontsize=11)
        axes[1, 1].set_ylabel('Value', fontsize=11)
        axes[1, 1].set_title('Value Distribution per Worker', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filepath: str):
        """
        Save optimization results to file.
        
        Parameters
        ----------
        filepath : str
            Path to save results (supports .csv, .pkl, .json)
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = self.trials_df
            df.to_csv(filepath, index=False)
        
        elif ext == '.pkl':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        
        elif ext == '.json':
            import json
            data = {
                'best_params': self.best_params,
                'best_value': self.best_value,
                'best_worker_id': self.best_worker_id,
                'n_trials': self.n_trials,
                'n_workers': self.n_workers,
                'direction': self.direction,
                'optimization_params': self.optimization_params
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .csv, .pkl, or .json")
    
    def __repr__(self):
        return (f"OptimizationResult(\n"
                f"  best_value={self.best_value:.6f},\n"
                f"  best_params={self.best_params},\n"
                f"  n_trials={self.n_trials},\n"
                f"  n_workers={self.n_workers},\n"
                f"  direction='{self.direction}'\n"
                f")")