import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from pathlib import Path
from typing import List

def load_task_dataset(task_name, results_dir="results"):
    """
    Load xarray dataset for a specific task.
    
    Parameters:
    task_name (str): Task name (e.g. 'mvis', 'mveb', ...)
    results_dir (str): Base directory containing results
    
    Returns:
    xarray.Dataset: Loaded dataset
    """
    dataset_path = Path(results_dir) / f"automated_eigenvalues_{task_name.lower()}" / f"{task_name.lower()}_dyca_results.nc"
    
    if dataset_path.exists():
        return xr.load_dataset(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def plot_both_bands_comparison(tasks: List[str]=['mvis', 'mveb'], results_dir="results"):
    """
    Plot comparison for both gamma and beta bands in subplots for multiple tasks.
    
    Parameters:
    tasks (List[str]): List of task names to compare
    results_dir (str): Base directory containing results
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    for band_idx, band in enumerate(['gamma', 'beta']):
        ax = axes[band_idx]
        try:
            # Load datasets
            datasets = {}
            for task in tasks:
                datasets[task] = load_task_dataset(task, results_dir)
            
            # Get common patients across all tasks
            patient_sets = [set(ds.coords['patient'].values) for ds in datasets.values()]
            common_patients = sorted(list(set.intersection(*patient_sets))) if patient_sets else []
            if not common_patients:
                print(f"No common patients for band {band}")
                continue
            
            # Check if band exists in all datasets
            if any(band not in ds.coords['band'].values for ds in datasets.values()):
                print(f"Band '{band}' not available in one or more datasets")
                continue
            
            # Extract eigenvalues
            task_first_eigs = {task: [] for task in tasks}
            for patient in common_patients:
                for task, ds in datasets.items():
                    try:
                        eig = ds['eigenvalues'].sel(
                            patient=patient, band=band, eigenvalue_rank=1
                        ).mean(dim='time_window', skipna=True).values
                    except Exception as e:
                        print(f"Warning: couldn't extract eigen for task={task}, patient={patient}: {e}")
                        eig = np.nan
                    task_first_eigs[task].append(np.asarray(eig).item() if np.ndim(eig) == 0 else np.nan)
            
            # Plot grouped bars on subplot
            x_pos = np.arange(len(common_patients))
            n_tasks = len(tasks)
            total_width = 0.8
            width = total_width / n_tasks
            offsets = (np.arange(n_tasks) - (n_tasks - 1) / 2) * width
            
            for i, task in enumerate(tasks):
                ax.bar(x_pos + offsets[i], task_first_eigs[task], width, label=task.upper(), alpha=0.8)
            
            ax.set_xlabel('Patients')
            ax.set_ylabel('First Eigenvalue (Mean)')
            ax.set_title(f'{band.upper()} Band')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(common_patients, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            print(f"Error plotting {band} band: {e}")
            continue
    
    plt.suptitle(f'First Eigenvalue Comparison: {" vs ".join([t.upper() for t in tasks])}')
    plt.tight_layout()
    tasks_tag = "_".join([t.lower() for t in tasks])
    plt.savefig(f'first_eigenvalue_comparison_{tasks_tag}_both_bands.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":    
    # Example: compare three tasks (change to desired task list)
    plot_both_bands_comparison(tasks=['mvis', 'mveb', "audi"])