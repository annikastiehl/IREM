import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from pathlib import Path

def load_task_dataset(task_name, results_dir="results"):
    """
    Load xarray dataset for a specific task.
    
    Parameters:
    task_name (str): Task name ('mvis' or 'mveb')
    results_dir (str): Base directory containing results
    
    Returns:
    xarray.Dataset: Loaded dataset
    """
    dataset_path = Path(results_dir) / f"automated_eigenvalues_{task_name.lower()}" / f"{task_name.lower()}_dyca_results.nc"
    
    if dataset_path.exists():
        return xr.load_dataset(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

def plot_first_eigenvalue_comparison(band='gamma', results_dir="results"):
    """
    Plot first eigenvalue comparison between MVIS and MVEB tasks for a specific band.
    
    Parameters:
    band (str): Frequency band ('gamma' or 'beta')
    results_dir (str): Base directory containing results
    """
    try:
        # Load datasets for both tasks
        mvis_dataset = load_task_dataset('mvis', results_dir)
        mveb_dataset = load_task_dataset('mveb', results_dir)
        
        # Get common patients between both datasets
        mvis_patients = set(mvis_dataset.coords['patient'].values)
        mveb_patients = set(mveb_dataset.coords['patient'].values)
        common_patients = sorted(list(mvis_patients.intersection(mveb_patients)))
        
        if not common_patients:
            print("No common patients found between MVIS and MVEB datasets")
            return
        
        print(f"Found {len(common_patients)} common patients: {common_patients}")
        
        # Check if the band exists in both datasets
        if band not in mvis_dataset.coords['band'].values:
            print(f"Band '{band}' not found in MVIS dataset. Available bands: {list(mvis_dataset.coords['band'].values)}")
            return
        if band not in mveb_dataset.coords['band'].values:
            print(f"Band '{band}' not found in MVEB dataset. Available bands: {list(mveb_dataset.coords['band'].values)}")
            return
        
        # Extract first eigenvalue (rank 0) for each patient and take mean across time windows
        mvis_first_eigs = []
        mveb_first_eigs = []
        
        for patient in common_patients:
            # MVIS first eigenvalue (mean across time windows)
            mvis_eig = mvis_dataset['eigenvalues'].sel(
                patient=patient, 
                band=band, 
                eigenvalue_rank=1  # rank 1 corresponds to first eigenvalue
            ).mean(dim='time_window', skipna=True).values
            
            # MVEB first eigenvalue (mean across time windows)
            mveb_eig = mveb_dataset['eigenvalues'].sel(
                patient=patient, 
                band=band, 
                eigenvalue_rank=1  # rank 1 corresponds to first eigenvalue
            ).mean(dim='time_window', skipna=True).values
            
            mvis_first_eigs.append(mvis_eig)
            mveb_first_eigs.append(mveb_eig)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(common_patients))
        width = 0.35
        
        # Create bars
        bars1 = plt.bar(x_pos - width/2, mvis_first_eigs, width, label='MVIS', alpha=0.8)
        bars2 = plt.bar(x_pos + width/2, mveb_first_eigs, width, label='MVEB', alpha=0.8)
        
        # Customize the plot
        plt.xlabel('Patients')
        plt.ylabel('First Eigenvalue (Mean across time)')
        plt.title(f'First Eigenvalue Comparison: MVIS vs MVEB ({band.upper()} Band)')
        plt.xticks(x_pos, common_patients, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f'first_eigenvalue_comparison_mvis_mveb_{band}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary for {band.upper()} band:")
        print(f"MVIS - Mean: {np.nanmean(mvis_first_eigs):.4f}, Std: {np.nanstd(mvis_first_eigs):.4f}")
        print(f"MVEB - Mean: {np.nanmean(mveb_first_eigs):.4f}, Std: {np.nanstd(mveb_first_eigs):.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the automated analysis for both MVIS and MVEB tasks first.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def plot_both_bands_comparison(results_dir="results"):
    """
    Plot comparison for both gamma and beta bands in subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    for band_idx, band in enumerate(['gamma', 'beta']):
        try:
            # Load datasets
            mvis_dataset = load_task_dataset('mvis', results_dir)
            mveb_dataset = load_task_dataset('mveb', results_dir)
            
            # Get common patients
            mvis_patients = set(mvis_dataset.coords['patient'].values)
            mveb_patients = set(mveb_dataset.coords['patient'].values)
            common_patients = sorted(list(mvis_patients.intersection(mveb_patients)))
            
            if not common_patients:
                continue
                
            # Check if band exists
            if band not in mvis_dataset.coords['band'].values or band not in mveb_dataset.coords['band'].values:
                print(f"Band '{band}' not available in one or both datasets")
                continue
            
            # Extract eigenvalues
            mvis_first_eigs = []
            mveb_first_eigs = []
            
            for patient in common_patients:
                mvis_eig = mvis_dataset['eigenvalues'].sel(
                    patient=patient, band=band, eigenvalue_rank=1
                ).mean(dim='time_window', skipna=True).values
                
                mveb_eig = mveb_dataset['eigenvalues'].sel(
                    patient=patient, band=band, eigenvalue_rank=1
                ).mean(dim='time_window', skipna=True).values
                
                mvis_first_eigs.append(mvis_eig)
                mveb_first_eigs.append(mveb_eig)
            
            # Plot on appropriate subplot
            ax = ax1 if band_idx == 0 else ax2
            
            x_pos = np.arange(len(common_patients))
            width = 0.35
            
            ax.bar(x_pos - width/2, mvis_first_eigs, width, label='MVIS', alpha=0.8)
            ax.bar(x_pos + width/2, mveb_first_eigs, width, label='MVEB', alpha=0.8)
            
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
    
    plt.suptitle('First Eigenvalue Comparison: MVIS vs MVEB')
    plt.tight_layout()
    plt.savefig('first_eigenvalue_comparison_both_bands.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":    
    # Uncomment to plot both bands in subplots
    plot_both_bands_comparison(results_dir="results")