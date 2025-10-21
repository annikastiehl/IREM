import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import dyca
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import warnings

def extract_patient_and_band_info(filepath):
    """Extract patient ID and frequency band from filename"""
    filename = os.path.basename(filepath)
    # Assuming format like "HID-Sub-001_MVIS_gamma_data.npz"
    parts = filename.replace('.npz', '').split('_')
    patient_id = parts[0]
    band = None
    
    # Look for gamma or beta in filename
    for part in parts:
        if 'gamma' in part.lower():
            band = 'gamma'
            break
        elif 'beta' in part.lower():
            band = 'beta'
            break
    
    return patient_id, band

def moving_window_dyca_analysis(raw_data, fs=64, window_sec=5, hop_sec=2, m=2, n=3, topk=10):
    """Perform moving window DyCA analysis"""
    
    window_len = int(window_sec * fs)
    hop = int(hop_sec * fs)
    
    # Sanity check
    if raw_data.shape[0] < window_len:
        warnings.warn(f"Data too short ({raw_data.shape[0]} samples) for window length {window_len}")
        return None, None
    
    # Compute window parameters
    start_idxs = np.arange(0, raw_data.shape[0] - window_len + 1, hop)
    n_windows = len(start_idxs)
    centers = start_idxs + window_len // 2
    time_centers = centers / fs
    
    # Storage for results
    eigs_top = np.full((n_windows, topk), np.nan)
    amplitudes_all = []
    
    # Time vector for DyCA
    tv_win = np.linspace(0, window_len / fs, window_len)
    
    print(f"  Running DyCA on {n_windows} windows (window={window_len} samples, hop={hop} samples)")
    
    for wi, s in tqdm(enumerate(start_idxs), total=n_windows, desc="  Processing windows"):
        win = raw_data[s : s + window_len, :]
        
        try:
            # Apply DyCA to window
            dyca_res_win = dyca.dyca(win, m=m, n=n, time_index=tv_win)
            
            # Extract eigenvalues
            gev = np.asarray(dyca_res_win.get('generalized_eigenvalues', []))
            
            if gev.size > 0:
                gev_sorted = np.sort(gev)[::-1]
                take = min(topk, gev_sorted.size)
                eigs_top[wi, :take] = gev_sorted[:take]
                
                # Store amplitudes for this window
                amplitudes_all.append(dyca_res_win['amplitudes'])
                
        except Exception as e:
            if wi == 0:
                print(f"    Warning: DyCA failed for first window. Error: {e}")
            amplitudes_all.append(None)
            continue
    
    return {
        'eigenvalues': eigs_top,
        'time_centers': time_centers,
        'amplitudes': amplitudes_all,
        'window_params': {
            'window_sec': window_sec,
            'hop_sec': hop_sec,
            'fs': fs,
            'n_windows': n_windows
        }
    }, dyca_res_win

def process_all_mvis_files(data_dir, output_dir="results/automated_mvis", 
                          window_sec=5, hop_sec=2, topk=10):
    """Process all MVIS files across patients and bands"""
    
    # Find all MVIS files
    mvis_pattern = os.path.join(data_dir, "MVIS", "**/*.npz")
    task_files = glob.glob(mvis_pattern, recursive=True)
    
    print(f"Found {len(task_files)} MVIS files")
    
    if len(task_files) == 0:
        print(f"No files found matching pattern: {mvis_pattern}")
        return None
    
    # Group files by patient and band
    files_by_patient_band = {}
    for file_path in task_files:
        patient_id, band = extract_patient_and_band_info(file_path)
        if patient_id and band:
            key = (patient_id, band)
            if key not in files_by_patient_band:
                files_by_patient_band[key] = []
            files_by_patient_band[key].append(file_path)
    
    print(f"Processing {len(files_by_patient_band)} patient-band combinations")
    
    # Storage for all results
    all_results = {}
    patient_ids = []
    bands = []
    
    # Process each patient-band combination
    for (patient_id, band), files in files_by_patient_band.items():
        print(f"\nProcessing Patient: {patient_id}, Band: {band}")
        
        # For now, take the first file if multiple exist
        file_path = files[0]
        if len(files) > 1:
            print(f"  Multiple files found, using: {os.path.basename(file_path)}")
        
        try:
            # Load data
            data = np.load(file_path, allow_pickle=True)
            raw_data = data['data']
            
            print(f"  Data shape: {raw_data.shape}")
            
            # Perform moving window analysis
            results, _ = moving_window_dyca_analysis(
                raw_data, 
                window_sec=window_sec, 
                hop_sec=hop_sec, 
                topk=topk
            )
            
            if results is not None:
                all_results[(patient_id, band)] = results
                if patient_id not in patient_ids:
                    patient_ids.append(patient_id)
                if band not in bands:
                    bands.append(band)
            
        except Exception as e:
            print(f"  Error processing {patient_id}-{band}: {e}")
            continue
    
    # Create xarray dataset
    if all_results:
        dataset = create_xarray_dataset(all_results, patient_ids, bands, topk)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save xarray dataset
        dataset_path = os.path.join(output_dir, "mvis_dyca_results.nc")
        dataset.to_netcdf(dataset_path)
        print(f"\nSaved xarray dataset to: {dataset_path}")
        
        # Save summary CSV
        summary_df = create_summary_dataframe(all_results)
        csv_path = os.path.join(output_dir, "mvis_dyca_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary CSV to: {csv_path}")
        
        return dataset, all_results
    
    else:
        print("No results to save")
        return None, None

def create_xarray_dataset(all_results, patient_ids, bands, topk):
    """Create xarray dataset from results"""
    
    # Find maximum time length across all results
    max_time_len = max([len(results['time_centers']) for results in all_results.values()])
    
    # Initialize data arrays
    eigenvalues_data = np.full((len(patient_ids), len(bands), max_time_len, topk), np.nan)
    time_data = np.full((len(patient_ids), len(bands), max_time_len), np.nan)
    
    # Fill data arrays
    for (patient_id, band), results in all_results.items():
        p_idx = patient_ids.index(patient_id)
        b_idx = bands.index(band)
        
        time_len = len(results['time_centers'])
        eigenvalues_data[p_idx, b_idx, :time_len, :] = results['eigenvalues']
        time_data[p_idx, b_idx, :time_len] = results['time_centers']
    
    # Create xarray dataset
    coords = {
        'patient': patient_ids,
        'band': bands,
        'time_window': np.arange(max_time_len),
        'eigenvalue_rank': np.arange(1, topk + 1)
    }
    
    dataset = xr.Dataset({
        'eigenvalues': (['patient', 'band', 'time_window', 'eigenvalue_rank'], eigenvalues_data),
        'time_centers': (['patient', 'band', 'time_window'], time_data)
    }, coords=coords)
    
    # Add attributes
    dataset.attrs['description'] = 'DyCA analysis results for MVIS task'
    dataset.attrs['window_sec'] = list(all_results.values())[0]['window_params']['window_sec']
    dataset.attrs['hop_sec'] = list(all_results.values())[0]['window_params']['hop_sec']
    dataset.attrs['topk_eigenvalues'] = topk
    
    return dataset

def create_summary_dataframe(all_results):
    """Create summary dataframe with key statistics"""
    
    summary_data = []
    
    for (patient_id, band), results in all_results.items():
        eigenvals = results['eigenvalues']
        
        # Calculate statistics
        mean_top_eigenval = np.nanmean(eigenvals[:, 0])  # Mean of top eigenvalue
        std_top_eigenval = np.nanstd(eigenvals[:, 0])
        mean_all_eigenvals = np.nanmean(eigenvals)
        n_windows = results['window_params']['n_windows']
        total_time = results['time_centers'][-1] if len(results['time_centers']) > 0 else 0
        
        summary_data.append({
            'patient_id': patient_id,
            'band': band,
            'n_windows': n_windows,
            'total_time_sec': total_time,
            'mean_top_eigenvalue': mean_top_eigenval,
            'std_top_eigenvalue': std_top_eigenval,
            'mean_all_eigenvalues': mean_all_eigenvals
        })
    
    return pd.DataFrame(summary_data)

def plot_results_summary(dataset, output_dir):
    """Generate summary plots from xarray dataset"""
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Plot 1: Mean eigenvalues by patient and band
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for b_idx, band in enumerate(dataset.coords['band'].values):
        ax = axes[b_idx]
        
        # Get mean eigenvalues across time for each patient
        mean_eigs = dataset['eigenvalues'].isel(band=b_idx, eigenvalue_rank=0).mean(dim='time_window')
        
        mean_eigs.plot(ax=ax, marker='o')
        ax.set_title(f'Mean Top Eigenvalue - {band.upper()} Band')
        ax.set_ylabel('Mean Eigenvalue')
        ax.tick_params(axis='x', rotation=45)

    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'mean_eigenvalues_by_patient.png'), dpi=300, bbox_inches='tight')
    plt.show()
    


if __name__ == "__main__":
    # Configuration
    data_dir = "/media/annika/Daten/Promotion/18_Marseille/03_Data/7tasks_raw/"
    output_dir = "results/automated_mvis"
    
    # Parameters for moving window analysis
    window_sec = 30
    hop_sec = 20
    topk = 10

    # Overwrite parameter for not calculating the same again
    overwrite = False
    
    print("Starting automated MVIS DyCA analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window parameters: {window_sec}s window, {hop_sec}s hop, top-{topk} eigenvalues")
    
    # Run analysis
    if not overwrite and Path(os.path.join(output_dir, "mvis_dyca_results.nc")).exists():
        print("Results already exist and overwrite is False. Loading existing dataset...")
        dataset = xr.load_dataset(os.path.join(output_dir, "mvis_dyca_results.nc"))
        all_results = None
    else:
        dataset, all_results = process_all_mvis_files(
            data_dir=data_dir,
            output_dir=output_dir,
            window_sec=window_sec,
            hop_sec=hop_sec,
            topk=topk
        )
    
    if dataset is not None:
        print(f"\nDataset shape: {dataset.dims}")
        print(f"Patients: {list(dataset.coords['patient'].values)}")
        print(f"Bands: {list(dataset.coords['band'].values)}")
        
        # Generate summary plots
        plot_results_summary(dataset, output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
    else:
        print("Analysis failed - no results generated")