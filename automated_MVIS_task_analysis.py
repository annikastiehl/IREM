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
import h5py
import gc


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


def moving_window_dyca_analysis_minimal(raw_data, fs=64, window_sec=5, hop_sec=2, m=2, n=3, topk=10):
    """Memory-efficient moving window DyCA analysis - only stores essential results"""

    window_len = int(window_sec * fs)
    hop = int(hop_sec * fs)

    # Sanity check
    if raw_data.shape[0] < window_len:
        warnings.warn(f"Data too short ({raw_data.shape[0]} samples) for window length {window_len}")
        return None

    # Compute window parameters
    start_idxs = np.arange(0, raw_data.shape[0] - window_len + 1, hop)
    n_windows = len(start_idxs)
    centers = start_idxs + window_len // 2
    time_centers = centers / fs

    # Storage for MINIMAL results only
    eigs_top = np.full((n_windows, topk), np.nan)
    singular_values_all = np.full((n_windows, topk), np.nan)
    reconstruction_cost_all = np.full(n_windows, np.nan)

    # Only store first 3 DyCA components to save memory
    n_store = min(n, 3)
    amplitudes_summary = np.full((n_windows, n_store, 5), np.nan)  # mean, std, min, max, median

    # Time vector for DyCA
    tv_win = np.linspace(0, window_len / fs, window_len)

    print(f"  Running memory-efficient DyCA on {n_windows} windows")

    for wi, s in tqdm(enumerate(start_idxs), total=n_windows, desc="  Processing windows"):
        win = raw_data[s: s + window_len, :]

        try:
            # Apply DyCA to window
            dyca_res_win = dyca.dyca(win, m=m, n=n, time_index=tv_win)

            # Get reconstruction (but don't store the full signals)
            reconstruction_dict = dyca.reconstruction(win.T, dyca_res_win['amplitudes'])

            # Extract and store eigenvalues
            gev = np.asarray(dyca_res_win.get('generalized_eigenvalues', []))
            if gev.size > 0:
                gev_sorted = np.sort(gev)[::-1]
                take = min(topk, gev_sorted.size)
                eigs_top[wi, :take] = gev_sorted[:take]

            # Extract and store singular values
            sv = np.asarray(dyca_res_win.get('singular_values', []))
            if sv.size > 0:
                sv_sorted = np.sort(sv)[::-1]
                take_sv = min(topk, sv_sorted.size)
                singular_values_all[wi, :take_sv] = sv_sorted[:take_sv]

            # Store only summary statistics of amplitudes
            if dyca_res_win['amplitudes'] is not None:
                amplitudes = dyca_res_win['amplitudes'].T  # Shape: (time, components)
                for comp in range(min(n_store, amplitudes.shape[1])):
                    comp_data = amplitudes[:, comp]
                    amplitudes_summary[wi, comp, 0] = np.mean(comp_data)
                    amplitudes_summary[wi, comp, 1] = np.std(comp_data)
                    amplitudes_summary[wi, comp, 2] = np.min(comp_data)
                    amplitudes_summary[wi, comp, 3] = np.max(comp_data)
                    amplitudes_summary[wi, comp, 4] = np.median(comp_data)

            # Store reconstruction cost only
            reconstruction_cost_all[wi] = reconstruction_dict.get('cost', np.nan)

            # Force garbage collection every 10 windows
            if wi % 10 == 0:
                gc.collect()

        except Exception as e:
            if wi == 0:
                print(f"    Warning: DyCA failed for first window. Error: {e}")
            continue

    return {
        'eigenvalues': eigs_top,
        'singular_values': singular_values_all,
        'amplitudes_summary': amplitudes_summary,
        'reconstruction_cost': reconstruction_cost_all,
        'time_centers': time_centers,
        'window_params': {
            'window_sec': window_sec,
            'hop_sec': hop_sec,
            'fs': fs,
            'n_windows': n_windows,
            'm': m,
            'n': n
        }
    }


def save_patient_results_hdf5(results, patient_id, band, output_file):
    """Save individual patient results to HDF5 file"""

    with h5py.File(output_file, 'a') as f:
        # Create group for this patient-band combination
        group_name = f"{patient_id}_{band}"

        if group_name in f:
            del f[group_name]  # Remove if exists

        grp = f.create_group(group_name)

        # Save data arrays
        grp.create_dataset('eigenvalues', data=results['eigenvalues'], compression='gzip')
        grp.create_dataset('singular_values', data=results['singular_values'], compression='gzip')
        grp.create_dataset('amplitudes_summary', data=results['amplitudes_summary'], compression='gzip')
        grp.create_dataset('reconstruction_cost', data=results['reconstruction_cost'], compression='gzip')
        grp.create_dataset('time_centers', data=results['time_centers'], compression='gzip')

        # Save metadata
        grp.attrs['patient_id'] = patient_id
        grp.attrs['band'] = band
        for key, value in results['window_params'].items():
            grp.attrs[key] = value


def process_all_mvis_files_memory_efficient(data_dir, output_dir="results/automated_mvis",
                                            window_sec=5, hop_sec=2, topk=10, params_electrodes=None):
    """Memory-efficient processing of all MVIS files"""

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # HDF5 file for storing results
    hdf5_file = os.path.join(output_dir, "mvis_dyca_results.h5")

    # Remove existing file
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)

    # Storage for metadata only
    metadata_list = []
    successful_patients = 0

    # Process each patient-band combination individually
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

            # Apply electrode selection if specified
            if params_electrodes is not None:
                cutting_half = params_electrodes['cutting_half_electrodes']
                offset = params_electrodes['offset_electrodes']
                random_elec = params_electrodes['random_electrodes']

                n_electrodes = raw_data.shape[1]
                if cutting_half:
                    half = n_electrodes // 2
                    if random_elec:
                        selected_indices = np.random.choice(n_electrodes, half, replace=False)
                        raw_data = raw_data[:, selected_indices]
                    else:
                        raw_data = raw_data[:, offset::2]

                    print(f"  Selected {half} electrodes (cutting half), new shape: {raw_data.shape}")

            # Perform moving window analysis (memory efficient)
            results = moving_window_dyca_analysis_minimal(
                raw_data,
                window_sec=window_sec,
                hop_sec=hop_sec,
                topk=topk
            )

            if results is not None:
                # Save to HDF5 immediately
                save_patient_results_hdf5(results, patient_id, band, hdf5_file)

                # Store metadata
                metadata_list.append({
                    'patient_id': patient_id,
                    'band': band,
                    'n_windows': results['window_params']['n_windows'],
                    'total_time_sec': results['time_centers'][-1] if len(results['time_centers']) > 0 else 0,
                    'mean_top_eigenvalue': np.nanmean(results['eigenvalues'][:, 0]),
                    'data_shape': raw_data.shape
                })

                successful_patients += 1
                print(f"  ✓ Saved results for {patient_id}-{band}")

            # Clear memory
            del data, raw_data, results
            gc.collect()

        except Exception as e:
            print(f"  ✗ Error processing {patient_id}-{band}: {e}")
            continue

    # Save metadata CSV
    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        csv_path = os.path.join(output_dir, "mvis_dyca_metadata.csv")
        metadata_df.to_csv(csv_path, index=False)
        print(f"\nSaved metadata CSV to: {csv_path}")

        print(f"Successfully processed {successful_patients} patient-band combinations")
        print(f"Results saved to HDF5 file: {hdf5_file}")

        return hdf5_file, metadata_df

    else:
        print("No results to save")
        return None, None


def load_results_from_hdf5(hdf5_file, patient_subset=None, band_subset=None):
    """Load specific results from HDF5 file"""

    results = {}

    with h5py.File(hdf5_file, 'r') as f:
        for group_name in f.keys():
            patient_id, band = group_name.split('_', 1)

            # Apply filters
            if patient_subset is not None and patient_id not in patient_subset:
                continue
            if band_subset is not None and band not in band_subset:
                continue

            grp = f[group_name]

            # Load data
            patient_results = {
                'eigenvalues': grp['eigenvalues'][:],
                'singular_values': grp['singular_values'][:],
                'amplitudes_summary': grp['amplitudes_summary'][:],
                'reconstruction_cost': grp['reconstruction_cost'][:],
                'time_centers': grp['time_centers'][:],
                'metadata': dict(grp.attrs)
            }

            results[(patient_id, band)] = patient_results

    return results


def create_summary_plots_from_hdf5(hdf5_file, metadata_df, output_dir):
    """Create summary plots from HDF5 data without loading everything into memory"""

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Get unique patients and bands
    patients = sorted(metadata_df['patient_id'].unique())
    bands = sorted(metadata_df['band'].unique())

    # Plot 1: Mean eigenvalues by patient and band
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 6))
    if len(bands) == 1:
        axes = [axes]

    for b_idx, band in enumerate(bands):
        ax = axes[b_idx]

        # Load data for this band only
        band_results = load_results_from_hdf5(hdf5_file, band_subset=[band])

        patient_means = []
        patient_labels = []

        for patient in patients:
            if (patient, band) in band_results:
                eigenvals = band_results[(patient, band)]['eigenvalues']
                mean_top = np.nanmean(eigenvals[:, 0])
                patient_means.append(mean_top)
                patient_labels.append(patient)

        if patient_means:
            ax.bar(range(len(patient_means)), patient_means)
            ax.set_xticks(range(len(patient_labels)))
            ax.set_xticklabels(patient_labels, rotation=45)
            ax.set_title(f'Mean Top Eigenvalue - {band.upper()} Band')
            ax.set_ylabel('Mean Eigenvalue')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'mean_eigenvalues_by_patient_memory_efficient.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Time courses for first few patients
    n_patients_plot = min(3, len(patients))
    fig, axes = plt.subplots(len(bands), n_patients_plot, figsize=(5 * n_patients_plot, 4 * len(bands)))
    if len(bands) == 1:
        axes = axes.reshape(1, -1)
    if n_patients_plot == 1:
        axes = axes.reshape(-1, 1)

    for b_idx, band in enumerate(bands):
        band_results = load_results_from_hdf5(hdf5_file,
                                              patient_subset=patients[:n_patients_plot],
                                              band_subset=[band])

        for p_idx, patient in enumerate(patients[:n_patients_plot]):
            ax = axes[b_idx, p_idx]

            if (patient, band) in band_results:
                results = band_results[(patient, band)]
                eigenvals = results['eigenvalues']
                time_centers = results['time_centers']

                # Plot top 3 eigenvalues
                for comp in range(min(3, eigenvals.shape[1])):
                    valid_mask = ~np.isnan(eigenvals[:, comp])
                    if valid_mask.sum() > 0:
                        ax.plot(time_centers[valid_mask], eigenvals[valid_mask, comp],
                                label=f'Eig {comp + 1}', marker='o', markersize=2)

                ax.set_title(f'{patient} - {band.upper()}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Eigenvalue')
                ax.legend()
                ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'eigenvalue_timecourses_memory_efficient.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Configuration
    data_dir = "/home/astiehl/05_Code/Promotion/marseille/IREM/7tasks_raw"  # "/media/annika/Daten/Promotion/18_Marseille/03_Data/7tasks_raw/"
    output_dir = "results/automated_mvis_2s_halfelectrodes"

    # Parameters for moving window analysis
    window_sec = 2
    hop_sec = 1
    topk = 10

    # Parameter for cutting the half of electrodes
    cutting_half_electrodes = True
    offset_electrodes = 0
    random_electrodes = False
    params_electrodes = {
        'cutting_half_electrodes': cutting_half_electrodes,
        'offset_electrodes': offset_electrodes,
        'random_electrodes': random_electrodes
    }

    # Overwrite parameter for not calculating the same again
    overwrite = False

    print("Starting memory-efficient automated MVIS DyCA analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window parameters: {window_sec}s window, {hop_sec}s hop, top-{topk} eigenvalues")

    # Check if results already exist
    hdf5_file = os.path.join(output_dir, "mvis_dyca_results.h5")
    metadata_file = os.path.join(output_dir, "mvis_dyca_metadata.csv")

    if not overwrite and os.path.exists(hdf5_file) and os.path.exists(metadata_file):
        print("Results already exist and overwrite is False. Loading existing metadata...")
        metadata_df = pd.read_csv(metadata_file)
        results_file = hdf5_file
    else:
        # Run analysis
        results_file, metadata_df = process_all_mvis_files_memory_efficient(
            data_dir=data_dir,
            output_dir=output_dir,
            window_sec=window_sec,
            hop_sec=hop_sec,
            topk=topk,
            params_electrodes=params_electrodes
        )

    if results_file is not None and metadata_df is not None:
        print(f"\nProcessed {len(metadata_df)} patient-band combinations")
        print(f"Patients: {sorted(metadata_df['patient_id'].unique())}")
        print(f"Bands: {sorted(metadata_df['band'].unique())}")

        # Generate memory-efficient plots
        create_summary_plots_from_hdf5(results_file, metadata_df, output_dir)

        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"HDF5 file: {results_file}")
        print(f"Metadata file: {metadata_file}")

        # Example of loading specific data
        print("\nExample: Loading first patient's results...")
        first_patient = metadata_df['patient_id'].iloc[0]
        first_band = metadata_df['band'].iloc[0]

        sample_results = load_results_from_hdf5(results_file,
                                                patient_subset=[first_patient],
                                                band_subset=[first_band])

        if sample_results:
            key = list(sample_results.keys())[0]
            print(f"Eigenvalues shape for {key}: {sample_results[key]['eigenvalues'].shape}")
            print(f"Amplitudes summary shape for {key}: {sample_results[key]['amplitudes_summary'].shape}")

    else:
        print("Analysis failed - no results generated")
