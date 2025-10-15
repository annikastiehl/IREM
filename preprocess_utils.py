import os
import json
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats, interpolate

from neo.rawio import ElanRawIO
from neo.io import ElanIO

def read_pos_file(path_pos):
    r = np.genfromtxt(path_pos)
    assert (r.ndim == 2) and (r.shape[1] == 3)
    return pd.DataFrame(r[:, 0:2].astype(int), columns=['sample', 'trigger'])

def parse_trials(df, trig_rules, trig_sfreq=64.0, trig_map=None):

    trials = []
    current_trial = {}

    stim_codes = trig_rules.get("stim_codes", [])
    secondary_codes = trig_rules.get("2nd_onset_codes", [])
    resp_codes = trig_rules.get("resp_codes", [])
    feedback_codes = trig_rules.get("feedback_codes", [])

    multi_onset_codes_to_filter = trig_rules.get("2nd_onset_codes", [])

    if multi_onset_codes_to_filter:

        is_multi_onset = np.isin(df['trigger'], multi_onset_codes_to_filter)
        prev_trigger = df['trigger'].shift(1)

        is_first_onset = is_multi_onset & (prev_trigger != df['trigger'])
        is_other_code = ~is_multi_onset

        keep_mask = is_other_code | is_first_onset
        df = df[keep_mask].copy()

    trials = []
    current_trial = {}

    stim_codes = trig_rules.get("stim_codes", [])
    secondary_codes = trig_rules.get("2nd_onset_codes", [])
    resp_codes = trig_rules.get("resp_codes", [])
    feedback_codes = trig_rules.get("feedback_codes", [])

    for _, row in df.iterrows():
        trig = int(row['trigger'])
        sample = int(row['sample'])

        if trig in stim_codes:
            if current_trial:
                trials.append(current_trial)
            current_trial = {'stim_code': trig, 'stim_sample': sample}

        elif trig in secondary_codes and current_trial and 'maint_code' not in current_trial:
            current_trial['maint_code'] = trig
            current_trial['maint_sample'] = sample

        elif trig in resp_codes and current_trial and 'resp_code' not in current_trial:
            current_trial['resp_code'] = trig
            current_trial['resp_sample'] = sample

        elif trig in feedback_codes and current_trial and 'resp_code' in current_trial and 'fb_code' not in current_trial:
            current_trial['fb_code'] = trig
            current_trial['fb_sample'] = sample
            trials.append(current_trial)
            current_trial = {}

    if current_trial:
        trials.append(current_trial)

    n_trials = len(trials)
    stim_codes = np.array([t['stim_code'] for t in trials])
    stim_samples = np.array([t['stim_sample'] for t in trials])
    maint_codes = np.array([t.get('maint_code', np.nan) for t in trials])
    maint_samples = np.array([t.get('maint_sample', np.nan) for t in trials])
    resp_codes = np.array([t.get('resp_code', np.nan) for t in trials])
    resp_samples = np.array([t.get('resp_sample', np.nan) for t in trials])
    fb_codes = np.array([t.get('fb_code', np.nan) for t in trials])
    fb_samples = np.array([t.get('fb_sample', np.nan) for t in trials])

    time_stim = stim_samples / trig_sfreq
    time_maint = maint_samples / trig_sfreq
    time_resp = resp_samples / trig_sfreq
    time_fb = fb_samples / trig_sfreq
    rt = time_resp - time_stim

    coords = {
        "trial": np.arange(n_trials),
        "stim_code": ("trial", stim_codes.astype(int)),
        "maint_code": ("trial", maint_codes),
        "resp_code": ("trial", resp_codes),
        "fb_code": ("trial", fb_codes),
        "time_stim": ("trial", time_stim),
        "time_maint": ("trial", time_maint),
        "time_resp": ("trial", time_resp),
        "time_fb": ("trial", time_fb),
        "rt": ("trial", rt),
    }

    if trig_map:
        def get_label(code):
            if pd.isna(code): return 'n/a'
            return trig_map.get(int(code), str(int(code)))

        coords["stim_label"] = ("trial", [get_label(c) for c in stim_codes])
        coords["maint_label"] = ("trial", [get_label(c) for c in maint_codes])
        coords["resp_label"] = ("trial", [get_label(c) for c in resp_codes])
        coords["fb_label"] = ("trial", [get_label(c) for c in fb_codes])

    return xr.DataArray(np.arange(n_trials), dims=("trial",),coords=coords,name="events")


def pad_array(arr, target_len, pad_value=np.nan):
    pad_width = target_len - len(arr)
    if pad_width > 0:
        arr = np.concatenate([arr, np.full(pad_width, pad_value, dtype=arr.dtype)])
    return arr

def clean_contacts(ch_names):
    cleaned_names = [n.split('.')[0].lower().strip().replace(' ', '').replace("'", "p") for n in ch_names]
    for i, name in enumerate(cleaned_names):
        if '-' in name: continue
        letter_part = re.findall(r'^[a-z]+', name)
        number_part = re.findall(r'\d+$', name)
        if letter_part and number_part and len(number_part[0]) == 1:
            cleaned_names[i] = f"{letter_part[0]}{int(number_part[0]):02d}"
    return np.asarray(cleaned_names)

def load_subject_data(subject_name, cohort, task_name, trig_rules, trig_map, root_dir, bad_contacts, freq_band, smoothing):

    anat_folder = os.path.join(root_dir, f'Cohort {cohort}', 'anat', subject_name)
    anat_file = glob.glob(f'{anat_folder}/*.csv')

    df_anat = pd.read_csv(anat_file[0], skiprows=2, sep='\t')
    df_anat['contact'] = clean_contacts(df_anat['contact'].values)

    data_folder = os.path.join(root_dir, f'Cohort {cohort}', 'seeg', subject_name)
    data_file = glob.glob(f'{data_folder}/*_{task_name}_{freq_band}_*_{smoothing}.eeg')[0]
    pos_file = glob.glob(f'{data_folder}/*_{task_name}_*.pos')[0]

    reader = ElanIO(data_file, posfile=pos_file)
    block = reader.read_block(lazy=False)
    segment = block.segments[0]
    signals = segment.analogsignals[0]
    data = signals.rescale('V').magnitude.astype(np.float32)
    sfreq = float(signals.sampling_rate.magnitude)
    header_ch_names = clean_contacts(reader.header['signal_channels']['name'])

    all_header_names = reader.header['signal_channels']['name']
    n_data_channels = data.shape[1]
    filtered_names_by_structure = [name for name in all_header_names if '.' in name]

    aligned_header_names = np.array(filtered_names_by_structure)
    header_ch_names = clean_contacts(aligned_header_names)

    valid_mask = np.isin(header_ch_names, df_anat['contact']) & ~np.isin(header_ch_names, bad_contacts)
    final_ch_names = header_ch_names[valid_mask]
    data = data[:, valid_mask]
    df_anat = df_anat.set_index('contact').loc[final_ch_names].reset_index()

    events = parse_trials(read_pos_file(pos_file), trig_rules=trig_rules, trig_map=trig_map)

    return data, final_ch_names, events, df_anat, sfreq


def create_epochs(data, channel_names, events, df_anat, tmin, tmax, event_query, time_coord, sfreq):

    selected_events = events.query(trial=event_query)
    if selected_events.trial.size == 0:
        return None

    reference_samples = selected_events[time_coord].values * sfreq
    valid_mask = ~np.isnan(reference_samples)
    reference_samples = reference_samples[valid_mask].astype(int)
    selected_events = selected_events.isel(trial=valid_mask)
    if selected_events.trial.size == 0:
        return None

    times = np.arange(int(tmin * sfreq), int(tmax * sfreq)) / sfreq
    epochs_list, valid_trials_idx = [], []
    for i, s in enumerate(reference_samples):
        start, end = s + int(tmin * sfreq), s + int(tmax * sfreq)
        if 0 <= start and end <= data.shape[0]:
            epochs_list.append(data[start:end, :].T)
            valid_trials_idx.append(i)

    if not epochs_list:
        return None

    epochs_data = np.stack(epochs_list, axis=0)
    final_events = selected_events.isel(trial=valid_trials_idx)

    epochs = xr.DataArray(epochs_data, dims=('trial', 'region', 'time'),
        coords={'trial': final_events.trial.values, 'region': list(df_anat['MarsAtlas']), 'time': times}, attrs={'sfreq': sfreq, 'tmin': tmin, 'tmax': tmax})
    epochs = epochs.assign_coords(contacts=('region', channel_names))    # Add contact names as a supplementary coordinate

    for name, val in final_events.coords.items():
        if name != 'trial':
            epochs = epochs.assign_coords({name: ('trial', val.values)})

    return epochs

def smooth_epochs(epochs, tend_orig=1, sfreq=64, freq_scale=2, order=3, lp_freq=20, nbins=100, remove_tails=False, axis=-1):

    tmin_orig = epochs.tmin
    tmax_orig = epochs.tmax
    nt_orig = epochs.time.shape[0]
    time_orig = np.linspace(tmin_orig, tmax_orig, nt_orig)
    time_extd = np.linspace(tmin_orig, tmax_orig, nt_orig*freq_scale)

    lowpass = signal.butter(order, lp_freq, 'lp', fs=sfreq, output='sos')
    epochs_processed = signal.sosfiltfilt(lowpass, epochs.data.copy(), axis=axis)
    interpFunc = interpolate.interp1d(time_orig, epochs_processed, kind='cubic')
    epochs_interp1d = interpFunc(time_extd)

    epoch_attrs = epochs.attrs.copy()
    epoch_attrs['sfreq'] = sfreq*freq_scale

    epochs_processed = xr.DataArray(epochs_interp1d, dims=epochs.dims, coords={k:v for k,v in epochs.coords.items() if k not in ['time']}, attrs=epoch_attrs)

    return epochs_processed

def process_WM_tasks(subj_name, coh_map, task_trules, task_tmap, smoothing, root_dir, save_array=True, output_dir=os.getcwd(), sub_dir='data/epochs/'):

    task_names = ['MVEB', 'MVIS']
    freq_dict = {'gamma': 'f50f150', 'beta': 'f8f24'}
    subj_coh = subj_coh_map[subj_name]

    task_coords = {}
    task_arrays = []

    for t_ind, task_name in enumerate(task_names):

        task_freq_arrays = []
        for freq_name, freq_band in freq_dict.items():

            subj_data, final_ch_names, task_events, df_anat, subj_sfreq = load_subject_data(subject_name=subj_name, cohort=subj_coh, task_name=task_name, trig_rules=task_trules, trig_map=task_tmap, root_dir=root_dir,
                                                                                            bad_contacts=[], freq_band=freq_band, smoothing=smoothing)

            epochs_encod = create_epochs(subj_data, final_ch_names, task_events, df_anat, tmin, tmax, event_query='time_stim.notnull()', time_coord='time_stim', sfreq=subj_sfreq)
            epochs_maint = create_epochs(subj_data, final_ch_names, task_events, df_anat, tmin, tmax, event_query='time_maint.notnull()', time_coord='time_maint', sfreq=subj_sfreq)

            subj_task_blocks = xr.concat([epochs_encod, epochs_maint], dim='event')
            subj_task_blocks.coords['event'] = ['encod', 'maint']

            if freq_name == 'beta':
                task_coords[f'{task_name}-feedback'] = subj_task_blocks.fb_label.to_numpy()
                task_coords[f'{task_name}-response'] = subj_task_blocks.resp_code.to_numpy()

            coords_to_remove = ['contacts', 'stim_code', 'maint_code', 'fb_code', 'time_stim', 'time_maint', 'time_resp',
                                'time_fb', 'rt', 'stim_label', 'maint_label', 'resp_label', 'fb_label', 'resp_code']

            subj_task_blocks = subj_task_blocks.drop_vars(coords_to_remove)
            task_freq_arrays.append(subj_task_blocks)

        subj_freq_arrays = xr.concat(task_freq_arrays, dim='band')
        subj_freq_arrays.coords['band'] = list(freq_dict.keys())
        task_arrays.append(subj_freq_arrays)

    feedback_list = [np.asarray(task_coords[f'{t}-feedback']) for t in task_names]
    response_list = [np.asarray(task_coords[f'{t}-response']) for t in task_names]
    max_trials = max(len(fbak) for fbak in feedback_list)

    feedback_padded = np.stack([pad_array(fbak, max_trials) for fbak in feedback_list])
    response_padded = np.stack([pad_array(resp, max_trials) for resp in response_list])

    subj_task_array = xr.concat(task_arrays, dim='task')
    subj_task_array.coords['task'] = task_names

    subj_task_array = subj_task_array.assign_coords(feedback=(["task", "trial"], feedback_padded),
                                                    response=(["task", "trial"], response_padded))

    if save_array:

        save_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        subj_task_array.to_netcdf(f'{save_dir}{subj_name}_MVIS_MVEB_Epochs.nc')

def process_WM_task(subj_name, task_name, coh_map, task_trules, task_tmap, smoothing, root_dir, freq_dict={'gamma':'f50f150', 'beta':'f8f24'}, fscale=4):

    subj_coh = coh_map[subj_name]
    task_freq_arrays = []
    task_coords = {}

    for freq_name, freq_band in freq_dict.items():
        subj_data, final_ch_names, task_events, df_anat, subj_sfreq = load_subject_data(subject_name=subj_name, cohort=subj_coh, task_name=task_name, trig_rules=task_trules,
                                                                                        trig_map=task_tmap, root_dir=root_dir, bad_contacts=[], freq_band=freq_band,
                                                                                        smoothing=smoothing)

        epochs_encod = create_epochs(subj_data, final_ch_names, task_events, df_anat, tmin, tmax, event_query='time_stim.notnull()', time_coord='time_stim', sfreq=subj_sfreq)
        epochs_maint = create_epochs(subj_data, final_ch_names, task_events, df_anat, tmin, tmax, event_query='time_maint.notnull()', time_coord='time_maint', sfreq=subj_sfreq)

        epochs_encod_int = smooth_epochs(epochs_encod, freq_scale=fscale)
        epochs_maint_int = smooth_epochs(epochs_maint, freq_scale=fscale)

        subj_task_blocks = xr.concat([epochs_encod_int, epochs_maint_int], dim='event')
        subj_task_blocks.coords['event'] = ['encod', 'maint']

        if freq_name == 'beta':
            task_coords['feedback'] = subj_task_blocks.fb_label.to_numpy()
            task_coords['response'] = subj_task_blocks.resp_code.to_numpy()

        coords_to_remove = ['contacts', 'stim_code', 'maint_code', 'fb_code', 'time_stim', 'time_maint', 'time_resp',
                            'time_fb', 'rt', 'stim_label', 'maint_label', 'resp_label', 'fb_label', 'resp_code']

        subj_task_blocks = subj_task_blocks.drop_vars(coords_to_remove)
        task_freq_arrays.append(subj_task_blocks)

    subj_freq_arrays = xr.concat(task_freq_arrays, dim='band')
    subj_freq_arrays.coords['band'] = list(freq_dict.keys())

    return subj_freq_arrays, task_coords

def stack_WM_tasks(subj_name, tasks_exist, coh_map, task_trules, task_tmap, smoothing, root_dir, save_array=True, output_dir=os.getcwd(),
                         sub_dir='data/epochs/', freq_dict={'gamma': 'f50f150', 'beta': 'f8f24'}):

    task_names = tasks_exist.task.to_numpy()

    processed_tasks = {}
    processed_coords = {}

    for task_name, exists in zip(task_names, tasks_exist):
        if exists:
            try:
                task_array, task_coord_data = process_WM_task(subj_name, task_name, coh_map, task_trules, task_tmap, smoothing, root_dir, freq_dict=freq_dict)
                processed_tasks[task_name] = task_array
                processed_coords[task_name] = task_coord_data
            finally:
                pass

    if not processed_tasks:
        print(f"No task data could be processed for subject {subj_name}.")
        return

    final_arrays = []
    template_xr = next(iter(processed_tasks.values()))

    for task_name in task_names:
        if task_name in processed_tasks:
            final_arrays.append(processed_tasks[task_name])
        else:
            nan_placeholder = xr.full_like(template_xr, np.nan)
            final_arrays.append(nan_placeholder)

    feedback_list = [np.asarray(processed_coords.get(t, {}).get('feedback', [])) for t in task_names]
    response_list = [np.asarray(processed_coords.get(t, {}).get('response', [])) for t in task_names]
    max_trials = max(len(arr) for arr in feedback_list) if any(arr.size > 0 for arr in feedback_list) else 0

    feedback_padded = np.stack([pad_array(fbak, max_trials) for fbak in feedback_list])
    response_padded = np.stack([pad_array(resp, max_trials) for resp in response_list])

    subj_task_array = xr.concat(final_arrays, dim='task')
    subj_task_array.coords['task'] = task_names
    subj_task_array = subj_task_array.assign_coords(feedback=(["task", "trial"], feedback_padded), response=(["task", "trial"], response_padded))

    if save_array:
        save_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        subj_task_array.to_netcdf(f'{save_dir}{subj_name}_MVIS_MVEB_Epochs.nc')
