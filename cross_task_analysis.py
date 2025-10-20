# Cross task analysis within the 7 cognitive tasks
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import dyca
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Set up paths
data_dir = "/media/annika/Daten/Promotion/18_Marseille/03_Data/7tasks_raw/"
tasks = ["AUDI",  "LEC1", "LEC2", "MCSE", "MVEB", "MVIS", "REST", "VISU"]

selected_task = "MVIS"

# Load data for the selected task
print(os.path.join(data_dir, f"{selected_task}/**.npz"))
task_files = glob.glob(os.path.join(data_dir, f"{selected_task}/**.npz"), recursive=True)

print(f"Found {len(task_files)} files for task {selected_task}")

# Only read the first file for demonstration
task_file = task_files[:1]
data = np.load(task_file[0], allow_pickle=True)
print(data.files)
print(data['data'].shape)

raw_data = data['data']  # shape (n_samples, n_features)

# removing the mean from the data
raw_data = raw_data - np.mean(raw_data, axis=0)

# rank of the data
data_rank = np.linalg.matrix_rank(raw_data)
print(f"Data rank: {data_rank} out of {raw_data.shape[1]} features")

# apply DyCA
m = 2
n = 3
time_vector = np.linspace(0, raw_data.shape[0] / 64, raw_data.shape[0])  # assuming 64 Hz sampling rate
dyca_result = dyca.dyca(raw_data, m=m, n=n, time_index=time_vector)
reconstructed_signal = dyca.reconstruction(raw_data.T, dyca_result['amplitudes']) 

# plot the results
# eigenvalues in barplot
eigenvalues = dyca_result['generalized_eigenvalues']
plt.figure(figsize=(8, 5))
sns.barplot(x=np.arange(1, len(eigenvalues) + 1), y=eigenvalues)
plt.xlabel('Component')
plt.ylabel('Generalized Eigenvalue')
plt.title(f'DyCA Generalized Eigenvalues for {selected_task} Task')
plt.show()
plt.savefig(f'figures/cross_task/dyca_eigenvalues_{selected_task}.png')

# plot the singular values as barplot
singular_values = dyca_result['singular_values']
plt.figure(figsize=(8, 5))
sns.barplot(x=np.arange(1, len(singular_values) + 1), y=singular_values)
plt.xlabel('Component')
plt.ylabel('Singular Value')
plt.title(f'DyCA Singular Values for {selected_task} Task')
plt.show()
plt.savefig(f'figures/cross_task/dyca_singular_values_{selected_task}.png')

# plot the trajectory of the first three DyCA components as 3d plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    trajectory = dyca_result['amplitudes'].T  # shape (n_samples, n_components)

    # Normalize time for color mapping
    t_norm = (time_vector - time_vector.min()) / (time_vector.max() - time_vector.min())

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Build line segments for 3D coloring
    points = trajectory[:, :3]  # (N, 3)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # shape (N-1, 2, 3)

    lc = Line3DCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(t_norm)
    lc.set_linewidth(2)

    ax.add_collection3d(lc)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    ax.set_xlabel('DyCA Component 1')
    ax.set_ylabel('DyCA Component 2')
    ax.set_zlabel('DyCA Component 3')
    ax.set_title(f'DyCA 3D Trajectory for {selected_task} Task')

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, label='Time')

    plt.tight_layout()
    plt.savefig(f'figures/cross_task/dyca_trajectory_{selected_task}.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"Could not plot 3D trajectory: {e}\nPlot 2D trajectory instead.")

    trajectory = dyca_result['amplitudes'].T
    t_norm = (time_vector - time_vector.min()) / (time_vector.max() - time_vector.min())

    fig, ax = plt.subplots(figsize=(8, 6))

    # Build line segments for 2D coloring
    points = trajectory[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(t_norm)
    lc.set_linewidth(2)

    ax.add_collection(lc)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())

    ax.set_xlabel('DyCA Component 1')
    ax.set_ylabel('DyCA Component 2')
    ax.set_title(f'DyCA 2D Trajectory for {selected_task} Task')

    cbar = plt.colorbar(lc, ax=ax, label='Time')
    plt.tight_layout()
    plt.savefig(f'figures/cross_task/dyca_2d_trajectory_{selected_task}.png', dpi=300)
    plt.show()

    

# plot the timeseries vs reconstruction of the signal 
random_channels = np.random.choice(raw_data.shape[1], size=10, replace=False)
cutted_length = min(2560, raw_data.shape[0])
reconstruction = reconstructed_signal['reconstruction'].T
plt.figure(figsize=(15, 10))
for i, ch in enumerate(random_channels):
    plt.subplot(5, 2, i + 1)
    plt.plot(raw_data[:cutted_length, ch], label='Original Signal', alpha=0.7)
    plt.plot(reconstruction[:cutted_length, ch], label='Reconstructed Signal', alpha=0.7)
    plt.title(f'Channel {ch}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f'figures/cross_task/dyca_reconstruction_{selected_task}.png')


# ------------- moving-window DyCA: top-3 eigenvalues over time -----------------
import math
from tqdm import tqdm  # optional, nice progress bar if installed

data = np.load(task_file[0], allow_pickle=True)
print(data.files)
print(data['data'].shape)

raw_data = data['data']  # shape (n_samples, n_features)

# Parameters
fs = 64                      # sampling rate (Hz) -- you already assumed this
window_sec = 2               # window length in seconds
hop_sec = 2                  # hop length in seconds (overlap = window - hop)
window_len = int(window_sec * fs)   # e.g. 320
hop = int(hop_sec * fs)             # e.g. 64

# sanity checks
if raw_data.shape[0] < window_len:
    raise ValueError(f"raw_data too short ({raw_data.shape[0]} samples) for window length {window_len}")

# compute window centers for plotting
start_idxs = np.arange(0, raw_data.shape[0] - window_len + 1, hop)
n_windows = len(start_idxs)
centers = start_idxs + window_len // 2
time_centers = centers / fs   # in seconds

# storage for top-3 eigenvalues per window
topk = 5
eigs_top = np.full((n_windows, topk), np.nan)

print(f"Running DyCA on {n_windows} windows (window={window_len} samples, hop={hop} samples)")

# iterate windows (use tqdm if available for progress)
iterator = tqdm(enumerate(start_idxs), total=n_windows) if 'tqdm' in globals() else enumerate(start_idxs)
for wi, s in iterator:
    win = raw_data[s : s + window_len, :]    # shape (window_len, n_features)
    # optional: remove mean per channel inside window to help stability
    win = win - np.mean(win, axis=0)

    # build a short time vector for dyca call (relative time)
    tv_win = np.linspace(s / fs, (s + window_len - 1) / fs, window_len)

    try:
        # call dyca for the window
        # keep m,n same as before; you can tune these if needed
        dyca_res_win = dyca.dyca(win, m=m, n=n, time_index=tv_win)

        # generalized eigenvalues
        gev = np.asarray(dyca_res_win.get('generalized_eigenvalues', []))

        # sort descending and keep top-k
        if gev.size > 0:
            gev_sorted = np.sort(gev)[::-1]
            # print(f"Window {wi+1}/{n_windows}: top eigenvalues: {gev_sorted[:topk]}")
            take = min(topk, gev_sorted.size)
            eigs_top[wi, :take] = gev_sorted[:take]
            # check if any of the top-k are > 1 (should not happen)
            if np.any(gev_sorted[:take] > 1.0):
                print(f"Warning: Window {wi+1}/{n_windows} has eigenvalues > 1: {gev_sorted[:take]}")

    except Exception as e:
        # if dyca fails on a window, leave NaNs and print once (or collect logs)
        if wi == 0:
            print(f"Warning: DyCA failed for a window. Error: {e}")
        continue

# Convert to seconds (time_centers) if you want the x-axis
t_plot = time_centers

# Plot top-3 eigenvalues over time
plt.figure(figsize=(12, 5))
labels = [f"Eig {i+1}" for i in range(topk)]
colors = sns.color_palette("tab10", n_colors=topk)

for k in range(topk):
    plt.plot(t_plot, eigs_top[:, k], label=labels[k], color=colors[k], marker='o', markersize=3, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Eigenvalue (top 3)")
plt.title(f"Top-3 DyCA Generalized Eigenvalues over time ({selected_task}, window={window_sec}s, hop={hop_sec}s)")
plt.legend()
# plt.ylim([0.5, 1.05])
plt.grid(alpha=0.3)
plt.tight_layout()

out_fn = f"figures/cross_task/dyca_top3_eigenvalues_over_time_{selected_task}.png"
os.makedirs(os.path.dirname(out_fn), exist_ok=True)
plt.savefig(out_fn, dpi=300)
plt.show()

print(f"Saved top-3 eigenvalue plot to: {out_fn}")
# ------------- end moving-window code -----------------


print(1)


