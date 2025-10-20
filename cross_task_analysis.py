# Cross task analysis within the 7 cognitive tasks
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import dyca


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

# rank of the data
data_rank = np.linalg.matrix_rank(raw_data)
print(f"Data rank: {data_rank} out of {raw_data.shape[1]} features")

# apply DyCA
m = 2
n = 3
dyca_result = dyca.dyca(raw_data, m=m, n=n)
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
trajectory = dyca_result['amplitudes'].T
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'gray')
ax.set_xlabel('DyCA Component 1')
ax.set_ylabel('DyCA Component 2')
ax.set_zlabel('DyCA Component 3')
ax.set_title(f'DyCA Trajectory for {selected_task} Task')
plt.show()
plt.savefig(f'figures/cross_task/dyca_trajectory_{selected_task}.png')

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


print(1)
