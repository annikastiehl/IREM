import numpy as np
import pandas as pd
import xarray as xr
import dyca
import matplotlib.pyplot as plt

test_array = xr.load_dataarray(
    "/home/astiehl/05_Code/Promotion/marseille/IREM/seeg_epochs/HID-Sub-001_MVIS_MVEB_Epochs.nc")
# count nans and notnans in the data
nans = np.isnan(test_array).sum().item()
notnans = np.count_nonzero(~np.isnan(test_array))

print(f"Number of NaNs in the data array: {nans}")
print(f"Number of non-NaNs in the data array: {notnans}")

# maybe i can reconstruct the number of datapoints
total_points = nans + notnans
total_points2 = test_array.size
total_points3 = 256*93*2*2*2*36
print(f"Total number of data points (nans):n): {total_points}")
print(f"Total number of data points (size): {total_points2}")
print(f"Total number of data points (manual calc): {total_points3}")
# JUHU - thats the same and I understood the data structure

# Where happens the NaNs?
# Get unique combinations of task, band, event, trial and region where NaNs are located (excluding time)
nan_mask = np.isnan(test_array)
# Check if any time point has NaN for each combination of other dimensions
nan_combinations = nan_mask.any(dim='time')
nan_indices = np.argwhere(nan_combinations.values)

unique_combinations = set()
for loc in nan_indices:
    task, band, event, trial, region = loc
    combination = (task, band, event, trial, region)
    unique_combinations.add(combination)

i = 0
for task, band, event, trial, region in sorted(unique_combinations):
    print(
        f"{i}. NaN found at - Task: {test_array.task[task].item()}, Band: {test_array.band[band].item()}, Event: {test_array.event[event].item()}, Trial: {test_array.trial[trial].item()}, Region: {test_array.region[region].item()}")
    i += 1
# Okay, when a NaN is present, it is present for all timepoints in that combination of other dimensions

# Print unique values for each dimension where NaNs are located
unique_tasks = set()
unique_bands = set()
unique_events = set()
unique_trials = set()
unique_regions = set()

for task, band, event, trial, region in unique_combinations:
    unique_tasks.add(test_array.task[task].item())
    unique_bands.add(test_array.band[band].item())
    unique_events.add(test_array.event[event].item())
    unique_trials.add(test_array.trial[trial].item())
    unique_regions.add(test_array.region[region].item())

print("\nUnique values where NaNs are found:")
print(f"Tasks: {sorted(unique_tasks)}")
print(f"Bands: {sorted(unique_bands)}")
print(f"Events: {sorted(unique_events)}")
print(f"Trials: {sorted(unique_trials)}")
print(f"Regions: {sorted(unique_regions)}")
# here you can see, that all the Nans happens in the Trial 35

# delete trial 35
cleaned_array = test_array.sel(trial=test_array.trial != 35)

# count nans and notnans in the cleaned data
nans_cleaned = np.isnan(cleaned_array).sum().item()
notnans_cleaned = np.count_nonzero(~np.isnan(cleaned_array))
print(f"Number of NaNs in the cleaned data array: {nans_cleaned}")
print(f"Number of non-NaNs in the cleaned data array: {notnans_cleaned}")

# look at the regions
print(cleaned_array.region.values)
unique_regions = set(cleaned_array.region.values)
print(unique_regions)


# look at the rank of every single trial
ranks = []
for trial in cleaned_array.trial.values:
    trial_data = cleaned_array.sel(trial=trial)
    for i in range(trial_data.shape[0]):  # task
        for j in range(trial_data.shape[1]):  # band
            for k in range(trial_data.shape[2]):  # event
                # shape (93, 256)
                region_time_matrix = trial_data[i, j, k, :, :].values
                if np.isnan(region_time_matrix).all():
                    rank = 0
                else:
                    rank = np.linalg.matrix_rank(
                        np.nan_to_num(region_time_matrix))
                ranks.append((trial, cleaned_array.task[i].item(
                ), cleaned_array.band[j].item(), cleaned_array.event[k].item(), rank))

ranks_df = pd.DataFrame(
    ranks, columns=['trial', 'task', 'band', 'event', 'rank'])
print(ranks_df)

# print the histogramm of the ranks
unique, counts = np.unique(ranks_df['rank'], return_counts=True)
rank_histogram = dict(zip(unique, counts))
print("\nHistogram of ranks:")
for rank, count in sorted(rank_histogram.items()):
    print(f"Rank {rank}: {count} occurrences")


# group the data by task, band, event, feedback (so we have at the end 24 arrays)
# Get unique values for each dimension
tasks = cleaned_array.task.values
bands = cleaned_array.band.values
events = cleaned_array.event.values

print(f"Tasks: {tasks}")
print(f"Bands: {bands}")
print(f"Events: {events}")
print(f"Shape: {cleaned_array.shape}")


# Create grouped arrays for each combination
grouped_arrays = {}

for task in tasks:
    for band in bands:
        for event in events:
            group_name = f"task_{task}_band_{band}_event_{event}"

            # Select data for this specific combination
            group_data = cleaned_array.sel(
                task=task,
                band=band,
                event=event
            )

            grouped_arrays[group_name] = group_data
            print(f"Created group: {group_name}, shape: {group_data.shape}")

print(f"\nTotal number of groups created: {len(grouped_arrays)}")


# calculate mean over the trials for each group
mean_grouped_arrays_trails = {}
for group_name, group_data in grouped_arrays.items():
    mean_data = group_data.mean(dim='trial')
    mean_grouped_arrays_trails[group_name] = mean_data
    print(f"Calculated mean for group: {group_name}, shape: {mean_data.shape}")


# calculate rank for each mean group
ranked_grouped_arrays = {}
for group_name, mean_data in mean_grouped_arrays_trails.items():
    rank_of_data = np.linalg.matrix_rank(mean_data)
    ranked_grouped_arrays[group_name] = rank_of_data
    print(
        f"Calculated rank for group: {group_name}, rank: {rank_of_data}, full shape: {mean_data.shape}")


# other idea: calculate the mean of every region over time (because some regions are the same, but from different electrodes)
# then calculate the rank of this mean matrix (regions x mean over time)
mean_over_regions_grouped_arrays = {}

for group_name, group_data in mean_grouped_arrays_trails.items():
    mean_over_regions_grouped_arrays[group_name] = {}
    # falls region Koordinate ist:
    unique_regions = np.unique(group_data['region'].values)

    for unique_region in unique_regions:
        region_sel = group_data.sel(region=unique_region)
        if len(region_sel.shape) == 1:
            # extent the shape
            region_sel = np.expand_dims(region_sel, axis=0)
            # print(region_sel.shape)
        region_mean = np.mean(region_sel, axis=0)  # mean over region
        if len(region_mean.shape) == 0:
            print(region_mean.shape)
            print(region_sel.shape)
        # print(region_mean.shape)
        mean_over_regions_grouped_arrays[group_name][unique_region] = region_mean

# now calculate the rank of each group
ranked_mean_over_regions_grouped_arrays = {}
for group_name, region_dict in mean_over_regions_grouped_arrays.items():
    # shape (n_regions, n_timepoints)
    region_means_matrix = np.array(list(region_dict.values()))
    rank_of_data = np.linalg.matrix_rank(region_means_matrix)
    ranked_mean_over_regions_grouped_arrays[group_name] = rank_of_data
    print(
        f"Calculated rank for group: {group_name}, rank: {rank_of_data}, full shape: {region_means_matrix.shape}")

##### The Variable `mean_over_regions_grouped_arrays` contains now a grouped array, full rank #####

# Run dyca on the first trial of each group
dyca_results = {}
for group_name, region_dict in mean_over_regions_grouped_arrays.items():
    # shape (n_regions, n_timepoints)
    data = np.array(list(region_dict.values()))
    print(f"Running dyca on group: {group_name}, data shape: {data.shape}")

    # Run dyca
    time = np.linspace(0, 1, 256)
    dyca_result = dyca.dyca(data.T, time_index=time, m=2, n=3)
    dyca_results[group_name] = dyca_result
    print(f"dyca completed for group: {group_name}")

# plot the dyca eigenvalues for each group
rows = 2
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = np.array(axes).flatten()
i = 0
for group_name, dyca_result in dyca_results.items():
    ax = axes[i]
    ax.bar(range(len(dyca_result['generalized_eigenvalues'])),
           dyca_result['generalized_eigenvalues'])
    ax.set_title(f"{group_name}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.grid()
    i += 1

fig.suptitle("DyCA Generalized Eigenvalues for Each Group")
plt.tight_layout()
plt.savefig(f"figures/dyca_eigenvalues.png")

# plot the singular values for each group
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = np.array(axes).flatten()
i = 0
for group_name, dyca_result in dyca_results.items():
    ax = axes[i]
    ax.bar(range(len(dyca_result['singular_values'])),
           dyca_result['singular_values'])
    ax.set_title(f"{group_name}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.grid()
    i += 1
fig.suptitle("DyCA Singular Values for Each Group")
plt.tight_layout()
plt.savefig(f"figures/dyca_singular_values.png")

# plot the dyca trajectories for each group
fig = plt.figure(figsize=(5 * cols, 4 * rows))
axes = []

for i in range(rows * cols):
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    axes.append(ax)

for i, (group_name, dyca_result) in enumerate(dyca_results.items()):
    ax = axes[i]
    traj = dyca_result['amplitudes']
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_title(f"{group_name}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.grid()

fig.suptitle("DyCA Trajectories for Each Group")
plt.tight_layout()
plt.savefig("figures/dyca_trajectories.png")
