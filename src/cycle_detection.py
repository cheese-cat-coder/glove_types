import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist

def time_normalize_cycle(
    cycle_data, time_col="time", moment_col="propulsion_moment", n_points=101
):
    """Time-normalize a single cycle to 0-100%"""
    time = cycle_data[time_col].values
    moment = cycle_data[moment_col].values

    # Normalize time to 0-100% based on actual time values
    # Note - for our dataset, time is evenly spaced, but just in case it's not in the future
    time_pct = (time - time[0]) / (time[-1] - time[0]) * 100
    time_norm = np.linspace(0, 100, n_points)

    f = interpolate.interp1d(time_pct, moment, kind="cubic", fill_value="extrapolate")
    moment_norm = f(time_norm)

    return moment_norm


def find_steady_state_cycles(
    df,
    cycle_col="cycle_id",
    time_col="time",
    moment_col="propulsion_moment",
    n_select=40,
):
    """
    Find the n_select most similar cycles (steady-state propulsion)
    Excludes acceleration/deceleration cycles with anomalous torque patterns

    For this project, we want to select 40 of them per the 60-second periods
    """

    cycle_ids = df[cycle_col].unique()
    n_cycles = len(cycle_ids)

    print(f"Analyzing {n_cycles} cycles...")

    # Time-normalize all cycles to compare shapes
    normalized_cycles = []
    for cycle_id in cycle_ids:
        cycle_data = df[df[cycle_col] == cycle_id]
        try:
            norm_cycle = time_normalize_cycle(cycle_data, time_col, moment_col)
            normalized_cycles.append(norm_cycle)
        except:
            print(f"Error: Couldn't find normalized data for cycle {cycle_id}")

    cycle_matrix = np.array(normalized_cycles)  # (n_cycles, 101)

    # Calculate pairwise RMSD between all cycles
    distances = cdist(cycle_matrix, cycle_matrix, metric="euclidean")

    # Find the n_select cycles with minimum average distance to each other
    mean_distances = distances.mean(axis=1)

    # Select cycles with lowest mean distance (most similar to others)
    steady_state_idx = np.argsort(mean_distances)[:n_select]
    selected_cycle_ids = cycle_ids[steady_state_idx]

    # Statistics for the selected steady-state cycles
    selected_cycles = cycle_matrix[steady_state_idx]
    mean_curve = selected_cycles.mean(axis=0)
    std_curve = selected_cycles.std(axis=0)

    # Calculate within-cluster consistency
    selected_distances = distances[steady_state_idx][:, steady_state_idx]
    mean_internal_distance = selected_distances[
        np.triu_indices_from(selected_distances, k=1)
    ].mean()

    print(f"Selected {n_select} steady-state cycles")
    print(f"Mean internal RMSD: {mean_internal_distance:.2f}")
    print(f"Excluded {n_cycles - n_select} cycles (likely acceleration/deceleration)")

    results = {
        "selected_cycle_ids": selected_cycle_ids,
        "excluded_cycle_ids": np.setdiff1d(cycle_ids, selected_cycle_ids),
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "mean_internal_rmsd": mean_internal_distance,
        "all_mean_distances": mean_distances,
        "selected_cycles_matrix": selected_cycles,
        "all_cycles_matrix": cycle_matrix,
        "all_cycle_ids": cycle_ids,
        "selected_idx": steady_state_idx,
    }

    return results
