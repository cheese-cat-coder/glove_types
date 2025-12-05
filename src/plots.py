import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cycle_detection import time_normalize_cycle

def plot_cycle_selection_analysis(
    df, results, time_col="time", moment_col="propulsion_moment", cycle_col="cycle_id"
):
    """
    Comprehensive visualization of cycle selection results
    """

    fig = plt.figure(figsize=(16, 12))

    # 1. Distance distribution - which cycles were selected?
    ax1 = plt.subplot(3, 3, 1)
    distances = results["all_mean_distances"]
    selected_idx = results["selected_idx"]

    ax1.hist(distances, bins=30, alpha=0.6, label="All cycles", color="gray")
    ax1.hist(
        distances[selected_idx],
        bins=30,
        alpha=0.8,
        label="Selected cycles",
        color="green",
    )
    ax1.axvline(
        distances[selected_idx].max(),
        color="red",
        linestyle="--",
        label=f"Selection threshold",
    )
    ax1.set_xlabel("Mean RMSD to all other cycles")
    ax1.set_ylabel("Count")
    ax1.set_title("Cycle Selection Based on Similarity")
    ax1.legend()

    # 2. All cycles overlaid (gray) with selected cycles highlighted
    ax2 = plt.subplot(3, 3, 2)
    time_norm = np.linspace(0, 100, 101)

    # Plot all cycles in gray
    for i, cycle in enumerate(results["all_cycles_matrix"]):
        if i not in selected_idx:
            ax2.plot(time_norm, cycle, color="lightgray", alpha=0.3, linewidth=0.5)

    # Plot selected cycles in color
    for cycle in results["selected_cycles_matrix"]:
        ax2.plot(time_norm, cycle, color="steelblue", alpha=0.3, linewidth=1)

    # Plot mean ± SD
    mean_curve = results["mean_curve"]
    std_curve = results["std_curve"]
    ax2.plot(time_norm, mean_curve, "r-", linewidth=2.5, label="Mean (selected)")
    ax2.fill_between(
        time_norm,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
        color="red",
        label="±1 SD",
    )

    ax2.set_xlabel("Cycle (%)")
    ax2.set_ylabel("Propulsion Moment")
    ax2.set_title("Selected vs Excluded Cycles")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Mean ± SD only (cleaner view)
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time_norm, mean_curve, "b-", linewidth=2.5, label="Mean")
    ax3.fill_between(
        time_norm,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.4,
        color="blue",
        label="±1 SD",
    )
    ax3.fill_between(
        time_norm,
        mean_curve - 2 * std_curve,
        mean_curve + 2 * std_curve,
        alpha=0.2,
        color="blue",
        label="±2 SD",
    )
    ax3.set_xlabel("Cycle (%)")
    ax3.set_ylabel("Propulsion Moment")
    ax3.set_title("Mean Steady-State Pattern")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Coefficient of variation across cycle
    ax4 = plt.subplot(3, 3, 4)
    cv = (std_curve / np.abs(mean_curve)) * 100
    ax4.plot(time_norm, cv, "purple", linewidth=2)
    ax4.axhline(
        cv.mean(), color="red", linestyle="--", label=f"Mean CV: {cv.mean():.1f}%"
    )
    ax4.set_xlabel("Cycle (%)")
    ax4.set_ylabel("Coefficient of Variation (%)")
    ax4.set_title("Variability Across Cycle Phase")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Time series view - original data with selected cycles highlighted
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(
        df[time_col], df[moment_col], "gray", alpha=0.5, linewidth=0.5, label="All data"
    )

    selected_df = df[df[cycle_col].isin(results["selected_cycle_ids"])]
    ax5.plot(
        selected_df[time_col],
        selected_df[moment_col],
        "blue",
        alpha=0.7,
        linewidth=1,
        label="Selected cycles",
    )

    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Propulsion Moment")
    ax5.set_title("Time Series: Selected vs All Cycles")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Peak moment comparison
    ax6 = plt.subplot(3, 3, 6)
    all_peaks = results["all_cycles_matrix"].max(axis=1)
    selected_peaks = results["selected_cycles_matrix"].max(axis=1)
    excluded_peaks = np.delete(all_peaks, selected_idx)

    data_to_plot = [selected_peaks, excluded_peaks]
    positions = [1, 2]
    bp = ax6.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=["Selected", "Excluded"],
    )
    bp["boxes"][0].set_facecolor("green")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("red")
    bp["boxes"][1].set_alpha(0.6)

    ax6.set_ylabel("Peak Propulsion Moment")
    ax6.set_title("Peak Moment: Selected vs Excluded")
    ax6.grid(True, alpha=0.3, axis="y")

    # 7. Cycle-by-cycle distance plot
    ax7 = plt.subplot(3, 3, 7)
    cycle_nums = np.arange(len(results["all_cycle_ids"]))
    colors = ["green" if i in selected_idx else "red" for i in range(len(distances))]
    ax7.scatter(cycle_nums, distances, c=colors, alpha=0.6, s=50)
    ax7.axhline(
        distances[selected_idx].max(),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Selection threshold",
    )
    ax7.set_xlabel("Cycle Number")
    ax7.set_ylabel("Mean RMSD")
    ax7.set_title("Similarity Score by Cycle")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Heatmap of pairwise distances (subset)
    ax8 = plt.subplot(3, 3, 8)
    # Show distances for selected cycles only
    selected_distances = cdist(
        results["selected_cycles_matrix"],
        results["selected_cycles_matrix"],
        metric="euclidean",
    )

    im = ax8.imshow(selected_distances, cmap="viridis", aspect="auto")
    ax8.set_xlabel("Cycle Index (selected)")
    ax8.set_ylabel("Cycle Index (selected)")
    ax8.set_title("Pairwise Distances\n(Selected Cycles Only)")
    plt.colorbar(im, ax=ax8, label="RMSD")

    # 9. Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    summary_text = f"""
    SUMMARY STATISTICS
    
    Total cycles detected: {len(results['all_cycle_ids'])}
    Selected cycles: {len(results['selected_cycle_ids'])}
    Excluded cycles: {len(results['excluded_cycle_ids'])}
    
    Selected cycles:
      Mean internal RMSD: {results['mean_internal_rmsd']:.2f}
      Mean peak moment: {selected_peaks.mean():.2f} ± {selected_peaks.std():.2f}
      CV of peaks: {(selected_peaks.std()/selected_peaks.mean())*100:.1f}%
    
    Excluded cycles:
      Mean peak moment: {excluded_peaks.mean():.2f} ± {excluded_peaks.std():.2f}
      CV of peaks: {(excluded_peaks.std()/excluded_peaks.mean())*100:.1f}%
    
    Mean CV across cycle: {cv.mean():.1f}%
    """

    ax9.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()
    return fig


def plot_individual_cycles_grid(
    df,
    results,
    cycle_col="cycle_id",
    time_col="time",
    moment_col="propulsion_moment",
    n_display=12,
):
    """
    Display individual selected cycles in a grid for detailed inspection
    """

    selected_ids = results["selected_cycle_ids"][:n_display]

    n_cols = 4
    n_rows = int(np.ceil(len(selected_ids) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    time_norm = np.linspace(0, 100, 101)
    mean_curve = results["mean_curve"]

    for idx, (ax, cycle_id) in enumerate(zip(axes, selected_ids)):
        cycle_data = df[df[cycle_col] == cycle_id]
        norm_cycle = time_normalize_cycle(cycle_data, time_col, moment_col)

        # Plot individual cycle
        ax.plot(time_norm, norm_cycle, "b-", linewidth=2, label="This cycle")
        ax.plot(time_norm, mean_curve, "r--", linewidth=1.5, alpha=0.7, label="Mean")

        # Calculate RMSD from mean
        rmsd = np.sqrt(((norm_cycle - mean_curve) ** 2).mean())

        ax.set_title(f"Cycle {cycle_id}\nRMSD: {rmsd:.2f}", fontsize=10)
        ax.set_xlabel("Cycle (%)", fontsize=8)
        ax.set_ylabel("Moment", fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(selected_ids), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
