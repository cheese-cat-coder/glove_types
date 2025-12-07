import pandas as pd
import numpy as np
from utils import add_summary_rows

SIDES = ["L", "R"]


# =============================================================================
# Helper function to compute key metrics, for when the glove is still touching
# =============================================================================
def contact_angle(x):
    if len(x) == 0:
        return np.nan
    else:
        return x.dropna().max() - x.dropna().min()


"""
Calculates the contact ratio %
Contact time relative to the duration of one cycle
Params:
  x - the series data for the angle
"""
def calculate_contact_time(x):
    return 100 * x.notna().mean()


# =============================== Calculation functions for the theta_cop =========================
def get_contact_params():
    return {
        "max_speed[km/h]": ("speed_R[km/h]", "max"),
        "contact_angle_R[deg]": ("theta_cop_R[deg]", contact_angle),
        "contact_angle_L[deg]": ("theta_cop_L[deg]", contact_angle),
        "contact_time_R[deg]": ("theta_cop_R[deg]", calculate_contact_time),
        "contact_time_L[deg]": ("theta_cop_L[deg]", calculate_contact_time),
    }
    
# Helper function to compute key metrics, for when the glove is still touching
def compute_cycle_metrics(df, side):
    """
    Compute key angles for each cycle where various metrics peak or change sign.
    """
    angle_col = f"theta_cop_{side}[deg]"
    torque_col = f"moment_z_{side}[Nm]"
    tangential_force_col = f"tangential_force_{side}[N]"
    radial_force_col = f"radial_force_{side}[N]"
    axle_force_col = f"axle_force_{side}[N]"

    df_copy = df[df[angle_col].notna()] 
    def peak_torque_angle(group):
        idx = group[torque_col].idxmax()
        return group.loc[idx, angle_col]
    
    def peak_power_angle(group):
        idx = group['power_z[W]'].idxmax()
        return group.loc[idx, angle_col]
    
    def peak_tangential_force_angle(group):
        idx = group[tangential_force_col].idxmax()
        return group.loc[idx, angle_col]
    
    def angle_of_radial_force_change(group):
        vals = group[radial_force_col].values
        angles = group[angle_col].values
        for i in range(1, len(vals)):
            if vals[i-1] > 0 and vals[i] < 0:
                return angles[i]

        return None
    
    def angle_of_axle_force_change(group):
        vals = group[axle_force_col].values
        angles = group[angle_col].values
        for i in range(1, len(vals)):
            if vals[i-1] > 0 and vals[i] < 0:
                return angles[i]
        return None
    
    results = df_copy.groupby('cycle[count]').apply(lambda g: pd.Series({
        f'peak_torque_angle_{side}[Nm]': peak_torque_angle(g),
        f'peak_power_angle_{side}[W]': peak_power_angle(g),
        f'peak_tangential_force_angle_{side}[N]': peak_tangential_force_angle(g),
        f'radial_force_change_angle_{side}[N]': angle_of_radial_force_change(g),
        f'axle_force_change_angle_{side}[N]': angle_of_axle_force_change(g)
    }),
    include_groups=False)
    
    return results



# ============= A summary statistic per each person that needs to be calculated =======================
def calculate_stroke_frequency(df):
    TOTAL_CYCLES = 25

    total_time = df["time[sec]"].max() - df["time[sec]"].min()
    return TOTAL_CYCLES / total_time

# =======================================================================================

def compute_kinematics_per_file(file_path, cycle_col="cycle[count]"):
    df = pd.read_csv(file_path)
    contact_params = get_contact_params()
    kinematics_summary = df.groupby(cycle_col).agg(**contact_params)

    # add power calculation
    df["power_z[W]"] = df["gyro_z_R[rad/s]"] * df["moment_z_R[Nm]"]

    # compute the cycle metrics
    cycle_metrics_R = compute_cycle_metrics(df, "R")
    cycle_metrics_L = compute_cycle_metrics(df, "L")

    # add all the metrics together
    results = pd.concat([kinematics_summary, cycle_metrics_R, cycle_metrics_L], axis=1)

    results = add_summary_rows(results)
    return results