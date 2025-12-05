import pandas as pd
import numpy as np
from config import BASE_DIR, OUTPUT_DIR

#==================================================================================================
# Helper functions for calculating the mean positive, negatives, peaks and valleys
#==================================================================================================
def mean_positive(x):
    positives = x[x > 0]
    if len(x) == 0:
        return np.nan
    return positives.mean()

def mean_negative(x):
    negatives = x[x < 0]
    if len(x) == 0:
        return np.nan
    return negatives.mean()

def nan_min(x):
    negatives = x[x < 0]
    if len(x) == 0:
        return np.nan
    return negatives.min()

def nan_max(x):
    positives = x[x >0]
    if len(x) == 0:
        return np.nan
    return positives.max()  


#==================================================================================================
# Other helper functions for aggregation calculations, 
#   loading data from the CSV per L and R wheel
#   computing forces, averages, aggregation
#==================================================================================================
# Building the parameter dictionary
# side is "L" or "R" for left or right sides
def build_parameter_dictionary(side):
    xy_force_params = {
        "tangential_force":"N" , 
        "radial_force": "N", 
        "axle_force": "N",
        "moment_z": "Nm",
        "power_z": "W"
    }
    
    # build up a dictionary containing the parameters we want to calculate
    agg_dict = {}
    for col, unit in xy_force_params.items():
        col_name = f"{col}_{side}" 
        og_column = f"{col_name}[{unit}]"
        agg_dict[f"{col_name}_pos[{unit}]"] = (og_column, mean_positive)
        agg_dict[f"{col_name}_neg[{unit}]"] = (og_column, mean_negative)
        agg_dict[f"{col_name}_pos_peak[{unit}]"] = (og_column, nan_max)
        agg_dict[f"{col_name}_neg_peak[{unit}]"] = (og_column, nan_min)

    # add the total force as well
    col = "total_force"
    unit = "N"
    col_name = f"{col}_{side}" 
    og_column = f"{col_name}[{unit}]"
    agg_dict[f"{col_name}[{unit}]"] = (og_column, mean_positive)
    agg_dict[f"{col_name}_peak[{unit}]"] = (og_column, nan_max)

    return agg_dict



# load and clean the data for a given side
def load_cycle_data(path, side):
    raw_df = pd.read_csv(path)

    # only use named columns and when the hand cycle is touching
    df = raw_df.drop(columns=[c for c in raw_df.columns if "Unnamed" in c])
    df = df[df[f'theta_cop_{side}[deg]'].notna()]

    return df

# another helper function to load the cycle data based on the side the hand is touching
def load_cycle_data_from_df(raw_df, side):
    df = raw_df.drop(columns=[c for c in raw_df.columns if "Unnamed" in c])
    df = df[df[f'theta_cop_{side}[deg]'].notna()]

    return df

# Add total_force and power calculations
def compute_force_columns(df, side):
    # add the total force
    df[f'total_force_{side}[N]'] = np.sqrt(
        df[f"tangential_force_{side}[N]"]**2 +
        df[f"radial_force_{side}[N]"]**2 +
        df[f"axle_force_{side}[N]"]**2
    )

    # add the power calculation
    df[f"power_z_{side}[W]"] = df[f"gyro_z_{side}[rad/s]"]*df[f"moment_z_{side}[Nm]"]
    return df

# post-aggregation averages
def get_averages_from_l_r(df):
    # with positives and negatives
    side_cols = {
        "tangential_force":"N" , 
        "radial_force": "N", 
        "axle_force": "N",
        "moment_z": "Nm"
    }

    for col, unit in side_cols.items():
        L_pos = f"{col}_L_pos[{unit}]"
        L_neg = f"{col}_L_neg[{unit}]"

        R_pos = f"{col}_R_pos[{unit}]"
        R_neg = f"{col}_R_neg[{unit}]"

        L_pos_peak = f"{col}_L_pos_peak[{unit}]"
        L_neg_peak = f"{col}_L_neg_peak[{unit}]"     

        R_pos_peak = f"{col}_R_pos_peak[{unit}]"
        R_neg_peak = f"{col}_R_neg_peak[{unit}]"

        # averages
        df[f"{col}_avg_pos[{unit}]"] = df[[L_pos, R_pos]].fillna(0).mean(axis=1)
        df[f"{col}_avg_neg[{unit}]"] = df[[L_neg, R_neg]].fillna(0).mean(axis=1)


        df[f"{col}_avg_pos_peak[{unit}]"] = df[[L_pos_peak, R_pos_peak]].fillna(0).mean(axis=1)
        df[f"{col}_avg_neg_peak[{unit}]"] = df[[L_neg_peak, R_neg_peak]].fillna(0).mean(axis=1)


    # add on the total force, that is not split into positive and negatives
    df['total_force_avg[N]'] = df[['total_force_R[N]', 'total_force_L[N]']].fillna(0).mean(axis=1)
    df['total_force_avg_peak[N]'] = df[['total_force_R_peak[N]', 'total_force_L_peak[N]']].fillna(0).mean(axis=1)

    return df


def aggregate_per_cycle(df, side):
    agg_dict = build_parameter_dictionary(side)
    return df.groupby("cycle[count]").agg(**agg_dict)


#==================================================================================================
# Equivalent of the "__main__" function per scenario (glove material, participant)
#==================================================================================================
def run_all(material, initials, dry_run=False, base_dir=BASE_DIR, output_dir=OUTPUT_DIR):
    input_file = f"{base_dir}/{material}/{initials}25{material}.csv"
    output_file = f"{output_dir}/{material}/{initials}25{material}_per_cycle.csv"

    print(f"Processing file for {initials} with gloves {material}")    

    # left side
    df_left = load_cycle_data(input_file, "L")
    df_left = compute_force_columns(df_left, "L")
    agg_df_left = aggregate_per_cycle(df_left, "L")
    
    # # right side
    df_right = load_cycle_data(input_file, "R")
    df_right = compute_force_columns(df_right, "R")
    agg_df_right = aggregate_per_cycle(df_right, "R")
    
    
    # put it all together into one aggregate datafarme
    full_df = pd.concat([agg_df_left, agg_df_right], axis=1)

    # calculate the averages
    result_df = get_averages_from_l_r(full_df)

    # print to CSV files
    if not dry_run:
        result_df.to_csv(output_file)
        print(f"Saved → {output_file}")

    return result_df


# Helper function to run the kinetics calculations on a dataframe
def run_kinetics_calculations(df):
    # left side
    df_left = load_cycle_data_from_df(df, "L")
    df_left = compute_force_columns(df_left, "L")
    agg_df_left = aggregate_per_cycle(df_left, "L")
    
    # # right side
    df_right = load_cycle_data_from_df(df, "R")
    df_right = compute_force_columns(df_right, "R")
    agg_df_right = aggregate_per_cycle(df_right, "R")
    
    
    # put it all together into one aggregate datafarme
    full_df = pd.concat([agg_df_left, agg_df_right], axis=1)

    # calculate the averages
    result_df = get_averages_from_l_r(full_df)

    return result_df
    