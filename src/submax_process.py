from cycle_detection import find_steady_state_cycles
from kinetics_calculations import run_kinetics_calculations
from plots import plot_cycle_selection_analysis
import os
import pandas as pd
import numpy as np
from utils import add_summary_rows

import seaborn as sns

from config import GLOVE_MATERIALS, PARTICIPANTS, VELOCITY_EFFORTS, SUBMAX_OUTPUT_DIR, SUBMAX_BASE_DIR
import matplotlib.pyplot as plt

# some styling for the plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

speed_subsection = {
    "60%": {"start": 0, "end": 60},
    "70%": {"start": 60, "end": 120},
    "80%": {"start": 120, "end": 180},
}

def steady_state_cycle_calculations(
    df, subsection, time_col="time[sec]", cycle_col="cycle[count]"
):
    if subsection not in speed_subsection:
        return ValueError("Not calculated")

    time_params = speed_subsection[subsection]
    start_time = time_params["start"]
    end_time = time_params["end"]

    # Get a subslice of the dataframe
    # 1. drop rows based on time constraints
    sub_df = df[(start_time < df[time_col]) & (df[time_col] < end_time)]
    # 2. drop cycles where they never touch right / left hands to the wheel
    sub_df = drop_cycles_all_missing_theta(sub_df)
    # 3. add the moment calculations, which we use for finding the steady states
    sub_df['moment_z_total[Nm]'] = sub_df['moment_z_R[Nm]'] + sub_df['moment_z_L[Nm]']
    
    results = find_steady_state_cycles(
        sub_df,
        cycle_col="cycle[count]",  # replace with your cycle ID column
        time_col="time[sec]",  # replace with your time column
        moment_col="moment_z_total[Nm]",  # replace with your propulsion moment column
    )

    # get the selected cycles only
    selected_cycle_ids = results["selected_cycle_ids"]
    print(f"Selected cycle ids: {selected_cycle_ids}")
    print(f"Unique cycle ids: {len(set(selected_cycle_ids))}")

    selected_cycle_df = df[df[cycle_col].isin(selected_cycle_ids)]

    # calculate the kinetics on this
    kinetics_calculations = run_kinetics_calculations(selected_cycle_df)
    
    kinetics_calculations = add_summary_rows(kinetics_calculations)

    # TODO: calculate the kinematics on this too
    return {"kinetics": kinetics_calculations, 
            "cycle_calculations": results,
            "cleaned_df": sub_df
            }


# helper function to drop cycles with hands not touching
def drop_cycles_all_missing_theta(df, cycle_col="cycle[count]"):
    cycles_all_missing = df.groupby(cycle_col)[
        ["theta_cop_R[deg]", "theta_cop_L[deg]"]
    ].apply(lambda x: x.isna().all().all())
    
    cycles_to_drop = cycles_all_missing[cycles_all_missing].index
    df = df[~df[cycle_col].isin(cycles_to_drop)]
    
    return df


# helper function to flip the positive/negative values for power and torque
def _clean_left_side_df(df):
    # swap out the positive and negative sided power
    df['moment_x_L[Nm]'] = -1 * df['moment_x_L[Nm]']
    df['moment_y_L[Nm]'] = -1 * df['moment_y_L[Nm]']
    df['moment_z_L[Nm]'] = -1 * df['moment_z_L[Nm]']
    return df


# File to run and get all the kinetics
def process_and_export_kinetics(base_output_dir=SUBMAX_OUTPUT_DIR):
    # Create directories
    os.makedirs(f"{base_output_dir}/data", exist_ok=True)
    os.makedirs(f"{base_output_dir}/plots", exist_ok=True)
    
    failed_computations = []
    
    for person in PARTICIPANTS:
        print(f"✨Processing person {person}...")
        
        excel_data = {}
        for material in GLOVE_MATERIALS:
            for percentile in VELOCITY_EFFORTS:
                sheet_name=f"{material}_{percentile}"
                input_file = f"{SUBMAX_BASE_DIR}/{material}/{person}_submax_{material}.csv"

                df = pd.read_csv(input_file)
                # flip the positive and negative values for the R and L sides
                df = _clean_left_side_df(df)
                
                try: 
                    kinetics_and_results = steady_state_cycle_calculations(df, f"{percentile}%")        
                    excel_data[sheet_name] = kinetics_and_results["kinetics"]
                    
                    output_plot_file =os.path.expanduser(f"{base_output_dir}/plots/{person}_{material}_{percentile}.png")
                    cleaned_df = kinetics_and_results["cleaned_df"]
                    cycle_calculation_results = kinetics_and_results["cycle_calculations"]
                    
                    plot_cycle_selection_analysis(cleaned_df, 
                                                cycle_calculation_results,
                                                time_col='time[sec]', 
                                                moment_col='moment_z_total[Nm]',
                                                cycle_col='cycle[count]')
                    plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                except ValueError as e:
                    print(f"Failed to calculate data for {person} at {percentile}%")
                    print(e)                                    
                    failed_computations.append(f"{person}_{material}_{percentile}")
                
        excel_path = f"{base_output_dir}/data/{person}_kinetics.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for sheet, data in excel_data.items():
                 # note: want to keep the cycle count data
                data.to_excel(writer, sheet_name=sheet, index=True)
                
        print(f"✅ Completed data processing for person {person} ❤️")
        
    return failed_computations
                
                
                
            
    
        
    