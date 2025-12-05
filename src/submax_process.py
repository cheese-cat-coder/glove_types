from cycle_detection import find_steady_state_cycles 
from kinetics_calculations import run_kinetics_calculations
# configuration of the speeds
speed_subsection = {
    "60%": {"start": 0, "end": 60},
    "70%": {"start": 60, "end": 120},
    "80%": {"start": 120, "end": 180},
}


def steady_state_cycle_calculations(df, 
                   subsection, 
                   time_col='time[sec]',
                   cycle_col='cycle[count]'):
    if subsection not in speed_subsection:
        return ValueError("Not calculated")

    time_params = speed_subsection[subsection]
    start_time = time_params["start"]
    end_time = time_params["end"]
    
    print("hello")
    # only use the time period
    sub_df = df[(start_time < df[time_col]) & (df[time_col] < end_time)]   
    results = find_steady_state_cycles(
        sub_df, 
        cycle_col='cycle[count]',  # replace with your cycle ID column
        time_col='time[sec]',    # replace with your time column
        moment_col='moment_z_total[Nm]', # replace with your propulsion moment column
    )

    
    # get the selected cycles only
    selected_cycle_ids = results["selected_cycle_ids"]    
    print(f"Selected cycle ids: {selected_cycle_ids}")
    print(f"Unique cycle ids: {len(set(selected_cycle_ids))}")

    selected_cycle_df = df[df[cycle_col].isin(selected_cycle_ids)]
    
    # calculate the kinetics on this
    kinetics_calculations = run_kinetics_calculations(selected_cycle_df)
    
    # TODO: calculate the kinematics on this too
    return {
        "kinetics": kinetics_calculations,
        "selected_cycle_ids": selected_cycle_ids
    }
    