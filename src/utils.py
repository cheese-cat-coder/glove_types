import pandas as pd
# =======================================================================================
# Add the summary rows of the standard deviation and mean, ignoring NANs
# =======================================================================================
def add_summary_rows(df):
    """
    Add rows with column-wise mean and standard deviation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with mean and std rows appended
    """
    # Calculate mean and std for numeric columns only
    mean_row = df.mean(numeric_only=True)
    std_row = df.std(numeric_only=True)
    
    # Convert to DataFrames so we can append them
    mean_df = pd.DataFrame([mean_row], index=['Mean'])
    std_df = pd.DataFrame([std_row], index=['StdDev'])
    
    # Concatenate original df with summary rows
    df_with_summary = pd.concat([df, mean_df, std_df])
    
    return df_with_summary
