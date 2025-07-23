import os
import sys
import pandas as pd

# Usage
# python data_integrity_check.py folder1/ folder2/ folder3/

def check_csv_files(folder_paths):
    """
    For each folder in folder_paths, recursively look for .csv files.
    For each .csv file, check:
      1) The minimum of emg0..emg7 is -128, and the maximum is 128.
      2) The gt column has only values in {0,1,2}.
    Print messages indicating whether or not each file meets these criteria.
    """
    print('Running integrity check')
    for folder_path in folder_paths:
        # Walk through all subfolders and files in the current folder_path
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith(".csv"):
                    file_path = os.path.join(root, file_name)
                    
                    # Read the CSV file into a pandas DataFrame
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        print(f"[ERROR] Could not read {file_path}. Exception: {e}")
                        continue
                    
                    # Verify required columns exist
                    required_columns = {"gt","emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7"}
                    if not required_columns.issubset(df.columns):
                        print(f"[WARNING] {file_path} is missing some required columns: {required_columns}")
                        continue
                    
                    # -------------------
                    # Check #1: EMG range
                    # -------------------
                    emg_cols = ["emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7"]
                    data_min = df[emg_cols].min().min()
                    data_max = df[emg_cols].max().max()
                    
                    # Flag for pass/fail
                    range_check_passed = (data_min >= -128) and (data_max <= 128)
                    
                    # ------------------------
                    # Check #2: Ground truth
                    # ------------------------
                    invalid_gts = df.loc[~df["gt"].isin([0,1,2,3,4,5,-1]), "gt"].unique()
                    gt_check_passed = (len(invalid_gts) == 0)
                    
                    # ------------------------
                    # Print results
                    # ------------------------
                    if range_check_passed and gt_check_passed:
                        pass
                    else:
                        # Print each failure reason
                        if not range_check_passed:
                            print(f"[FAIL] {file_path}: Expected min=-128, max=128, got min={data_min}, max={data_max}.")
                        if not gt_check_passed:
                            print(f"[FAIL] {file_path}: Found invalid gt values -> {invalid_gts}")


if __name__ == "__main__":
    """
    Usage example:
       python script_name.py folder1 folder2 ...
    This will recursively check all CSV files in folder1, folder2, ...
    """
    if len(sys.argv) < 2:
        print("Please provide at least one folder path.")
        sys.exit(1)
    
    # Collect folder paths from command-line arguments
    folders = sys.argv[1:]
    
    # Pass them to our checking function
    check_csv_files(folders)
