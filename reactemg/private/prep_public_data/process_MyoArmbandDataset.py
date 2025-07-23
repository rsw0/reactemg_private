import os
import shutil
import numpy as np
import pandas as pd

###############################################################################
# STEP 1: COPY ONLY PreTrainingDataset/EvaluationDataset INTO EMG_DATASET_DIR #
###############################################################################

def copy_relevant_data(original_main_dir, emg_dataset_dir):
    """
    Copies ONLY the subfolders 'PreTrainingDataset' and 'EvaluationDataset'
    from 'original_main_dir' into 'emg_dataset_dir'.
    """

    # Ensure EMG_DATASET_DIR exists
    os.makedirs(emg_dataset_dir, exist_ok=True)

    relevant_subfolders = ["PreTrainingDataset", "EvaluationDataset"]
    
    for sub in relevant_subfolders:
        src_path = os.path.join(original_main_dir, sub)
        dst_path = os.path.join(emg_dataset_dir, sub)
        if os.path.isdir(src_path):
            print(f"[copy_relevant_data] Copying {src_path} --> {dst_path}")
            shutil.copytree(src=src_path, dst=dst_path, dirs_exist_ok=True)
        else:
            print(f"[copy_relevant_data] WARNING: {src_path} does not exist, skipping.")
    
    print("[copy_relevant_data] Done copying relevant subfolders.\n")


###############################################################################
# STEP 2: CLEAN DATASET (KEEP ONLY OPEN/CLOSE DAT, CONVERT TO CSV, REMOVE DAT)#
###############################################################################

def get_gesture_index(filename):
    """
    Extracts the integer index from filenames like 'classe_5.dat'.
    Returns None if format is not recognized.
    """
    base, ext = os.path.splitext(filename)
    if ext.lower() != '.dat':
        return None
    parts = base.split('_')
    if len(parts) != 2 or parts[0].lower() != 'classe':
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def convert_dat_to_csv(dat_filepath, csv_filepath, gesture_label):
    """
    Loads a .dat file (16-bit int, 8 channels) -> float32 -> (num_samples, 8).
    Places 'gt' as the FIRST column: 1=HandOpen, 2=HandClose.
    Saves to CSV.
    """
    data_int16 = np.fromfile(dat_filepath, dtype=np.int16)
    data_float32 = data_int16.astype(np.float32)

    num_samples = data_float32.shape[0] // 8
    if num_samples * 8 != data_float32.shape[0]:
        raise ValueError(f"File {dat_filepath} has a length not divisible by 8.")

    # Reshape into (num_samples, 8)
    data_2d = data_float32.reshape((num_samples, 8))

    column_names = [f"emg{i}" for i in range(8)]
    df = pd.DataFrame(data_2d, columns=column_names)

    # Insert gt as the first column
    df.insert(0, "gt", gesture_label)

    df.to_csv(csv_filepath, index=False)
    print(f"[convert_dat_to_csv] {dat_filepath} -> {csv_filepath} (gt={gesture_label})")

def clean_dataset_of_non_open_close(folder_path):
    """
    Recursively scans 'folder_path' for 'classe_i.dat'.
    - i % 7 == 5 => HandClose => label=2
    - i % 7 == 6 => HandOpen  => label=1
    - else => remove .dat

    Converts .dat -> .csv (with gt as first column), removes the .dat.
    """
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if not filename.lower().startswith("classe_") or not filename.lower().endswith(".dat"):
                continue
            
            gesture_index = get_gesture_index(filename)
            if gesture_index is None:
                continue
            
            dat_path = os.path.join(dirpath, filename)
            mod_val = gesture_index % 7
            
            if mod_val == 0:
                # Hand Close => label=2
                gesture_label = 0
            elif mod_val == 1:
                # Hand Open => label=1
                gesture_label = 1
            elif mod_val == 2:
                # Hand Open => label=1
                gesture_label = 1
            elif mod_val == 3:
                # Hand Open => label=1
                gesture_label = 1
            elif mod_val == 4:
                # Hand Open => label=1
                gesture_label = 1
            elif mod_val == 5:
                # Hand Close => label=2
                gesture_label = 2
            elif mod_val == 6:
                # Hand Open => label=1
                gesture_label = 1
            else:
                # Not open/close => remove
                os.remove(dat_path)
                print(f"[clean_dataset_of_non_open_close] Removed non-open/close: {dat_path}")
                continue

            # Convert valid .dat to .csv
            csv_filename = filename.replace(".dat", ".csv")
            csv_path = os.path.join(dirpath, csv_filename)
            try:
                convert_dat_to_csv(dat_path, csv_path, gesture_label)
                os.remove(dat_path)  # remove the original .dat
                print(f"[clean_dataset_of_non_open_close] Removed .dat after conversion: {dat_path}")
            except Exception as e:
                print(f"ERROR converting {dat_path}:\n{e}")


###############################################################################
# STEP 3: RENAME/MERGE USER FOLDERS + FLATTEN training0, Test0, Test1 STRUCT. #
###############################################################################

def rename_and_merge_folders(emg_dataset_dir):
    """
    - Moves each user folder from PreTrainingDataset or EvaluationDataset to top-level,
      renaming it from:
          PreTrainingDataset/XYZ -> PreTrainingDataset_XYZ
          EvaluationDataset/ABC  -> EvaluationDataset_ABC
    - Then flattens subfolders like 'training0', 'Test0', 'Test1':
        * Move all .csv to the user folder (rename them to subfoldername_filename).
        * Remove subfolders.
        * Remove non-CSV files.
        * Remove all experiment.csv files specifically.
    """

    main_subfolders = ["PreTrainingDataset", "EvaluationDataset"]
    for main_sub in main_subfolders:
        original_path = os.path.join(emg_dataset_dir, main_sub)
        if not os.path.isdir(original_path):
            print(f"[rename_and_merge_folders] {original_path} does not exist, skipping.")
            continue
        
        # For each user folder under (PreTrainingDataset or EvaluationDataset)
        for user_name in os.listdir(original_path):
            user_dir_path = os.path.join(original_path, user_name)
            if not os.path.isdir(user_dir_path):
                continue  # skip files

            # New name, e.g. "PreTrainingDataset_Male6"
            new_user_folder_name = f"{main_sub}_{user_name}"
            new_user_folder_path = os.path.join(emg_dataset_dir, new_user_folder_name)

            print(f"[rename_and_merge_folders] Moving {user_dir_path} -> {new_user_folder_path}")
            shutil.move(user_dir_path, new_user_folder_path)

            # ---- Flatten subfolders (training0, Test0, Test1, etc.) ----
            flatten_subfolders(new_user_folder_path)

        # Try removing the now-empty original subfolder
        try:
            os.rmdir(original_path)
            print(f"[rename_and_merge_folders] Removed empty folder: {original_path}\n")
        except OSError:
            pass  # not empty or other error


def flatten_subfolders(user_folder_path):
    """
    In the newly moved user folder (e.g. EvaluationDataset_Male6), we might have:
       experiment.csv
       training0/
       Test0/
       Test1/
       ...
    We want to:
       1) Remove any file named 'experiment.csv' in user_folder_path immediately.
       2) Move all .csv files from each subfolder => user_folder_path
          rename them to subfoldername_filename if not already.
       3) Remove all 'experiment.csv' files if found in subfolders.
       4) Remove non-CSV files from the subfolder.
       5) Remove the empty subfolder after.
    """

    # 1) Remove any top-level experiment.csv
    for item in os.listdir(user_folder_path):
        if item.lower() == "experiment.csv":
            exp_path = os.path.join(user_folder_path, item)
            print(f"[flatten_subfolders] Removing top-level experiment.csv: {exp_path}")
            os.remove(exp_path)

    # 2) Process each subfolder
    for sub in os.listdir(user_folder_path):
        subpath = os.path.join(user_folder_path, sub)
        if not os.path.isdir(subpath):
            # Already a file in user folder, skip
            continue

        # 'sub' might be training0, Test0, Test1, etc.
        # We walk through subpath to move or remove files
        for root, _, files in os.walk(subpath, topdown=False):
            for f in files:
                old_file_path = os.path.join(root, f)

                # 3) Remove if it is 'experiment.csv'
                if f.lower() == "experiment.csv":
                    print(f"[flatten_subfolders] Removing experiment.csv: {old_file_path}")
                    os.remove(old_file_path)
                    continue

                if f.lower().endswith(".csv"):
                    # Move CSV
                    if not f.startswith(sub + "_"):
                        new_file_name = f"{sub}_{f}"
                    else:
                        new_file_name = f
                    new_file_path = os.path.join(user_folder_path, new_file_name)

                    print(f"[flatten_subfolders] Moving CSV: {old_file_path} -> {new_file_path}")
                    shutil.move(old_file_path, new_file_path)
                else:
                    # Non-CSV file => remove
                    print(f"[flatten_subfolders] Removing non-CSV: {old_file_path}")
                    os.remove(old_file_path)

        # 5) Remove the now-empty subfolder
        try:
            os.rmdir(subpath)
            print(f"[flatten_subfolders] Removed empty subfolder: {subpath}")
        except OSError:
            # If not empty, ignore or handle
            pass


###############################################################################
# FINAL PIPELINE                                                               #
###############################################################################

def create_emg_dataset_pipeline(original_main_dir, emg_dataset_dir):
    """
    1) Copy only 'PreTrainingDataset' and 'EvaluationDataset' from 'original_main_dir'
       to 'emg_dataset_dir'.
    2) Clean the copied data so that only open/close remain (converted to CSV with
       gt as the FIRST column).
    3) Merge user folders (e.g. PreTrainingDataset_Male6) and flatten subfolder
       structure (training0, Test0, etc.) so that only CSV files remain at the
       user folder level.
    4) Remove all files explicitly named 'experiment.csv' in each user directory,
       do not remove anything else (besides non-CSV from subfolders).
    """
    # Step 1: Copy
    copy_relevant_data(original_main_dir, emg_dataset_dir)

    # Step 2: Clean => keep open/close .dat => CSV
    for subfolder in ["PreTrainingDataset", "EvaluationDataset"]:
        subfolder_path = os.path.join(emg_dataset_dir, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"[create_emg_dataset_pipeline] Cleaning data in: {subfolder_path}")
            clean_dataset_of_non_open_close(subfolder_path)

    # Step 3: Rename/merge and flatten subfolders
    rename_and_merge_folders(emg_dataset_dir)

    print("[create_emg_dataset_pipeline] Done! Final dataset in:", emg_dataset_dir)


###############################################################################
# USAGE EXAMPLE
###############################################################################
if __name__ == "__main__":
    ORIGINAL_DATASET_DIR = "MyoArmbandDataset"
    EMG_DATASET_DIR      = "All_EMG_datasets/MyoArmbandDataset_new"

    create_emg_dataset_pipeline(ORIGINAL_DATASET_DIR, EMG_DATASET_DIR)
