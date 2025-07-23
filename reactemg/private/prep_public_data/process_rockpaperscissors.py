import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def copy_dataset_into_dir(original_dataset_dir, target_dataset_dir):
    """
    1) Looks inside original_dataset_dir for a folder named 'dataset'.
       If not found, raises an error.
    2) Copies everything from 'dataset/' into target_dataset_dir, skipping:
       - .github folder
       - README.md
    3) The result is that the subfolders of 'dataset/' appear directly
       at the top level of target_dataset_dir (no extra 'dataset' folder).
    """
    dataset_subdir = os.path.join(original_dataset_dir, "dataset")
    if not os.path.isdir(dataset_subdir):
        raise FileNotFoundError(f"No 'dataset' folder found under {original_dataset_dir}")

    os.makedirs(target_dataset_dir, exist_ok=True)

    print(f"[copy_dataset_into_dir] Copying {dataset_subdir} -> {target_dataset_dir}")
    for root, dirs, files in os.walk(dataset_subdir):
        # Remove .github from the list of subdirs to walk
        dirs[:] = [d for d in dirs if d.lower() != '.github']

        # Relative path from .../dataset
        rel_path = os.path.relpath(root, dataset_subdir)
        target_subdir_path = os.path.join(target_dataset_dir, rel_path)

        # Make the subdir if needed
        os.makedirs(target_subdir_path, exist_ok=True)

        for f in files:
            # Skip README.md
            if f.lower() == "readme.md":
                continue

            src_file_path = os.path.join(root, f)
            dst_file_path = os.path.join(target_subdir_path, f)
            shutil.copy2(src_file_path, dst_file_path)

    print("[copy_dataset_into_dir] Done copying.\n")


def flatten_users_and_remove_irrelevant(dataset_dir):
    """
    1) For each folder like 's7_r_2', splits on underscore => user_id='s7'.
       Moves all CSVs into a single folder named 's7'.
    2) Removes 'orientation.csv', 'pose.csv', and '.yaml' files along the way.
    3) Removes empty folders once files are moved.
    """
    entries = os.listdir(dataset_dir)
    for entry in entries:
        full_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(full_path):
            continue  # skip files at top level

        # Example: entry='s7_r_2' => user_id='s7'
        user_id = entry.split('_')[0]
        user_folder = os.path.join(dataset_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)

        # Move all relevant files from s7_r_2 => s7
        for root, dirs, files in os.walk(full_path, topdown=False):
            for f in files:
                # Remove orientation.csv, pose.csv, and .yaml
                if (f.lower().endswith('orientation.csv')
                    or f.lower().endswith('pose.csv')
                    or f.lower().endswith('.yaml')):
                    to_remove = os.path.join(root, f)
                    print(f"[flatten_users] Removing irrelevant file: {to_remove}")
                    os.remove(to_remove)
                    continue

                src_file_path = os.path.join(root, f)
                dst_file_path = os.path.join(user_folder, f)
                print(f"[flatten_users] Moving {src_file_path} -> {dst_file_path}")
                shutil.move(src_file_path, dst_file_path)

            # Remove empty subfolders
            for d in dirs:
                subd_path = os.path.join(root, d)
                try:
                    os.rmdir(subd_path)
                except OSError:
                    pass

        # Finally remove s7_r_2 if empty
        try:
            os.rmdir(full_path)
            print(f"[flatten_users] Removed empty folder: {full_path}")
        except OSError:
            pass


def preprocess_csv_files(dataset_dir):
    """
    Recursively:
      - Renames columns [0..7] => [emg0..emg7] if the CSV has [index, timestamp, 0..7].
      - If 'scissors' in filename => add gt=-1, rename => *_unlabeled.csv.
      - For 'paper' or 'rock' CSVs => remove all non-emg0..emg7 columns, 
        find the steepest change among those 8 signals *within the first 150 timesteps*, 
        label all rows up to (and including) that transition as gt=0, 
        and the rest as gt=1 (paper) or gt=2 (rock).
      - Overwrite the CSV with the updated content, unless it's scissors => rename to *_unlabeled.csv.
    """
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith('.csv'):
                continue

            csv_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[preprocess_csv_files] Could not read {csv_path}: {e}")
                continue

            # Check if it has [index, timestamp, 0..7]
            expected_cols = ['index', 'timestamp'] + [str(i) for i in range(8)]
            if not all(col in df.columns for col in expected_cols):
                # If it doesn't contain the expected columns, skip
                continue

            # Rename 0..7 => emg0..emg7
            rename_map = {str(i): f"emg{i}" for i in range(8)}
            df.rename(columns=rename_map, inplace=True)

            # Remove any column not among emg0..emg7
            emg_columns = [f"emg{i}" for i in range(8)]
            df = df[[col for col in df.columns if col in emg_columns]]

            # --- If SCISSORS, add gt=-1, rename => *_unlabeled.csv ---
            if 'scissors' in filename.lower():
                df['gt'] = -1
                # Move 'gt' to the first column
                df = df[['gt'] + [c for c in df.columns if c != 'gt']]

                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_unlabeled{ext}"
                new_path = os.path.join(root, new_filename)

                df.to_csv(new_path, index=False)
                os.remove(csv_path)
                print(f"[preprocess_csv_files] SCISSORS => {filename} -> {new_filename}")

            else:
                # --- Not scissors => we do the new 'paper/rock' labeling ---

                # 1) If it's paper or rock => find the steepest change in the first 150 timesteps
                if 'paper' in filename.lower() or 'rock' in filename.lower():
                    # Create a gt column (default 0)
                    df['gt'] = 0

                    # We only measure the transition within the first 150 rows
                    limit = min(len(df), 60)

                    # Compute sum of abs differences across all EMG channels,
                    # from row i to row i-1, for i in [1..limit-1]
                    diffs = []
                    for i in range(1, limit):
                        row_diff = sum(
                            abs(df.loc[i, col] - df.loc[i-1, col])
                            for col in emg_columns
                        )
                        diffs.append(row_diff)
                    
                    if len(diffs) > 0:
                        # Index in diffs that has the maximum difference
                        transition_sub_idx = np.argmax(diffs)
                        # That corresponds to df row = transition_sub_idx + 1
                        transition_idx = transition_sub_idx + 1
                    else:
                        # If there's only 1 row or no data, or limit=1 => no real "transition"
                        transition_idx = 0

                    # Determine final label
                    if 'paper' in filename.lower():
                        final_label = 1
                    else:  # rock
                        final_label = 2

                    # Label everything BEFORE (and including) transition_idx as 0
                    # everything AFTER as final_label
                    df.loc[:transition_idx, 'gt'] = 0
                    df.loc[transition_idx+1:, 'gt'] = final_label

                # Move 'gt' to the first column before saving
                if 'gt' in df.columns:
                    df = df[['gt'] + [c for c in df.columns if c != 'gt']]

                # Overwrite the original CSV in place (for non-scissors)
                df.to_csv(csv_path, index=False)
                print(f"[preprocess_csv_files] Processed (non-scissors) CSV: {filename}")


def run_full_preprocess_pipeline(src_dir, dst_dir):
    """
    1) Copy everything from 'src_dir/dataset' into 'dst_dir', skipping .github & README.md.
    2) Flatten so each user is in a single folder. Remove orientation/pose CSVs, and .yaml files.
    3) Rename columns [0..7] => [emg0..emg7].
    4) If 'scissors' => add gt=-1, rename => *_unlabeled.csv.
    5) For 'paper'/'rock' => remove non-EMG columns, find steepest EMG change
       in the first 150 timesteps => define transition => label gt.
    """
    copy_dataset_into_dir(src_dir, dst_dir)
    flatten_users_and_remove_irrelevant(dst_dir)
    preprocess_csv_files(dst_dir)
    print("[run_full_preprocess_pipeline] Finished!")


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Point this to your "myo-dataset" folder, which contains "dataset/" at the top level.
    original_dir = "myo-dataset"
    # Where to place the cleaned/flattened dataset:
    dataset_dir  = "All_EMG_datasets/RPS"

    run_full_preprocess_pipeline(original_dir, dataset_dir)
