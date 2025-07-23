import os
import shutil
import pandas as pd

################################################################################
# CONFIG / MAPPINGS
################################################################################

# MOTION-ID to MOTION-NAME mapping
MOTION_MAP = {
    "M8":  "HC",          # Hand Closing
    "M9":  "HC & WF",     # Hand Closing + Wrist Flexion
    "M10": "HC & WE",     # Hand Closing + Wrist Extension
    "M11": "HC & RD",     # Hand Closing + Radial Deviation
    "M12": "HC & UD",     # Hand Closing + Ulnar Deviation
    "M13": "HC & FP",     # Hand Closing + Forearm Pronation
    "M14": "HC & FS",     # Hand Closing + Forearm Supination
}

################################################################################
# 1) COPY ONLY "sub*" DIRECTORIES (RECURSIVELY)
################################################################################

def copy_user_dirs_only(src_main_dir, dst_main_dir):
    """
    Recursively scans src_main_dir for directories whose names start with 'sub'
    (e.g., 'sub7', 'sub8', etc.) and copies them (and all their contents) into
    dst_main_dir, preserving the relative structure initially.

    Skips any files or directories that do not start with 'sub'.
    """
    if not os.path.exists(dst_main_dir):
        os.makedirs(dst_main_dir)

    # Walk the entire src_main_dir tree
    for root, dirs, files in os.walk(src_main_dir):
        for d in dirs:
            if d.startswith("sub"):
                # Construct full path to the source sub directory
                src_path = os.path.join(root, d)
                # Compute relative path from src_main_dir so we can preserve folder structure
                rel_path = os.path.relpath(src_path, src_main_dir)
                # Construct the corresponding destination path
                dst_path = os.path.join(dst_main_dir, rel_path)

                print(f"[INFO] Copying {src_path} -> {dst_path}")
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

################################################################################
# 2) PREPROCESS CSVs IN-PLACE
################################################################################

def rename_columns_to_emg(df):
    """
    Given a DataFrame with 8 columns but no headers,
    rename columns to emg0..emg7.
    """
    if len(df.columns) != 8:
        raise ValueError("Expected exactly 8 columns in the CSV!")
    new_col_names = [f"emg{i}" for i in range(8)]
    df.columns = new_col_names
    return df

def parse_motion_id(filename):
    """
    Given a filename like 'M9T1.csv', return 'M9'.
    Returns None if not in the expected pattern.
    """
    base, ext = os.path.splitext(filename)
    if not base.startswith('M') or ext.lower() != '.csv':
        return None
    idx_t = base.find('T')
    if idx_t < 0:
        return None
    motion_id = base[:idx_t]  # e.g. "M9"
    return motion_id

def preprocess_csvs_in_place(dst_main_dir):
    """
    1. Walk through each folder under dst_main_dir (recursively).
    2. Keep only CSVs whose motion_id M# is in MOTION_MAP.
    3. Rename columns -> emg0..emg7.
    4. Scale from [-1,1] to [-128,128].
    5. Append "_open" or "_close" to the filename.
    6. Remove files that do not match the motion map or fail to load.
    7. After all that, add a 'gt' column as the FIRST column, based on the
       heuristic described in the question.
    """
    for root, dirs, files in os.walk(dst_main_dir):
        for filename in files:
            if not filename.lower().endswith('.csv'):
                continue  # skip non-CSV

            full_path = os.path.join(root, filename)
            motion_id = parse_motion_id(filename)
            if motion_id is None or motion_id not in MOTION_MAP:
                # Not in the form M#T# or not in our keep list => remove
                os.remove(full_path)
                continue

            # If we get here, it's a CSV we want to keep. Let's load & rename columns.
            try:
                df = pd.read_csv(full_path, header=None)  # no header
                df = rename_columns_to_emg(df)
                
                # -----------------------------------------------------------------------
                # [NEW] SCALE THE EMG SIGNALS FROM [-1, 1] TO [-128, 128]
                # -----------------------------------------------------------------------
                df = df * 128
                df = df.astype(float)
                # -----------------------------------------------------------------------

            except Exception as e:
                print(f"[ERROR] Could not read/rename {full_path}: {e}")
                os.remove(full_path)
                continue

            # Determine if we add "_open" or "_close" to the filename
            motion_name = MOTION_MAP[motion_id]  # e.g. "HO", "HC", "HC & WF", ...
            if motion_name == "HO":
                suffix = "_open"
            else:
                # If it's "HC" or "HC & ???", treat as "close"
                suffix = "_close"

            base_no_ext, _ = os.path.splitext(filename)
            new_filename = base_no_ext + suffix + ".csv"
            new_full_path = os.path.join(root, new_filename)

            # -----------------------------------------------------------------------
            # [NEW] HEURISTIC TO FIND TWO TRANSITIONS AND CREATE 'gt' LABELS
            #
            #  1) From rows 0..400, find index of max change.
            #  2) From rows 700..1000, find index of max change.
            #  3) If '_open': label 0 -> 1 -> 0
            #     If '_close': label 0 -> 2 -> 0
            # -----------------------------------------------------------------------
            # First, get the row-by-row sum of absolute differences:
            df_diff = df.diff().abs().sum(axis=1)

            # Safeguards if file shorter than 1000 rows, you might want your own checks:
            max_row = len(df) - 1
            end_1 = min(350, max_row)      # handle short files
            start_2 = min(900, max_row)
            end_2 = min(1000, max_row)

            transition1 = df_diff.iloc[:end_1].idxmax()  # largest change in [0..400)
            transition2 = df_diff.iloc[start_2:end_2].idxmax()  # largest change in [700..1000)

            # Initialize gt with all zeros
            df["gt"] = 0

            if suffix == "_open":
                # rows [transition1..transition2] = 1
                df.loc[transition1:transition2, "gt"] = 1
            else:
                # '_close' => rows [transition1..transition2] = 2
                df.loc[transition1:transition2, "gt"] = 2

            # Move 'gt' to be the FIRST column
            all_cols = df.columns.tolist()
            all_cols.remove("gt")
            df = df[["gt"] + all_cols]
            # -----------------------------------------------------------------------

            # Finally, save the processed DataFrame
            df.to_csv(new_full_path, index=False)

            # Remove old file if we changed its name
            if new_filename != filename:
                os.remove(full_path)

################################################################################
# 3) FLATTEN THE STRUCTURE
################################################################################

def flatten_subdirs(dst_main_dir):
    """
    After copying subfolders (which might be nested like:
        All_EMG_datasets/SS-STM_for_MyoDatasets/data/sub7
    we want to flatten so that sub7, sub8, etc. end up directly under
    dst_main_dir (e.g. All_EMG_datasets/SS-STM_for_MyoDatasets/sub7).

    Steps:
      1. Find all 'sub*' directories below dst_main_dir (including nested).
      2. For each, compute a top-level target path (dst_main_dir/subX).
      3. If that path differs, move the entire directory up.
      4. Remove now-empty parent directories if possible.
    """
    # We'll gather (current_full_path -> top_level_destination_path) for all subX
    moves = []

    for root, dirs, files in os.walk(dst_main_dir):
        for d in dirs:
            if d.startswith("sub"):
                cur_sub_path = os.path.join(root, d)
                # The immediate top-level path for subX:
                top_level_sub_path = os.path.join(dst_main_dir, d)
                if cur_sub_path != top_level_sub_path:
                    moves.append((cur_sub_path, top_level_sub_path))

    # Sort by descending path length so we move deeper subfolders first
    moves.sort(key=lambda x: len(x[0]), reverse=True)

    for src_path, dst_path in moves:
        if os.path.exists(dst_path):
            # If there's a name collision, you may need a custom rename or handle
            print(f"[WARNING] Destination {dst_path} already exists; skipping move.")
            continue
        print(f"[FLATTEN] Moving {src_path} -> {dst_path}")
        shutil.move(src_path, dst_path)

    # Optionally, remove any empty dirs that remain, from bottom up:
    for root, dirs, files in os.walk(dst_main_dir, topdown=False):
        # don't remove the main dst_main_dir itself
        if root == dst_main_dir:
            continue
        # if empty, remove
        if not dirs and not files:
            try:
                os.rmdir(root)
                print(f"[FLATTEN] Removed empty dir: {root}")
            except OSError:
                pass

################################################################################
# 4) MAIN PIPELINE
################################################################################

def main():
    src_main_dir = "SS-STM_for_MyoDatasets"                  # <-- CHANGE if needed
    dst_main_dir = "All_EMG_datasets/SS-STM_for_MyoDatasets" # <-- CHANGE if needed

    # 1) Copy only user dirs that start with 'sub'
    copy_user_dirs_only(src_main_dir, dst_main_dir)

    # 2) Preprocess CSVs in place
    preprocess_csvs_in_place(dst_main_dir)

    # 3) Flatten so that subX directories are directly under dst_main_dir
    flatten_subdirs(dst_main_dir)

    print("[INFO] Done! Check your flattened structure in:", dst_main_dir)


if __name__ == "__main__":
    main()
