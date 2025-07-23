import os
import shutil
import pandas as pd

def copy_dataset(src_dir, dst_dir):
    """
    Recursively copy from src_dir to dst_dir, maintaining the structure.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        # Compute the destination path:
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel_path)

        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)


def flatten_and_preprocess(dst_dir):
    """
    1. Walk through the copied dataset in dst_dir.
    2. For each user folder (e.g., "User 7"), look into subfolders:
       - Move all .csv files up to the user folder level.
       - Remove subfolders if empty.
    3. Preprocess each CSV:
       - Rename emg1..emg8 => emg0..emg7.
       - Remove all columns except emg0..emg7 (+ gt if it exists).
       - If the CSV is from "Index Finger Extension" or "Middle Finger Extension" subfolders,
         then add/overwrite gt=-1 for all rows, place 'gt' first, and rename the file to *_unlabeled.csv.
       - NEW REQUIREMENT: If it belongs to "closed" or "cylindrical" (detected by filename)
         and doesn't have a 'gt' column, remove the file entirely.
    """

    # 1) Identify all user folders (top-level subdirs in dst_dir).
    for user_name in os.listdir(dst_dir):
        user_path = os.path.join(dst_dir, user_name)
        if not os.path.isdir(user_path):
            continue  # skip files at top level

        # 2) Recursively gather all CSV files in subfolders and move them to user_path
        csv_files_info = []  # list of (csv_full_path, subfolder_relative_path)

        for root, dirs, files in os.walk(user_path, topdown=False):
            for f in files:
                if f.lower().endswith('.csv'):
                    csv_full_path = os.path.join(root, f)
                    # Relative path from the user folder
                    rel_path = os.path.relpath(root, user_path)
                    csv_files_info.append((csv_full_path, rel_path))

        # Move them up to user_path
        for csv_full_path, rel_subdir in csv_files_info:
            # final name (same filename, new location):
            final_csv_path = os.path.join(user_path, os.path.basename(csv_full_path))
            shutil.move(csv_full_path, final_csv_path)

        # Remove empty subfolders (after moving CSVs up)
        for root, dirs, files in os.walk(user_path, topdown=False):
            if root == user_path:
                continue  # don't remove the user folder itself
            try:
                os.rmdir(root)  # remove if empty
            except OSError:
                pass

    # 3) Now, preprocess each CSV in each user folder at the top level
    for user_name in os.listdir(dst_dir):
        user_path = os.path.join(dst_dir, user_name)
        if not os.path.isdir(user_path):
            continue

        for f in os.listdir(user_path):
            if not f.lower().endswith('.csv'):
                continue

            csv_path = os.path.join(user_path, f)

            # Detect if file name suggests "Index/Middle" or "Closed/Cylindrical"
            lower_f = f.lower()
            is_rest  = ('rest' in lower_f)
            is_index_or_middle = ('index' in lower_f or 'middle' in lower_f)
            is_closed_or_cyl = ('closed' in lower_f or 'cylind' in lower_f)

            # ---- READ CSV ----
            df = pd.read_csv(csv_path)

            # ---------------- NEW REQUIREMENT ----------------
            # If file is from "closed" or "cylindrical" and does NOT have a 'gt' column, remove it.
            if is_closed_or_cyl and 'gt' not in [c.lower() for c in df.columns]:
                print(f"[INFO] Removing '{csv_path}' because it's closed/cylindrical with no GT.")
                os.remove(csv_path)
                continue  # skip further processing of this file
            # -------------------------------------------------

            # rename emg1..emg8 => emg0..emg7
            rename_map = {f"emg{i}": f"emg{i-1}" for i in range(1, 9)}
            new_column_names = {}
            for col in df.columns:
                lower_col = col.lower().strip()
                if lower_col == 'slno':
                    # Drop it
                    continue
                elif lower_col == 'time':
                    # Drop it
                    continue
                elif lower_col.startswith('emg'):
                    # e.g. "emg1" => "emg0", etc.
                    old_idx_str = lower_col.replace('emg', '')
                    try:
                        old_idx = int(old_idx_str)
                        new_column_names[col] = f"emg{old_idx - 1}"  # shift down by 1
                    except:
                        pass
                elif lower_col == 'gt':
                    new_column_names[col] = 'gt'
                # otherwise ignore

            df = df.rename(columns=new_column_names)

            # Keep only {emg0..emg7, gt}
            keep_cols = set(new_column_names.values())
            valid_emg = [f"emg{i}" for i in range(8)]
            final_cols = []
            if 'gt' in keep_cols:
                final_cols.append('gt')
            final_cols += [c for c in valid_emg if c in keep_cols]

            df = df[final_cols]

            # If CSV is from "Index Finger Extension" or "Middle Finger Extension",
            # add or overwrite gt=-1, rename file to *unlabeled.csv
            renamed_filename = f
            if is_index_or_middle:
                df['gt'] = -1
                # Put gt first
                col_order = ['gt'] + [c for c in df.columns if c != 'gt']
                df = df[col_order]

                base, ext = os.path.splitext(f)
                if not base.endswith('_unlabeled'):
                    renamed_filename = base + '_unlabeled' + ext
            if is_rest:
                df['gt'] = 0
                # Put gt first
                col_order = ['gt'] + [c for c in df.columns if c != 'gt']
                df = df[col_order]

            # Save final CSV
            final_csv_path = os.path.join(user_path, renamed_filename)
            df.to_csv(final_csv_path, index=False)

            # If the file was renamed, remove old file
            if renamed_filename != f:
                os.remove(csv_path)

    print("[INFO] Flattening & Preprocessing complete.")


def run_pipeline(src_dir, dst_dir):
    """
    High-level runner:
      1) Copy dataset from src_dir to dst_dir.
      2) Flatten & preprocess the CSV files in dst_dir.
    """
    print(f"[INFO] Copying dataset from '{src_dir}' to '{dst_dir}' ...")
    copy_dataset(src_dir, dst_dir)
    print("[INFO] Starting flatten + preprocess ...")
    flatten_and_preprocess(dst_dir)
    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    # Example usage:
    source_dir = "d4y7fm3g79-1"
    destination_dir = "All_EMG_datasets/Mendeley"

    run_pipeline(source_dir, destination_dir)
