import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------------------------------------------------------
# Task Mappings for E1, E2, E3, E4
# ---------------------------------------------------------
# - Any task label not in the dictionary for its exercise
#   will be assigned groundtruth=-1 and "_unlabeled" appended
#   to the filename.
TASK_MAPPING = {
    'E1': {
        # Fill in any known mappings if desired
        # e.g. 1: 'open', 2: 'close', etc.
        # If empty, everything in E1 becomes unlabeled (-1)
    },
    'E2': {
        5:  'open',   # -> gt=1
        6:  'close',  # -> gt=2
        9:  'open',
        10: 'open',
        11: 'open',
        12: 'open',
        17: 'close',
    },
    'E3': {
        1:  'close',
        3:  'close',  # -> gt=1
        5:  'close',  # -> gt=2
        10: 'close',
        18: 'close',
    },
    'E4': {
        # Fill in any known mappings if desired
        # If empty, everything in E4 becomes unlabeled (-1)
    }
}

def copy_db5_to_target(source_dir, target_dir):
    """
    Recursively copy the folder structure from source_dir to target_dir,
    copying ONLY .mat files. This ensures the structure is the same in
    target_dir, ready for preprocessing.
    """
    for root, dirs, files in os.walk(source_dir):
        # figure out the sub-path, e.g. s1, s2, ...
        relative_path = os.path.relpath(root, source_dir)
        dest_path = os.path.join(target_dir, relative_path)
        os.makedirs(dest_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(".mat"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dst_file)

def rotate_counterclockwise_8channels(df, start_col=0, rotation_amount=1):
    """
    Rotates 8 consecutive EMG columns in a ring by 'rotation_amount'.
    e.g. for start_col=0: (emg0->emg1, emg1->emg2, ..., emg7->emg0)
         for start_col=8: (emg8->emg9, emg9->emg10, ..., emg15->emg8)
    """
    ring_indices = list(range(start_col, start_col + 8))
    ring_cols = [f"emg{i}" for i in ring_indices]
    
    # Temporary rename so there's no conflict
    placeholder_mapping = {col: f"tmp_{col}" for col in ring_cols}
    df = df.rename(columns=placeholder_mapping)
    
    # Final rename
    final_mapping = {}
    for i in ring_indices:
        old_placeholder = f"tmp_emg{i}"
        ring_offset = i - start_col
        new_offset  = (ring_offset + rotation_amount) % 8
        new_idx     = start_col + new_offset
        final_mapping[old_placeholder] = f"emg{new_idx}"
    
    df = df.rename(columns=final_mapping)
    return df

def split_by_task_middle_zero(emg_data, restimulus):
    """
    Splits (emg_data, restimulus) into runs of tasks, using midpoint
    of zero regions between tasks. The first task starts at index=0,
    the last ends at the final index, etc.
    """
    emg_data = np.asarray(emg_data)
    restimulus = np.asarray(restimulus).flatten()
    n_samples = len(restimulus)
    
    runs = []
    i = 0
    while i < n_samples:
        current_label = restimulus[i]
        if current_label == 0:
            i += 1
            continue
        
        # Found a non-zero label
        task_label = current_label
        start_idx = i
        
        while i < n_samples and restimulus[i] == task_label:
            i += 1
        end_idx = i - 1
        runs.append((int(task_label), start_idx, end_idx))
    
    # Create segments from midpoint logic
    out_segments = {}
    for idx in range(len(runs)):
        t_label, run_start, run_end = runs[idx]
        
        # The first run starts at 0
        if idx == 0:
            run_start_for_this = 0
        else:
            _, prev_start, prev_end = runs[idx - 1]
            zero_start = prev_end + 1
            zero_end   = run_start - 1
            if zero_start <= zero_end:
                mid_zero = (zero_start + zero_end) // 2
                run_start_for_this = mid_zero + 1
            else:
                run_start_for_this = run_start
        
        # The last run ends at the final sample
        if idx == len(runs) - 1:
            run_end_for_this = n_samples - 1
        else:
            _, next_start, next_end = runs[idx + 1]
            zero_start = run_end + 1
            zero_end   = next_start - 1
            if zero_start <= zero_end:
                mid_zero = (zero_start + zero_end) // 2
                run_end_for_this = mid_zero
            else:
                run_end_for_this = run_end
        
        if run_start_for_this <= run_end_for_this:
            seg_emg  = emg_data[run_start_for_this : run_end_for_this + 1]
            seg_rest = restimulus[run_start_for_this : run_end_for_this + 1]
            out_segments.setdefault(t_label, []).append(
                np.column_stack([seg_emg, seg_rest])
            )
    
    # Concatenate partial segments
    for t in out_segments:
        out_segments[t] = np.vstack(out_segments[t])
    
    return out_segments

def transform_task_segment(df, label_str):
    """
    Convert restimulus based on label_str in {open, close}.
      - 'open'  -> non-zero => 1
      - 'close' -> non-zero => 2
      - otherwise => non-zero => -1 (unlabeled)
    Returns (df, is_unlabeled) 
    """
    is_unlabeled = False
    
    if label_str == 'open':
        df.loc[df['restimulus'] != 0, 'restimulus'] = 1
    elif label_str == 'close':
        df.loc[df['restimulus'] != 0, 'restimulus'] = 2
    else:
        df.loc[df['restimulus'] != -1, 'restimulus'] = -1
        is_unlabeled = True
    
    return df, is_unlabeled

def process_directory_for_user(user_dir_path, user_name):
    """
    Processes one user's directory, reading all .mat files for E1, E2, E3, or E4.
    For each recognized task segment, it creates TWO CSVs:
        1) [gt, emg0..emg7]   => "myo1"
        2) [gt, emg0..emg7]   => "myo2" (renamed from original emg8..emg15)
    Both are saved directly under this user's folder.
    After that, remove each .mat file so none remain in the final directory.
    """
    for filename in os.listdir(user_dir_path):
        if not filename.lower().endswith(".mat"):
            continue
        
        # Determine which exercise (E1, E2, E3, E4) from filename
        lower_f = filename.lower()
        if "e1" in lower_f:
            exercise_key = "E1"
        elif "e2" in lower_f:
            exercise_key = "E2"
        elif "e3" in lower_f:
            exercise_key = "E3"
        elif "e4" in lower_f:
            exercise_key = "E4"
        else:
            # If it doesn't match any known exercise, skip or handle as needed
            continue
        
        mat_path = os.path.join(user_dir_path, filename)
        data = loadmat(mat_path)
        
        # Adjust keys if your .mat structure differs
        emg    = data['emg'][:, :16]      
        restim = data['restimulus'].flatten()
        
        # Split into task segments
        task_segments = split_by_task_middle_zero(emg, restim)
        base_name = os.path.splitext(filename)[0]  # e.g. "S1_E2_A1"
        mapping_dict = TASK_MAPPING.get(exercise_key, {})
        
        for t_label, segment_data in task_segments.items():
            columns = [f"emg{i}" for i in range(16)] + ["restimulus"]
            df_seg = pd.DataFrame(segment_data, columns=columns)
            
            # Rotate 0..7, then 8..15
            df_seg = rotate_counterclockwise_8channels(df_seg, start_col=0, rotation_amount=1)
            df_seg = rotate_counterclockwise_8channels(df_seg, start_col=8, rotation_amount=2)
            
            # Identify how to label this movement
            label_str = mapping_dict.get(t_label, None)  # None => unlabeled
            df_transformed, is_unlabeled = transform_task_segment(df_seg, label_str)
            
            # Rename "restimulus" -> "gt" and make it the FIRST column
            df_transformed.rename(columns={'restimulus': 'gt'}, inplace=True)
            df_transformed['gt'] = df_transformed['gt'].astype(int)
            
            ordered_cols = ['gt'] + [f"emg{i}" for i in range(16)]
            df_transformed = df_transformed[ordered_cols]
            
            # 1) myo1 uses emg0..emg7 (unchanged)
            df_myo1 = df_transformed[['gt'] + [f"emg{i}" for i in range(8)]]
            
            # 2) myo2 originally emg8..emg15, but rename them to emg0..emg7
            df_myo2_temp = df_transformed[['gt'] + [f"emg{i}" for i in range(8,16)]].copy()
            rename_map = {f"emg{i}": f"emg{i-8}" for i in range(8,16)}
            df_myo2_temp.rename(columns=rename_map, inplace=True)
            df_myo2 = df_myo2_temp
            
            # Build the output filenames
            if is_unlabeled:
                out_base = f"{base_name}_task{t_label}_unlabeled"
            else:
                out_base = f"{base_name}_task{t_label}"
            
            out_path_myo1 = os.path.join(user_dir_path, f"{out_base}_myo1.csv")
            out_path_myo2 = os.path.join(user_dir_path, f"{out_base}_myo2.csv")
            
            df_myo1.to_csv(out_path_myo1, index=False)
            df_myo2.to_csv(out_path_myo2, index=False)
            
            print(f"[{user_name}] Saved: {out_path_myo1}")
            print(f"[{user_name}] Saved: {out_path_myo2}")
        
        # Remove the .mat file after processing
        os.remove(mat_path)

def process_all_users(original_dir, processed_dir):
    """
    1) Copy the entire `original_dir` structure (only .mat files) into `processed_dir`.
    2) Process all .mat files in `processed_dir` (rotations, rename restimulus->gt,
       split into two CSVs, etc.), then remove the .mat files in the process.
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Step 1: Copy all .mat files from original_dir to processed_dir
    print(f"Copying *.mat from '{original_dir}' to '{processed_dir}'...")
    copy_db5_to_target(original_dir, processed_dir)
    
    # Step 2: Process them in the new location
    print(f"Processing all users in '{processed_dir}'...")
    for user_folder in os.listdir(processed_dir):
        user_dir_path = os.path.join(processed_dir, user_folder)
        if os.path.isdir(user_dir_path):
            user_name = user_folder
            print(f"--- Processing user directory: {user_dir_path} ---")
            process_directory_for_user(user_dir_path, user_name)

if __name__ == "__main__":
    # Example usage:
    original_directory = "DB5"               # e.g. has subfolders s1, s2, ...
    processed_directory = "All_EMG_datasets/DB5"  # new "target" dir to store + process
    process_all_users(original_directory, processed_directory)
