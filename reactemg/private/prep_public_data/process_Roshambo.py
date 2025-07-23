import os
import re
import shutil
import numpy as np

def copy_dataset(input_dir, output_dir):
    """
    Copies all files from input_dir to output_dir, including .npy files and subdirectories.
    If output_dir does not exist, it is created.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Copy everything from input_dir to output_dir (including subdirs & files).
    shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)

def remove_scissor_with_half_zeros(data):
    """
    Given a combined array 'data' of shape (N, 9), where:
      - data[:, 0] = ground truth labels (0=none, 1=paper, 2=rock, 3=scissor)
      - data[:, 1..8] = EMG channels

    Returns a boolean mask of length N indicating which rows to KEEP.

    Logic:
      - For each contiguous scissor block, remove it entirely.
      - Identify the contiguous zeros that 'bridge' this scissor block to adjacent gestures 1 or 2.
      - On each side of the scissor block:
          * Keep the half of zeros closest to the non-scissor gesture (i.e. away from the scissor),
            and remove the other half (closest to scissor).
    """

    labels = data[:, 0]
    N = len(labels)
    mask = np.ones(N, dtype=bool)  # start by keeping everything, then remove as needed

    i = 0
    while i < N:
        if labels[i] == 3:
            # Found the start of a scissor block
            sc_start = i
            while i < N and labels[i] == 3:
                i += 1
            sc_end = i - 1  # last index of scissor block

            # Remove the entire scissor block
            mask[sc_start:sc_end+1] = False

            # -------------------
            # Handle bridging zeros on the LEFT side of scissor
            # We look backward from sc_start-1 while label == 0
            left = sc_start - 1
            while left >= 0 and labels[left] == 0 and mask[left]:
                left -= 1
            # bridging zeros are indices (left+1 .. sc_start-1)
            bridging_start = left + 1
            bridging_end = sc_start - 1
            bridging_len = bridging_end - bridging_start + 1
            if bridging_len > 0:
                # Keep the 'far' half, remove the 'near' half to scissor
                # Example: if bridging_len=5, we keep the first 3 zeros, remove the last 2 zeros
                # (the ones closest to scissor).
                keep_count = (bridging_len + 1) // 2  # round up for odd lengths
                remove_start = bridging_start + keep_count
                remove_end = bridging_end
                if remove_start <= remove_end:
                    mask[remove_start:remove_end+1] = False

            # -------------------
            # Handle bridging zeros on the RIGHT side of scissor
            # i is now sc_end+1
            # So bridging zeros are (sc_end+1 .. ???) while label == 0
            right = sc_end + 1
            while right < N and labels[right] == 0 and mask[right]:
                right += 1
            bridging_start2 = sc_end + 1
            bridging_end2 = right - 1
            bridging_len2 = bridging_end2 - bridging_start2 + 1
            if bridging_len2 > 0:
                # Keep the half closest to the next gesture (the "right" half),
                # remove the half closer to scissor.
                # E.g. bridging_len2=5 => remove first 2 zeros, keep last 3 zeros.
                remove_count2 = (bridging_len2) // 2  # floor for the left half
                remove_start2 = bridging_start2
                remove_end2 = bridging_start2 + remove_count2 - 1
                if remove_start2 <= remove_end2:
                    mask[remove_start2:remove_end2+1] = False

            # Continue from 'right'
            i = right
        else:
            i += 1

    return mask

def segment_data_by_mask(data, mask):
    """
    Given data (N, 9) and a boolean mask of length N,
    returns a list of contiguous segments (each segment is (k, 9)) for all 'True' indices in the mask.

    i.e., we group consecutive True rows into separate segments.
    """
    segments = []
    N = len(mask)
    start_idx = None

    for idx in range(N):
        if mask[idx]:
            if start_idx is None:
                start_idx = idx
        else:
            # we hit a False
            if start_idx is not None:
                # end the previous segment
                segments.append(data[start_idx:idx])
                start_idx = None
    # if we ended with a True region
    if start_idx is not None:
        segments.append(data[start_idx:])

    return segments

def convert_npy_to_segmented_csv(root_dir):
    """
    1) Searches root_dir for all *_emg.npy and *_ann.npy pairs.
    2) Loads the EMG (shape=(num_samples, 8)) + annotation (shape=(num_samples,)).
    3) Maps annotations to numeric codes: 0=none, 1=paper, 2=rock, 3=scissor
    4) Combines them => shape (num_samples, 9) with [gt, emg0..emg7].
    5) Remove scissor blocks entirely, plus keep only half of bridging zeros on each side.
       Then extract contiguous segments of the remaining data (each segment gets its own CSV).
    6) Saves the segments as subjectXX_sessionYY_segZZ.csv in subject-specific folder.
    7) Deletes the original .npy files.
    """
    gesture_map = {
        'none':    0,
        'paper':   1,
        'rock':    2,
        'scissor': 3
    }

    all_files = os.listdir(root_dir)

    for file_name in all_files:
        if file_name.endswith("_emg.npy"):
            # Example: "subject01_session01_emg.npy"
            match = re.match(r"(subject\d+)_session(\d+)_emg.npy", file_name)
            if not match:
                continue

            subject_str = match.group(1)  # e.g. "subject01"
            session_str = match.group(2)  # e.g. "01"

            # Build the matching annotation file
            ann_file_name = f"{subject_str}_session{session_str}_ann.npy"
            if ann_file_name not in all_files:
                print(f"[WARN] No matching annotation for {file_name}. Skipping.")
                continue

            emg_path = os.path.join(root_dir, file_name)
            ann_path = os.path.join(root_dir, ann_file_name)

            # Load data
            emg_data = np.load(emg_path)  # shape: (num_samples, 8)
            ann_data = np.load(ann_path)  # shape: (num_samples,)

            # Normalize annotation to string
            ann_str_list = []
            for g in ann_data:
                if isinstance(g, bytes):
                    ann_str_list.append(g.decode('utf-8'))
                else:
                    ann_str_list.append(g)

            # Map to numeric
            try:
                ann_numeric = np.array([gesture_map[label] for label in ann_str_list], dtype=float)
            except KeyError as e:
                print(f"[ERROR] Unknown label '{e.args[0]}' in '{ann_file_name}'. Skipping file.")
                continue

            # Combine: shape (num_samples, 9)
            combined_data = np.column_stack([ann_numeric, emg_data])

            # Remove scissor + half bridging zeros
            mask = remove_scissor_with_half_zeros(combined_data)

            # Extract contiguous segments from the kept rows
            segments = segment_data_by_mask(combined_data, mask)

            # Prepare subject subfolder
            subject_dir = os.path.join(root_dir, subject_str)
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)

            # Save each segment
            if not segments:
                print(f"[INFO] All data for {subject_str}_session{session_str} was removed (scissor or bridging).")
            for idx, seg in enumerate(segments, start=1):
                csv_filename = f"{subject_str}_session{session_str}_seg{idx:02d}.csv"
                csv_path = os.path.join(subject_dir, csv_filename)
                header = "gt,emg0,emg1,emg2,emg3,emg4,emg5,emg6,emg7"
                np.savetxt(csv_path, seg, delimiter=",", header=header, comments="", fmt="%.4f")
                print(f"[INFO] Saved: {csv_path} (shape={seg.shape}).")

            # Delete the original .npy files
            os.remove(emg_path)
            os.remove(ann_path)
            print(f"[INFO] Deleted {emg_path} and {ann_path}.")

def main():
    """
    By default:
      1) Copies data from 'Roshambo' to 'All_EMG_datasets/Roshambo'.
      2) Converts the .npy files => CSV segments that remove scissor entirely,
         plus half the bridging zeros on each side of scissor.
      3) Deletes the .npy files from the output directory.
    """
    input_dir = "Roshambo"
    output_dir = "All_EMG_datasets/Roshambo"

    print(f"[INFO] Copying data from '{input_dir}' to '{output_dir}'...")
    copy_dataset(input_dir, output_dir)
    print("[INFO] Finished copying.")

    print("[INFO] Converting .npy files to segmented CSV (scissor removed, half bridging zeros removed)...")
    convert_npy_to_segmented_csv(output_dir)
    print("[INFO] Done!")
    
if __name__ == "__main__":
    main()

