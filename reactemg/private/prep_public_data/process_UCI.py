import os
import glob
import pandas as pd

# Global min/max across *all* files
OLD_MIN, OLD_MAX = -0.00128, 0.00128
# Desired new range
NEW_MIN, NEW_MAX = -128, 128

def rescale_to_128(x):
    """
    Rescales value x from [OLD_MIN, OLD_MAX] to [NEW_MIN, NEW_MAX].
    """
    return NEW_MIN + ((x - OLD_MIN) * (NEW_MAX - NEW_MIN) / (OLD_MAX - OLD_MIN))

def segment_data(df, base_name, save_dir):
    """
    1) Find all transition indices i where gt changes from 0 to a non-zero label.
    2) Create boundaries: [0, transition_1, transition_2, ..., transition_n, len(df)].
    3) For each segment [boundary_i .. boundary_(i+1)):
       - Check the set of unique gt labels.
       - If it's {0, x} for exactly one x>0, and x in {2, 7}, save that segment.
         The filename is <baseName>_<gestureLabel>_<gestureName>_<segIndex>.csv
         (where gestureName is 'close' for 2, 'open' for 7)
       - Otherwise, skip it.
    """
    df.reset_index(drop=True, inplace=True)

    # 1) Find transitions 0 -> (something != 0)
    transitions = []
    for i in range(1, len(df)):
        if df.loc[i-1, "gt"] == 0 and df.loc[i, "gt"] != 0:
            transitions.append(i - 1600)

    # 2) Build boundaries
    boundaries = [0] + transitions + [len(df)]
    print("boundaries", boundaries)

    # 3) Create segments & save
    seg_count = 0
    
    # Create a mapping from label -> gesture name
    gesture_name_map = {
        2: "close",
        7: "open"
    }

    for b_idx in range(len(boundaries) - 1):
        seg_start = boundaries[b_idx]
        seg_end   = boundaries[b_idx + 1]
        segment_df = df.iloc[seg_start:seg_end].copy()

        # Check unique labels in this segment
        unique_labels = set(segment_df["gt"].unique())

        # We only allow exactly one non-zero label, plus 0
        non_zero_labels = unique_labels - {0}
        if len(non_zero_labels) == 1:
            gesture_label = list(non_zero_labels)[0]

            # Only keep if it's 2 or 7
            if gesture_label in [2, 7]:
                # Get the descriptive name ("close" or "open")
                gesture_name = gesture_name_map[gesture_label]

                out_name = f"{base_name}_{gesture_label}_{gesture_name}_{seg_count}.csv"
                out_path = os.path.join(save_dir, out_name)
                segment_df.to_csv(out_path, index=False)

                print(f"[INFO] Saved segment: {out_name} | Rows [{seg_start}:{seg_end}) | Labels={unique_labels}")
                seg_count += 1
            else:
                print(f"[INFO] Segment has gesture {gesture_label} not in [2,7], skipping.")

        # NEW: If the segment contains only 0 (no non-zero labels), save as "relax"
        elif len(non_zero_labels) == 0:
            gesture_label = 0
            gesture_name = "relax"
            out_name = f"{base_name}_{gesture_label}_{gesture_name}_{seg_count}.csv"
            out_path = os.path.join(save_dir, out_name)
            segment_df.to_csv(out_path, index=False)

            print(f"[INFO] Saved segment: {out_name} | Rows [{seg_start}:{seg_end}) | Labels={unique_labels}")
            seg_count += 1

        else:
            print(f"[INFO] Segment has multiple or no gestures: {unique_labels}, skipping.")

def convert_txt_to_csv_with_rescaling(root_dir, output_dir=None):
    """
    1) Recursively finds .txt files in 'root_dir'.
    2) Reads them as whitespace-delimited DataFrame with columns:
       [time, channel1..channel8, class].
    3) Rename channel1..8 -> emg0..7, class->gt.
    4) Convert all label 1 -> 0 (in your current code).
    5) Rescale emg0..emg7 from [-0.00128,0.00128] to [-128,128].
    6) Split into segments by transitions (0->x>0).
    7) Only keep segments if they have exactly one non-zero label,
       which must be either 2 or 7, or if they contain only 0.
       In the all-0 case, label the output filename "relax".
    """
    txt_files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)

    for txt_path in txt_files:
        try:
            df = pd.read_csv(txt_path, delim_whitespace=True)
        except Exception as e:
            print(f"[WARNING] Could not read {txt_path}: {e}")
            continue

        # Ensure columns exist
        expected_cols = ["time"] + [f"channel{i}" for i in range(1, 9)] + ["class"]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            print(f"[WARNING] {txt_path} missing columns {missing}, skipping.")
            continue

        # Rename channelX -> emgX-1, class->gt
        rename_dict = {}
        for i in range(1, 9):
            rename_dict[f"channel{i}"] = f"emg{i-1}"
        rename_dict["class"] = "gt"
        df.rename(columns=rename_dict, inplace=True)

        # If your intent is to map label 1 -> 0:
        df.loc[df["gt"] == 1, "gt"] = 0

        # Rescale each emg column
        for i in range(8):
            emg_col = f"emg{i}"
            df[emg_col] = df[emg_col].apply(rescale_to_128)

        # Decide where to store CSV segments
        if output_dir is None:
            save_dir = os.path.dirname(txt_path)
        else:
            rel_path = os.path.relpath(os.path.dirname(txt_path), root_dir)
            save_dir = os.path.join(output_dir, rel_path)
            os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(txt_path))[0]

        # Segment & save
        segment_data(df, base_name, save_dir)

if __name__ == "__main__":
    # 1) Root folder containing .txt files
    root_directory = "EMG_data_for_gestures-master"

    # 2) Output folder
    output_directory = "All_EMG_datasets/UCI"

    convert_txt_to_csv_with_rescaling(root_directory, output_dir=output_directory)
