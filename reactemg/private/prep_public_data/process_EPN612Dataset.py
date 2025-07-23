import os
import shutil
import json
import numpy as np
import pandas as pd

##############################################################################
# STEP 1: COPY AND FLATTEN trainingJSON & testingJSON INTO A SINGLE DIRECTORY
##############################################################################

def copy_epn_data(original_main_dir, epn_dataset_dir):
    """
    Copies the user folders from 'trainingJSON'/'testingJSON' into 'epn_dataset_dir',
    with each user folder renamed as 'trainingJSON_userX' or 'testingJSON_userY'.
    That way, each user has their own directory under 'epn_dataset_dir'.
    """

    os.makedirs(epn_dataset_dir, exist_ok=True)

    for subfolder in ['trainingJSON', 'testingJSON']:
        src_path = os.path.join(original_main_dir, subfolder)

        if not os.path.isdir(src_path):
            print(f"[copy_epn_data] WARNING: {src_path} not found; skipping.")
            continue

        for user_dirname in os.listdir(src_path):
            user_src_dir = os.path.join(src_path, user_dirname)
            if not os.path.isdir(user_src_dir):
                # Skip if it's not actually a user directory
                continue

            # The new folder name: e.g. "trainingJSON_user289"
            new_user_folder_name = f"{subfolder}_{user_dirname}"
            dst_user_dir = os.path.join(epn_dataset_dir, new_user_folder_name)

            print(f"[copy_epn_data] Copying user folder:\n"
                  f"   {user_src_dir}  =>  {dst_user_dir}")
            shutil.copytree(src=user_src_dir, dst=dst_user_dir, dirs_exist_ok=True)

    print("[copy_epn_data] Done copying user folders.\n")


##############################################################################
# TOP-LEVEL FUNCTION TO PARSE ONE SAMPLE                                     #
##############################################################################

def parse_sample(
    sample_data, sample_name, sample_type,
    valid_gestures, label_mapping,
    epn_dataset_dir, json_file
):
    """
    sample_data: dict with e.g. {
      'emg': { 'ch1': [...], 'ch2': [...], ... },
      'groundTruth': [...],
      'gestureName': 'open' or 'fist' or 'noGesture' or None,
      ...
    }
    sample_name: e.g. 'idx_1'
    sample_type: 'trainingSamples' or 'testingSamples'
    valid_gestures: list of strings (e.g. ["open", "fist", "noGesture"])
    label_mapping: dict mapping gestures to ints (e.g. {"open":1, "fist":2, "noGesture":0})
    epn_dataset_dir: path to the directory where we save the CSV
    json_file: name of the JSON we're parsing (for logging only)
    """

    gesture_name = sample_data.get('gestureName', None)

    # If gestureName is None => unlabeled => label=-1
    if gesture_name is None:
        gesture_name = "unlabeled"  # So we can build a filename
        label_val = -1
    else:
        if gesture_name not in valid_gestures:
            return None
        label_val = label_mapping.get(gesture_name, None)
        if label_val is None:
            # Not recognized => skip
            return None

    if gesture_name == "unlabeled":
        # fill with -1
        ch1_data = sample_data['emg']['ch1']
        ground_truth = np.full_like(ch1_data, fill_value=-1, dtype=int)
        gesture_str_for_filename = "unlabeled"
    elif gesture_name == "noGesture":
        ch1_data = sample_data['emg']['ch1']
        ground_truth = np.zeros_like(ch1_data, dtype=int)
        gesture_str_for_filename = "noGesture"
    else:
        # For other gestures, read groundTruth from JSON
        ground_truth = np.array(sample_data['groundTruth'], dtype=int)
        ground_truth[ground_truth == 1] = label_val
        gesture_str_for_filename = gesture_name


    # Now read EMG
    emg_dict = sample_data.get('emg', {})
    all_channels = []
    for i in range(1, 9):
        ch_name = f"ch{i}"
        channel_data = emg_dict.get(ch_name, [])
        all_channels.append(channel_data)
    emg_array = np.array(all_channels, dtype=np.float32).T  # shape: (T, 8)

    T = len(ground_truth)
    if emg_array.shape[0] != T:
        print(f"[parse_sample] EMG/GT length mismatch for {json_file}, {sample_name}, skipping.")
        return None

    # Build DataFrame
    df_dict = {"gt": ground_truth}
    for ch_idx in range(8):
        df_dict[f"emg{ch_idx}"] = emg_array[:, ch_idx]
    df = pd.DataFrame(df_dict)

    # CSV name: e.g. "trainingSamples_idx_1_open.csv"
    csv_filename = f"{sample_type}_{sample_name}_{gesture_str_for_filename}.csv"
    csv_path = os.path.join(epn_dataset_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    return csv_path


##############################################################################
# STEP 2: PARSE JSON => GENERATE CSV FOR OPEN / FIST / noGesture / UNLABELED
##############################################################################

def process_epn_dataset(epn_dataset_dir, valid_gestures=None):
    """
    1) Now all JSON files are in 'epn_dataset_dir', (or subdirs) named like:
        trainingJSON_user289.json or testingJSON_user42.json
    2) For each JSON file, read + parse it.
    3) For each sample (trainingSamples/testingSamples -> idx_1..N):
       - If gestureName is absent => treat as unlabeled => gt = -1
       - If gestureName is 'open' => label=1
         If gestureName is 'fist' => label=2
         If gestureName is 'noGesture' => label=0
       - groundTruth & EMG => DataFrame => columns: gt, emg0..emg7
       - Save each sample as CSV (one CSV per idx_*).
    """



    label_mapping = {
        "open": 1,
        "fist": 2,
        "noGesture": 0,
        "waveIn": 3,
        "waveOut": 4,
        "pinch":5
    }

    # ----------------------------------------------------------------------
    # ONLY CHANGE: use os.walk to find all matching JSON files recursively
    # ----------------------------------------------------------------------
    json_files = []
    for root, dirs, files in os.walk(epn_dataset_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    # Now parse each JSON file we found
    for json_file_path in json_files:
        json_file = os.path.basename(json_file_path)
        print(f"[process_epn_dataset] Parsing: {json_file}")

        # We'll treat anything that starts with "trainingJSON_" as training,
        # "testingJSON_" as testing
        if json_file.startswith("trainingJSON_"):
            subfolder_tag = "trainingJSON"
        else:
            subfolder_tag = "testingJSON"

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # We now parse data['trainingSamples'] and data['testingSamples'] if they exist
        for sample_type in ["trainingSamples", "testingSamples"]:
            if sample_type not in data:
                continue

            samples_dict = data[sample_type]  # e.g. { 'idx_1': {...}, 'idx_2': {...} }
            for sample_key, sample_data in samples_dict.items():
                parse_sample(
                    sample_data=sample_data, 
                    sample_name=sample_key, 
                    sample_type=sample_type,
                    valid_gestures=valid_gestures,
                    label_mapping=label_mapping,
                    epn_dataset_dir=os.path.dirname(json_file_path),
                    json_file=json_file
                )

        print(f"[process_epn_dataset] Done processing {json_file}\n")


##############################################################################
# MASTER PIPELINE FUNCTION                                                   #
##############################################################################

def create_epn_dataset_pipeline(original_main_dir, epn_dataset_dir):
    """
    1) Flatten data by copying all userX.json from 'trainingJSON' / 'testingJSON'
       into 'epn_dataset_dir' as 'trainingJSON_userX.json' or 'testingJSON_userX.json'.
    2) Parse each flattened JSON, generating CSV for each sample.
       - If gestureName is None => unlabeled => gt=-1
       - If 'open' => 1, 'fist' => 2, 'noGesture' => 0
       - If groundTruth missing => skip sample
       - 'gt' is first column, then emg0..emg7
       - CSVs are placed in the same epn_dataset_dir.
    """
    # Step 1: Flatten/copy
    copy_epn_data(original_main_dir, epn_dataset_dir)

    # Step 2: Process
    process_epn_dataset(epn_dataset_dir, valid_gestures=["open", "fist", "noGesture", "waveIn", "waveOut", "pinch"])


##############################################################################
# USAGE EXAMPLE                                                              #
##############################################################################

if __name__ == "__main__":
    ORIGINAL_EPN_DIR = "EMG-EPN612 Dataset"   # has subfolders trainingJSON, testingJSON
    EPN_DATASET_DIR  = "All_EMG_datasets/EMG-EPN612_Dataset"  # new folder

    create_epn_dataset_pipeline(ORIGINAL_EPN_DIR, EPN_DATASET_DIR)

