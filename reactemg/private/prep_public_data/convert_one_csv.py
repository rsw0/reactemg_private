import os
import sys
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    csv_path = args.csv_path
    csv_dir = args.csv_dir
    output_dir = args.output_dir

    # Compute relative path to preserve folder structure
    rel_path = os.path.relpath(csv_path, start=csv_dir)
    # Replace ".csv" with ".npy"
    npy_rel_path = rel_path.rsplit('.', 1)[0] + ".npy"
    # Final path to save the .npy
    npy_path = os.path.join(output_dir, npy_rel_path)

    # Ensure output subdirectory exists
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)

    # Read the CSV (no chunking)
    # Adjust 'usecols' if you have different column names
    df = pd.read_csv(
        csv_path,
        usecols=['gt','emg0','emg1','emg2','emg3','emg4','emg5','emg6','emg7'],
        dtype=np.float32
    )

    # Convert to NumPy array
    arr = df.to_numpy()

    # Save as .npy
    np.save(npy_path, arr)

    print(f"Converted {csv_path} -> {npy_path} (rows={len(df)})")

if __name__ == "__main__":
    main()
