import os
import subprocess

def main():
    csv_dir = "../all_emg_datasets"
    output_dir = "all_emg_datasets_npy"
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all subdirectories under csv_dir
    tasks = []
    for root, dirs, files in os.walk(csv_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_path = os.path.join(root, filename)
                tasks.append(csv_path)

    print(f"Found {len(tasks)} CSV files.")

    # Process each CSV in a *fresh* subprocess
    for i, csv_path in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Converting: {csv_path}")
        cmd = [
            "python", "convert_one_csv.py",
            "--csv_path", csv_path,
            "--csv_dir", csv_dir,
            "--output_dir", output_dir
        ]
        subprocess.run(cmd, check=True)

    print("All conversions complete!")

if __name__ == "__main__":
    main()
