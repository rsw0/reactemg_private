"""
Compute coactivation profiles for ROAM-EMG dataset using Kendall's tau-b correlation.

For each subject in data/ROAM_EMG/, computes coactivation profiles for:
- open (label=1)
- close (label=2)
- relax (label=0)

Processes files: *_static_reaching.csv, *_static_unsupported.csv,
                *_static_hanging.csv, *_static_resting.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Import coactivation functions
# Add the parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))
from coactivation_kendall import (
    compute_profile_from_window,
    log_euclidean_spd_mean,
    mean_then_renormalize,
)

# Configuration
DATA_ROOT = Path("/home/rsw1/Workspace/reactemg_private/data/ROAM_EMG")
OUTPUT_DIR = Path("coactivation_profiles_output_roam_kendall")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "per_subject_profiles").mkdir(exist_ok=True)

# Sampling rate for ROAM dataset
FS = 200.0

# Gesture classes - mapping label to gesture name
GESTURES = {
    0: "relax",
    1: "open",
    2: "close"
}

# File patterns to process
FILE_PATTERNS = [
    "_static_reaching.csv",
    "_static_unsupported.csv",
    "_static_hanging.csv",
    "_static_resting.csv"
]


def load_and_filter_gesture_data_csv(file_path, target_label):
    """
    Load CSV file and filter based on labels.

    For open/close: keep only samples with the target label (active gesture)
    For relax: keep all label=0 samples

    Returns:
        X: (C, T) EMG data (channels x time)

    Data format: CSV with 'gt' column (label) and 'emg0'-'emg7' columns (8 EMG channels)
    """
    df = pd.read_csv(file_path)

    # Extract ground truth labels and EMG channels
    labels = df['gt'].values  # (T,)
    emg_cols = [f'emg{i}' for i in range(8)]
    emg = df[emg_cols].values  # (T, 8)

    if target_label == 0:
        # For relax, use all samples with label=0
        mask = labels == 0
    else:
        # For open/close, use only samples with the target label
        mask = labels == target_label

    filtered_emg = emg[mask]

    if len(filtered_emg) == 0:
        return None

    # Transpose to (C, T) as expected by coactivation functions
    X = filtered_emg.T  # (C, T) where C=8

    return X


def compute_subject_profiles(subject_dir):
    """
    Compute coactivation profiles for a single subject across all valid files.

    Returns:
        dict: {gesture_name: {'R': coactivation_matrix, 'a': relative_strength}}
    """
    subject_profiles = {}

    for label, gesture_name in GESTURES.items():
        all_R = []
        all_a = []
        n_trials = 0

        # Find all files matching the patterns
        for pattern in FILE_PATTERNS:
            files = list(subject_dir.glob(f"*{pattern}"))

            for file_path in files:
                try:
                    X = load_and_filter_gesture_data_csv(file_path, label)

                    if X is None or X.shape[1] < 10:
                        continue

                    # Compute profile using Kendall correlation
                    _, _, _, a, R, _, _, _ = compute_profile_from_window(X, FS, method="kendall")

                    all_R.append(R)
                    all_a.append(a)
                    n_trials += 1

                except Exception as e:
                    print(f"Warning: Failed to process {file_path.name}: {e}")
                    continue

        if n_trials > 0:
            # Aggregate across all trials for this gesture
            R_agg, _ = log_euclidean_spd_mean(all_R, renormalize_correlation=True)
            a_agg = mean_then_renormalize(all_a)

            subject_profiles[gesture_name] = {
                'R': R_agg,
                'a': a_agg,
                'n_trials': n_trials
            }

    return subject_profiles


def plot_coactivation_matrices(profiles, title_prefix, output_path):
    """
    Plot coactivation matrices for all three gestures with consistent styling.
    Creates separate files for matrices and vectors.
    """
    # Set seaborn style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    # Define consistent color palette - viridis_r so lighter = higher correlation
    cmap = plt.cm.viridis_r

    # Get all vectors for adaptive scale
    all_vectors = []
    gesture_list = ["relax", "open", "close"]
    for gesture in gesture_list:
        if gesture in profiles:
            all_vectors.append(profiles[gesture]['a'])

    # Adaptive vector scale
    vmin_vec = 0.0
    vmax_vec = max([np.max(a) for a in all_vectors])

    # Plot 1: Coactivation Matrices (keep diagonal for visualization)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            R = profiles[gesture]['R'].copy()

            # Scale based on off-diagonal values only (diagonal is always 1.0)
            off_diag_mask = ~np.eye(R.shape[0], dtype=bool)
            vmin_mat = np.min(R[off_diag_mask])
            vmax_mat = np.max(R[off_diag_mask])

            # Plot matrix with adaptive scale (lighter = higher correlation)
            im = axes[idx].imshow(R, cmap=cmap, vmin=vmin_mat, vmax=vmax_mat,
                                  aspect='auto', interpolation='nearest')

            axes[idx].set_title(f"{gesture.capitalize()}", fontsize=14, fontweight='bold', pad=15)
            axes[idx].set_xlabel("EMG Channel", fontsize=12)
            axes[idx].set_ylabel("EMG Channel", fontsize=12)
            axes[idx].set_xticks(np.arange(8))
            axes[idx].set_yticks(np.arange(8))

            # Add colorbar with adaptive range
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label('Kendall Tau', rotation=270, labelpad=20, fontsize=11)

    if "Population-Level" in title_prefix:
        suptitle = "Coactivation Matrices - Kendall Tau (ROAM)"
    else:
        suptitle = f"{title_prefix} - Coactivation Matrices - Kendall Tau (ROAM)"

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save matrix plot
    matrix_path = output_path.parent / f"{output_path.stem}_matrices.png"
    plt.savefig(matrix_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {matrix_path}")

    # Plot 2: Relative Strength Vectors as bar plots (now 5 subplots)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Plot the three base gestures
    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            a = profiles[gesture]['a']

            # Map values to colors - higher values get lighter colors (same as matrices)
            norm = plt.Normalize(vmin=vmin_vec, vmax=vmax_vec)
            colors = cmap(norm(a))

            # Create bar plot
            bars = axes[idx].bar(np.arange(8), a, color=colors, edgecolor='black', linewidth=1.2)

            axes[idx].set_title(f"{gesture.capitalize()}", fontsize=14, fontweight='bold', pad=15)
            axes[idx].set_xlabel("EMG Channel", fontsize=12)
            axes[idx].set_ylabel("Relative Strength", fontsize=12)
            axes[idx].set_xticks(np.arange(8))
            axes[idx].set_ylim([vmin_vec, vmax_vec * 1.05])
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[idx].set_axisbelow(True)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}',
                              ha='center', va='bottom', fontsize=9, rotation=0)

    # Plot 4: Difference open - relax
    if 'open' in profiles and 'relax' in profiles:
        diff_open = profiles['open']['a'] - profiles['relax']['a']

        # Color bars based on sign (positive = green, negative = red)
        bar_colors = ['green' if x >= 0 else 'red' for x in diff_open]

        bars = axes[3].bar(np.arange(8), diff_open, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[3].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[3].set_title("Open - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[3].set_xlabel("EMG Channel", fontsize=12)
        axes[3].set_ylabel("Difference", fontsize=12)
        axes[3].set_xticks(np.arange(8))
        axes[3].grid(axis='y', alpha=0.3, linestyle='--')
        axes[3].set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, diff_open)):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # Plot 5: Difference close - relax
    if 'close' in profiles and 'relax' in profiles:
        diff_close = profiles['close']['a'] - profiles['relax']['a']

        # Color bars based on sign
        bar_colors = ['green' if x >= 0 else 'red' for x in diff_close]

        bars = axes[4].bar(np.arange(8), diff_close, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[4].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[4].set_title("Close - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[4].set_xlabel("EMG Channel", fontsize=12)
        axes[4].set_ylabel("Difference", fontsize=12)
        axes[4].set_xticks(np.arange(8))
        axes[4].grid(axis='y', alpha=0.3, linestyle='--')
        axes[4].set_axisbelow(True)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, diff_close)):
            height = bar.get_height()
            axes[4].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    if "Population-Level" in title_prefix:
        suptitle = "Relative Strength Vectors and Differences - Kendall Tau (ROAM)"
    else:
        suptitle = f"{title_prefix} - Relative Strength Vectors and Differences - Kendall Tau (ROAM)"

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save vector plot
    vector_path = output_path.parent / f"{output_path.stem}_vectors.png"
    plt.savefig(vector_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {vector_path}")

    # Reset style
    sns.reset_defaults()


def main():
    print("="*60)
    print("ROAM-EMG COACTIVATION PROFILE COMPUTATION (KENDALL TAU)")
    print("="*60)
    print()

    # Get all subject directories
    subject_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith('s')])
    print(f"Found {len(subject_dirs)} subjects")
    print()

    # Step 1: Test on single subject first
    print("Step 1: Testing on single subject")
    test_subject_dir = subject_dirs[0]
    print(f"Testing on: {test_subject_dir.name}")

    test_profiles = compute_subject_profiles(test_subject_dir)

    for gesture, data in test_profiles.items():
        print(f"  {gesture}: {data['n_trials']} trials aggregated")

    # Plot and save single subject
    plot_coactivation_matrices(
        test_profiles,
        f"Single Subject ({test_subject_dir.name})",
        OUTPUT_DIR / "single_subject_coactivation.png"
    )

    # Save single subject profiles
    single_npz_path = OUTPUT_DIR / f"single_subject_profiles_{test_subject_dir.name}.npz"
    save_dict = {}
    for gesture, data in test_profiles.items():
        save_dict[f"{gesture}_R"] = data['R']
        save_dict[f"{gesture}_a"] = data['a']
        save_dict[f"{gesture}_n_trials"] = data['n_trials']
    np.savez(single_npz_path, **save_dict)
    print(f"Saved: {single_npz_path}")
    print()

    # Step 2: Process all subjects
    print("Step 2: Processing all subjects")
    all_subject_profiles = {}

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        try:
            profiles = compute_subject_profiles(subject_dir)
            if profiles:
                all_subject_profiles[subject_dir.name] = profiles
        except Exception as e:
            print(f"\nError processing {subject_dir.name}: {e}")
            continue

    print(f"Successfully processed {len(all_subject_profiles)} subjects")
    print()

    # Save per-subject profiles
    for subject_name, profiles in tqdm(all_subject_profiles.items(), desc="Saving profiles"):
        save_dict = {}
        for gesture, data in profiles.items():
            save_dict[f"{gesture}_R"] = data['R']
            save_dict[f"{gesture}_a"] = data['a']
            save_dict[f"{gesture}_n_trials"] = data['n_trials']

        npz_path = OUTPUT_DIR / "per_subject_profiles" / f"{subject_name}_profiles.npz"
        np.savez(npz_path, **save_dict)
        print(f"Saved: {npz_path}")
    print()

    # Step 3: Compute population-level profiles
    print("Step 3: Computing population-level profiles")
    population_profiles = {}

    for gesture in ["relax", "open", "close"]:
        all_R = []
        all_a = []

        for subject_name, profiles in all_subject_profiles.items():
            if gesture in profiles:
                all_R.append(profiles[gesture]['R'])
                all_a.append(profiles[gesture]['a'])

        if len(all_R) > 0:
            R_pop, _ = log_euclidean_spd_mean(all_R, renormalize_correlation=True)
            a_pop = mean_then_renormalize(all_a)

            population_profiles[gesture] = {
                'R': R_pop,
                'a': a_pop,
                'n_subjects': len(all_R)
            }

            print(f"{gesture}: aggregated {len(all_R)} subjects")

    # Save population profiles
    pop_save_dict = {}
    for gesture, data in population_profiles.items():
        pop_save_dict[f"{gesture}_R"] = data['R']
        pop_save_dict[f"{gesture}_a"] = data['a']
        pop_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    pop_npz_path = OUTPUT_DIR / "population_profiles.npz"
    np.savez(pop_npz_path, **pop_save_dict)
    print(f"Saved: {pop_npz_path}")

    # Plot population results
    plot_coactivation_matrices(
        population_profiles,
        "Population-Level",
        OUTPUT_DIR / "population_coactivation.png"
    )

    print()
    print("="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
