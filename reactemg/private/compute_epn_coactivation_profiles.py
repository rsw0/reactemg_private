#!/usr/bin/env python3
"""
Compute per-subject and population-level coactivation profiles for EMG-EPN-612 dataset.

For each subject, compute coactivation matrices and relative strength vectors for:
- open gesture (using non-zero labeled samples only)
- fist gesture (using non-zero labeled samples only)
- noGesture (using all samples)

Then aggregate across all subjects to get population-level profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Import coactivation functions
sys.path.insert(0, str(Path(__file__).parent))
from reactemg.coactivation import (
    compute_profile_from_window,
    log_euclidean_spd_mean,
    mean_then_renormalize,
)


# Constants
DATA_ROOT = Path("data/EMG-EPN-612")
OUTPUT_DIR = Path("coactivation_profiles_output")
SAMPLING_RATE = 200.0  # Hz (typical for EPN-612)
GESTURES = ["open", "fist", "noGesture"]


def load_and_filter_gesture_data(file_path, gesture_name):
    """
    Load .npy file and filter based on labels.

    For open/fist: keep only non-zero labeled samples (active gesture)
    For noGesture: keep all samples

    Returns:
        X: (C, T) EMG data (channels x time)
    """
    data = np.load(file_path, allow_pickle=True)  # Shape: (T, C+1)

    # Last column is label, first C columns are EMG
    emg = data[:, :-1]  # (T, C)
    labels = data[:, -1]  # (T,)

    if gesture_name == "noGesture":
        # Use all samples
        filtered_emg = emg
    else:
        # For open/fist, keep only non-zero labels (active gesture)
        active_mask = labels != 0
        filtered_emg = emg[active_mask]

    # Transpose to (C, T) as expected by coactivation functions
    X = filtered_emg.T  # (C, T)

    return X


def compute_subject_gesture_profile(subject_dir, gesture_name):
    """
    Compute aggregated coactivation profile for one subject and one gesture.

    Returns:
        R_subject: (C, C) coactivation matrix
        a_subject: (C,) relative strength vector
        n_trials: number of trials aggregated
    """
    # Find all files for this gesture
    pattern = f"*_{gesture_name}.npy"
    files = sorted(subject_dir.glob(pattern))

    if len(files) == 0:
        return None, None, 0

    Rs_trials = []
    As_trials = []

    for file_path in files:
        try:
            X = load_and_filter_gesture_data(file_path, gesture_name)

            # Skip if no samples after filtering
            if X.shape[1] < 10:  # Need at least some samples
                continue

            # Compute profile for this trial
            _, _, _, a, R, _, _, _ = compute_profile_from_window(X, SAMPLING_RATE)

            Rs_trials.append(R)
            As_trials.append(a)
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue

    if len(Rs_trials) == 0:
        return None, None, 0

    # Aggregate across trials for this subject
    Rs_trials = np.stack(Rs_trials, axis=0)  # (N_trials, C, C)
    As_trials = np.stack(As_trials, axis=0)  # (N_trials, C)

    # Log-Euclidean mean for coactivation matrices
    R_subject, _ = log_euclidean_spd_mean(Rs_trials, renormalize_correlation=True)

    # Arithmetic mean then renormalize for relative strength
    a_subject = mean_then_renormalize(As_trials)

    return R_subject, a_subject, len(Rs_trials)


def process_single_subject(subject_dir, verbose=False):
    """
    Process one subject to compute profiles for all three gestures.

    Returns:
        profiles: dict with keys 'open', 'fist', 'noGesture'
                  each containing {'R': (C,C), 'a': (C,), 'n_trials': int}
    """
    profiles = {}

    for gesture in GESTURES:
        R, a, n_trials = compute_subject_gesture_profile(subject_dir, gesture)

        if R is not None:
            profiles[gesture] = {
                'R': R,
                'a': a,
                'n_trials': n_trials
            }
            if verbose:
                print(f"  {gesture}: {n_trials} trials aggregated")
        else:
            if verbose:
                print(f"  {gesture}: No valid data")

    return profiles


def process_all_subjects():
    """
    Process all subjects and compute per-subject profiles.

    Returns:
        all_subject_profiles: dict[subject_id] -> profiles dict
    """
    subject_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("trainingJSON_user")])

    print(f"Found {len(subject_dirs)} subjects")

    all_subject_profiles = {}

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_id = subject_dir.name
        profiles = process_single_subject(subject_dir, verbose=False)

        if len(profiles) > 0:
            all_subject_profiles[subject_id] = profiles

    print(f"Successfully processed {len(all_subject_profiles)} subjects")
    return all_subject_profiles


def aggregate_population_profiles(all_subject_profiles):
    """
    Aggregate profiles across all subjects to get population-level means.

    Returns:
        population_profiles: dict with keys 'open', 'fist', 'noGesture'
                            each containing {'R': (C,C), 'a': (C,), 'n_subjects': int}
    """
    population_profiles = {}

    for gesture in GESTURES:
        Rs_all_subjects = []
        As_all_subjects = []

        for subject_id, profiles in all_subject_profiles.items():
            if gesture in profiles:
                Rs_all_subjects.append(profiles[gesture]['R'])
                As_all_subjects.append(profiles[gesture]['a'])

        if len(Rs_all_subjects) > 0:
            Rs_all_subjects = np.stack(Rs_all_subjects, axis=0)  # (N_subjects, C, C)
            As_all_subjects = np.stack(As_all_subjects, axis=0)  # (N_subjects, C)

            # Log-Euclidean mean for coactivation matrices
            R_population, _ = log_euclidean_spd_mean(Rs_all_subjects, renormalize_correlation=True)

            # Arithmetic mean then renormalize for relative strength
            a_population = mean_then_renormalize(As_all_subjects)

            population_profiles[gesture] = {
                'R': R_population,
                'a': a_population,
                'n_subjects': len(Rs_all_subjects)
            }

            print(f"{gesture}: aggregated {len(Rs_all_subjects)} subjects")
        else:
            print(f"{gesture}: No subjects with valid data")

    return population_profiles


def plot_coactivation_matrices(profiles, title_prefix, output_path):
    """
    Plot coactivation matrices for all three gestures with relative strength bars above.
    """
    fig = plt.figure(figsize=(15, 6))

    for idx, gesture in enumerate(GESTURES):
        if gesture in profiles:
            R = profiles[gesture]['R'].copy()
            a = profiles[gesture]['a']

            # Create subplot with 2 rows: one for strength bar, one for matrix
            # Using gridspec for custom height ratios
            ax_bar = plt.subplot2grid((8, 3), (0, idx), rowspan=1)
            ax_mat = plt.subplot2grid((8, 3), (1, idx), rowspan=7)

            # Plot relative strength as horizontal color bar
            a_2d = a.reshape(1, -1)  # Shape: (1, C)
            im_bar = ax_bar.imshow(a_2d, cmap='viridis', aspect='auto', vmin=0, vmax=a.max() * 1.1)
            ax_bar.set_xticks(np.arange(len(a)))
            ax_bar.set_xticklabels([])
            ax_bar.set_yticks([])
            ax_bar.set_title(f"{gesture}", fontsize=12, fontweight='bold', pad=10)

            # Set diagonal to NaN to exclude from visualization
            np.fill_diagonal(R, np.nan)

            # Find min/max excluding diagonal
            R_no_diag = R[~np.isnan(R)]
            vmin = R_no_diag.min()
            vmax = R_no_diag.max()

            # Plot coactivation matrix
            im_mat = ax_mat.imshow(R, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
            ax_mat.set_xlabel("Channel", fontsize=10)
            ax_mat.set_ylabel("Channel", fontsize=10)

            # Add colorbar for matrix
            plt.colorbar(im_mat, ax=ax_mat, fraction=0.046, pad=0.04)
        else:
            ax = plt.subplot(1, 3, idx + 1)
            ax.text(0.5, 0.5, f"No data\n{gesture}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Remove "Population-Level -" prefix if present, just use "Coactivation Matrices"
    if "Population-Level" in title_prefix:
        suptitle = "Coactivation Matrices"
    else:
        suptitle = f"{title_prefix} - Coactivation Matrices"

    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_profiles_to_disk(profiles, output_path):
    """
    Save profiles to disk as .npz file.
    """
    data_to_save = {}
    for gesture, profile in profiles.items():
        data_to_save[f"{gesture}_R"] = profile['R']
        data_to_save[f"{gesture}_a"] = profile['a']
        if 'n_trials' in profile:
            data_to_save[f"{gesture}_n_trials"] = profile['n_trials']
        if 'n_subjects' in profile:
            data_to_save[f"{gesture}_n_subjects"] = profile['n_subjects']

    np.savez(output_path, **data_to_save)
    print(f"Saved: {output_path}")


def main():
    """Main execution function."""

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Step 1: Testing on single subject")
    print("="*60)

    # Test on first subject
    test_subject_dir = sorted([d for d in DATA_ROOT.iterdir()
                               if d.is_dir() and d.name.startswith("trainingJSON_user")])[0]
    print(f"Testing on: {test_subject_dir.name}")

    test_profiles = process_single_subject(test_subject_dir, verbose=True)

    # Plot single subject results
    plot_coactivation_matrices(
        test_profiles,
        f"Single Subject ({test_subject_dir.name})",
        OUTPUT_DIR / "single_subject_coactivation_matrices.png"
    )

    # Save single subject profiles
    save_profiles_to_disk(
        test_profiles,
        OUTPUT_DIR / f"single_subject_profiles_{test_subject_dir.name}.npz"
    )

    print("\n" + "="*60)
    print("Step 2: Processing all subjects")
    print("="*60)

    # Process all subjects
    all_subject_profiles = process_all_subjects()

    # Save per-subject profiles
    per_subject_dir = OUTPUT_DIR / "per_subject_profiles"
    per_subject_dir.mkdir(exist_ok=True)

    print("\nSaving per-subject profiles...")
    for subject_id, profiles in tqdm(all_subject_profiles.items(), desc="Saving profiles"):
        save_profiles_to_disk(
            profiles,
            per_subject_dir / f"{subject_id}_profiles.npz"
        )

    print("\n" + "="*60)
    print("Step 3: Computing population-level profiles")
    print("="*60)

    # Aggregate to population level
    population_profiles = aggregate_population_profiles(all_subject_profiles)

    # Save population profiles
    save_profiles_to_disk(
        population_profiles,
        OUTPUT_DIR / "population_profiles.npz"
    )

    # Plot population results
    plot_coactivation_matrices(
        population_profiles,
        "Population-Level",
        OUTPUT_DIR / "population_coactivation_matrices.png"
    )

    print("\n" + "="*60)
    print("All done! Results saved to:", OUTPUT_DIR)
    print("="*60)

    # Summary statistics
    print("\nSummary:")
    print(f"  Total subjects processed: {len(all_subject_profiles)}")
    for gesture in GESTURES:
        if gesture in population_profiles:
            n = population_profiles[gesture]['n_subjects']
            print(f"  {gesture}: {n} subjects")


if __name__ == "__main__":
    main()
