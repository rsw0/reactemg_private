"""
ROAM-EMG Relative Muscle Strength Analysis

This module provides a complete pipeline for computing relative muscle strength
profiles from surface EMG (sEMG) data using compositional data analysis.

Key Features:
- EMG preprocessing: bandpass filtering, rectification, envelope extraction
- Relative strength computation: L1-normalized muscle activation profiles
- Compositional analysis: CLR transform and Aitchison distance for proper
  comparison of muscle activation patterns
- ROAM dataset processing: batch computation across subjects and gestures
- Visualization: strength profiles and CLR compositional differences

Typical Usage:
    python coactivation_analysis.py

This will process all subjects in the ROAM-EMG dataset, compute relative strength
profiles for relax/open/close gestures, and generate visualizations.

Author: ReactEMG Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("/home/rsw1/Workspace/reactemg_private/data/ROAM_EMG")

# Stroke patient directories (added for P4 and P15)
STROKE_PATIENT_DIRS = [
    Path("/home/rsw1/Workspace/reactemg_private/reactemg/private/2025_09_04_p4"),
    Path("/home/rsw1/Workspace/reactemg_private/reactemg/private/2025_09_04_p15")
]

# Save outputs in the same directory as this script
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "coactivation_profiles_output"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "per_subject_profiles").mkdir(exist_ok=True)
(OUTPUT_DIR / "leave_one_out_analysis").mkdir(exist_ok=True)

FS = 200.0  # Sampling rate for ROAM dataset (Hz)

GESTURES = {
    0: "relax",
    1: "open",
    2: "close"
}

FILE_PATTERNS = [
    "_static_reaching.csv",
    "_static_unsupported.csv",
    "_static_hanging.csv",
    "_static_resting.csv"
]


# ============================================================
# EMG PREPROCESSING
# ============================================================

def preprocess_emg_envelope(
    X,
    fs,
    bp_low=10.0,
    bp_high=95.0,
    bp_order=4,
    lp_cutoff=6.0,
    lp_order=2,
    winsor_lo=0.5,
    winsor_hi=99.5,
):
    """
    Extract EMG envelope with bandpass, rectification, lowpass, and winsorization.

    Processing Pipeline:
    1. Bandpass filter (10-95 Hz) to remove noise and motion artifacts
    2. Rectify (absolute value) to capture muscle activation magnitude
    3. Lowpass filter (6 Hz) to smooth and create envelope
    4. Winsorize to clip extreme outliers

    Parameters
    ----------
    X : ndarray, shape (C, T)
        Raw EMG data (channels x time)
    fs : float
        Sampling frequency in Hz
    bp_low : float
        Bandpass low cutoff (Hz)
    bp_high : float
        Bandpass high cutoff (Hz)
    bp_order : int
        Bandpass Butterworth filter order
    lp_cutoff : float
        Lowpass cutoff for envelope (Hz)
    lp_order : int
        Lowpass Butterworth filter order
    winsor_lo : float
        Lower percentile for winsorization (0-100)
    winsor_hi : float
        Upper percentile for winsorization (0-100)

    Returns
    -------
    E_smooth : ndarray, shape (C, T)
        Smoothed envelope (before winsorization)
    E_wins : ndarray, shape (C, T)
        Winsorized envelope (clipped outliers)
    """
    X = np.asarray(X, dtype=np.float64)

    # Zero-phase bandpass filtering
    sos_bp = signal.butter(bp_order, [bp_low, bp_high], btype="bandpass", fs=fs, output="sos")
    F = signal.sosfiltfilt(sos_bp, X, axis=1)

    # Rectify
    R = np.abs(F)

    # Zero-phase lowpass filtering for envelope
    sos_lp = signal.butter(lp_order, lp_cutoff, btype="lowpass", fs=fs, output="sos")
    E_smooth = signal.sosfiltfilt(sos_lp, R, axis=1)

    # Winsorize per channel to clip outliers
    lo = np.percentile(E_smooth, winsor_lo, axis=1, keepdims=True)
    hi = np.percentile(E_smooth, winsor_hi, axis=1, keepdims=True)
    E_wins = np.clip(E_smooth, lo, hi)

    return E_smooth, E_wins


# ============================================================
# MUSCLE STRENGTH COMPUTATION
# ============================================================

def relative_strength(E, eps=1e-12):
    """
    Compute absolute and relative muscle activation strength.

    Relative strength is the L1-normalized activation profile, representing
    the proportional contribution of each muscle to total activation.

    Parameters
    ----------
    E : ndarray, shape (C, T)
        EMG envelope (winsorized recommended)
    eps : float
        Small value to prevent division by zero

    Returns
    -------
    s : ndarray, shape (C,)
        Absolute mean amplitude per channel
    a : ndarray, shape (C,)
        L1-normalized relative strength (sums to ~1)
    """
    E = np.asarray(E, dtype=np.float64)
    s = E.mean(axis=1)
    total = s.sum()
    a = s / (total + eps)
    return s, a


# ============================================================
# COMPOSITIONAL DATA ANALYSIS: Aitchison Geometry
# ============================================================
#
# OVERVIEW:
# Relative muscle strengths are compositional data - they represent proportions
# that sum to 1. Such data live on the probability simplex, not in Euclidean
# space. Using standard operations (arithmetic mean, Euclidean distance) on
# compositional data violates their fundamental constraint and can lead to
# incorrect conclusions.
#
# THE CLR (Centered Log-Ratio) SOLUTION:
# The CLR transform is an isometric mapping from the simplex to Euclidean space:
#   CLR(x) = log(x) - mean(log(x)) = log(x / geometric_mean(x))
#
# Key properties:
# 1. CLR vectors sum to 0 (isometric log-ratio coordinates)
# 2. Euclidean operations in CLR space are valid
# 3. CLR respects scale-invariance: only relative proportions matter
# 4. The L2 norm of CLR differences = Aitchison distance (the natural metric)
#
# WHY NOT OTHER METHODS:
# - Arithmetic mean: Violates simplex constraint, can produce spurious correlations
# - Raw log-ratios: Include arbitrary scale shifts, not centered
# - Euclidean distance: Sensitive to closure (forced sum-to-1), not scale-invariant
#
# PROPER WORKFLOW FOR COMPOSITIONAL EMG DATA:
# 1. Aggregation: Use compositional_mean() to average multiple compositions
# 2. Comparison: Use compositional_compare() to compare two compositions
#    - Returns CLR transforms, differences, and Aitchison distance in one call
#
# References:
# - Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
# - Pawlowsky-Glahn, V., et al. (2015). Modeling and Analysis of Compositional Data.

def compositional_mean(A_stack, eps=1e-12):
    """
    Geometric mean for compositional data (proper aggregation method).

    Computes the geometric mean of compositional vectors, which is the
    unique center that minimizes the sum of squared Aitchison distances.
    This respects the simplex geometry and is scale-invariant.

    Parameters
    ----------
    A_stack : ndarray, shape (N, C)
        Stack of N compositional vectors (each row sums to 1)
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    a_bar : ndarray, shape (C,)
        L1-normalized geometric mean (sums to 1)

    Mathematical Details
    --------------------
    The geometric mean is computed as:
        a_bar = exp(mean(log(A_stack))) / sum(exp(mean(log(A_stack))))

    This is equivalent to:
        a_bar = argmin_x sum_i d_Aitchison(x, A_stack[i])^2
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    X = np.clip(A_stack, eps, None)
    logX = np.log(X)
    g = np.exp(logX.mean(axis=0))
    return g / (g.sum() + eps)


def compositional_compare(a, b, eps=1e-12, normalize=True):
    """
    Complete compositional comparison between two vectors (single efficient pass).

    Computes CLR transforms, differences, and Aitchison distance in one pass,
    avoiding redundant computations. This is the ONLY function you need for
    comparing two compositional vectors.

    Parameters
    ----------
    a, b : ndarray, shape (C,)
        Compositional vectors (proportions that sum to 1)
    eps : float
        Small value to avoid log(0)
    normalize : bool
        If True, L1-normalize inputs before comparison

    Returns
    -------
    results : dict
        {
            'clr_a': CLR transform of a (ndarray, shape (C,)),
            'clr_b': CLR transform of b (ndarray, shape (C,)),
            'clr_diff': clr_a - clr_b (ndarray, shape (C,)),
            'aitchison_distance': ||clr_diff||_2 (float)
        }

    Mathematical Details
    --------------------
    CLR transform:
        CLR(x) = log(x) - mean(log(x)) = log(x / geometric_mean(x))

    CLR difference:
        CLR(a) - CLR(b) = log(a/b) - mean(log(a/b))

    Aitchison distance (natural metric on the simplex):
        d_A(a, b) = ||CLR(a) - CLR(b)||_2
                  = sqrt(sum_i [CLR(a)_i - CLR(b)_i]^2)

    Properties:
        - Scale-invariant: d_A(c*a, c*b) = d_A(a, b) for c > 0
        - Subcompositional coherence: operations on parts are consistent
        - Metric properties: d_A(a,b) >= 0, d_A(a,b) = 0 iff a = b
        - Triangle inequality: d_A(a,c) <= d_A(a,b) + d_A(b,c)

    Examples
    --------
    >>> a = np.array([0.2, 0.3, 0.5])
    >>> b = np.array([0.1, 0.4, 0.5])
    >>> result = compositional_compare(a, b)
    >>> print(f"Aitchison distance: {result['aitchison_distance']:.4f}")
    >>> print(f"CLR difference: {result['clr_diff']}")
    """
    # Convert to arrays and normalize if requested
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

    if normalize:
        a = a / (a.sum() + eps)
        b = b / (b.sum() + eps)

    # Compute CLR transforms (only once!)
    log_a = np.log(np.clip(a, eps, None))
    log_b = np.log(np.clip(b, eps, None))

    clr_a = log_a - log_a.mean()
    clr_b = log_b - log_b.mean()

    # Compute difference
    clr_diff = clr_a - clr_b

    # Compute Aitchison distance (L2 norm of CLR difference)
    aitchison_dist = float(np.linalg.norm(clr_diff, ord=2))

    return {
        'clr_a': clr_a,
        'clr_b': clr_b,
        'clr_diff': clr_diff,
        'aitchison_distance': aitchison_dist
    }


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def compute_profile_from_window(X, fs):
    """
    Complete pipeline: preprocess EMG and compute relative strength profile.

    Convenience function that chains preprocessing and strength computation
    for a single window of EMG data.

    Parameters
    ----------
    X : ndarray, shape (C, T)
        Raw EMG data (channels x time)
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    E_smooth : ndarray, shape (C, T)
        Smoothed envelope
    E_wins : ndarray, shape (C, T)
        Winsorized envelope
    s : ndarray, shape (C,)
        Absolute strength per channel
    a : ndarray, shape (C,)
        Relative strength (L1-normalized composition)
    """
    E_smooth, E_wins = preprocess_emg_envelope(X, fs)
    s, a = relative_strength(E_wins)
    return E_smooth, E_wins, s, a


# ============================================================
# DATA LOADING: ROAM Dataset Specific
# ============================================================

def load_gesture_segments_csv(file_path, target_label, min_length=200):
    """
    Load ROAM-EMG CSV file and extract individual gesture repetitions.

    Instead of concatenating all repetitions, this extracts each continuous
    segment separately. Each segment represents one gesture repetition (~5 seconds).

    Parameters
    ----------
    file_path : Path
        Path to ROAM-EMG CSV file
    target_label : int
        Gesture label (0=relax, 1=open, 2=close)
    min_length : int
        Minimum segment length in samples (default 200 = 1 second @ 200Hz)

    Returns
    -------
    segments : list of ndarray
        List of EMG segments, each shape (C, T)
        Returns empty list if no valid segments found
    """
    df = pd.read_csv(file_path)

    # Extract ground truth and EMG channels
    labels = df['gt'].values
    emg_cols = [f'emg{i}' for i in range(8)]
    emg = df[emg_cols].values  # (T, 8)

    # Find all indices with target label
    indices = np.where(labels == target_label)[0]

    if len(indices) == 0:
        return []

    # Find continuous segments (breaks where diff > 1)
    breaks = np.where(np.diff(indices) > 1)[0]

    # Extract all continuous segments
    segments = []
    if len(breaks) == 0:
        # Only one continuous segment
        segment_indices = indices
        if len(segment_indices) >= min_length:
            segment_emg = emg[segment_indices, :]  # (T, 8)
            segments.append(segment_emg.T)  # Transpose to (8, T)
    else:
        # Multiple segments
        start_idx = 0
        for break_idx in breaks:
            segment_indices = indices[start_idx:break_idx+1]
            if len(segment_indices) >= min_length:
                segment_emg = emg[segment_indices, :]  # (T, 8)
                segments.append(segment_emg.T)  # Transpose to (8, T)
            start_idx = break_idx + 1

        # Last segment
        segment_indices = indices[start_idx:]
        if len(segment_indices) >= min_length:
            segment_emg = emg[segment_indices, :]  # (T, 8)
            segments.append(segment_emg.T)  # Transpose to (8, T)

    return segments


# ============================================================
# PROFILE COMPUTATION: Subject & Population Level
# ============================================================

def compute_subject_profiles(subject_dir):
    """
    Compute relative strength profiles for one subject across all repetitions.

    PROPER AGGREGATION PROCEDURE:
    1. For each file, extract individual gesture segments (repetitions)
    2. Process each segment separately to get one relative strength vector
    3. Aggregate all vectors using compositional_mean (geometric mean)

    This properly treats each repetition as an independent measurement and
    aggregates them using the theoretically correct method for compositional data.

    Parameters
    ----------
    subject_dir : Path
        Directory containing subject's CSV files

    Returns
    -------
    profiles : dict
        {gesture_name: {'a': relative_strength, 'n_repetitions': int}}
    """
    subject_profiles = {}

    for label, gesture_name in GESTURES.items():
        all_a = []  # Collect relative strength vectors from all repetitions

        # Find all files matching the patterns
        for pattern in FILE_PATTERNS:
            files = list(subject_dir.glob(f"*{pattern}"))

            for file_path in files:
                try:
                    # Load individual gesture segments (NOT concatenated)
                    segments = load_gesture_segments_csv(file_path, label)

                    if len(segments) == 0:
                        continue

                    # Process each segment separately
                    for segment in segments:
                        # segment shape: (C, T)
                        if segment.shape[1] < 200:  # Skip very short segments
                            continue

                        # Compute relative strength for this repetition
                        _, _, _, a = compute_profile_from_window(segment, FS)
                        all_a.append(a)

                except Exception as e:
                    print(f"Warning: Failed to process {file_path.name}: {e}")
                    continue

        if len(all_a) > 0:
            # Aggregate using geometric mean (proper for compositional data)
            a_agg = compositional_mean(np.array(all_a))

            subject_profiles[gesture_name] = {
                'a': a_agg,
                'n_repetitions': len(all_a)
            }

    return subject_profiles


# ============================================================
# VISUALIZATION: Plotting Functions
# ============================================================

def plot_relative_strength_profiles(profiles, title_prefix, output_path):
    """
    Visualize relative strength profiles with CLR compositional differences.

    Creates visualization showing:
    - Relative strength vectors for each gesture
    - CLR compositional differences between gestures

    Parameters
    ----------
    profiles : dict
        {gesture_name: {'a': relative_strength_vector}}
    title_prefix : str
        Title prefix (e.g., "Subject s01" or "Population-Level")
    output_path : Path
        Path for saving the plot
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    cmap = plt.cm.viridis_r

    gesture_list = ["relax", "open", "close"]

    # Get adaptive scale for vectors
    all_vectors = []
    for gesture in gesture_list:
        if gesture in profiles:
            all_vectors.append(profiles[gesture]['a'])

    vmin_vec = 0.0
    vmax_vec = max([np.max(a) for a in all_vectors])

    # ===== 3 Rows: Relative Strength + CLR Differences + Fold-Changes =====
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    channels = np.arange(8)

    # ROW 0: Plot base gestures (relax, open, close)
    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            a = profiles[gesture]['a']

            norm = plt.Normalize(vmin=vmin_vec, vmax=vmax_vec)
            colors = cmap(norm(a))

            bars = axes[0, idx].bar(channels, a, color=colors, edgecolor='black', linewidth=1.2)

            axes[0, idx].set_title(f"{gesture.capitalize()}", fontsize=14, fontweight='bold', pad=15)
            axes[0, idx].set_xlabel("EMG Channel", fontsize=12)
            axes[0, idx].set_ylabel("Relative Strength", fontsize=12)
            axes[0, idx].set_xticks(channels)
            axes[0, idx].set_ylim([vmin_vec, vmax_vec * 1.05])
            axes[0, idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[0, idx].set_axisbelow(True)

            for bar in bars:
                height = bar.get_height()
                axes[0, idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}',
                              ha='center', va='bottom', fontsize=9, rotation=0)

    # Compute CLR compositional differences
    if all(g in profiles for g in ['relax', 'open', 'close']):
        comp_open_relax = compositional_compare(profiles['open']['a'], profiles['relax']['a'])
        comp_close_relax = compositional_compare(profiles['close']['a'], profiles['relax']['a'])
        clr_open_relax = comp_open_relax['clr_diff']
        clr_close_relax = comp_close_relax['clr_diff']

        # Exponentiate CLR differences to get fold-changes
        fold_open_relax = np.exp(clr_open_relax)
        fold_close_relax = np.exp(clr_close_relax)

        # ROW 1: CLR differences
        # Hide first column (aligned with relax)
        axes[1, 0].axis('off')

        # Plot 1.1: CLR difference (open - relax), aligned with "open"
        colors_clr_open = ['green' if x >= 0 else 'red' for x in clr_open_relax]
        bars = axes[1, 1].bar(channels, clr_open_relax, color=colors_clr_open,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_title("CLR Difference: Open - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[1, 1].set_xlabel("EMG Channel", fontsize=12)
        axes[1, 1].set_ylabel("CLR Difference", fontsize=12)
        axes[1, 1].set_xticks(channels)
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1, 1].set_axisbelow(True)

        for bar, val in zip(bars, clr_open_relax):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

        # Plot 1.2: CLR difference (close - relax), aligned with "close"
        colors_clr_close = ['green' if x >= 0 else 'red' for x in clr_close_relax]
        bars = axes[1, 2].bar(channels, clr_close_relax, color=colors_clr_close,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 2].set_title("CLR Difference: Close - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[1, 2].set_xlabel("EMG Channel", fontsize=12)
        axes[1, 2].set_ylabel("CLR Difference", fontsize=12)
        axes[1, 2].set_xticks(channels)
        axes[1, 2].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1, 2].set_axisbelow(True)

        for bar, val in zip(bars, clr_close_relax):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

        # ROW 2: Fold-changes
        # Hide first column (aligned with relax)
        axes[2, 0].axis('off')

        # Plot 2.1: Fold-change (open / relax), aligned with "open"
        colors_fold_open = ['green' if x >= 1.0 else 'red' for x in fold_open_relax]
        bars = axes[2, 1].bar(channels, fold_open_relax, color=colors_fold_open,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[2, 1].axhline(y=1.0, color='black', linestyle='-', linewidth=1)
        axes[2, 1].set_title("Fold-Change: Open / Relax", fontsize=14, fontweight='bold', pad=15)
        axes[2, 1].set_xlabel("EMG Channel", fontsize=12)
        axes[2, 1].set_ylabel("Fold-Change (exp(CLR diff))", fontsize=12)
        axes[2, 1].set_xticks(channels)
        axes[2, 1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[2, 1].set_axisbelow(True)

        for bar, val in zip(bars, fold_open_relax):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}x',
                        ha='center', va='bottom' if val >= 1.0 else 'top', fontsize=8)

        # Plot 2.2: Fold-change (close / relax), aligned with "close"
        colors_fold_close = ['green' if x >= 1.0 else 'red' for x in fold_close_relax]
        bars = axes[2, 2].bar(channels, fold_close_relax, color=colors_fold_close,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[2, 2].axhline(y=1.0, color='black', linestyle='-', linewidth=1)
        axes[2, 2].set_title("Fold-Change: Close / Relax", fontsize=14, fontweight='bold', pad=15)
        axes[2, 2].set_xlabel("EMG Channel", fontsize=12)
        axes[2, 2].set_ylabel("Fold-Change (exp(CLR diff))", fontsize=12)
        axes[2, 2].set_xticks(channels)
        axes[2, 2].grid(axis='y', alpha=0.3, linestyle='--')
        axes[2, 2].set_axisbelow(True)

        for bar, val in zip(bars, fold_close_relax):
            height = bar.get_height()
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}x',
                        ha='center', va='bottom' if val >= 1.0 else 'top', fontsize=8)

    if "Population-Level" in title_prefix:
        suptitle = "Relative Strength & CLR Compositional Differences (ROAM)"
    else:
        suptitle = f"{title_prefix} - Relative Strength & CLR Compositional Differences (ROAM)"

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    vector_path = output_path.parent / f"{output_path.stem}_vectors.png"
    plt.savefig(vector_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {vector_path}")

    sns.reset_defaults()


def plot_subject_vs_population_per_gesture(subject_profiles, population_profiles,
                                           subject_name, output_path):
    """
    Plot side-by-side comparison of subject vs LOO population for each gesture.

    Shows relative strength vectors for subject and population mean for each
    of the three gestures (relax, open, close).

    Parameters
    ----------
    subject_profiles : dict
        Subject's profiles {gesture_name: {'a': vector, 'R': matrix}}
    population_profiles : dict
        Population LOO profiles {gesture_name: {'a': vector, 'R': matrix}}
    subject_name : str
        Name of the subject
    output_path : Path
        Path for saving the plot
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    gesture_list = ["relax", "open", "close"]
    channels = np.arange(8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, gesture in enumerate(gesture_list):
        if gesture in subject_profiles and gesture in population_profiles:
            a_subject = subject_profiles[gesture]['a']
            a_population = population_profiles[gesture]['a']

            # Compute Aitchison distance for this gesture
            comp_result = compositional_compare(a_subject, a_population)
            dist = comp_result['aitchison_distance']

            # Bar positions for grouped bars
            x = channels
            width = 0.35

            bars1 = axes[idx].bar(x - width/2, a_subject, width, label=subject_name,
                                 color='#3498db', edgecolor='black', linewidth=1, alpha=0.8)
            bars2 = axes[idx].bar(x + width/2, a_population, width, label='Population (LOO)',
                                 color='#e74c3c', edgecolor='black', linewidth=1, alpha=0.8)

            axes[idx].set_title(f"{gesture.capitalize()}\nAitchison Distance: {dist:.4f}",
                               fontsize=13, fontweight='bold', pad=15)
            axes[idx].set_xlabel("EMG Channel", fontsize=11)
            axes[idx].set_ylabel("Relative Strength", fontsize=11)
            axes[idx].set_xticks(channels)
            axes[idx].legend(fontsize=9, loc='upper right')
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[idx].set_axisbelow(True)

    plt.suptitle(f"{subject_name} vs Population (Leave-One-Out) - Per-Gesture Comparison",
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()


def compute_and_plot_aitchison_distances(subject_profiles, population_profiles,
                                         subject_name, output_path, subtitle=""):
    """
    Compute and visualize Aitchison distances between subject and population.

    The Aitchison distance is the natural distance metric for compositional data:
    - Distance = 0: Identical muscle activation patterns
    - Higher distance: More dissimilar muscle activation patterns
    - Lower distance: More similar muscle activation patterns

    Parameters
    ----------
    subject_profiles : dict
        Subject's profiles {gesture_name: {'a': vector}}
    population_profiles : dict
        Population profiles {gesture_name: {'a': vector}}
    subject_name : str
        Name of the subject
    output_path : Path
        Path for saving the plot
    subtitle : str
        Additional subtitle text (e.g., "LOO" or "Full Population")

    Returns
    -------
    distances : dict
        {gesture_name: aitchison_distance}
    """
    gesture_list = ["relax", "open", "close"]

    # Compute Aitchison distances
    distances = {}
    for gesture in gesture_list:
        if gesture in subject_profiles and gesture in population_profiles:
            comp_result = compositional_compare(
                subject_profiles[gesture]['a'],
                population_profiles[gesture]['a']
            )
            distances[gesture] = comp_result['aitchison_distance']

    # Create bar plot
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    gestures = list(distances.keys())
    dist_values = [distances[g] for g in gestures]

    colors = ['#3498db', '#2ecc71', '#e74c3c']  # blue, green, red
    bars = ax.bar(gestures, dist_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    title = f"Aitchison Distance: {subject_name} vs Population"
    if subtitle:
        title += f" ({subtitle})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Gesture", fontsize=12)
    ax.set_ylabel("Aitchison Distance\n(0=identical, higher=more dissimilar)", fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, dist_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()

    return distances


def leave_one_out_analysis(all_subject_profiles, subject_to_hold_out):
    """
    Perform leave-one-out analysis: compute population mean excluding one subject.

    Parameters
    ----------
    all_subject_profiles : dict
        {subject_name: {gesture_name: {'a': relative_strength_vector}}}
    subject_to_hold_out : str
        Name of subject to exclude from population mean

    Returns
    -------
    population_profiles_loo : dict
        Population profiles computed without the held-out subject
    """
    population_profiles_loo = {}

    for gesture in ["relax", "open", "close"]:
        all_a = []

        for subject_name, profiles in all_subject_profiles.items():
            if subject_name == subject_to_hold_out:
                continue  # Skip the held-out subject

            if gesture in profiles:
                all_a.append(profiles[gesture]['a'])

        if len(all_a) > 0:
            a_pop = compositional_mean(all_a)

            population_profiles_loo[gesture] = {
                'a': a_pop,
                'n_subjects': len(all_a)
            }

    return population_profiles_loo


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution: process all ROAM subjects and compute population profiles.

    Pipeline:
    1. Test on single subject to verify processing
    2. Process all subjects in parallel
    3. Save per-subject profiles
    4. Aggregate to population level using log-Euclidean mean
    5. Generate visualizations
    """
    print("="*60)
    print("ROAM-EMG COACTIVATION PROFILE COMPUTATION")
    print("="*60)
    print()

    # Get all subject directories (ROAM healthy subjects ONLY)
    subject_dirs = sorted([d for d in DATA_ROOT.iterdir()
                          if d.is_dir() and d.name.startswith('s')])
    print(f"Found {len(subject_dirs)} healthy subjects")
    print()

    # Step 1: Test on single subject
    print("Step 1: Testing on single subject")
    test_subject_dir = subject_dirs[0]
    print(f"Testing on: {test_subject_dir.name}")

    test_profiles = compute_subject_profiles(test_subject_dir)

    for gesture, data in test_profiles.items():
        print(f"  {gesture}: {data['n_repetitions']} repetitions aggregated")

    # Plot and save single subject
    plot_relative_strength_profiles(
        test_profiles,
        f"Single Subject ({test_subject_dir.name})",
        OUTPUT_DIR / "single_subject_coactivation.png"
    )

    # Save single subject profiles
    single_npz_path = OUTPUT_DIR / f"single_subject_profiles_{test_subject_dir.name}.npz"
    save_dict = {}
    for gesture, data in test_profiles.items():
        save_dict[f"{gesture}_a"] = data['a']
        save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']
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
            save_dict[f"{gesture}_a"] = data['a']
            save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']

        npz_path = OUTPUT_DIR / "per_subject_profiles" / f"{subject_name}_profiles.npz"
        np.savez(npz_path, **save_dict)

    print()

    # Step 3: Compute population-level profiles
    print("Step 3: Computing population-level profiles")
    population_profiles = {}

    for gesture in ["relax", "open", "close"]:
        all_a = []

        for subject_name, profiles in all_subject_profiles.items():
            if gesture in profiles:
                all_a.append(profiles[gesture]['a'])

        if len(all_a) > 0:
            a_pop = compositional_mean(all_a)

            population_profiles[gesture] = {
                'a': a_pop,
                'n_subjects': len(all_a)
            }

            print(f"{gesture}: aggregated {len(all_a)} subjects")

    # Save population profiles
    pop_save_dict = {}
    for gesture, data in population_profiles.items():
        pop_save_dict[f"{gesture}_a"] = data['a']
        pop_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    pop_npz_path = OUTPUT_DIR / "population_profiles.npz"
    np.savez(pop_npz_path, **pop_save_dict)
    print(f"Saved: {pop_npz_path}")

    # Plot population results (now includes compositional differences)
    plot_relative_strength_profiles(
        population_profiles,
        "Population-Level",
        OUTPUT_DIR / "population_coactivation.png"
    )

    print()

    # Step 4: Leave-One-Out Analysis
    print("="*60)
    print("Step 4: Leave-One-Out Analysis (Per-Gesture Comparisons)")
    print("="*60)

    # Select a subject for leave-one-out analysis (use first subject)
    loo_subject = list(all_subject_profiles.keys())[0]
    print(f"\nHeld-out subject: {loo_subject}")

    # Compute population profiles excluding this subject (LOO)
    population_profiles_loo = leave_one_out_analysis(all_subject_profiles, loo_subject)

    print(f"\nPopulation profiles computed without {loo_subject} (LOO):")
    for gesture, data in population_profiles_loo.items():
        print(f"  {gesture}: aggregated {data['n_subjects']} subjects")

    # Save leave-one-out population profiles
    loo_save_dict = {}
    for gesture, data in population_profiles_loo.items():
        loo_save_dict[f"{gesture}_a"] = data['a']
        loo_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    loo_npz_path = OUTPUT_DIR / "leave_one_out_analysis" / f"population_without_{loo_subject}.npz"
    np.savez(loo_npz_path, **loo_save_dict)
    print(f"\nSaved: {loo_npz_path}")

    # Plot leave-one-out population profiles (now includes compositional differences)
    plot_relative_strength_profiles(
        population_profiles_loo,
        f"Population (LOO without {loo_subject})",
        OUTPUT_DIR / "leave_one_out_analysis" / f"population_without_{loo_subject}_coactivation.png"
    )

    # Step 5: Subject vs Population Comparison (Per-Gesture)
    print("\n" + "="*60)
    print("Step 5: Per-Gesture Comparison (Subject vs LOO Population)")
    print("="*60)

    # Plot held-out subject's profiles (now includes compositional differences)
    print(f"\nGenerating visualizations for {loo_subject}...")
    plot_relative_strength_profiles(
        all_subject_profiles[loo_subject],
        f"Subject {loo_subject}",
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_coactivation.png"
    )

    # NEW: Per-gesture comparison plot (relax, open, close separately)
    print(f"\nGenerating per-gesture comparison plot...")
    plot_subject_vs_population_per_gesture(
        all_subject_profiles[loo_subject],
        population_profiles_loo,
        loo_subject,
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_vs_loo_per_gesture.png"
    )

    # Step 6: Aitchison Distance Analysis
    print("\n" + "="*60)
    print("Step 6: Aitchison Distance Analysis")
    print("="*60)
    print("Note: Higher distance = more dissimilar, Lower = more similar")

    # Compare held-out subject to leave-one-out population (PRIMARY comparison)
    print(f"\nComparing {loo_subject} to LOO population mean (excluding self)...")
    distances_loo = compute_and_plot_aitchison_distances(
        all_subject_profiles[loo_subject],
        population_profiles_loo,
        loo_subject,
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_vs_loo_population_aitchison.png",
        subtitle="LOO"
    )

    print(f"\nAitchison Distances ({loo_subject} vs LOO Population):")
    for gesture, dist in distances_loo.items():
        print(f"  {gesture}: {dist:.6f} (0=identical, higher=more dissimilar)")

    # Save Aitchison distances to JSON
    import json
    aitchison_results = {
        "subject": loo_subject,
        "vs_loo_population": distances_loo,
        "interpretation": "Distance of 0 means identical patterns. Higher values indicate more dissimilar muscle activation patterns."
    }

    json_path = OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_aitchison_distances.json"
    with open(json_path, 'w') as f:
        json.dump(aitchison_results, f, indent=2)
    print(f"\nSaved Aitchison distances: {json_path}")

    # Step 7: Stroke Patient Analysis (P4 and P15 vs Healthy Population)
    print("\n" + "="*60)
    print("Step 7: Stroke Patient Analysis (vs Healthy Population)")
    print("="*60)

    # Create output directory for stroke patients
    (OUTPUT_DIR / "stroke_patients").mkdir(exist_ok=True)

    for stroke_dir in STROKE_PATIENT_DIRS:
        if not stroke_dir.exists() or not stroke_dir.is_dir():
            print(f"\nWarning: Stroke patient directory not found: {stroke_dir}")
            continue

        stroke_id = stroke_dir.name  # e.g., "2025_09_04_p4"
        print(f"\nProcessing stroke patient: {stroke_id}")
        print("-" * 60)

        # Compute profiles for stroke patient
        try:
            stroke_profiles = compute_subject_profiles(stroke_dir)

            if not stroke_profiles:
                print(f"  Warning: No profiles computed for {stroke_id}")
                continue

            print(f"  Successfully computed profiles for {stroke_id}")
            for gesture, data in stroke_profiles.items():
                print(f"    {gesture}: {data['n_repetitions']} repetitions")

            # Save stroke patient profiles
            stroke_save_dict = {}
            for gesture, data in stroke_profiles.items():
                stroke_save_dict[f"{gesture}_a"] = data['a']
                stroke_save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']

            stroke_npz_path = OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_profiles.npz"
            np.savez(stroke_npz_path, **stroke_save_dict)
            print(f"  Saved: {stroke_npz_path}")

            # Plot stroke patient profiles
            plot_relative_strength_profiles(
                stroke_profiles,
                f"Stroke Patient {stroke_id}",
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_coactivation.png"
            )

            # Compare stroke patient to FULL healthy population (no LOO)
            print(f"\n  Comparing {stroke_id} to healthy population...")
            plot_subject_vs_population_per_gesture(
                stroke_profiles,
                population_profiles,  # Full healthy population
                stroke_id,
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_vs_healthy_population_per_gesture.png"
            )

            # Compute and plot Aitchison distances
            distances_stroke = compute_and_plot_aitchison_distances(
                stroke_profiles,
                population_profiles,  # Full healthy population
                stroke_id,
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_vs_healthy_population_aitchison.png",
                subtitle="Healthy Population"
            )

            print(f"\n  Aitchison Distances ({stroke_id} vs Healthy Population):")
            for gesture, dist in distances_stroke.items():
                print(f"    {gesture}: {dist:.6f}")

            # Save Aitchison distances to JSON
            stroke_aitchison_results = {
                "stroke_patient": stroke_id,
                "vs_healthy_population": distances_stroke,
                "interpretation": "Distance of 0 means identical patterns. Higher values indicate more dissimilar muscle activation patterns."
            }

            stroke_json_path = OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_aitchison_distances.json"
            with open(stroke_json_path, 'w') as f:
                json.dump(stroke_aitchison_results, f, indent=2)
            print(f"  Saved Aitchison distances: {stroke_json_path}")

        except Exception as e:
            print(f"  Error processing {stroke_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print(f"Leave-one-out analysis saved to: {(OUTPUT_DIR / 'leave_one_out_analysis').absolute()}")
    print(f"Stroke patient analysis saved to: {(OUTPUT_DIR / 'stroke_patients').absolute()}")


if __name__ == "__main__":
    main()
