"""
ROAM-EMG Kendall Tau-b Co-activation Analysis

This module provides a complete pipeline for computing Kendall tau-b co-activation
maps from surface EMG (sEMG) data using rank-based correlation analysis.

Key Features:
- EMG preprocessing: bandpass filtering, rectification, envelope extraction (same as relative strength)
- Rank-based co-activation: Kendall tau-b correlation matrices (C x C) across channels
- Geometry-aware aggregation: Gaussian copula + Fisher z-transform for proper pooling
- ROAM dataset processing: batch computation across subjects and gestures
- Visualization: tau-b heatmaps and difference maps
- Leave-one-out analysis: subject vs population comparison

Differences from Relative Strength Analysis:
- Relative strength: (C,) vector - one value per channel (how much each muscle activates)
- Kendall tau-b: (C x C) matrix - pairwise monotonic relationships (how muscles co-activate)

Typical Usage:
    python coactivation_kendall_analysis.py

This will process all subjects in the ROAM-EMG dataset, compute Kendall tau-b
co-activation maps for relax/open/close gestures, and generate visualizations.

Author: ReactEMG Project
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import signal, stats
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path("/home/rsw1/Workspace/reactemg_private/data/ROAM_EMG")

# Stroke patient directories (added for P4 and P15)
STROKE_PATIENT_DIRS = [
    Path("/home/rsw1/Workspace/reactemg_private/reactemg/private/2025_09_04_p4"),
    Path("/home/rsw1/Workspace/reactemg_private/reactemg/private/2025_09_04_p15")
]

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "coactivation_kendall_output"
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

# File patterns for stroke patients (subset of files to use)
STROKE_FILE_PATTERNS = [
    "_close_only.csv",
    "_open_only.csv",
    "_static_resting.csv",
    "_static_hanging.csv",
    "_static_unsupported.csv"
]


# =============================================================================
# (1) Preprocessing: same as your relative-strength pipeline
# =============================================================================

def preprocess_emg_envelope(
    X: np.ndarray,
    fs: float,
    bp_low: float = 10.0,
    bp_high: float = 95.0,
    bp_order: int = 4,
    lp_cutoff: float = 6.0,
    lp_order: int = 2,
    winsor_lo: float = 0.5,
    winsor_hi: float = 99.5,
) -> np.ndarray:
    """
    Convert raw EMG (C x T) to a linear envelope (C x T).

    Pipeline:
      - zero-phase Butterworth band-pass [bp_low, bp_high] Hz
      - full-wave rectification
      - zero-phase low-pass at lp_cutoff Hz
      - per-channel winsorization at [winsor_lo, winsor_hi] percentiles

    Returns
    -------
    E_wins : np.ndarray, shape (C, T)
        Winsorized envelope.
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2, "X must be (channels, time)"

    # Band-pass (zero-phase)
    sos_bp = signal.butter(bp_order, [bp_low, bp_high], btype="bandpass", fs=fs, output="sos")
    F = signal.sosfiltfilt(sos_bp, X, axis=1)

    # Rectify
    R = np.abs(F)

    # Low-pass (zero-phase)
    sos_lp = signal.butter(lp_order, lp_cutoff, btype="lowpass", fs=fs, output="sos")
    E_smooth = signal.sosfiltfilt(sos_lp, R, axis=1)

    # Winsorize per channel
    lo = np.percentile(E_smooth, winsor_lo, axis=1, keepdims=True)
    hi = np.percentile(E_smooth, winsor_hi, axis=1, keepdims=True)
    E_wins = np.clip(E_smooth, lo, hi)
    return E_wins


# =============================================================================
# (2) Full-envelope ranking (ties preserved) and (3) Kendall tau-b map
# =============================================================================

def rank_envelope_full(E: np.ndarray) -> np.ndarray:
    """
    Rank-transform the entire envelope per channel along time.

    Parameters
    ----------
    E : (C, T) float array

    Returns
    -------
    R : (C, T) float array
        Ranks with ties assigned the average rank ("mid-ranks").

    Notes
    -----
    - Kendall tau-b only depends on order relations (and ties). Ranking preserves
      those relations, so tau-b on ranks == tau-b on the raw values (ties retained).
    """
    E = np.asarray(E, dtype=np.float64)
    assert E.ndim == 2, "E must be (channels, time)"

    # Use scipy.stats.rankdata channel-wise (ties -> average rank)
    R = np.apply_along_axis(stats.rankdata, 1, E, method="average")
    return R


def kendall_tau_b_matrix_from_ranks(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Kendall tau-b across channels given ranked series.

    Parameters
    ----------
    R : (C, T) array
        Rank-transformed envelope (ties preserved).

    Returns
    -------
    tau : (C, C) float array in [-1, 1]
        Kendall tau-b matrix (symmetric; diag = 1).
    pval : (C, C) float array
        Two-sided p-values (warning: standard p-values do NOT account for time-series
        autocorrelation; they can be anti-conservative and should not be used for
        formal hypothesis testing).

    Implementation notes
    --------------------
    - Uses scipy.stats.kendalltau(..., variant='b', nan_policy='omit').
    - If a channel is (nearly) constant over the whole gesture, tau will be NaN with that
      channel; we set those entries to 0.0 to avoid propagating NaNs; diag reset to 1.0.
    """
    R = np.asarray(R, dtype=np.float64)
    C, T = R.shape
    tau = np.zeros((C, C), dtype=np.float64)
    pval = np.ones((C, C), dtype=np.float64)

    for i in range(C):
        tau[i, i] = 1.0
        pval[i, i] = 0.0
        for j in range(i + 1, C):
            t, p = stats.kendalltau(R[i], R[j], variant="b", nan_policy="omit")
            if not np.isfinite(t):
                t = 0.0
            if not np.isfinite(p):
                p = np.nan
            tau[i, j] = tau[j, i] = float(t)
            pval[i, j] = pval[j, i] = float(p)

    return tau, pval


def kendall_tau_b_map_from_raw(
    X: np.ndarray,
    fs: float,
    **preproc_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for one gesture:
      raw EMG -> envelope -> ranks -> Kendall tau-b map.

    Returns
    -------
    tau : (C, C) Kendall tau-b
    pval: (C, C) p-values (see caveat above)
    """
    E = preprocess_emg_envelope(X, fs, **preproc_kwargs)
    R = rank_envelope_full(E)
    tau, pval = kendall_tau_b_matrix_from_ranks(R)
    return tau, pval


# =============================================================================
# (4) Aggregation of Kendall maps across repetitions and across subjects
#      (geometry-aware)
# =============================================================================

def _tau_to_rho_gaussian_copula(tau: np.ndarray) -> np.ndarray:
    """
    Map Kendall tau to the latent Pearson correlation 'rho' under a Gaussian copula:
        rho = sin( (pi/2) * tau )
    This is a standard bridge used in meta-analyses of rank correlations.
    """
    return np.sin(0.5 * np.pi * np.clip(tau, -1.0, 1.0))


def _rho_to_tau_gaussian_copula(rho: np.ndarray) -> np.ndarray:
    """Inverse of the mapping: tau = (2/pi) * arcsin(rho)."""
    return (2.0 / np.pi) * np.arcsin(np.clip(rho, -1.0, 1.0))


def _fisher_z_mean(rho_list: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None,
                   eps: float = 1e-6) -> np.ndarray:
    """
    Fisher z-average of latent correlations (rho) with optional weights.
    Works elementwise on square matrices of identical shape.

    We clip rho to (-1+eps, 1-eps) to avoid atanh blowups.
    Diagonals should be handled by the caller (set to NaN and restored later).
    """
    Zs = [np.arctanh(np.clip(r, -1.0 + eps, 1.0 - eps)) for r in rho_list]
    Zs = np.stack(Zs, axis=0)  # (N, C, C)
    if weights is None:
        Zbar = np.nanmean(Zs, axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
        Zbar = np.nansum(w * Zs, axis=0) / np.maximum(np.nansum(w, axis=0), eps)
    return np.tanh(Zbar)


def aggregate_kendall_maps(
    tau_maps: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Geometry-aware aggregation of multiple Kendall tau-b maps.

    Strategy:
      1) Map tau -> latent rho via Gaussian copula: rho = sin( (pi/2) * tau )
      2) Fisher z-average rho (optionally weighted)
      3) Map back: tau = (2/pi) * arcsin(rho)

    Why this:
      - Tau is bounded and non-additive; averaging tau directly can be biased.
      - The copula bridge moves tau to a latent Pearson scale; Fisher z
        provides an approximately normal additive scale for pooling.

    Parameters
    ----------
    tau_maps : list of (C, C) tau-b matrices
    weights  : optional list of weights (e.g., T per repetition)

    Returns
    -------
    tau_agg : (C, C) aggregated tau-b

    Implementation detail:
      - Diagonal entries (self-edges) are set back to 1.0 after aggregation.
    """
    assert len(tau_maps) > 0, "Need at least one map to aggregate"
    C = tau_maps[0].shape[0]
    # Mask diagonal to avoid atanh(inf) via rho=1
    tau_stack = [np.array(t, dtype=np.float64) for t in tau_maps]
    for t in tau_stack:
        assert t.shape == (C, C), "All tau maps must have same shape"
        np.fill_diagonal(t, np.nan)

    rhos = [_tau_to_rho_gaussian_copula(t) for t in tau_stack]
    rho_agg = _fisher_z_mean(rhos, weights=weights, eps=eps)
    tau_agg = _rho_to_tau_gaussian_copula(rho_agg)

    # Restore diagonals
    np.fill_diagonal(tau_agg, 1.0)
    # Symmetrize (guarding against tiny asymmetries from NaNs/weights)
    tau_agg = 0.5 * (tau_agg + tau_agg.T)
    return tau_agg


# Convenience: two-level aggregation

def aggregate_repetitions_to_subject(
    repetition_tau_maps: Sequence[np.ndarray],
    repetition_lengths: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Aggregate repetitions of the SAME subject and gesture to a subject-level map.

    weights:
      - If provided, use repetition_lengths (e.g., T per repetition) to weight the Fisher z-mean.
      - Else, equal weights.
    """
    return aggregate_kendall_maps(repetition_tau_maps, weights=repetition_lengths)


def aggregate_subjects_to_population(
    subject_tau_maps: Sequence[np.ndarray],
    subject_weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Aggregate subject-level maps to a population map (same gesture).

    weights:
      - Optional; use equal weights unless you have a reason to weight by duration N or data quality.
    """
    return aggregate_kendall_maps(subject_tau_maps, weights=subject_weights)


# =============================================================================
# (5) Comparing two co-activation maps (focus on off-diagonals)
# =============================================================================

def _upper_triangle_vec(M: np.ndarray, k: int = 1) -> np.ndarray:
    """Vectorize the upper triangle (exclude diagonal by default)."""
    iu = np.triu_indices(M.shape[0], k=k)
    return M[iu]


def tau_map_delta(
    tau_A: np.ndarray,
    tau_B: np.ndarray,
    eps: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Compute interpretable deltas between two tau-b maps.

    Returns
    -------
    out : dict with
        'delta_tau' : elementwise difference (A - B) in tau space (C x C)
        'delta_z'   : elementwise difference in Fisher-z of latent rho (C x C)

    Interpretation
    --------------
    - delta_tau[i,j] > 0: edge (i,j) shows stronger monotonic co-activation in A.
    - delta_z accounts for the proper pooling geometry (via rho+Fisher z); it is
      a good basis for distances and rankings of change magnitude.

    Notes
    -----
    - Diagonal is set to 0 in both outputs to emphasize off-diagonals.
    """
    A = np.array(tau_A, dtype=np.float64)
    B = np.array(tau_B, dtype=np.float64)
    assert A.shape == B.shape and A.shape[0] == A.shape[1], "Tau maps must be same CxC shape"

    # Simple tau difference:
    d_tau = A - B
    np.fill_diagonal(d_tau, 0.0)

    # Geometry-aware difference via rho->Fisher z
    rhoA = _tau_to_rho_gaussian_copula(A)
    rhoB = _tau_to_rho_gaussian_copula(B)
    zA = np.arctanh(np.clip(rhoA, -1.0 + eps, 1.0 - eps))
    zB = np.arctanh(np.clip(rhoB, -1.0 + eps, 1.0 - eps))
    d_z = zA - zB
    np.fill_diagonal(d_z, 0.0)

    return {"delta_tau": 0.5 * (d_tau + d_tau.T), "delta_z": 0.5 * (d_z + d_z.T)}


def tau_map_similarity(
    tau_A: np.ndarray,
    tau_B: np.ndarray,
    use_geometry: bool = True,
    eps: float = 1e-6
) -> Dict[str, float]:
    """
    Single-number similarities/distances between two tau-b maps (off-diagonals).

    Returns
    -------
    {
      'pearson':   Pearson correlation of off-diagonal entries,
      'pearson_z': Pearson correlation after tau->rho->Fisher z (recommended),
      'fro_z':     Frobenius distance of z-maps (normalized by sqrt(#edges)),
      'cosine_z':  Cosine similarity in z-space
    }
    """
    A = np.array(tau_A, dtype=np.float64)
    B = np.array(tau_B, dtype=np.float64)
    assert A.shape == B.shape and A.shape[0] == A.shape[1]

    vA = _upper_triangle_vec(A, k=1)
    vB = _upper_triangle_vec(B, k=1)

    # Pearson directly on tau
    vAc = vA - np.nanmean(vA)
    vBc = vB - np.nanmean(vB)
    denom = (np.linalg.norm(vAc) * np.linalg.norm(vBc)) + 1e-12
    pearson = float(np.dot(vAc, vBc) / denom) if denom > 0 else 0.0

    if use_geometry:
        rhoA = _tau_to_rho_gaussian_copula(A)
        rhoB = _tau_to_rho_gaussian_copula(B)
        zA = np.arctanh(np.clip(_upper_triangle_vec(rhoA, 1), -1.0 + eps, 1.0 - eps))
        zB = np.arctanh(np.clip(_upper_triangle_vec(rhoB, 1), -1.0 + eps, 1.0 - eps))

        zAc = zA - np.nanmean(zA)
        zBc = zB - np.nanmean(zB)
        denom_z = (np.linalg.norm(zAc) * np.linalg.norm(zBc)) + 1e-12
        pearson_z = float(np.dot(zAc, zBc) / denom_z) if denom_z > 0 else 0.0

        # Frobenius distance in z-space (normalized)
        fro_z = float(np.linalg.norm(zA - zB) / np.sqrt(zA.size))

        # Cosine similarity in z-space
        denom_cos = (np.linalg.norm(zA) * np.linalg.norm(zB)) + 1e-12
        cosine_z = float(np.dot(zA, zB) / denom_cos) if denom_cos > 0 else 0.0
    else:
        pearson_z, fro_z, cosine_z = np.nan, np.nan, np.nan

    return {"pearson": pearson, "pearson_z": pearson_z, "fro_z": fro_z, "cosine_z": cosine_z}


# =============================================================================
# DATA LOADING: ROAM Dataset Specific (same as coactivation_analysis.py)
# =============================================================================

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


# =============================================================================
# PROFILE COMPUTATION: Subject & Population Level
# =============================================================================

def compute_subject_kendall_profiles(subject_dir, file_patterns=None):
    """
    Compute Kendall tau-b co-activation maps for one subject across all repetitions.

    PROPER AGGREGATION PROCEDURE:
    1. For each file, extract individual gesture segments (repetitions)
    2. Process each segment separately to get one tau-b map (C x C)
    3. Aggregate all maps using geometry-aware aggregation (Gaussian copula + Fisher z)

    This properly treats each repetition as an independent measurement and
    aggregates them using the theoretically correct method for rank correlations.

    Parameters
    ----------
    subject_dir : Path
        Directory containing subject's CSV files
    file_patterns : list of str, optional
        File patterns to match. If None, uses FILE_PATTERNS (default for healthy subjects)

    Returns
    -------
    profiles : dict
        {gesture_name: {'tau': tau_b_matrix (C,C), 'n_repetitions': int}}
    """
    if file_patterns is None:
        file_patterns = FILE_PATTERNS

    subject_profiles = {}

    for label, gesture_name in GESTURES.items():
        all_tau_maps = []  # Collect tau-b maps from all repetitions
        all_lengths = []   # Track length of each repetition for weighting

        # Find all files matching the patterns
        for pattern in file_patterns:
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

                        # Compute Kendall tau-b map for this repetition
                        tau, _ = kendall_tau_b_map_from_raw(segment, FS)
                        all_tau_maps.append(tau)
                        all_lengths.append(segment.shape[1])  # T samples

                except Exception as e:
                    print(f"Warning: Failed to process {file_path.name}: {e}")
                    continue

        if len(all_tau_maps) > 0:
            # Aggregate using geometry-aware method (Gaussian copula + Fisher z)
            tau_agg = aggregate_repetitions_to_subject(all_tau_maps, all_lengths)

            subject_profiles[gesture_name] = {
                'tau': tau_agg,
                'n_repetitions': len(all_tau_maps)
            }

    return subject_profiles


# =============================================================================
# VISUALIZATION: Plotting Functions
# =============================================================================

def _add_wraparound(tau):
    """
    Add wraparound row and column (duplicate channel 0 at position 8).

    Useful for circular electrode arrays to visualize continuity.

    Parameters
    ----------
    tau : ndarray, shape (C, C)
        Original tau-b matrix

    Returns
    -------
    tau_wrap : ndarray, shape (C+1, C+1)
        Matrix with wraparound (channel 0 duplicated at end)
    """
    C = tau.shape[0]
    tau_wrap = np.zeros((C + 1, C + 1), dtype=tau.dtype)

    # Copy original matrix
    tau_wrap[:C, :C] = tau

    # Add wraparound: duplicate row 0 at position C
    tau_wrap[C, :C] = tau[0, :]

    # Add wraparound: duplicate column 0 at position C
    tau_wrap[:C, C] = tau[:, 0]

    # Corner: duplicate (0, 0) at (C, C)
    tau_wrap[C, C] = tau[0, 0]

    return tau_wrap


def plot_kendall_tau_heatmap(tau, title, output_path, annotate=False, fmt='.2f',
                             triangle_only=True):
    """
    Plot a single Kendall tau-b co-activation heatmap.

    Parameters
    ----------
    tau : ndarray, shape (C, C)
        Kendall tau-b matrix (values in [-1, 1])
    title : str
        Plot title
    output_path : Path
        Path for saving the plot
    annotate : bool
        Whether to show values in cells
    fmt : str
        Format string for annotations
    triangle_only : bool
        If True, show only lower triangle (matrix is symmetric)
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    # Apply transformations
    tau_plot = tau.copy()
    C_plot = tau_plot.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create masked array
    tau_masked = np.ma.array(tau_plot, mask=False)

    # Mask diagonal
    np.fill_diagonal(tau_masked.mask, True)

    # Mask upper triangle if requested
    if triangle_only:
        tau_masked.mask[np.triu_indices(C_plot, k=1)] = True

    # Compute dynamic color scale based on off-diagonal values
    offdiag_values = tau_masked.compressed()  # Get non-masked values
    if len(offdiag_values) > 0:
        max_abs = np.max(np.abs(offdiag_values))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = -1, 1

    # Use diverging colormap
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='white')  # Masked values will be white

    # Create heatmap with dynamic range
    im = ax.imshow(tau_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Kendall's τ-b", rotation=270, labelpad=20, fontsize=12)

    # Set ticks and labels
    channel_labels = [str(i) for i in range(C_plot)]

    ax.set_xticks(np.arange(C_plot))
    ax.set_yticks(np.arange(C_plot))
    ax.set_xticklabels(channel_labels)
    ax.set_yticklabels(channel_labels)

    # Add grid
    ax.set_xticks(np.arange(C_plot + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(C_plot + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Annotate with values if requested (skip masked cells)
    if annotate:
        for i in range(C_plot):
            for j in range(C_plot):
                if not tau_masked.mask[i, j]:  # Only annotate non-masked cells
                    text = ax.text(j, i, f'{tau_plot[i, j]:{fmt}}',
                                 ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("EMG Channel", fontsize=12)
    ax.set_ylabel("EMG Channel", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()


def plot_gesture_tau_maps(profiles, title_prefix, output_path, triangle_only=True):
    """
    Visualize Kendall tau-b maps for all three gestures side-by-side.

    Parameters
    ----------
    profiles : dict
        {gesture_name: {'tau': tau_matrix (C,C)}}
    title_prefix : str
        Title prefix (e.g., "Subject s01" or "Population-Level")
    output_path : Path
        Path for saving the plot
    triangle_only : bool
        If True, show only lower triangle
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.1)

    gesture_list = ["relax", "open", "close"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Compute global color scale across all gestures for consistency
    all_offdiag_values = []
    for gesture in gesture_list:
        if gesture in profiles:
            tau = profiles[gesture]['tau'].copy()
            # Extract off-diagonal values (exclude diagonal)
            mask = ~np.eye(tau.shape[0], dtype=bool)
            all_offdiag_values.extend(tau[mask])

    if len(all_offdiag_values) > 0:
        max_abs = np.max(np.abs(all_offdiag_values))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = -1, 1

    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            tau = profiles[gesture]['tau'].copy()
            C_plot = tau.shape[0]

            # Create masked array
            tau_masked = np.ma.array(tau, mask=False)
            np.fill_diagonal(tau_masked.mask, True)

            # Mask upper triangle if requested
            if triangle_only:
                tau_masked.mask[np.triu_indices(C_plot, k=1)] = True

            # Use diverging colormap
            cmap = plt.cm.RdBu_r.copy()
            cmap.set_bad(color='white')

            # Create heatmap with dynamic range
            im = axes[idx].imshow(tau_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                 aspect='auto', interpolation='nearest')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            if idx == 2:  # Only label the rightmost colorbar
                cbar.set_label("Kendall's τ-b", rotation=270, labelpad=20, fontsize=11)

            # Set ticks and labels
            channel_labels = [str(i) for i in range(C_plot)]

            axes[idx].set_xticks(np.arange(C_plot))
            axes[idx].set_yticks(np.arange(C_plot))
            axes[idx].set_xticklabels(channel_labels)
            axes[idx].set_yticklabels(channel_labels)

            # Add grid
            axes[idx].set_xticks(np.arange(C_plot + 1) - 0.5, minor=True)
            axes[idx].set_yticks(np.arange(C_plot + 1) - 0.5, minor=True)
            axes[idx].grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

            axes[idx].set_title(f"{gesture.capitalize()}", fontsize=13, fontweight='bold', pad=12)
            axes[idx].set_xlabel("EMG Channel", fontsize=11)
            if idx == 0:
                axes[idx].set_ylabel("EMG Channel", fontsize=11)

    if "Population-Level" in title_prefix:
        suptitle = "Kendall τ-b Co-activation Maps (ROAM)"
    else:
        suptitle = f"{title_prefix} - Kendall τ-b Co-activation Maps"

    plt.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()


def plot_subject_vs_population_tau_maps(subject_profiles, population_profiles,
                                        subject_name, output_path, triangle_only=True):
    """
    Plot comparison of subject vs population tau-b maps for each gesture.

    Creates a 3x2 grid showing subject and population maps side-by-side.

    Parameters
    ----------
    subject_profiles : dict
        Subject's profiles {gesture_name: {'tau': matrix}}
    population_profiles : dict
        Population profiles {gesture_name: {'tau': matrix}}
    subject_name : str
        Name of the subject
    output_path : Path
        Path for saving the plot
    triangle_only : bool
        If True, show only lower triangle
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    gesture_list = ["relax", "open", "close"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Compute global color scale across all gestures and both subject/population for consistency
    all_offdiag_values = []
    for gesture in gesture_list:
        if gesture in subject_profiles and gesture in population_profiles:
            for tau in [subject_profiles[gesture]['tau'], population_profiles[gesture]['tau']]:
                # Extract off-diagonal values (exclude diagonal)
                mask = ~np.eye(tau.shape[0], dtype=bool)
                all_offdiag_values.extend(tau[mask])

    if len(all_offdiag_values) > 0:
        max_abs = np.max(np.abs(all_offdiag_values))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = -1, 1

    for row_idx, gesture in enumerate(gesture_list):
        if gesture in subject_profiles and gesture in population_profiles:
            tau_subject = subject_profiles[gesture]['tau'].copy()
            tau_population = population_profiles[gesture]['tau'].copy()

            # Compute similarity metrics (before transformation)
            sim = tau_map_similarity(tau_subject, tau_population)

            C_plot = tau_subject.shape[0]

            # Create masked arrays
            tau_subject_masked = np.ma.array(tau_subject, mask=False)
            tau_population_masked = np.ma.array(tau_population, mask=False)
            np.fill_diagonal(tau_subject_masked.mask, True)
            np.fill_diagonal(tau_population_masked.mask, True)

            # Mask upper triangle if requested
            if triangle_only:
                tau_subject_masked.mask[np.triu_indices(C_plot, k=1)] = True
                tau_population_masked.mask[np.triu_indices(C_plot, k=1)] = True

            # Use diverging colormap
            cmap = plt.cm.RdBu_r.copy()
            cmap.set_bad(color='white')

            # Plot subject map with dynamic range
            im1 = axes[row_idx, 0].imshow(tau_subject_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                         aspect='auto', interpolation='nearest')
            axes[row_idx, 0].set_title(f"{gesture.capitalize()} - {subject_name}",
                                      fontsize=12, fontweight='bold', pad=10)

            # Plot population map with dynamic range
            im2 = axes[row_idx, 1].imshow(tau_population_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                         aspect='auto', interpolation='nearest')
            axes[row_idx, 1].set_title(f"{gesture.capitalize()} - Population (LOO)\n" +
                                      f"Pearson(z): {sim['pearson_z']:.3f}, Fro: {sim['fro_z']:.3f}",
                                      fontsize=12, fontweight='bold', pad=10)

            # Set ticks and labels for both
            for col_idx in [0, 1]:
                channel_labels = [str(i) for i in range(C_plot)]

                axes[row_idx, col_idx].set_xticks(np.arange(C_plot))
                axes[row_idx, col_idx].set_yticks(np.arange(C_plot))
                axes[row_idx, col_idx].set_xticklabels(channel_labels)
                axes[row_idx, col_idx].set_yticklabels(channel_labels)

                # Add grid
                axes[row_idx, col_idx].set_xticks(np.arange(C_plot + 1) - 0.5, minor=True)
                axes[row_idx, col_idx].set_yticks(np.arange(C_plot + 1) - 0.5, minor=True)
                axes[row_idx, col_idx].grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

                axes[row_idx, col_idx].set_xlabel("EMG Channel", fontsize=10)
                axes[row_idx, col_idx].set_ylabel("EMG Channel", fontsize=10)

                # Add colorbar
                cbar = plt.colorbar(im1 if col_idx == 0 else im2, ax=axes[row_idx, col_idx],
                                  fraction=0.046, pad=0.04)
                if row_idx == 1:  # Middle row
                    cbar.set_label("τ-b", rotation=270, labelpad=15, fontsize=10)

    plt.suptitle(f"{subject_name} vs Population (LOO) - Kendall τ-b Co-activation",
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()


def compute_and_plot_tau_similarities(subject_profiles, population_profiles,
                                     subject_name, output_path, subtitle=""):
    """
    Compute and visualize similarity metrics between subject and population tau-b maps.

    Similarity metrics:
    - Pearson (z-space): Correlation after tau->rho->Fisher z transform (recommended)
    - Frobenius distance (z-space): L2 distance in z-space (lower = more similar)
    - Cosine similarity (z-space): Cosine of angle in z-space (1 = identical)

    Parameters
    ----------
    subject_profiles : dict
        Subject's profiles {gesture_name: {'tau': matrix}}
    population_profiles : dict
        Population profiles {gesture_name: {'tau': matrix}}
    subject_name : str
        Name of the subject
    output_path : Path
        Path for saving the plot
    subtitle : str
        Additional subtitle text (e.g., "LOO")

    Returns
    -------
    similarities : dict
        {gesture_name: {metric_name: value}}
    """
    gesture_list = ["relax", "open", "close"]

    # Compute similarities
    similarities = {}
    for gesture in gesture_list:
        if gesture in subject_profiles and gesture in population_profiles:
            sim = tau_map_similarity(
                subject_profiles[gesture]['tau'],
                population_profiles[gesture]['tau']
            )
            similarities[gesture] = sim

    # Create bar plots
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    gestures = list(similarities.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # blue, green, red

    # Plot 1: Pearson correlation (z-space)
    pearson_z_values = [similarities[g]['pearson_z'] for g in gestures]
    bars = axes[0].bar(gestures, pearson_z_values, color=colors,
                      edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_title("Pearson Correlation\n(Fisher z-space)", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel("Correlation [-1, 1]", fontsize=11)
    axes[0].set_ylim([-1, 1])
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    for bar, val in zip(bars, pearson_z_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')

    # Plot 2: Frobenius distance (z-space)
    fro_z_values = [similarities[g]['fro_z'] for g in gestures]
    bars = axes[1].bar(gestures, fro_z_values, color=colors,
                      edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].set_title("Frobenius Distance\n(Fisher z-space)", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_ylabel("Distance (lower = more similar)", fontsize=11)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    for bar, val in zip(bars, fro_z_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Cosine similarity (z-space)
    cosine_z_values = [similarities[g]['cosine_z'] for g in gestures]
    bars = axes[2].bar(gestures, cosine_z_values, color=colors,
                      edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[2].set_title("Cosine Similarity\n(Fisher z-space)", fontsize=12, fontweight='bold', pad=12)
    axes[2].set_ylabel("Similarity [0, 1]", fontsize=11)
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].set_axisbelow(True)
    for bar, val in zip(bars, cosine_z_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    title = f"Kendall τ-b Map Similarity: {subject_name} vs Population"
    if subtitle:
        title += f" ({subtitle})"
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    sns.reset_defaults()

    return similarities


def leave_one_out_analysis(all_subject_profiles, subject_to_hold_out):
    """
    Perform leave-one-out analysis: compute population mean excluding one subject.

    Parameters
    ----------
    all_subject_profiles : dict
        {subject_name: {gesture_name: {'tau': tau_matrix}}}
    subject_to_hold_out : str
        Name of subject to exclude from population mean

    Returns
    -------
    population_profiles_loo : dict
        Population profiles computed without the held-out subject
    """
    population_profiles_loo = {}

    for gesture in ["relax", "open", "close"]:
        all_tau_maps = []

        for subject_name, profiles in all_subject_profiles.items():
            if subject_name == subject_to_hold_out:
                continue  # Skip the held-out subject

            if gesture in profiles:
                all_tau_maps.append(profiles[gesture]['tau'])

        if len(all_tau_maps) > 0:
            tau_pop = aggregate_subjects_to_population(all_tau_maps)

            population_profiles_loo[gesture] = {
                'tau': tau_pop,
                'n_subjects': len(all_tau_maps)
            }

    return population_profiles_loo


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution: process all ROAM subjects and compute population Kendall tau-b maps.

    Pipeline:
    1. Test on single subject to verify processing
    2. Process all subjects
    3. Save per-subject tau-b maps
    4. Aggregate to population level using geometry-aware method
    5. Generate visualizations
    6. Perform leave-one-out analysis
    """
    print("="*60)
    print("ROAM-EMG KENDALL TAU-B CO-ACTIVATION ANALYSIS")
    print("="*60)
    print()

    # =============================================================================
    # VISUALIZATION CONFIGURATION
    # =============================================================================
    # Kendall's tau-b naturally ranges from -1 to +1:
    #   - Positive values: Co-activation (synergistic muscles)
    #   - Negative values: Anti-coactivation (antagonistic muscles, reciprocal inhibition)
    #
    # TRIANGLE_ONLY: Show only lower triangle (matrix is symmetric)
    #
    TRIANGLE_ONLY = True   # Show only lower triangle (reduces redundancy)
    # =============================================================================

    print("Visualization settings:")
    print("  - Values: Signed τ (preserves positive/negative)")
    print("  - Colormap: Diverging Red-White-Blue")
    print("  - Range: [-max_abs, +max_abs] (dynamic)")
    print("  - Note: Red=negative, White=zero, Blue=positive")
    print(f"  - Triangle: {'Lower only' if TRIANGLE_ONLY else 'Full matrix'}")
    print("  - Diagonal: White (self-correlations masked)")
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

    test_profiles = compute_subject_kendall_profiles(test_subject_dir)

    for gesture, data in test_profiles.items():
        print(f"  {gesture}: {data['n_repetitions']} repetitions aggregated")
        print(f"    Tau-b map shape: {data['tau'].shape}")
        print(f"    Tau-b range: [{data['tau'].min():.3f}, {data['tau'].max():.3f}]")

    # Plot and save single subject
    plot_gesture_tau_maps(
        test_profiles,
        f"Single Subject ({test_subject_dir.name})",
        OUTPUT_DIR / "single_subject_kendall.png",
        triangle_only=TRIANGLE_ONLY
    )

    # Save single subject profiles
    single_npz_path = OUTPUT_DIR / f"single_subject_kendall_{test_subject_dir.name}.npz"
    save_dict = {}
    for gesture, data in test_profiles.items():
        save_dict[f"{gesture}_tau"] = data['tau']
        save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']
    np.savez(single_npz_path, **save_dict)
    print(f"Saved: {single_npz_path}")
    print()

    # Step 2: Process all subjects
    print("Step 2: Processing all subjects")
    all_subject_profiles = {}

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        try:
            profiles = compute_subject_kendall_profiles(subject_dir)
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
            save_dict[f"{gesture}_tau"] = data['tau']
            save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']

        npz_path = OUTPUT_DIR / "per_subject_profiles" / f"{subject_name}_kendall.npz"
        np.savez(npz_path, **save_dict)

    print()

    # Step 3: Compute population-level profiles
    print("Step 3: Computing population-level Kendall tau-b maps")
    population_profiles = {}

    for gesture in ["relax", "open", "close"]:
        all_tau_maps = []

        for subject_name, profiles in all_subject_profiles.items():
            if gesture in profiles:
                all_tau_maps.append(profiles[gesture]['tau'])

        if len(all_tau_maps) > 0:
            tau_pop = aggregate_subjects_to_population(all_tau_maps)

            population_profiles[gesture] = {
                'tau': tau_pop,
                'n_subjects': len(all_tau_maps)
            }

            print(f"{gesture}: aggregated {len(all_tau_maps)} subjects")
            print(f"  Tau-b range: [{tau_pop.min():.3f}, {tau_pop.max():.3f}]")

    # Save population profiles
    pop_save_dict = {}
    for gesture, data in population_profiles.items():
        pop_save_dict[f"{gesture}_tau"] = data['tau']
        pop_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    pop_npz_path = OUTPUT_DIR / "population_kendall.npz"
    np.savez(pop_npz_path, **pop_save_dict)
    print(f"Saved: {pop_npz_path}")

    # Plot population results
    plot_gesture_tau_maps(
        population_profiles,
        "Population-Level",
        OUTPUT_DIR / "population_kendall.png",
        triangle_only=TRIANGLE_ONLY
    )

    print()

    # Step 4: Leave-One-Out Analysis
    print("="*60)
    print("Step 4: Leave-One-Out Analysis")
    print("="*60)

    # Select a subject for leave-one-out analysis (use first subject)
    loo_subject = list(all_subject_profiles.keys())[0]
    print(f"\nHeld-out subject: {loo_subject}")

    # Compute population profiles excluding this subject (LOO)
    population_profiles_loo = leave_one_out_analysis(all_subject_profiles, loo_subject)

    print(f"\nPopulation Kendall tau-b maps computed without {loo_subject} (LOO):")
    for gesture, data in population_profiles_loo.items():
        print(f"  {gesture}: aggregated {data['n_subjects']} subjects")

    # Save leave-one-out population profiles
    loo_save_dict = {}
    for gesture, data in population_profiles_loo.items():
        loo_save_dict[f"{gesture}_tau"] = data['tau']
        loo_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    loo_npz_path = OUTPUT_DIR / "leave_one_out_analysis" / f"population_without_{loo_subject}.npz"
    np.savez(loo_npz_path, **loo_save_dict)
    print(f"\nSaved: {loo_npz_path}")

    # Plot leave-one-out population profiles
    plot_gesture_tau_maps(
        population_profiles_loo,
        f"Population (LOO without {loo_subject})",
        OUTPUT_DIR / "leave_one_out_analysis" / f"population_without_{loo_subject}_kendall.png",
        triangle_only=TRIANGLE_ONLY
    )

    # Step 5: Subject vs Population Comparison
    print("\n" + "="*60)
    print("Step 5: Subject vs Population Comparison")
    print("="*60)

    # Plot held-out subject's profiles
    print(f"\nGenerating visualizations for {loo_subject}...")
    plot_gesture_tau_maps(
        all_subject_profiles[loo_subject],
        f"Subject {loo_subject}",
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_kendall.png",
        triangle_only=TRIANGLE_ONLY
    )

    # Side-by-side comparison plot
    print(f"\nGenerating subject vs population comparison...")
    plot_subject_vs_population_tau_maps(
        all_subject_profiles[loo_subject],
        population_profiles_loo,
        loo_subject,
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_vs_loo_kendall.png",
        triangle_only=TRIANGLE_ONLY
    )

    # Step 6: Similarity Analysis
    print("\n" + "="*60)
    print("Step 6: Kendall Tau-b Map Similarity Analysis")
    print("="*60)

    # Compare held-out subject to leave-one-out population
    print(f"\nComparing {loo_subject} to LOO population...")
    similarities_loo = compute_and_plot_tau_similarities(
        all_subject_profiles[loo_subject],
        population_profiles_loo,
        loo_subject,
        OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_vs_loo_similarities.png",
        subtitle="LOO"
    )

    print(f"\nSimilarity Metrics ({loo_subject} vs LOO Population):")
    for gesture, metrics in similarities_loo.items():
        print(f"  {gesture}:")
        print(f"    Pearson (z-space): {metrics['pearson_z']:.4f}")
        print(f"    Frobenius dist (z): {metrics['fro_z']:.4f}")
        print(f"    Cosine similarity: {metrics['cosine_z']:.4f}")

    # Save similarity metrics to JSON
    similarity_results = {
        "subject": loo_subject,
        "vs_loo_population": similarities_loo,
        "interpretation": {
            "pearson_z": "Correlation in Fisher z-space. Range [-1,1]. Higher = more similar pattern.",
            "fro_z": "Frobenius distance in z-space. Lower = more similar. 0 = identical.",
            "cosine_z": "Cosine similarity in z-space. Range [0,1]. 1 = identical direction."
        }
    }

    json_path = OUTPUT_DIR / "leave_one_out_analysis" / f"{loo_subject}_kendall_similarities.json"
    with open(json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2, default=float)
    print(f"\nSaved similarity metrics: {json_path}")

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

        # Compute Kendall profiles for stroke patient using specific file patterns
        try:
            stroke_profiles = compute_subject_kendall_profiles(stroke_dir, file_patterns=STROKE_FILE_PATTERNS)

            if not stroke_profiles:
                print(f"  Warning: No profiles computed for {stroke_id}")
                continue

            print(f"  Successfully computed Kendall profiles for {stroke_id}")
            for gesture, data in stroke_profiles.items():
                print(f"    {gesture}: {data['n_repetitions']} repetitions")
                print(f"      Tau-b range: [{data['tau'].min():.3f}, {data['tau'].max():.3f}]")

            # Save stroke patient profiles
            stroke_save_dict = {}
            for gesture, data in stroke_profiles.items():
                stroke_save_dict[f"{gesture}_tau"] = data['tau']
                stroke_save_dict[f"{gesture}_n_repetitions"] = data['n_repetitions']

            stroke_npz_path = OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_kendall.npz"
            np.savez(stroke_npz_path, **stroke_save_dict)
            print(f"  Saved: {stroke_npz_path}")

            # Plot stroke patient Kendall maps
            plot_gesture_tau_maps(
                stroke_profiles,
                f"Stroke Patient {stroke_id}",
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_kendall.png",
                triangle_only=TRIANGLE_ONLY
            )

            # Compare stroke patient to FULL healthy population (no LOO)
            print(f"\n  Comparing {stroke_id} to healthy population...")
            plot_subject_vs_population_tau_maps(
                stroke_profiles,
                population_profiles,  # Full healthy population
                stroke_id,
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_vs_healthy_population_kendall.png",
                triangle_only=TRIANGLE_ONLY
            )

            # Compute and plot similarity metrics
            similarities_stroke = compute_and_plot_tau_similarities(
                stroke_profiles,
                population_profiles,  # Full healthy population
                stroke_id,
                OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_vs_healthy_population_similarities.png",
                subtitle="Healthy Population"
            )

            print(f"\n  Similarity Metrics ({stroke_id} vs Healthy Population):")
            for gesture, metrics in similarities_stroke.items():
                print(f"    {gesture}:")
                print(f"      Pearson (z-space): {metrics['pearson_z']:.4f}")
                print(f"      Frobenius dist (z): {metrics['fro_z']:.4f}")
                print(f"      Cosine similarity: {metrics['cosine_z']:.4f}")

            # Save similarity metrics to JSON
            stroke_similarity_results = {
                "stroke_patient": stroke_id,
                "vs_healthy_population": similarities_stroke,
                "interpretation": {
                    "pearson_z": "Correlation in Fisher z-space. Range [-1,1]. Higher = more similar pattern.",
                    "fro_z": "Frobenius distance in z-space. Lower = more similar. 0 = identical.",
                    "cosine_z": "Cosine similarity in z-space. Range [0,1]. 1 = identical direction."
                }
            }

            stroke_json_path = OUTPUT_DIR / "stroke_patients" / f"{stroke_id}_kendall_similarities.json"
            with open(stroke_json_path, 'w') as f:
                json.dump(stroke_similarity_results, f, indent=2, default=float)
            print(f"  Saved similarity metrics: {stroke_json_path}")

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
