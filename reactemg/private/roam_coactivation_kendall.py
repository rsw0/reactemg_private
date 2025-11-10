"""
ROAM-EMG Coactivation Analysis using Kendall's Tau-b Correlation

This module provides a complete pipeline for computing muscle coactivation profiles
from surface EMG (sEMG) data. It combines signal processing, robust correlation
analysis, and Riemannian geometry on the manifold of Symmetric Positive Definite
(SPD) matrices.

Key Features:
- EMG preprocessing: bandpass filtering, rectification, envelope extraction
- Robust coactivation: Kendall's tau-b and Spearman correlation with SPD guarantees
- SPD aggregation: log-Euclidean mean across subjects/trials
- Compositional analysis: relative muscle strength vectors with Aitchison distance
- ROAM dataset processing: batch computation across subjects and gestures

Typical Usage:
    python roam_coactivation_kendall.py

This will process all subjects in the ROAM-EMG dataset, compute coactivation
profiles for relax/open/close gestures, and generate visualizations.

Author: ReactEMG Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import signal, stats
from numpy.linalg import eigh


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("/home/rsw1/Workspace/reactemg_private/data/ROAM_EMG")
# Save outputs in the same directory as this script
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "coactivation_profiles_output_roam_kendall"
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
# UTILITY FUNCTIONS: SPD Matrix Operations
# ============================================================

def _is_symmetric(a, tol=1e-12):
    """Check if matrix is symmetric within tolerance."""
    return np.allclose(a, a.T, atol=tol, rtol=0)


def _symmetrize(a):
    """Force matrix to be symmetric by averaging with transpose."""
    return 0.5 * (a + a.T)


def _is_spd(a, tol=1e-12):
    """
    Check if matrix is Symmetric Positive Definite (SPD).

    A matrix is SPD if it's symmetric and all eigenvalues are positive.
    """
    if not _is_symmetric(a, tol):
        return False
    w = np.linalg.eigvalsh(a)
    return np.min(w) > tol


def _spd_logm(a, eps=1e-12):
    """
    Matrix logarithm for SPD matrices via eigendecomposition.

    For SPD matrix A = V * diag(w) * V^T, computes log(A) = V * diag(log(w)) * V^T.

    Parameters
    ----------
    a : ndarray, shape (C, C)
        SPD matrix
    eps : float
        Minimum eigenvalue threshold for numerical stability

    Returns
    -------
    log_a : ndarray, shape (C, C)
        Matrix logarithm of a

    Raises
    ------
    ValueError
        If input is not SPD
    """
    a = _symmetrize(a)
    if not _is_spd(a, tol=eps):
        raise ValueError("spd_logm: input is not SPD.")
    w, v = eigh(a)
    w = np.maximum(w, eps)
    lw = np.log(w)
    return (v * lw) @ v.T


def _spd_expm(s):
    """
    Matrix exponential for symmetric matrices via eigendecomposition.

    For symmetric matrix S = V * diag(w) * V^T, computes exp(S) = V * diag(exp(w)) * V^T.
    The exponential of a symmetric matrix is always SPD.

    Parameters
    ----------
    s : ndarray, shape (C, C)
        Symmetric matrix

    Returns
    -------
    exp_s : ndarray, shape (C, C)
        SPD matrix exponential of s
    """
    s = _symmetrize(s)
    w, v = eigh(s)
    ew = np.exp(w)
    return (v * ew) @ v.T


def _renormalize_to_correlation(a, eps=1e-12):
    """
    Normalize SPD matrix to have unit diagonal (correlation-like).

    Applies transformation: D^{-1/2} * A * D^{-1/2} where D = diag(A).
    This preserves SPD property and forces diagonal to 1.

    Parameters
    ----------
    a : ndarray, shape (C, C)
        SPD matrix
    eps : float
        Minimum diagonal value for numerical stability

    Returns
    -------
    corr : ndarray, shape (C, C)
        Correlation-like matrix with diag = 1
    """
    a = _symmetrize(a)
    d = np.diag(a).copy()
    d = np.maximum(d, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    c = D_inv_sqrt @ a @ D_inv_sqrt
    return _symmetrize(c)


def _adaptive_shrink_lambda_for_spd(R_raw, lam_floor, eps=1e-8, lam_cap=0.99):
    """
    Compute adaptive shrinkage parameter to make matrix SPD.

    Chooses lambda >= lam_floor such that R = (1-lambda)*R_raw + lambda*I
    has all eigenvalues >= eps. Uses analytical formula based on minimum
    eigenvalue of R_raw.

    Parameters
    ----------
    R_raw : ndarray, shape (C, C)
        Raw correlation matrix (may not be SPD)
    lam_floor : float
        Minimum shrinkage (typically max(0.05, C/T))
    eps : float
        Target minimum eigenvalue
    lam_cap : float
        Maximum allowed shrinkage (default 0.99)

    Returns
    -------
    lam : float
        Final shrinkage parameter used
    lam_needed : float
        Minimum shrinkage needed to achieve SPD
    """
    R_raw = _symmetrize(R_raw)
    lam_needed = 0.0
    lambda_min = float(np.min(np.linalg.eigvalsh(R_raw)))

    if lambda_min < eps:
        denom = max(1.0 - lambda_min, eps)
        lam_needed = (eps - lambda_min) / denom

    lam = max(lam_floor, lam_needed)
    lam = float(np.clip(lam, 0.0, lam_cap))
    return lam, lam_needed


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
# COACTIVATION MATRICES (Kendall & Spearman)
# ============================================================

def spearman_coactivation_spd(E, lam_floor=None, eps=1e-8, lam_cap=0.99):
    """
    Compute Spearman rank correlation coactivation matrix with SPD guarantee.

    Spearman correlation measures monotonic relationships between channels
    and is robust to outliers. Adaptive shrinkage ensures the result is SPD.

    Parameters
    ----------
    E : ndarray, shape (C, T)
        EMG envelope
    lam_floor : float or None
        Minimum shrinkage; if None, uses max(0.05, C/T)
    eps : float
        Target minimum eigenvalue for SPD
    lam_cap : float
        Maximum shrinkage (default 0.99)

    Returns
    -------
    R : ndarray, shape (C, C)
        SPD coactivation matrix (diag = 1)
    R_raw : ndarray, shape (C, C)
        Raw Spearman correlation before shrinkage
    lam : float
        Shrinkage parameter used
    info : dict
        Diagnostic information (eigenvalues, dimensions)
    """
    E = np.asarray(E, dtype=np.float64)
    C, T = E.shape

    # Rank-transform each channel across time
    Q = np.empty_like(E)
    for i in range(C):
        Q[i, :] = stats.rankdata(E[i, :], method="average")

    # Row-standardize
    Qm = Q.mean(axis=1, keepdims=True)
    Qs = Q.std(axis=1, keepdims=True)
    Qs = np.where(Qs < eps, 1.0, Qs)
    Z = (Q - Qm) / Qs

    # Raw Spearman correlation (Pearson on ranks)
    denom = max(T - 1, 1)
    R_raw = (Z @ Z.T) / denom
    R_raw = _symmetrize(R_raw)
    np.fill_diagonal(R_raw, 1.0)

    # Adaptive shrinkage toward identity
    if lam_floor is None:
        lam_floor = max(0.05, C / max(T, 1))
    lam, lam_needed = _adaptive_shrink_lambda_for_spd(R_raw, lam_floor, eps=eps, lam_cap=lam_cap)
    R = (1.0 - lam) * R_raw + lam * np.eye(C)
    R = _symmetrize(R)

    # Final SPD check
    if not _is_spd(R, tol=eps):
        lam2 = min(lam_cap, lam + 1e-3)
        R = (1.0 - lam2) * R_raw + lam2 * np.eye(C)
        lam = lam2

    info = {
        "C": C, "T": T,
        "lam_floor": lam_floor, "lam_needed": lam_needed,
        "min_eig_raw": float(np.min(np.linalg.eigvalsh(R_raw))),
        "min_eig_R": float(np.min(np.linalg.eigvalsh(R))),
    }
    return R, R_raw, float(lam), info


def kendall_coactivation_spd(E, lam_floor=None, eps=1e-8, lam_cap=0.99, nan_to_zero=True):
    """
    Compute Kendall's tau-b coactivation matrix with SPD guarantee.

    Kendall's tau-b is a robust rank correlation that properly handles ties
    and is less sensitive to outliers than Pearson correlation. Adaptive
    shrinkage ensures the result is SPD for downstream Riemannian operations.

    Parameters
    ----------
    E : ndarray, shape (C, T)
        EMG envelope (winsorized recommended)
    lam_floor : float or None
        Minimum shrinkage; if None, uses max(0.05, C/T)
    eps : float
        Target minimum eigenvalue for SPD
    lam_cap : float
        Maximum shrinkage (default 0.99)
    nan_to_zero : bool
        Replace NaN tau values (from constant channels) with 0

    Returns
    -------
    R : ndarray, shape (C, C)
        SPD coactivation matrix (diag = 1)
    R_raw : ndarray, shape (C, C)
        Raw Kendall tau-b matrix before shrinkage
    lam : float
        Shrinkage parameter used
    info : dict
        Diagnostic information (eigenvalues, dimensions)
    """
    E = np.asarray(E, dtype=np.float64)
    C, T = E.shape

    R_raw = np.eye(C, dtype=np.float64)

    # Pairwise Kendall tau-b (handles ties)
    for i in range(C):
        for j in range(i+1, C):
            tau, _ = stats.kendalltau(E[i, :], E[j, :], nan_policy='omit', method='auto')
            if np.isnan(tau):
                tau = 0.0 if nan_to_zero else np.nan
            R_raw[i, j] = R_raw[j, i] = float(tau)

    # Adaptive shrinkage to SPD
    if lam_floor is None:
        lam_floor = max(0.05, C / max(T, 1))
    lam, lam_needed = _adaptive_shrink_lambda_for_spd(R_raw, lam_floor, eps=eps, lam_cap=lam_cap)
    R = (1.0 - lam) * R_raw + lam * np.eye(C)
    R = _symmetrize(R)

    # Final SPD check
    if not _is_spd(R, tol=eps):
        lam2 = min(lam_cap, lam + 1e-3)
        R = (1.0 - lam2) * R_raw + lam2 * np.eye(C)
        lam = lam2

    info = {
        "C": C, "T": T,
        "lam_floor": lam_floor, "lam_needed": lam_needed,
        "min_eig_raw": float(np.min(np.linalg.eigvalsh(R_raw))),
        "min_eig_R": float(np.min(np.linalg.eigvalsh(R))),
    }
    return R, R_raw, float(lam), info


# ============================================================
# SPD AGGREGATION: Log-Euclidean Mean & Distance
# ============================================================

def log_euclidean_spd_mean(
    mats,
    renormalize_correlation=True,
    ensure_input_spd=True,
    eps=1e-8,
    lam_floor=0.05,
    lam_cap=0.99,
):
    """
    Compute log-Euclidean mean of SPD matrices on the Riemannian manifold.

    The log-Euclidean mean is computed as:
        Mean(A_1, ..., A_N) = exp(mean(log(A_1), ..., log(A_N)))

    This respects the geometry of the SPD manifold and is the proper way
    to average coactivation matrices across trials/subjects.

    Parameters
    ----------
    mats : ndarray, shape (N, C, C)
        Stack of N SPD matrices
    renormalize_correlation : bool
        If True, force output to have diag = 1
    ensure_input_spd : bool
        If True, apply shrinkage to ensure each input is SPD
    eps : float
        Minimum eigenvalue threshold
    lam_floor : float
        Minimum shrinkage for fixing non-SPD inputs
    lam_cap : float
        Maximum shrinkage

    Returns
    -------
    R_bar : ndarray, shape (C, C)
        SPD mean matrix
    details : dict
        Diagnostic information (number fixed, shrinkage values)
    """
    mats = np.asarray(mats, dtype=np.float64)
    assert mats.ndim == 3 and mats.shape[1] == mats.shape[2], "mats must be (N, C, C)"
    N, C, _ = mats.shape

    logs = np.zeros((N, C, C), dtype=np.float64)
    spd_fixed = 0
    used_lams = []

    for k in range(N):
        A = _symmetrize(mats[k])

        if ensure_input_spd and not _is_spd(A, tol=eps):
            # Check if correlation-like (diag ~ 1)
            diag = np.diag(A)
            if np.allclose(diag, np.ones_like(diag), atol=1e-4, rtol=0):
                # Shrink toward identity
                lam, _ = _adaptive_shrink_lambda_for_spd(A, lam_floor, eps=eps, lam_cap=lam_cap)
                A = (1.0 - lam) * A + lam * np.eye(C)
                used_lams.append(lam)
            else:
                # Covariance-like: add jitter
                wmin = float(np.min(np.linalg.eigvalsh(A)))
                jitter = (eps - wmin) + 1e-8 if wmin < eps else 0.0
                A = A + jitter * np.eye(C)
                used_lams.append(np.nan)
            spd_fixed += 1

        logs[k] = _spd_logm(A, eps=eps)

    # Mean in log space
    Lbar = np.mean(logs, axis=0)
    R_bar = _spd_expm(Lbar)
    R_bar = _symmetrize(R_bar)

    if renormalize_correlation:
        R_bar = _renormalize_to_correlation(R_bar)

    details = {
        "N": N, "C": C,
        "spd_fixed_count": spd_fixed,
        "used_lams": used_lams,
    }
    return R_bar, details


def log_euclidean_spd_distance(A, B, eps=1e-8):
    """
    Compute log-Euclidean distance between two SPD matrices.

    Distance is defined as: d_LE(A, B) = ||log(A) - log(B)||_F
    where ||.||_F is the Frobenius norm.

    Parameters
    ----------
    A, B : ndarray, shape (C, C)
        SPD matrices
    eps : float
        Minimum eigenvalue threshold

    Returns
    -------
    distance : float
        Log-Euclidean distance

    Raises
    ------
    ValueError
        If either input is not SPD
    """
    A = _symmetrize(A)
    B = _symmetrize(B)
    if not _is_spd(A, tol=eps) or not _is_spd(B, tol=eps):
        raise ValueError("Inputs to log_euclidean_spd_distance must be SPD.")
    LA = _spd_logm(A, eps=eps)
    LB = _spd_logm(B, eps=eps)
    return float(np.linalg.norm(LA - LB, ord="fro"))


# ============================================================
# COMPOSITIONAL DATA: Relative Strength Aggregation
# ============================================================

def mean_then_renormalize(A_stack, eps=1e-12):
    """
    Arithmetic mean of relative strength vectors with L1 renormalization.

    Simple averaging approach suitable when vectors are already on the
    probability simplex (sum to 1).

    Parameters
    ----------
    A_stack : ndarray, shape (N, C)
        Stack of N relative strength vectors
    eps : float
        Small value to prevent division by zero

    Returns
    -------
    a_bar : ndarray, shape (C,)
        L1-normalized mean vector
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    m = A_stack.mean(axis=0)
    total = m.sum()
    return m / (total + eps)


def clr_mean(A_stack, eps=1e-12):
    """
    Geometric mean for compositional data using centered log-ratio (CLR).

    Proper way to average compositional data that respects the Aitchison
    geometry of the probability simplex.

    Parameters
    ----------
    A_stack : ndarray, shape (N, C)
        Stack of N relative strength vectors
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    a_bar : ndarray, shape (C,)
        L1-normalized geometric mean
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    X = np.clip(A_stack, eps, None)
    logX = np.log(X)
    g = np.exp(logX.mean(axis=0))
    return g / (g.sum() + eps)


def clr(x, eps=1e-12, normalize=True):
    """
    Centered log-ratio transform for compositional data.

    CLR maps compositional data from the simplex to Euclidean space where
    standard statistical operations are valid.

    Parameters
    ----------
    x : ndarray, shape (C,)
        Compositional vector (relative strength)
    eps : float
        Small value to avoid log(0)
    normalize : bool
        If True, L1-normalize x before CLR

    Returns
    -------
    z : ndarray, shape (C,)
        CLR-transformed vector (sums to 0)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if normalize:
        s = x.sum()
        x = x / (s + eps)
    lx = np.log(np.clip(x, eps, None))
    return lx - lx.mean()


def log_ratio_diff(a, b, eps=1e-12, normalize=True):
    """
    Component-wise log-ratio difference between two compositions.

    Computes log(a_i / b_i) for each component i. Useful for comparing
    muscle activation patterns between gestures.

    Parameters
    ----------
    a, b : ndarray, shape (C,)
        Compositional vectors
    eps : float
        Small value to avoid log(0)
    normalize : bool
        If True, L1-normalize before computing ratios

    Returns
    -------
    r : ndarray, shape (C,)
        Log-ratio differences
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    if normalize:
        a = a / (a.sum() + eps)
        b = b / (b.sum() + eps)
    return np.log(np.clip(a, eps, None)) - np.log(np.clip(b, eps, None))


def clr_difference(a, b, eps=1e-12, normalize=True):
    """
    CLR difference between two compositions.

    Equals clr(a) - clr(b), which is equivalent to log(a/b) minus its mean.

    Parameters
    ----------
    a, b : ndarray, shape (C,)
        Compositional vectors
    eps : float
        Small value to avoid log(0)
    normalize : bool
        If True, L1-normalize inputs

    Returns
    -------
    d : ndarray, shape (C,)
        CLR difference (sums to 0)
    """
    return clr(a, eps=eps, normalize=normalize) - clr(b, eps=eps, normalize=normalize)


def aitchison_distance(a, b, eps=1e-12, normalize=True):
    """
    Aitchison distance between two compositional vectors.

    The natural distance metric for compositional data, defined as:
        d_A(a, b) = ||clr(a) - clr(b)||_2

    Parameters
    ----------
    a, b : ndarray, shape (C,)
        Compositional vectors (relative strengths)
    eps : float
        Small value to avoid log(0)
    normalize : bool
        If True, L1-normalize inputs

    Returns
    -------
    distance : float
        Aitchison distance
    """
    dvec = clr_difference(a, b, eps=eps, normalize=normalize)
    return float(np.linalg.norm(dvec, ord=2))


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def compute_profile_from_window(X, fs, method="kendall"):
    """
    Complete pipeline: preprocess EMG and compute coactivation profile.

    Convenience function that chains preprocessing, strength computation,
    and coactivation analysis for a single window of EMG data.

    Parameters
    ----------
    X : ndarray, shape (C, T)
        Raw EMG data (channels x time)
    fs : float
        Sampling frequency (Hz)
    method : str
        'kendall' for Kendall's tau-b or 'spearman' for Spearman correlation

    Returns
    -------
    E_smooth : ndarray, shape (C, T)
        Smoothed envelope
    E_wins : ndarray, shape (C, T)
        Winsorized envelope
    s : ndarray, shape (C,)
        Absolute strength per channel
    a : ndarray, shape (C,)
        Relative strength (L1-normalized)
    R : ndarray, shape (C, C)
        SPD coactivation matrix
    R_raw : ndarray, shape (C, C)
        Raw correlation before shrinkage
    lam : float
        Shrinkage parameter used
    info : dict
        Diagnostic information
    """
    E_smooth, E_wins = preprocess_emg_envelope(X, fs)
    s, a = relative_strength(E_wins)

    if method.lower() == "kendall":
        R, R_raw, lam, info = kendall_coactivation_spd(E_wins)
    elif method.lower() == "spearman":
        R, R_raw, lam, info = spearman_coactivation_spd(E_wins)
    else:
        raise ValueError("method must be 'kendall' or 'spearman'")

    return E_smooth, E_wins, s, a, R, R_raw, lam, info


# ============================================================
# DATA LOADING: ROAM Dataset Specific
# ============================================================

def load_and_filter_gesture_data_csv(file_path, target_label):
    """
    Load ROAM-EMG CSV file and filter by gesture label.

    For open/close gestures, extracts only samples with the target label
    to capture active gesture periods. For relax, uses all label=0 samples.

    Parameters
    ----------
    file_path : Path
        Path to ROAM-EMG CSV file
    target_label : int
        Gesture label (0=relax, 1=open, 2=close)

    Returns
    -------
    X : ndarray, shape (C, T) or None
        Filtered EMG data (channels x time), or None if no valid samples
    """
    df = pd.read_csv(file_path)

    # Extract ground truth and EMG channels
    labels = df['gt'].values
    emg_cols = [f'emg{i}' for i in range(8)]
    emg = df[emg_cols].values  # (T, 8)

    # Filter by label
    if target_label == 0:
        mask = labels == 0  # All relax samples
    else:
        mask = labels == target_label  # Only active gesture samples

    filtered_emg = emg[mask]

    if len(filtered_emg) == 0:
        return None

    # Transpose to (C, T) as expected by coactivation functions
    X = filtered_emg.T
    return X


# ============================================================
# PROFILE COMPUTATION: Subject & Population Level
# ============================================================

def compute_subject_profiles(subject_dir):
    """
    Compute coactivation profiles for one subject across all trials.

    Aggregates coactivation matrices and relative strength vectors from
    multiple trials (different arm postures) for each gesture.

    Parameters
    ----------
    subject_dir : Path
        Directory containing subject's CSV files

    Returns
    -------
    profiles : dict
        {gesture_name: {'R': coactivation_matrix, 'a': relative_strength, 'n_trials': int}}
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
            # Aggregate across all trials using log-Euclidean mean
            R_agg, _ = log_euclidean_spd_mean(all_R, renormalize_correlation=True)
            a_agg = mean_then_renormalize(all_a)

            subject_profiles[gesture_name] = {
                'R': R_agg,
                'a': a_agg,
                'n_trials': n_trials
            }

    return subject_profiles


# ============================================================
# VISUALIZATION: Plotting Functions
# ============================================================

def plot_coactivation_matrices(profiles, title_prefix, output_path):
    """
    Generate comprehensive visualization of coactivation profiles.

    Creates two plots:
    1. Coactivation matrices: Kendall tau heatmaps for each gesture
    2. Relative strength vectors: Bar plots with difference comparisons

    Parameters
    ----------
    profiles : dict
        {gesture_name: {'R': matrix, 'a': vector}}
    title_prefix : str
        Title prefix (e.g., "Subject s01" or "Population-Level")
    output_path : Path
        Base path for saving plots (will create _matrices.png and _vectors.png)
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    cmap = plt.cm.viridis_r  # Lighter = higher correlation

    gesture_list = ["relax", "open", "close"]

    # Get adaptive scale for vectors
    all_vectors = []
    for gesture in gesture_list:
        if gesture in profiles:
            all_vectors.append(profiles[gesture]['a'])

    vmin_vec = 0.0
    vmax_vec = max([np.max(a) for a in all_vectors])

    # ===== PLOT 1: Coactivation Matrices =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            R = profiles[gesture]['R'].copy()

            # Scale based on off-diagonal values (diagonal always 1.0)
            off_diag_mask = ~np.eye(R.shape[0], dtype=bool)
            vmin_mat = np.min(R[off_diag_mask])
            vmax_mat = np.max(R[off_diag_mask])

            im = axes[idx].imshow(R, cmap=cmap, vmin=vmin_mat, vmax=vmax_mat,
                                  aspect='auto', interpolation='nearest')

            axes[idx].set_title(f"{gesture.capitalize()}", fontsize=14, fontweight='bold', pad=15)
            axes[idx].set_xlabel("EMG Channel", fontsize=12)
            axes[idx].set_ylabel("EMG Channel", fontsize=12)
            axes[idx].set_xticks(np.arange(8))
            axes[idx].set_yticks(np.arange(8))

            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label('Kendall Tau', rotation=270, labelpad=20, fontsize=11)

    if "Population-Level" in title_prefix:
        suptitle = "Coactivation Matrices - Kendall Tau (ROAM)"
    else:
        suptitle = f"{title_prefix} - Coactivation Matrices - Kendall Tau (ROAM)"

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    matrix_path = output_path.parent / f"{output_path.stem}_matrices.png"
    plt.savefig(matrix_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {matrix_path}")

    # ===== PLOT 2: Relative Strength Vectors + Compositional Differences =====
    fig, axes = plt.subplots(1, 7, figsize=(35, 5))

    channels = np.arange(8)

    # Plot base gestures (columns 0-2)
    for idx, gesture in enumerate(gesture_list):
        if gesture in profiles:
            a = profiles[gesture]['a']

            norm = plt.Normalize(vmin=vmin_vec, vmax=vmax_vec)
            colors = cmap(norm(a))

            bars = axes[idx].bar(channels, a, color=colors, edgecolor='black', linewidth=1.2)

            axes[idx].set_title(f"{gesture.capitalize()}", fontsize=14, fontweight='bold', pad=15)
            axes[idx].set_xlabel("EMG Channel", fontsize=12)
            axes[idx].set_ylabel("Relative Strength", fontsize=12)
            axes[idx].set_xticks(channels)
            axes[idx].set_ylim([vmin_vec, vmax_vec * 1.05])
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[idx].set_axisbelow(True)

            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}',
                              ha='center', va='bottom', fontsize=9, rotation=0)

    # Compute compositional differences
    if all(g in profiles for g in ['relax', 'open', 'close']):
        clr_open_relax = clr_difference(profiles['open']['a'], profiles['relax']['a'])
        clr_close_relax = clr_difference(profiles['close']['a'], profiles['relax']['a'])
        lr_open_relax = log_ratio_diff(profiles['open']['a'], profiles['relax']['a'])
        lr_close_relax = log_ratio_diff(profiles['close']['a'], profiles['relax']['a'])

        # Plot 4: CLR difference (open - relax)
        colors_clr_open = ['green' if x >= 0 else 'red' for x in clr_open_relax]
        bars = axes[3].bar(channels, clr_open_relax, color=colors_clr_open,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[3].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[3].set_title("CLR: Open - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[3].set_xlabel("EMG Channel", fontsize=12)
        axes[3].set_ylabel("CLR Difference", fontsize=12)
        axes[3].set_xticks(channels)
        axes[3].grid(axis='y', alpha=0.3, linestyle='--')
        axes[3].set_axisbelow(True)

        for bar, val in zip(bars, clr_open_relax):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

        # Plot 5: CLR difference (close - relax)
        colors_clr_close = ['green' if x >= 0 else 'red' for x in clr_close_relax]
        bars = axes[4].bar(channels, clr_close_relax, color=colors_clr_close,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[4].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[4].set_title("CLR: Close - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[4].set_xlabel("EMG Channel", fontsize=12)
        axes[4].set_ylabel("CLR Difference", fontsize=12)
        axes[4].set_xticks(channels)
        axes[4].grid(axis='y', alpha=0.3, linestyle='--')
        axes[4].set_axisbelow(True)

        for bar, val in zip(bars, clr_close_relax):
            height = bar.get_height()
            axes[4].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

        # Plot 6: Log-ratio difference (open - relax)
        colors_lr_open = ['green' if x >= 0 else 'red' for x in lr_open_relax]
        bars = axes[5].bar(channels, lr_open_relax, color=colors_lr_open,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[5].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[5].set_title("Log-Ratio: Open - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[5].set_xlabel("EMG Channel", fontsize=12)
        axes[5].set_ylabel("Log-Ratio Difference", fontsize=12)
        axes[5].set_xticks(channels)
        axes[5].grid(axis='y', alpha=0.3, linestyle='--')
        axes[5].set_axisbelow(True)

        for bar, val in zip(bars, lr_open_relax):
            height = bar.get_height()
            axes[5].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

        # Plot 7: Log-ratio difference (close - relax)
        colors_lr_close = ['green' if x >= 0 else 'red' for x in lr_close_relax]
        bars = axes[6].bar(channels, lr_close_relax, color=colors_lr_close,
                          edgecolor='black', linewidth=1.2, alpha=0.7)
        axes[6].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[6].set_title("Log-Ratio: Close - Relax", fontsize=14, fontweight='bold', pad=15)
        axes[6].set_xlabel("EMG Channel", fontsize=12)
        axes[6].set_ylabel("Log-Ratio Difference", fontsize=12)
        axes[6].set_xticks(channels)
        axes[6].grid(axis='y', alpha=0.3, linestyle='--')
        axes[6].set_axisbelow(True)

        for bar, val in zip(bars, lr_close_relax):
            height = bar.get_height()
            axes[6].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

    if "Population-Level" in title_prefix:
        suptitle = "Relative Strength & Compositional Differences (ROAM)"
    else:
        suptitle = f"{title_prefix} - Relative Strength & Compositional Differences (ROAM)"

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
            dist = aitchison_distance(a_subject, a_population)

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
            dist = aitchison_distance(
                subject_profiles[gesture]['a'],
                population_profiles[gesture]['a']
            )
            distances[gesture] = dist

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
        {subject_name: {gesture_name: {'R': matrix, 'a': vector}}}
    subject_to_hold_out : str
        Name of subject to exclude from population mean

    Returns
    -------
    population_profiles_loo : dict
        Population profiles computed without the held-out subject
    """
    population_profiles_loo = {}

    for gesture in ["relax", "open", "close"]:
        all_R = []
        all_a = []

        for subject_name, profiles in all_subject_profiles.items():
            if subject_name == subject_to_hold_out:
                continue  # Skip the held-out subject

            if gesture in profiles:
                all_R.append(profiles[gesture]['R'])
                all_a.append(profiles[gesture]['a'])

        if len(all_R) > 0:
            R_pop, _ = log_euclidean_spd_mean(all_R, renormalize_correlation=True)
            a_pop = mean_then_renormalize(all_a)

            population_profiles_loo[gesture] = {
                'R': R_pop,
                'a': a_pop,
                'n_subjects': len(all_R)
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
    print("ROAM-EMG COACTIVATION PROFILE COMPUTATION (KENDALL TAU)")
    print("="*60)
    print()

    # Get all subject directories
    subject_dirs = sorted([d for d in DATA_ROOT.iterdir()
                          if d.is_dir() and d.name.startswith('s')])
    print(f"Found {len(subject_dirs)} subjects")
    print()

    # Step 1: Test on single subject
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

    # Plot population results (now includes compositional differences)
    plot_coactivation_matrices(
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
        loo_save_dict[f"{gesture}_R"] = data['R']
        loo_save_dict[f"{gesture}_a"] = data['a']
        loo_save_dict[f"{gesture}_n_subjects"] = data['n_subjects']

    loo_npz_path = OUTPUT_DIR / "leave_one_out_analysis" / f"population_without_{loo_subject}.npz"
    np.savez(loo_npz_path, **loo_save_dict)
    print(f"\nSaved: {loo_npz_path}")

    # Plot leave-one-out population profiles (now includes compositional differences)
    plot_coactivation_matrices(
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
    plot_coactivation_matrices(
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

    print()
    print("="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print(f"Leave-one-out analysis saved to: {(OUTPUT_DIR / 'leave_one_out_analysis').absolute()}")


if __name__ == "__main__":
    main()
