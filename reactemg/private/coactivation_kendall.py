import numpy as np
from scipy import signal, stats
from numpy.linalg import eigh


# ============================================================
# Utilities
# ============================================================

def _is_symmetric(a, tol=1e-12):
    return np.allclose(a, a.T, atol=tol, rtol=0)

def _symmetrize(a):
    return 0.5 * (a + a.T)

def _is_spd(a, tol=1e-12):
    if not _is_symmetric(a, tol):
        return False
    w = np.linalg.eigvalsh(a)
    return np.min(w) > tol

def _spd_logm(a, eps=1e-12):
    """
    Matrix log for SPD matrices via eigen-decomposition.
    Input must be SPD.
    """
    a = _symmetrize(a)
    if not _is_spd(a, tol=eps):
        raise ValueError("spd_logm: input is not SPD.")
    w, v = eigh(a)               # a = v diag(w) v^T
    w = np.maximum(w, eps)
    lw = np.log(w)
    return (v * lw) @ v.T        # v diag(lw) v^T

def _spd_expm(s):
    """
    Matrix exp for symmetric matrices via eigen-decomposition.
    exp of a symmetric matrix is SPD.
    """
    s = _symmetrize(s)
    w, v = eigh(s)
    ew = np.exp(w)
    return (v * ew) @ v.T

def _renormalize_to_correlation(a, eps=1e-12):
    """
    Force diag=1 (correlation-like) via D^{-1/2} A D^{-1/2}. Preserves SPD.
    """
    a = _symmetrize(a)
    d = np.diag(a).copy()
    d = np.maximum(d, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    c = D_inv_sqrt @ a @ D_inv_sqrt
    return _symmetrize(c)

def _adaptive_shrink_lambda_for_spd(R_raw: np.ndarray, lam_floor: float,
                                    eps: float = 1e-8, lam_cap: float = 0.99):
    """
    Choose lambda >= lam_floor so that R = (1-lambda) R_raw + lambda I is SPD
    with eigenvalues >= eps. Uses the min eigenvalue of R_raw analytically.

    Want: (1-lam)*lambda_min + lam >= eps => lam >= (eps - lambda_min) / (1 - lambda_min)
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
# Preprocessing: Bandpass 10–95 Hz, rectify, LP 6 Hz, winsorize
# ============================================================

def preprocess_emg_envelope(
    X: np.ndarray,
    fs: float,
    bp_low: float = 10.0,
    bp_high: float = 95.0,
    bp_order: int = 4,
    lp_cutoff: float = 6.0,
    lp_order: int = 2,
    winsor_lo: float = 0.5,   # percent
    winsor_hi: float = 99.5,  # percent
):
    """
    EMG preprocessing -> envelope (rectified + 6 Hz LP) and winsorized envelope.

    Parameters
    ----------
    X : (C, T) raw EMG. Processed in float64.
    fs : sampling rate (Hz).

    Returns
    -------
    E_smooth : (C, T) smoothed envelope
    E_wins   : (C, T) winsorized envelope
    """
    X = np.asarray(X, dtype=np.float64)

    # Zero-phase band-pass
    sos_bp = signal.butter(bp_order, [bp_low, bp_high], btype="bandpass", fs=fs, output="sos")
    F = signal.sosfiltfilt(sos_bp, X, axis=1)

    # Rectify
    R = np.abs(F)

    # Zero-phase low-pass (envelope)
    sos_lp = signal.butter(lp_order, lp_cutoff, btype="lowpass", fs=fs, output="sos")
    E_smooth = signal.sosfiltfilt(sos_lp, R, axis=1)

    # Winsorize per channel
    lo = np.percentile(E_smooth, winsor_lo, axis=1, keepdims=True)
    hi = np.percentile(E_smooth, winsor_hi, axis=1, keepdims=True)
    E_wins = np.clip(E_smooth, lo, hi)

    return E_smooth, E_wins


# ============================================================
# Relative strength (absolute s, L1-normalized a)
# ============================================================

def relative_strength(E: np.ndarray, eps: float = 1e-12):
    """
    Compute per-channel absolute mean s and L1-normalized relative vector a.

    Parameters
    ----------
    E : (C, T) envelope (winsorized recommended).

    Returns
    -------
    s : (C,) absolute mean amplitude per channel
    a : (C,) L1-normalized s (sum ~ 1)
    """
    E = np.asarray(E, dtype=np.float64)
    s = E.mean(axis=1)               # (C,)
    total = s.sum()
    a = s / (total + eps)            # (C,)
    return s, a


# ============================================================
# Coactivation matrices (Spearman or Kendall), SPD-guaranteed
# ============================================================

def spearman_coactivation_spd(E: np.ndarray,
                              lam_floor: float | None = None,
                              eps: float = 1e-8,
                              lam_cap: float = 0.99):
    """
    Spearman coactivation (correlation on ranks) with adaptive shrinkage to SPD.

    Returns
    -------
    R : (C, C) SPD correlation-like matrix (diag=1)
    R_raw : (C, C) raw Spearman matrix before shrinkage
    lam : float, final shrinkage used
    info : dict diagnostics
    """
    E = np.asarray(E, dtype=np.float64)
    C, T = E.shape

    # Rank-transform each channel across time
    Q = np.empty_like(E)
    for i in range(C):
        Q[i, :] = stats.rankdata(E[i, :], method="average")  # ties handled

    # Row-standardize
    Qm = Q.mean(axis=1, keepdims=True)   # (C, 1)
    Qs = Q.std(axis=1, keepdims=True)    # (C, 1)
    Qs = np.where(Qs < eps, 1.0, Qs)
    Z  = (Q - Qm) / Qs                   # (C, T)

    # Raw Spearman correlation (Pearson on ranks)
    denom = max(T - 1, 1)
    R_raw = (Z @ Z.T) / denom
    R_raw = _symmetrize(R_raw)
    np.fill_diagonal(R_raw, 1.0)

    # Adaptive shrinkage
    if lam_floor is None:
        lam_floor = max(0.05, C / max(T, 1))
    lam, lam_needed = _adaptive_shrink_lambda_for_spd(R_raw, lam_floor, eps=eps, lam_cap=lam_cap)
    R = (1.0 - lam) * R_raw + lam * np.eye(C)
    R = _symmetrize(R)

    # Final SPD check (should pass)
    if not _is_spd(R, tol=eps):
        # Increase lam a hair, capped
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


def kendall_coactivation_spd(E: np.ndarray,
                             lam_floor: float | None = None,
                             eps: float = 1e-8,
                             lam_cap: float = 0.99,
                             nan_to_zero: bool = True):
    """
    Kendall's tau-b coactivation (pairwise, tie-corrected), with adaptive
    shrinkage to ensure SPD.

    Parameters
    ----------
    E : (C, T) envelope (winsorized recommended).
    lam_floor : if None, defaults to max(0.05, C/T).
    lam_cap : upper bound on lambda (0.99 by default to always allow SPD).
    nan_to_zero : if True, replace NaN taus (e.g., constant channel) with 0.

    Returns
    -------
    R : (C, C) SPD correlation-like matrix (diag=1)
    R_raw : (C, C) raw Kendall tau-b matrix before shrinkage
    lam : float, final shrinkage used
    info : dict diagnostics
    """
    E = np.asarray(E, dtype=np.float64)
    C, T = E.shape

    R_raw = np.eye(C, dtype=np.float64)
    # Pairwise Kendall tau-b (SciPy handles ties)
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

    # Final SPD check (should pass)
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
# Log-Euclidean SPD mean & distance
# ============================================================

def log_euclidean_spd_mean(
    mats: np.ndarray,
    renormalize_correlation: bool = True,
    ensure_input_spd: bool = True,
    eps: float = 1e-8,
    lam_floor: float = 0.05,
    lam_cap: float = 0.99,
):
    """
    Log-Euclidean mean of a stack of SPD matrices (N, C, C).

    If ensure_input_spd=True, each matrix is first made SPD via
    minimal shrinkage towards identity (keeps diag=1 for correlation-like inputs).

    Returns
    -------
    R_bar : (C, C) SPD mean (optionally renormalized to diag=1)
    details : dict with diagnostics
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
            # If correlation-like (diag ~ 1), shrink toward I without changing diag
            diag = np.diag(A)
            if np.allclose(diag, np.ones_like(diag), atol=1e-4, rtol=0):
                lam, _ = _adaptive_shrink_lambda_for_spd(A, lam_floor, eps=eps, lam_cap=lam_cap)
                A = (1.0 - lam) * A + lam * np.eye(C)
                used_lams.append(lam)
            else:
                # Covariance-like -> add jitter to lift eigenvalues
                wmin = float(np.min(np.linalg.eigvalsh(A)))
                jitter = (eps - wmin) + 1e-8 if wmin < eps else 0.0
                A = A + jitter * np.eye(C)
                used_lams.append(np.nan)
            spd_fixed += 1

        logs[k] = _spd_logm(A, eps=eps)

    Lbar = np.mean(logs, axis=0)   # (C, C)
    R_bar = _spd_expm(Lbar)        # (C, C), SPD
    R_bar = _symmetrize(R_bar)

    if renormalize_correlation:
        R_bar = _renormalize_to_correlation(R_bar)

    details = {
        "N": N, "C": C,
        "spd_fixed_count": spd_fixed,
        "used_lams": used_lams,
    }
    return R_bar, details


def log_euclidean_spd_distance(A: np.ndarray, B: np.ndarray, eps: float = 1e-8):
    """
    Log-Euclidean distance: d_LE(A,B) = || log(A) - log(B) ||_F
    """
    A = _symmetrize(A)
    B = _symmetrize(B)
    if not _is_spd(A, tol=eps) or not _is_spd(B, tol=eps):
        raise ValueError("Inputs to log_euclidean_spd_distance must be SPD.")
    LA = _spd_logm(A, eps=eps)
    LB = _spd_logm(B, eps=eps)
    return float(np.linalg.norm(LA - LB, ord="fro"))


# ============================================================
# Aggregators for relative strength vectors
# ============================================================

def mean_then_renormalize(A_stack: np.ndarray, eps: float = 1e-12):
    """
    Arithmetic mean of relative strength vectors then L1-renormalize.

    Parameters
    ----------
    A_stack : (N, C) each row sums ~ 1

    Returns
    -------
    a_bar : (C,) L1-normalized mean
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    m = A_stack.mean(axis=0)
    total = m.sum()
    return m / (total + eps)

def clr_mean(A_stack: np.ndarray, eps: float = 1e-12):
    """
    CLR/geometric mean for compositions (relative strength vectors).

    a_bar[i] ∝ exp( mean_k log( max(a_k[i], eps) ) ), then L1-normalize.
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    X = np.clip(A_stack, eps, None)
    logX = np.log(X)
    g = np.exp(logX.mean(axis=0))
    return g / (g.sum() + eps)


# ============================================================
# One-stop helper
# ============================================================

def compute_profile_from_window(X: np.ndarray, fs: float, method: str = "kendall"):
    """
    Convenience pipeline for a single window.

    Parameters
    ----------
    X : (C, T) raw EMG
    fs : Hz
    method : 'kendall' (tau-b) or 'spearman'

    Returns
    -------
    E_smooth : (C, T)
    E_wins   : (C, T)
    s        : (C,)
    a        : (C,)
    R        : (C, C) SPD
    R_raw    : (C, C)
    lam      : float
    info     : dict
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
