import numpy as np
from scipy import signal, stats
from numpy.linalg import eigh


"""
Some examples of using this file
# X: (C, T) raw EMG for one gesture window, fs ~ 200.0
E_smooth, E_wins, s, a, R, R_raw, lam, info = compute_profile_from_window(X, fs)

# Multiple windows for the same subject (same gesture):
Rs_subject = []  # collect (C, C)
As_subject = []  # collect (C,)
for Xk in subject_windows:
    _, Ew, sk, ak, Rk, _, _, _ = compute_profile_from_window(Xk, fs)
    Rs_subject.append(Rk)
    As_subject.append(ak)
Rs_subject = np.stack(Rs_subject, axis=0)   # (N_trials, C, C)
As_subject = np.stack(As_subject, axis=0)   # (N_trials, C)

# Subject-level means
R_subject_mean, _ = log_euclidean_spd_mean(Rs_subject, renormalize_correlation=True)
a_subject_mean_arith = mean_then_renormalize(As_subject)
a_subject_mean_clr   = clr_mean(As_subject)

# Grand (cohort) means across subjects
R_grand_mean, _ = log_euclidean_spd_mean(np.stack(Rs_all_subject_means, axis=0), renormalize_correlation=True)
a_grand_arith = mean_then_renormalize(np.stack(a_all_subject_means_arith, axis=0))
a_grand_clr   = clr_mean(np.stack(a_all_subject_means_clr,   axis=0))

# One-number difference between two coactivation matrices:
d_LE = log_euclidean_spd_distance(R_trial, R_subject_mean)

"""


# ------------------------------------------------------------
# 0) Utilities
# ------------------------------------------------------------

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
    Input must be symmetric positive definite.
    Returns a symmetric real matrix.
    """
    a = _symmetrize(a)
    if not _is_spd(a, tol=eps):
        raise ValueError("spd_logm: input is not SPD.")
    w, v = eigh(a)  # a = v diag(w) v^T
    w = np.maximum(w, eps)
    lw = np.log(w)
    return (v * lw) @ v.T  # v diag(lw) v^T

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
    Convert a symmetric SPD matrix to a correlation-like matrix (diag=1)
    by D^{-1/2} A D^{-1/2}. Preserves SPD.
    """
    a = _symmetrize(a)
    d = np.diag(a).copy()
    d = np.maximum(d, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    c = D_inv_sqrt @ a @ D_inv_sqrt
    return _symmetrize(c)

# ------------------------------------------------------------
# 1) Preprocess: Bandpass 10–95 Hz, rectify, LP 6 Hz, winsorize
# ------------------------------------------------------------

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
    X : (C, T) raw EMG (int8/uint8/float). Will be processed in float64.
    fs : sampling rate (Hz).
    bp_low, bp_high, bp_order : bandpass settings.
    lp_cutoff, lp_order : low-pass settings for envelope.
    winsor_lo, winsor_hi : per-channel percentile clipping (0..100).

    Returns
    -------
    E_smooth : (C, T) smoothed envelope (float64)
    E_wins   : (C, T) winsorized envelope (float64)
    """
    X = np.asarray(X, dtype=np.float64)
    C, T = X.shape

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

# ------------------------------------------------------------
# 2) Relative strength (absolute s and L1-normalized a)
# ------------------------------------------------------------

def relative_strength(E: np.ndarray, eps: float = 1e-12):
    """
    Compute per-channel absolute mean s and L1-normalized relative vector a.

    Parameters
    ----------
    E : (C, T) envelope (winsorized recommended).
    eps : small constant to avoid zero division.

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

# ------------------------------------------------------------
# 3) Coactivation via Spearman correlation with guaranteed SPD
# ------------------------------------------------------------

def _adaptive_shrink_lambda_for_spd(R_raw: np.ndarray, lam_floor: float, eps: float = 1e-8, lam_cap: float = 0.5):
    """
    Choose lambda >= lam_floor so that R = (1-lambda) R_raw + lambda I is SPD
    with eigenvalues >= eps. Uses the eigenvalues of R_raw analytically:
    eig(R) = (1-lambda)*eig(R_raw) + lambda.

    Returns
    -------
    lam : chosen shrinkage in [lam_floor, lam_cap]
    lam_needed : minimal lambda needed to lift the smallest eigenvalue to eps
    """
    R_raw = _symmetrize(R_raw)
    w = np.linalg.eigvalsh(R_raw)
    lam_needed = 0.0
    lambda_min = np.min(w)
    # Want (1-lam)*lambda_min + lam >= eps  => lam >= (eps - lambda_min)/(1 - lambda_min)
    if lambda_min < eps:
        denom = max(1.0 - lambda_min, eps)
        lam_needed = (eps - lambda_min) / denom
    lam = max(lam_floor, lam_needed)
    lam = float(np.clip(lam, 0.0, lam_cap))
    return lam, lam_needed

def spearman_coactivation_spd(E: np.ndarray, lam_floor: float | None = None, eps: float = 1e-8):
    """
    Spearman coactivation matrix (correlation on ranks) with adaptive shrinkage
    to guarantee SPD (and keep diagonal = 1).

    Parameters
    ----------
    E : (C, T) envelope (winsorized recommended).
    lam_floor : if None, defaults to max(0.05, C/T); otherwise a nonnegative floor.
    eps : small constant for numerical stability and SPD floor.

    Returns
    -------
    R : (C, C) SPD correlation-like matrix (diag=1)
    R_raw : (C, C) raw Spearman correlation before shrinkage
    lam : float, final shrinkage used
    info : dict with diagnostics
    """
    E = np.asarray(E, dtype=np.float64)
    C, T = E.shape

    # 1) Rank each channel across time (Spearman)
    Q = np.empty_like(E)
    for i in range(C):
        Q[i, :] = stats.rankdata(E[i, :], method="average")

    # 2) Standardize row-wise
    Qm = Q.mean(axis=1, keepdims=True)   # (C, 1)
    Qs = Q.std(axis=1, keepdims=True)    # (C, 1)
    # Handle constant rows -> std ~ 0
    Qs = np.where(Qs < eps, 1.0, Qs)
    Z = (Q - Qm) / Qs                    # (C, T)

    # 3) Raw Spearman correlation (Pearson on ranks)
    denom = max(T - 1, 1)
    R_raw = (Z @ Z.T) / denom
    R_raw = _symmetrize(R_raw)
    # Force exact diagonal = 1 for correlation semantics
    np.fill_diagonal(R_raw, 1.0)

    # 4) Adaptive shrinkage to guarantee SPD: R = (1-lam)R_raw + lam*I
    if lam_floor is None:
        lam_floor = max(0.05, C / max(T, 1))  # simple heuristic floor
    lam, lam_needed = _adaptive_shrink_lambda_for_spd(R_raw, lam_floor, eps=eps, lam_cap=0.5)
    R = (1.0 - lam) * R_raw + lam * np.eye(C)
    R = _symmetrize(R)  # guard tiny asymmetry

    # Sanity: ensure SPD
    if not _is_spd(R, tol=eps):
        # As a last resort, add tiny jitter (does not change diag exactly).
        # But adding jitter would change the diagonal; we avoid that.
        # Instead, slightly increase lam within cap:
        lam2 = min(0.5, lam + 1e-3)
        R = (1.0 - lam2) * R_raw + lam2 * np.eye(C)
        lam = lam2

    info = {
        "C": C,
        "T": T,
        "lam_floor": lam_floor,
        "lam_needed": lam_needed,
        "min_eig_raw": float(np.min(np.linalg.eigvalsh(R_raw))),
        "min_eig_R": float(np.min(np.linalg.eigvalsh(R))),
    }
    return R, R_raw, float(lam), info

# ------------------------------------------------------------
# 4) Log-Euclidean SPD mean & distance
# ------------------------------------------------------------

def log_euclidean_spd_mean(
    mats: np.ndarray,
    renormalize_correlation: bool = True,
    ensure_input_spd: bool = True,
    eps: float = 1e-8,
    lam_floor: float = 0.05,
):
    """
    Log-Euclidean mean of a stack of SPD matrices (N, C, C).

    If `ensure_input_spd` is True, each matrix is first made SPD via
    a minimal shrinkage towards identity that preserves diag=1 when possible.
    If inputs are correlation-like (diag~1), set `renormalize_correlation=True`
    to reimpose diag=1 after the mean.

    Parameters
    ----------
    mats : (N, C, C) array of SPD (or near-SPD) matrices
    renormalize_correlation : re-normalize diagonal to 1 after mean
    ensure_input_spd : enforce SPD on each input by small shrink if needed
    eps : SPD floor
    lam_floor : base shrink floor if SPD enforcement is needed

    Returns
    -------
    R_bar : (C, C) SPD mean
    details : dict with counts and diagnostics
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
            # Try shrinkage towards I while preserving diag=1 if it's correlation-like
            # Detect if diagonal is ~1
            diag = np.diag(A)
            if np.allclose(diag, np.ones_like(diag), atol=1e-4, rtol=0):
                # correlation-like -> use adaptive shrink that keeps diag=1
                lam, _ = _adaptive_shrink_lambda_for_spd(A, lam_floor, eps=eps, lam_cap=0.5)
                A = (1.0 - lam) * A + lam * np.eye(C)
                used_lams.append(lam)
            else:
                # covariance-like -> add jitter towards identity
                wmin = float(np.min(np.linalg.eigvalsh(A)))
                jitter = (eps - wmin) + 1e-8 if wmin < eps else 0.0
                A = A + jitter * np.eye(C)
                used_lams.append(np.nan)
            spd_fixed += 1

        logs[k] = _spd_logm(A, eps=eps)

    Lbar = np.mean(logs, axis=0)         # (C, C), symmetric
    R_bar = _spd_expm(Lbar)              # (C, C), SPD
    R_bar = _symmetrize(R_bar)

    if renormalize_correlation:
        R_bar = _renormalize_to_correlation(R_bar)

    details = {
        "N": N,
        "C": C,
        "spd_fixed_count": spd_fixed,
        "used_lams": used_lams,
    }
    return R_bar, details

def log_euclidean_spd_distance(A: np.ndarray, B: np.ndarray, eps: float = 1e-8):
    """
    Log-Euclidean distance between two SPD matrices:
        d_LE(A,B) = || log(A) - log(B) ||_F

    Parameters
    ----------
    A, B : (C, C) SPD matrices
    eps : SPD floor

    Returns
    -------
    d : float
    """
    A = _symmetrize(A)
    B = _symmetrize(B)
    if not _is_spd(A, tol=eps) or not _is_spd(B, tol=eps):
        raise ValueError("Inputs to log_euclidean_spd_distance must be SPD.")
    LA = _spd_logm(A, eps=eps)
    LB = _spd_logm(B, eps=eps)
    return float(np.linalg.norm(LA - LB, ord="fro"))

# ------------------------------------------------------------
# 5) Aggregators for relative strength vectors
# ------------------------------------------------------------

def mean_then_renormalize(A_stack: np.ndarray, eps: float = 1e-12):
    """
    Arithmetic mean of relative strength vectors then L1-renormalize.

    Parameters
    ----------
    A_stack : (N, C) array of relative strength vectors (each sums ~1)
    eps : small constant

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
    CLR / geometric mean for compositions (relative strength vectors).

    Equivalent to L1-normalized geometric mean across samples:
        g[i] = exp( mean_k log(a_k[i]) )
        a_bar = g / sum(g)

    Parameters
    ----------
    A_stack : (N, C), each row nonnegative (ideally strictly positive)
    eps : small positive floor to avoid log(0)

    Returns
    -------
    a_bar : (C,) CLR/geometric-mean aggregator (sums to 1)
    """
    A_stack = np.asarray(A_stack, dtype=np.float64)
    assert A_stack.ndim == 2, "A_stack must be (N, C)"
    X = np.clip(A_stack, eps, None)  # avoid log(0)
    logX = np.log(X)
    g = np.exp(logX.mean(axis=0))
    return g / (g.sum() + eps)

# ------------------------------------------------------------
# 6) One-stop helpers (optional)
# ------------------------------------------------------------

def compute_profile_from_window(X: np.ndarray, fs: float):
    """
    Convenience: full pipeline for one window.

    Returns
    -------
    E_smooth : (C, T), E_wins : (C, T)
    s : (C,), a : (C,)
    R : (C, C), R_raw : (C, C), lam : float, info : dict
    """
    E_smooth, E_wins = preprocess_emg_envelope(X, fs)
    s, a = relative_strength(E_wins)
    R, R_raw, lam, info = spearman_coactivation_spd(E_wins)  # SPD guaranteed
    return E_smooth, E_wins, s, a, R, R_raw, lam, info
