"""
Single‑mode optical homodyne tomography via Maximum Likelihood Estimation (MLE)
==============================================================================

This script builds a practical MLE reconstructor for a single bosonic mode
using QuTiP. It supports raw-shot binning by LO phase, constructs an
approximately informationally complete discretized POVM based on position
projectors |x><x| rotated by the phase operator, and performs the EM‑like
(Hradil) likelihood‑increasing iterations with optional dilution.

Key references:
- U. Leonhardt, Measuring the Quantum State of Light (Cambridge, 1997)
- Z. Hradil, Phys. Rev. A 55, R1561 (1997)
- K. Banaszek et al., Phys. Rev. A 61, 010304 (1999)

Author notes:
- The position eigenkets are represented in a finite Fock space using Hermite
  functions. Discretization uses ∑_i |x_i><x_i| Δx ≈ I on the truncated space.
- For high accuracy, ensure x‑range and grid are wide/fine enough so the
  approximate resolution of identity holds well for your chosen cutoff d.
- If you can assume Gaussian states, prefer a parametric MLE on moments.

Usage (minimal):
----------------
1) Prepare inputs: Fock cutoff `n_max`, LO phases `theta_list`, and raw shots
   `(x_samples, theta_samples)` OR a pre-binned 2D histogram `counts` on a grid
   `x_grid`.
2) Build POVMs with `build_rotated_povm(...)`.
3) Run `rho_hat, info = mle_homodyne(counts, povm, ...)`.

See the `if __name__ == "__main__":` block for a simulated end‑to‑end example.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from math import sqrt, pi
from scipy.special import eval_hermite, factorial
import qutip as qt


# ----------------------------- Utilities ----------------------------------- #

def hermite_gauss(n: int, x: np.ndarray) -> np.ndarray:
    """Harmonic‑oscillator eigenfunction ψ_n(x) in dimensionless units.
    ψ_n(x) = 1 / sqrt(2^n n! sqrt(pi)) * H_n(x) * exp(-x^2/2).
    Returns an array matching x.
    """
    Hn = eval_hermite(n, x)
    coeff = 1.0 / np.sqrt((2.0**n) * factorial(n) * np.sqrt(pi))
    return coeff * Hn * np.exp(-0.5 * x**2)


def position_ket_in_fock(d: int, x: float) -> qt.Qobj:
    """Return the (improper) position eigenket |x> represented in the Fock basis
    truncated to dimension `d`. Builds the amplitude vector ψ_n(x) for n=0..d-1.
    Note: Use the resulting |x⟩ only inside |x⟩⟨x| Δx (POVM discretization).
    """
    x = float(x)
    n = np.arange(d, dtype=int)
    # Compute Hermite polynomials H_n(x) for all n at this scalar x
    H = np.array([eval_hermite(int(k), x) for k in n], dtype=np.float64)
    # Normalization coefficients for harmonic-oscillator eigenfunctions
    coeffs = 1.0 / np.sqrt((2.0 ** n) * factorial(n) * np.sqrt(pi))
    psi = coeffs * H * np.exp(-0.5 * x * x)  # shape (d,)
    return qt.Qobj(psi, dims=[[d], [1]])  # ket


def rotation_operator(d: int, theta: float) -> qt.Qobj:
    """Phase‑space rotation R(θ) = exp(i θ a† a) acting in Fock basis.
    In the Fock basis, R is diagonal with entries e^{i n θ}.
    """
    phases = np.exp(1j * theta * np.arange(d))
    return qt.Qobj(np.diag(phases), dims=[[d], [d]])


@dataclass
class POVM:
    """Container for discretized homodyne POVM: E[j][i] with shape (K, M).
    Each element is a Qobj of shape (d, d), satisfying approximately ∑_{j,i} E[j][i] ≈ K * I.
    """
    elements: List[List[qt.Qobj]]  # K × M list
    thetas: np.ndarray              # (K,)
    x_grid: np.ndarray              # (M,)
    dx: float
    d: int


# ----------------------- POVM construction --------------------------------- #

def build_rotated_povm(d: int,
                        thetas: ArrayLike,
                        x_grid: ArrayLike,
                        dx: Optional[float] = None) -> POVM:
    """Construct discretized homodyne POVM E_{θ, x} = R(θ)† |x⟩⟨x| R(θ) Δx
    on a finite grid of x and phases θ. Returns a `POVM` object.

    Args:
        d: Fock space dimension (n_max + 1).
        thetas: iterable of LO phases in radians (use [0, π)).
        x_grid: sorted array of x sample points covering KL support.
        dx: optional bin width; if None, uses uniform spacing from x_grid.

    Notes:
        - Choose x_grid wide enough: e.g., x_max ≈ 3 * sqrt(2 * n_max + 1).
        - Use Δx small enough for ∑ |x⟩⟨x| Δx ≈ I on the truncated space.
    """
    thetas = np.asarray(thetas, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if dx is None:
        spacings = np.diff(x_grid)
        if not np.allclose(spacings, spacings[0], rtol=1e-4, atol=1e-10):
            raise ValueError("x_grid must be uniform spacing if dx is None.")
        dx = float(spacings[0])

    # Precompute non‑rotated projectors |x><x| Δx in Fock basis.
    proj_x = []  # length M
    for x in x_grid:
        ket_x = position_ket_in_fock(d, float(x))
        proj = ket_x * ket_x.dag()  # |x><x|
        proj_x.append((proj + proj.dag()) * 0.5 * dx)  # hermitize, scale by Δx

    # Rotate them for each θ.
    E = []
    for th in thetas:
        R = rotation_operator(d, th)
        Rdag = R.dag()
        row = [Rdag * P * R for P in proj_x]
        E.append(row)

    return POVM(elements=E, thetas=thetas, x_grid=x_grid, dx=dx, d=d)


# ------------------------- Data preparation -------------------------------- #

def bin_homodyne_data(x_samples: ArrayLike,
                      theta_samples: ArrayLike,
                      thetas: ArrayLike,
                      x_grid: ArrayLike) -> np.ndarray:
    """Bin raw homodyne shots into a (K×M) count matrix.

    Args:
        x_samples: array of outcomes x_k.
        theta_samples: array of LO phases θ_k (radians) for each shot.
        thetas: list/array of phase *bins* (centers); closest bin is used.
        x_grid: positions (bin centers). Shots are binned to nearest x_grid point.

    Returns:
        counts[K, M]: integer counts per (θ_j, x_i).

    Notes:
        This simple nearest‑neighbor binning works well if `x_grid` spacing is
        the same Δx used to create POVM elements. For production, consider true
        histogram bin edges; just keep POVM bins consistent with the histogram.
    """
    x_samples = np.asarray(x_samples, dtype=float)
    theta_samples = np.asarray(theta_samples, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    K = len(thetas); M = len(x_grid)
    counts = np.zeros((K, M), dtype=int)

    # Map each sample to closest θ bin in [0, π), account for periodicity π.
    thetas_mod = (thetas % np.pi)
    ts = (theta_samples % np.pi)
    # Use argmin over circular distance on [0, π)
    def wrap_dist(a, b):
        d = np.abs(a - b)
        return np.minimum(d, np.pi - d)

    # Vectorized mapping via broadcasting (may be memory heavy for huge N).
    # For very large datasets, loop in chunks.
    idx_theta = np.argmin(wrap_dist(ts[:, None], thetas[None, :]), axis=1)

    # Map x to nearest grid point.
    idx_x = np.searchsorted(x_grid, x_samples)
    idx_x = np.clip(idx_x, 1, M - 1)
    left = x_grid[idx_x - 1]
    right = x_grid[idx_x]
    choose_left = (np.abs(x_samples - left) <= np.abs(x_samples - right))
    idx_x = idx_x - choose_left.astype(int)

    # Accumulate counts
    for j, i in zip(idx_theta, idx_x):
        counts[j, i] += 1
    return counts


# ------------------------ Likelihood & MLE core ----------------------------- #

def compute_R_operator(rho: qt.Qobj, counts: np.ndarray, povm: POVM, eps: float = 1e-16) -> qt.Qobj:
    """State-dependent operator R(ρ) per Reháček–Hradil–Knill–Lvovsky (2007):
        R(ρ) = (1/N) * sum_{j,i} [ n_{j,i} / Tr(E_{j,i} ρ) ] * E_{j,i}
    where N = total counts. """
    d = povm.d

    # Backward-compat for old API: 'dilute'≈convex mixing → map to diluted ε
    if dilute is not None:
        try:
            dval = float(dilute)
            # clamp to (0,1)
            dval = min(max(dval, 1e-6), 1.0 - 1e-9)
            epsilon = max(min_epsilon, dval / (1.0 - dval))
            update_rule = "diluted" if update_rule is None else update_rule
            print(f"[mle_homodyne] 'dilute' is deprecated; mapped to diluted MLE with epsilon={epsilon:.3g}.")
        except Exception:
            pass

    # Initialize ρ
    if init is None:
        rho = qt.qeye(d) / d
    else:
        A = init.full()
        A = (A + A.conj().T) * 0.5
        evals, evecs = np.linalg.eigh(A)
        evals = np.clip(evals, 0.0, None)
        evals = evals / max(np.sum(evals), eps)
        rho = qt.Qobj((evecs @ np.diag(evals) @ evecs.conj().T), dims=[[d], [d]])

    ll_hist: List[float] = []
    eps_hist: List[float] = []
    prev_ll = log_likelihood(rho, counts, povm, eps=eps)
    ll_hist.append(prev_ll)

    for it in range(1, max_iter + 1):
        Rop = compute_R_operator(rho, counts, povm, eps=eps)

        if update_rule.lower() in ("rr", "r*r", "rhor"):
            rho_candidate = rr_update(rho, Rop, eps=eps)
            ll = log_likelihood(rho_candidate, counts, povm, eps=eps)
            eps_hist.append(float('inf'))  # RR corresponds to ε→∞
        else:
            # Diluted update with backtracking to ensure monotonicity
            eps_try = max(epsilon, min_epsilon)
            for _ in range(50):  # safeguard
                rho_candidate = diluted_update(rho, Rop, epsilon=eps_try, eps=eps)
                ll = log_likelihood(rho_candidate, counts, povm, eps=eps)
                if (not ensure_monotone) or ll + 1e-12 >= prev_ll:
                    break
                eps_try *= backtrack
                if eps_try < min_epsilon:
                    # Fall back to RR step if diluted can't improve
                    rho_candidate = rr_update(rho, Rop, eps=eps)
                    ll = log_likelihood(rho_candidate, counts, povm, eps=eps)
                    eps_try = float('inf')
                    break
            eps_hist.append(eps_try)

        # Convergence check
        rel = abs(ll - prev_ll) / max(1.0, abs(prev_ll))
        rho = rho_candidate
        ll_hist.append(ll)
        if rel < tol:
            return rho, MLEInfo(ll_history=ll_hist, iter=it, converged=True,
                                eps_history=eps_hist, update_rule=update_rule)
        prev_ll = ll

    return rho, MLEInfo(ll_history=ll_hist, iter=it, converged=False,
                        eps_history=eps_hist, update_rule=update_rule)


# ---------------------------- Simulation helper ---------------------------- #

def simulate_counts(rho_true: qt.Qobj,
                    povm: POVM,
                    shots_per_phase: int,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulate homodyne counts from a true state using the given POVM.

    Args:
        rho_true: true density matrix (Qobj d×d).
        povm: discretized POVM.
        shots_per_phase: number of shots for each θ_j.
        rng: optional numpy Generator for reproducibility.

    Returns:
        counts[K, M]
    """
    if rng is None:
        rng = np.random.default_rng()

    K = len(povm.thetas)
    M = len(povm.x_grid)
    counts = np.zeros((K, M), dtype=int)

    for j in range(K):
        ps = np.array([float((E * rho_true).tr().real) for E in povm.elements[j]])
        ps = np.maximum(ps, 0.0)
        s = ps.sum()
        if s <= 0:
            raise RuntimeError("POVM probabilities sum to zero (bad grid/cutoff).")
        ps /= s
        draws = rng.multinomial(shots_per_phase, ps)
        counts[j, :] = draws
    return counts


# ------------------------------ Recommendations --------------------------- #

def recommended_x_grid(n_max: int, dx: float = 0.05, width_sigma: float = 3.0) -> np.ndarray:
    """Heuristic x‑grid for Fock cutoff n_max.
    Position support ≈ ±sqrt(2 n_max + 1). Use a multiple `width_sigma` of that.
    """
    x_max = width_sigma * np.sqrt(2.0 * n_max + 1.0)
    M = int(np.ceil(2 * x_max / dx)) + 1
    xs = np.linspace(-x_max, x_max, M)
    return xs


def uniform_thetas(n_max: int, oversample: float = 1.5) -> np.ndarray:
    """Phase set on [0, π) with ≥ (2 n_max + 1) points, optionally oversampled."""
    K_min = 2 * n_max + 1
    K = int(np.ceil(oversample * K_min))
    thetas = np.linspace(0.0, np.pi, K, endpoint=False)
    return thetas


# ------------------------------ Demo / Example ----------------------------- #
if __name__ == "__main__":
    # Example: reconstruct a displaced‑squeezed vacuum from simulated data.
    n_max = 12
    d = n_max + 1

    # True state ρ = D(α) S(r) |0⟩
    a = qt.destroy(d)
    alpha = 0.8 + 0.3j
    r = 0.5  # squeeze along x
    S = qt.squeezing(d, r)
    D = qt.displace(d, alpha)
    psi = D * S * qt.basis(d, 0)
    rho_true = qt.ket2dm(psi)

    # Discretization
    thetas = uniform_thetas(n_max, oversample=1.5)  # K phases
    x_grid = recommended_x_grid(n_max, dx=0.05, width_sigma=3.0)  # M positions
    povm = build_rotated_povm(d, thetas, x_grid)

    # Simulate counts (e.g., 2e4 shots per phase)
    counts = simulate_counts(rho_true, povm, shots_per_phase=20000, rng=np.random.default_rng(7))

    # Reconstruct by MLE
    rho_hat, info = mle_homodyne(counts, povm, max_iter=200, tol=1e-6, dilute=0.7)

    # Report
    print(f"Converged: {info.converged} in {info.iter} iters; final logL = {info.ll_history[-1]:.3f}")
    # Trace distance error (within truncated space)
    diff = rho_hat - rho_true
    eigvals = (diff.dag() * diff).sqrtm().tr().real  # ||ρ−σ||_1
    T = 0.5 * eigvals
    print(f"Trace distance T(ρ_hat, ρ_true) ≈ {T:.3e}")
