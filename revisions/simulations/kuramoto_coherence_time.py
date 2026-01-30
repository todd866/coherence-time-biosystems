#!/usr/bin/env python3
"""
Kuramoto coherence-time validation suite.

Goal
----
Validate the scaling law from Eq. (4) of the paper:

    τ_coh = (1/Δω) p₁(ε, κ(r))^{-α(M-1)}

where:
- M: number of semi-independent modules that must align
- r: inter-module coherence (r_inter, not global r)
- κ(r): von Mises concentration from r via r = I₁(κ)/I₀(κ)
- p₁(ε, κ): window probability ∫_{-ε/2}^{ε/2} vonMises(θ; κ) dθ
- ε: phase alignment tolerance, FULL window width (radians)
- Δω: attempt rate ≈ 1/τ_mix (s⁻¹), NOT carrier frequency
- α: effective independence factor (1 = independent, <1 = correlated)

Two fitting approaches:
1. Surrogate (linearized): log(τ·Δω) ~ α·c·(1-r̄)(M-1) where c = log(2π/ε)
2. Theory-consistent: log(τ·Δω) ~ α·(M-1)·(-log p₁) using r_inter and p₁

This script:
- Computes both global r and inter-module r_inter
- Uses Kaplan-Meier survival analysis to handle right-censored data
- Falls back to naive estimates if lifelines package unavailable
- Reports hit fraction (fraction of trials reaching coherence)
- Generates publication-ready validation figures with censoring indicators

Dependencies:
- Core: numpy, scipy, matplotlib
- Optional: lifelines (for proper censoring handling; install via `pip install lifelines`)

Author: Ian Todd
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def mean_resultant(theta: np.ndarray) -> Tuple[float, float]:
    """Return (r, psi) for a 1D array of phases."""
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z)), float(np.angle(z))


def r_to_kappa(r: float, max_kappa: float = 50.0) -> float:
    """
    Convert mean resultant length r to von Mises concentration κ.

    Uses numerical inversion of r = I_1(κ) / I_0(κ).
    """
    from scipy.special import i0, i1
    from scipy.optimize import brentq

    if r <= 0:
        return 0.0
    if r >= 1.0:
        return max_kappa

    def residual(kappa):
        return i1(kappa) / i0(kappa) - r

    # For small r, use approximation: r ≈ κ/2 for κ << 1
    if r < 0.1:
        return 2.0 * r

    try:
        return brentq(residual, 0.0, max_kappa)
    except ValueError:
        return max_kappa


def compute_p1(epsilon: float, kappa: float) -> float:
    """
    Compute the von Mises window probability p1(ε, κ).

    p1 = ∫_{-ε/2}^{ε/2} (e^{κ cos θ} / (2π I_0(κ))) dθ

    This is the probability that a single phase drawn from von Mises(0, κ)
    falls within ±ε/2 of the mode.

    Parameters
    ----------
    epsilon : float
        Full window width in radians (paper convention: ε = π/2 for ±45°)
    kappa : float
        von Mises concentration parameter

    Returns
    -------
    float
        Window probability p1 ∈ (0, 1)
    """
    from scipy.special import i0
    from scipy.integrate import quad

    if kappa < 1e-10:
        # Uniform distribution: p1 = ε / (2π)
        return epsilon / (2 * np.pi)

    normalization = 2 * np.pi * i0(kappa)

    def integrand(theta):
        return np.exp(kappa * np.cos(theta))

    result, _ = quad(integrand, -epsilon / 2, epsilon / 2)
    return result / normalization


# ----------------------------
# Model
# ----------------------------

@dataclass
class SimParams:
    N: int = 100                 # total oscillators
    M: int = 10                  # modules
    K: float = 1.0               # coupling gain
    omega_std: float = 0.3       # natural frequency spread
    sigma: float = 0.1           # phase diffusion strength (rad / sqrt(s))
    topology: str = "modular"    # all-to-all | modular | sparse

    # modular topology knobs
    K_intra: float = 1.0
    K_inter: float = 0.15
    p_sparse: float = 0.03

    # simulation
    T: float = 60.0
    dt: float = 0.005
    seed: int = 0

    # coherence event definition
    r_threshold: float = 0.6
    epsilon: float = 2.0                 # phase tolerance (radians)
    dwell_time: float = 0.03             # must hold coherence for this long (s)

    # how to compute r̄
    r_avg_start: float = 0.25            # fraction of trajectory to skip as transient

    def validate(self) -> None:
        if self.M < 2:
            raise ValueError("M must be ≥ 2.")
        if self.N < self.M:
            raise ValueError("N must be ≥ M.")


class KuramotoNetwork:
    """Kuramoto network with modular coupling."""

    def __init__(self, params: SimParams):
        params.validate()
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        self.omega = self.rng.normal(loc=0.0, scale=params.omega_std, size=params.N)
        self.theta0 = self.rng.uniform(0.0, 2 * np.pi, size=params.N)
        self.modules = np.repeat(np.arange(params.M), int(np.ceil(params.N / params.M)))[:params.N]
        self.A = self._build_A()

    def _build_A(self) -> np.ndarray:
        N, M = self.p.N, self.p.M
        topo = self.p.topology.lower()

        if topo == "all-to-all":
            A = np.ones((N, N), dtype=float) - np.eye(N, dtype=float)
            A /= (N - 1)
        elif topo == "modular":
            same = (self.modules[:, None] == self.modules[None, :]).astype(float)
            np.fill_diagonal(same, 0.0)
            A = self.p.K_inter * (1.0 - same) + self.p.K_intra * same
            np.fill_diagonal(A, 0.0)
            row_sum = A.sum(axis=1, keepdims=True)
            A = A / np.maximum(row_sum, 1e-12)
        elif topo == "sparse":
            prob = float(self.p.p_sparse)
            A = (self.rng.random((N, N)) < prob).astype(float)
            np.fill_diagonal(A, 0.0)
            row_sum = A.sum(axis=1, keepdims=True)
            A = A / np.maximum(row_sum, 1e-12)
        else:
            raise ValueError(f"Unknown topology: {self.p.topology}")
        return A

    def step(self, theta: np.ndarray) -> np.ndarray:
        diff = theta[None, :] - theta[:, None]
        drift = self.omega + self.p.K * np.sum(self.A * np.sin(diff), axis=1)
        if self.p.sigma > 0:
            noise = self.p.sigma * self.rng.standard_normal(self.p.N) * np.sqrt(self.p.dt)
        else:
            noise = 0.0
        theta_next = theta + drift * self.p.dt + noise
        return np.mod(theta_next, 2 * np.pi)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(np.floor(self.p.T / self.p.dt)) + 1
        t = np.arange(n_steps) * self.p.dt  # Fix: use arange for consistent dt
        theta = np.zeros((n_steps, self.p.N), dtype=float)
        theta[0] = self.theta0.copy()
        for k in range(1, n_steps):
            theta[k] = self.step(theta[k - 1])
        self.theta_history = theta  # Store for post-hoc analysis
        return t, theta

    def module_stats(self, theta_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_m = np.zeros(self.p.M, dtype=float)
        psi_m = np.zeros(self.p.M, dtype=float)
        for m in range(self.p.M):
            idx = (self.modules == m)
            r, psi = mean_resultant(theta_t[idx])
            r_m[m] = r
            psi_m[m] = psi
        return r_m, psi_m

    def coherence_event(self, theta_t: np.ndarray) -> bool:
        """
        Detect alignment event using mean-phase reference (Case C in paper).

        All modules must fall within ε/2 of the mean phase, where ε is the
        full phase window width (paper convention). This is equivalent to
        the anchored reference case up to an O(1) constant; the (M-1) scaling
        exponent is preserved.
        """
        r_m, psi_m = self.module_stats(theta_t)
        if not np.all(r_m >= self.p.r_threshold):
            return False
        _, psi_bar = mean_resultant(psi_m)
        phase_err = np.abs(wrap_angle(psi_m - psi_bar))
        # ε is full window width, so check against ε/2 (half-width from mean)
        return bool(np.max(phase_err) <= self.p.epsilon / 2.0)

    def measure_tau_coh(self, t: np.ndarray, theta: np.ndarray) -> Optional[float]:
        dwell_steps = int(np.ceil(self.p.dwell_time / self.p.dt))
        if dwell_steps <= 1:
            for k in range(theta.shape[0]):
                if self.coherence_event(theta[k]):
                    return float(t[k])
            return None

        consec = 0
        for k in range(theta.shape[0]):
            if self.coherence_event(theta[k]):
                consec += 1
                if consec >= dwell_steps:
                    return float(t[k - dwell_steps + 1])
            else:
                consec = 0
        return None

    def mean_global_r(self, theta: np.ndarray) -> float:
        """Compute mean global coherence (all oscillators)."""
        start_idx = int(self.p.r_avg_start * theta.shape[0])
        z = np.mean(np.exp(1j * theta[start_idx:]), axis=1)
        r = np.abs(z)
        return float(np.mean(r))

    def mean_inter_module_r(self, theta: np.ndarray) -> float:
        """
        Compute mean INTER-MODULE coherence r_inter from module phases.

        This is the quantity relevant for Eq. (4) in the paper:
        r_inter(t) = |1/M sum_m exp(i psi_m(t))|
        where psi_m is module m's aggregate phase.

        Returns mean r_inter over the trajectory (after transient).
        """
        start_idx = int(self.p.r_avg_start * theta.shape[0])
        n_steps = theta.shape[0] - start_idx

        r_inter_vals = np.zeros(n_steps)
        for k, t_idx in enumerate(range(start_idx, theta.shape[0])):
            _, psi_m = self.module_stats(theta[t_idx])
            z = np.mean(np.exp(1j * psi_m))
            r_inter_vals[k] = np.abs(z)

        return float(np.mean(r_inter_vals))

    def delta_omega_proxy(self) -> float:
        """
        Static proxy for phase exploration rate (used if no trajectory available).

        NOTE: This is a dimensionless heuristic combining frequency spread and noise.
        For quantitative work, use measure_effective_domega() which computes the
        actual spread of instantaneous frequencies from the trajectory.

        Returns a rough estimate in rad/s, but the effective Δω from simulation
        is more accurate and should be preferred for fitting.
        """
        # omega_std is rad/s; we use it as the primary estimate
        # The noise term (sigma) contributes to phase diffusion but has different units
        # We omit it here; use measure_effective_domega() for accurate measurement
        return float(np.std(self.omega))

    def measure_effective_domega(self, theta: np.ndarray) -> float:
        """
        Measure the effective phase exploration rate (Δω) dynamically.

        Rather than using the static natural frequency spread (omega_std),
        this measures the standard deviation of instantaneous frequencies
        averaged over the simulation. This accounts for coupling suppression
        of phase drift and noise diffusion.

        Returns the mean spread of instantaneous frequencies across oscillators (rad/s).
        """
        # Calculate instantaneous phase velocities
        # theta is wrapped (0, 2pi), so we wrap the difference to (-pi, pi)
        d_theta = wrap_angle(np.diff(theta, axis=0)) / self.p.dt

        # Calculate the spread (std) of velocities across the population at each time step
        # Discard first 10% as transient
        start_idx = int(0.1 * d_theta.shape[0])
        instantaneous_spreads = np.std(d_theta[start_idx:], axis=1)

        return float(np.mean(instantaneous_spreads))

    def estimate_attempt_rate(self, theta: np.ndarray, method: str = "autocorr") -> float:
        """
        Estimate attempt rate λ (s⁻¹) from phase mixing dynamics.

        This measures the effective "alignment attempt rate" - the inverse of the
        time it takes for inter-module coherence to decorrelate. This is the quantity
        Δω/(2π) in the theory, where Δω is the phase exploration rate in rad/s.

        method="autocorr": Computes decorrelation time of the inter-module complex
        order parameter z_inter(t) = mean(exp(iψ_m(t))). Uses 1/e crossing of
        autocorrelation of |z_inter(t)|.

        Args:
            theta: Full trajectory (n_steps × N)
            method: "autocorr" (decorrelation time) or "freq_std" (frequency spread proxy)

        Returns:
            λ in s⁻¹ (Hz) - the effective alignment attempt rate
        """
        start_idx = int(self.p.r_avg_start * theta.shape[0])
        theta_post = theta[start_idx:]

        if method == "autocorr":
            # Compute inter-module complex order parameter z(t) = mean(exp(iψ_m(t)))
            # This directly measures inter-module phase coherence decorrelation
            n_steps = len(theta_post)
            z_inter = np.zeros(n_steps, dtype=complex)

            for i in range(n_steps):
                _, psi = self.module_stats(theta_post[i])
                z_inter[i] = np.mean(np.exp(1j * psi))

            # Autocorrelation of |z_inter(t)| (magnitude of order parameter)
            r_inter_t = np.abs(z_inter)
            r_inter_centered = r_inter_t - np.mean(r_inter_t)

            # Autocorrelation
            acf = np.correlate(r_inter_centered, r_inter_centered, mode='full')
            acf = acf[len(acf)//2:]  # keep positive lags only
            acf = acf / (acf[0] + 1e-12)  # normalize (avoid div by 0)

            # Find decorrelation time (1/e point)
            tau_mix_steps = np.argmax(acf < 1.0/np.e)
            if tau_mix_steps == 0:  # didn't decorrelate within trajectory
                tau_mix_steps = len(acf) // 2  # use half the trajectory as upper bound

            tau_mix = tau_mix_steps * self.p.dt
            lambda_attempt = 1.0 / tau_mix if tau_mix > 0 else 1.0 / self.p.T

        elif method == "freq_std":
            # Fallback: use frequency spread as proxy
            # Δω (rad/s) ≈ std of instantaneous frequencies
            # λ ≈ Δω / (2π)
            domega_rad_s = self.measure_effective_domega(theta)
            lambda_attempt = domega_rad_s / (2 * np.pi)

        else:
            raise ValueError(f"Unknown method: {method}")

        return float(lambda_attempt)

    def r_inter_at_time(self, theta_t: np.ndarray) -> float:
        """
        Compute inter-module coherence r_inter at a single timepoint.

        r_inter(t) = |1/M sum_m exp(i psi_m(t))|
        """
        _, psi_m = self.module_stats(theta_t)
        z = np.mean(np.exp(1j * psi_m))
        return float(np.abs(z))


# ----------------------------
# Experiments with proper stats
# ----------------------------

@dataclass
class TrialResult:
    """Per-trial result for proper alignment of τ with covariates."""
    hit: bool
    tau: float           # observed τ if hit, else T (right-censored)
    censored: bool       # True if tau == T (did not hit)
    r_global: float      # trajectory-mean global coherence (diagnostic)
    r_inter: float       # trajectory-mean inter-module coherence
    r_inter_at_hit: float  # r_inter measured near hit (or at end if censored)
    lambda_attempt: float  # attempt rate in s⁻¹ from mixing time
    omega_spread: float  # diagnostic: std of inst. frequencies (rad/s)
    seed: int            # random seed used for this trial (for reproducibility)


def compute_survival_stats(trials: List[TrialResult]) -> Dict[str, float]:
    """
    Compute survival-analysis-based statistics handling right-censoring.

    Uses Kaplan-Meier estimator to properly account for censored observations.
    Falls back to naive estimates if lifelines not available.

    Returns:
        dict with tau_median_km, tau_q25_km, tau_q75_km, and km_available flag
    """
    durations = np.array([tr.tau for tr in trials])
    events = np.array([tr.hit for tr in trials])  # 1 if hit, 0 if censored
    n_hit = int(events.sum())
    n_total = len(trials)

    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events)

        # Extract quantiles from survival function
        # KM survival function gives P(T > t), so median is where S(t) = 0.5
        tau_median_km = kmf.median_survival_time_

        # Get confidence intervals if available
        try:
            ci = kmf.confidence_interval_survival_function_
            # Extract quantiles by finding where survival crosses thresholds
            survival_func = kmf.survival_function_
            times = survival_func.index.values
            surv_vals = survival_func.values.flatten()

            # Q25: where survival = 0.75
            idx_q25 = np.argmax(surv_vals <= 0.75) if any(surv_vals <= 0.75) else 0
            tau_q25_km = times[idx_q25] if idx_q25 > 0 else float("nan")

            # Q75: where survival = 0.25
            idx_q75 = np.argmax(surv_vals <= 0.25) if any(surv_vals <= 0.25) else 0
            tau_q75_km = times[idx_q75] if idx_q75 > 0 else float("nan")
        except:
            tau_q25_km = float("nan")
            tau_q75_km = float("nan")

        return {
            "tau_median_km": float(tau_median_km) if tau_median_km is not None else float("nan"),
            "tau_q25_km": float(tau_q25_km),
            "tau_q75_km": float(tau_q75_km),
            "km_available": True,
            "km_note": f"KM estimate with {n_hit}/{n_total} events"
        }

    except ImportError:
        # Fallback: simple estimates (biased if censored)
        hit_trials = [tr for tr in trials if tr.hit]
        if len(hit_trials) > 0:
            taus = np.array([tr.tau for tr in hit_trials])
            return {
                "tau_median_km": float(np.median(taus)),
                "tau_q25_km": float(np.percentile(taus, 25)),
                "tau_q75_km": float(np.percentile(taus, 75)),
                "km_available": False,
                "km_note": f"Naive estimate (lifelines not available), {n_hit}/{n_total} events"
            }
        else:
            return {
                "tau_median_km": float("nan"),
                "tau_q25_km": float("nan"),
                "tau_q75_km": float("nan"),
                "km_available": False,
                "km_note": "No hits, cannot estimate"
            }


def run_trials(base: SimParams, n_trials: int, seeds: Optional[List[int]] = None) -> Dict:
    """
    Run repeated trials and return summary stats with median/IQR.

    Key improvement: stores per-trial records so covariates (r, Δω) align with
    outcomes (τ). Non-hits are treated as right-censored at T.
    """
    trials: List[TrialResult] = []

    if seeds is None:
        seeds = list(range(base.seed, base.seed + n_trials))

    for s in seeds[:n_trials]:
        p = SimParams(**{**asdict(base), "seed": int(s)})
        net = KuramotoNetwork(p)
        t, theta = net.simulate()
        tau = net.measure_tau_coh(t, theta)

        hit = tau is not None

        # Measure r_inter near the hit (or at end if censored)
        if hit:
            # Find the hit time index and measure r_inter in a window around it
            hit_idx = np.searchsorted(t, tau)
            # Use a 50ms window before hit (or available data)
            window_steps = max(1, int(0.05 / p.dt))
            start = max(0, hit_idx - window_steps)
            end = min(len(theta), hit_idx + 1)
            r_inter_vals = [net.r_inter_at_time(theta[i]) for i in range(start, end)]
            r_inter_at_hit_val = float(np.mean(r_inter_vals))
        else:
            # For censored trials, use r_inter from the last 10% of trajectory
            start = int(0.9 * len(theta))
            r_inter_vals = [net.r_inter_at_time(theta[i]) for i in range(start, len(theta))]
            r_inter_at_hit_val = float(np.mean(r_inter_vals))

        trials.append(TrialResult(
            hit=hit,
            tau=float(tau) if hit else float(p.T),
            censored=not hit,
            r_global=net.mean_global_r(theta),
            r_inter=net.mean_inter_module_r(theta),
            r_inter_at_hit=r_inter_at_hit_val,
            lambda_attempt=net.estimate_attempt_rate(theta, method="autocorr"),
            omega_spread=net.measure_effective_domega(theta),
            seed=int(s),
        ))

    # Extract arrays
    hit_trials = [tr for tr in trials if tr.hit]
    taus_hit = np.array([tr.tau for tr in hit_trials]) if hit_trials else np.array([])
    n_hit = len(hit_trials)
    hit_fraction = n_hit / n_trials

    # Compute statistics on HITS ONLY (conditional estimates)
    # Note: these are biased if hit_fraction < 1; we flag this below
    if n_hit > 0:
        tau_median = float(np.median(taus_hit))
        tau_q25 = float(np.percentile(taus_hit, 25))
        tau_q75 = float(np.percentile(taus_hit, 75))
        tau_mean = float(np.mean(taus_hit))
        tau_std = float(np.std(taus_hit))
        # Covariates for HIT trials only (aligned with τ)
        r_mean_hits = float(np.mean([tr.r_global for tr in hit_trials]))
        r_inter_mean_hits = float(np.mean([tr.r_inter for tr in hit_trials]))
        r_inter_at_hit_mean = float(np.mean([tr.r_inter_at_hit for tr in hit_trials]))
        lambda_mean_hits = float(np.mean([tr.lambda_attempt for tr in hit_trials]))
        domega_mean_hits = float(np.mean([tr.omega_spread for tr in hit_trials]))
    else:
        tau_median = tau_q25 = tau_q75 = tau_mean = tau_std = float("nan")
        r_mean_hits = r_inter_mean_hits = r_inter_at_hit_mean = lambda_mean_hits = domega_mean_hits = float("nan")

    # Also compute covariates over ALL trials (for diagnostics)
    r_mean_all = float(np.mean([tr.r_global for tr in trials]))
    r_inter_mean_all = float(np.mean([tr.r_inter for tr in trials]))
    r_inter_at_hit_all = float(np.mean([tr.r_inter_at_hit for tr in trials]))
    lambda_mean_all = float(np.mean([tr.lambda_attempt for tr in trials]))
    domega_mean_all = float(np.mean([tr.omega_spread for tr in trials]))

    # Compute survival-analysis-based estimates (handles censoring properly)
    survival_stats = compute_survival_stats(trials)

    # Flag if median is unreliable due to censoring
    # If hit_fraction < 0.5, the true median exceeds T
    median_censored = hit_fraction < 0.5

    # Prefer KM estimate if available, otherwise use naive
    tau_median_best = survival_stats["tau_median_km"] if survival_stats["km_available"] else tau_median
    tau_q25_best = survival_stats["tau_q25_km"] if survival_stats["km_available"] else tau_q25
    tau_q75_best = survival_stats["tau_q75_km"] if survival_stats["km_available"] else tau_q75

    return {
        # Naive estimates (hits only, biased if censored)
        "tau_median": tau_median if not median_censored else float("nan"),
        "tau_median_note": "reliable" if not median_censored else f"censored (hit_frac={hit_fraction:.2f} < 0.5)",
        "tau_q25": tau_q25,
        "tau_q75": tau_q75,
        # Survival-analysis estimates (handles censoring properly)
        "tau_median_km": tau_median_best,
        "tau_q25_km": tau_q25_best,
        "tau_q75_km": tau_q75_best,
        "km_available": survival_stats["km_available"],
        "km_note": survival_stats["km_note"],
        # Other stats
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "n_hit": n_hit,
        "n_trials": n_trials,
        "hit_fraction": hit_fraction,
        # Covariates aligned with hit trials (for fitting)
        "r_mean": r_mean_hits,
        "r_inter_mean": r_inter_mean_hits,
        "r_inter_at_hit_mean": r_inter_at_hit_mean,
        "lambda_attempt_mean": lambda_mean_hits,
        "domega_mean": domega_mean_hits,
        # Covariates over all trials (for diagnostics)
        "r_mean_all": r_mean_all,
        "r_inter_mean_all": r_inter_mean_all,
        "r_inter_at_hit_all": r_inter_at_hit_all,
        "lambda_attempt_all": lambda_mean_all,
        "domega_mean_all": domega_mean_all,
        # Legacy compatibility (DEPRECATED - use lambda_attempt_mean instead)
        "domega_static": base.omega_std,  # Fix: use actual omega_std
        "domega_effective": domega_mean_hits,
        "taus_raw": [tr.tau for tr in hit_trials],
        # Full per-trial data for detailed analysis
        "trials": [asdict(tr) for tr in trials],
    }


def sweep_M(base: SimParams, M_values: List[int], n_trials: int) -> Dict[str, np.ndarray]:
    out = {
        "M": [], "tau_median": [], "tau_q25": [], "tau_q75": [],
        "tau_median_km": [], "tau_q25_km": [], "tau_q75_km": [],  # KM estimates
        "tau_mean": [], "tau_std": [], "n_hit": [], "n_trials": [],
        "hit_fraction": [], "r_mean": [], "r_inter_mean": [], "r_inter_at_hit_mean": [],
        "r_inter_mean_all": [],  # unbiased: trajectory-mean r_inter across ALL trials
        "lambda_attempt_mean": [],
        "domega_static": [], "domega_effective": [], "domega_mean": []
    }
    for M in M_values:
        print(f"  M={M}...", end=" ", flush=True)
        p = SimParams(**{**asdict(base), "M": int(M)})
        stats = run_trials(p, n_trials=n_trials)
        out["M"].append(M)
        for k in out.keys():
            if k != "M" and k in stats:
                out[k].append(stats[k])

        # Handle tau_median display with censoring awareness
        tau_str = f"{stats['tau_median']:.2f}" if np.isfinite(stats['tau_median']) else f">T (censored)"
        r_inter_str = f"{stats['r_inter_mean']:.2f}" if np.isfinite(stats['r_inter_mean']) else "N/A"
        print(f"hit={stats['n_hit']}/{stats['n_trials']}, τ_med={tau_str}, r_inter={r_inter_str}")

    return {k: np.asarray(v) for k, v in out.items()}


def fit_alpha_from_sweep(results: Dict[str, np.ndarray], epsilon: float) -> Dict[str, float]:
    """
    Fit α using median τ values with linearized surrogate model.

    This uses the approximation log(τ·λ) ≈ α·c·(1-r̄)(M-1) where c = log(2π/ε).
    This is a linearization valid when r is not too close to 0 or 1.

    Prefers KM estimates (handles censoring), falls back to naive median.
    Prefers r_inter_mean if available, falls back to r_mean.
    """
    M = results["M"].astype(float)

    # Prefer KM estimate (handles censoring), fallback to naive
    if "tau_median_km" in results and not np.all(np.isnan(results["tau_median_km"])):
        tau = results["tau_median_km"].astype(float)
        tau_source = "KM"
    else:
        tau = results["tau_median"].astype(float)
        tau_source = "naive"

    # Use stationary r_inter (all trials, unbiased), fallback to hit-only, then r_mean
    if "r_inter_mean_all" in results and not np.all(np.isnan(results["r_inter_mean_all"])):
        rbar = results["r_inter_mean_all"].astype(float)
    elif "r_inter_mean" in results and not np.all(np.isnan(results["r_inter_mean"])):
        rbar = results["r_inter_mean"].astype(float)
    else:
        rbar = results["r_mean"].astype(float)

    # Prefer lambda_attempt_mean, fallback to domega/(2π)
    if "lambda_attempt_mean" in results and not np.all(np.isnan(results["lambda_attempt_mean"])):
        attempt_rate = results["lambda_attempt_mean"].astype(float)
    else:
        domega = results["domega_mean"].astype(float)
        attempt_rate = domega / (2 * np.pi)

    mask = np.isfinite(tau) & (tau > 0) & np.isfinite(attempt_rate) & (attempt_rate > 0)
    if mask.sum() < 3:
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(mask.sum())}

    x = (1.0 - rbar[mask]) * (M[mask] - 1.0)
    y = np.log(tau[mask] * attempt_rate[mask])
    c = np.log(2 * np.pi / float(epsilon))

    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = beta
    alpha_hat = float(slope / c) if c != 0 else float("nan")

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {
        "alpha_hat": alpha_hat,
        "r2": float(r2),
        "n": int(mask.sum()),
        "intercept": float(intercept),
        "tau_source": tau_source,
        "method": "surrogate"
    }


def fit_alpha_theory_consistent(results: Dict[str, np.ndarray], epsilon: float) -> Dict[str, float]:
    """
    Fit α using the exact form of Eq. (4) with p1(ε, κ(r_inter)).

    This is more theory-consistent than the linearized surrogate:
    log(τ·λ) = -α(M-1)·log(p1) + intercept

    Uses:
    - tau_median_km (KM survival estimate) if available, else tau_median
    - r_inter (inter-module coherence) if available, else r_mean
    - lambda_attempt (s⁻¹) if available, else domega_mean (rad/s) converted to λ

    Note: When r_inter is very high (>0.95), -log(p1) approaches 0 and the
    regression becomes numerically unstable. In such cases, the surrogate
    fit is more reliable.
    """
    M = results["M"].astype(float)

    # Prefer KM estimate (handles censoring), fallback to naive
    if "tau_median_km" in results and not np.all(np.isnan(results["tau_median_km"])):
        tau = results["tau_median_km"].astype(float)
        tau_source = "KM"
    else:
        tau = results["tau_median"].astype(float)
        tau_source = "naive"

    # Prefer lambda_attempt (s⁻¹) if available
    if "lambda_attempt_mean" in results and not np.all(np.isnan(results["lambda_attempt_mean"])):
        attempt_rate = results["lambda_attempt_mean"].astype(float)
        rate_label = "lambda"
    else:
        # Fall back to domega / (2π) if lambda not available
        domega = results["domega_mean"].astype(float)
        attempt_rate = domega / (2 * np.pi)
        rate_label = "domega/(2pi)"

    # Use stationary r_inter (trajectory mean across ALL trials, including censored).
    # r_inter_at_hit is selection-biased (high precisely because alignment occurred),
    # which inflates p1, shrinks -log(p1), and forces alpha_hat upward.
    # r_inter_mean_all is the unbiased estimate of the stationary distribution
    # that the theory (Eq. 4) actually describes.
    if "r_inter_mean_all" in results and not np.all(np.isnan(results["r_inter_mean_all"])):
        rbar = results["r_inter_mean_all"].astype(float)
        r_label = "r_inter_all"
    elif "r_inter_mean" in results:
        rbar = results["r_inter_mean"].astype(float)
        r_label = "r_inter"
    else:
        rbar = results["r_mean"].astype(float)
        r_label = "r_global"

    mask = np.isfinite(tau) & (tau > 0) & np.isfinite(attempt_rate) & (attempt_rate > 0) & (rbar > 0.01)
    if mask.sum() < 3:
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(mask.sum()),
                "note": "insufficient valid data points"}

    # Compute p1 for each r value
    neg_log_p1 = np.zeros(mask.sum())
    p1_values = np.zeros(mask.sum())
    for i, r_val in enumerate(rbar[mask]):
        kappa = r_to_kappa(float(r_val))
        p1 = compute_p1(epsilon, kappa)
        p1_values[i] = p1
        neg_log_p1[i] = -np.log(p1) if p1 > 0 and p1 < 1 else np.nan

    # Check for numerical stability: if -log(p1) is too small, regression is unreliable
    # When r > ~0.95, p1 > ~0.95, and -log(p1) < ~0.05, making x values tiny
    valid = np.isfinite(neg_log_p1) & (neg_log_p1 > 0.01)  # require meaningful variation
    if valid.sum() < 3:
        mean_p1 = float(np.mean(p1_values[np.isfinite(p1_values)])) if any(np.isfinite(p1_values)) else float("nan")
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(valid.sum()),
                "note": f"high coherence regime (mean p1={mean_p1:.3f}); use surrogate fit",
                "mean_p1": mean_p1}

    # Regress: log(τ·λ) = α·(M-1)·(-log p1) + intercept
    M_valid = M[mask][valid]
    tau_valid = tau[mask][valid]
    rate_valid = attempt_rate[mask][valid]
    neg_log_p1_valid = neg_log_p1[valid]

    x = (M_valid - 1.0) * neg_log_p1_valid
    y = np.log(tau_valid * rate_valid)

    # Check for sufficient variation in x
    if np.std(x) < 1e-6:
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(valid.sum()),
                "note": "insufficient variation in predictor (x ≈ constant)"}

    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_hat, intercept = beta

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {
        "alpha_hat": float(alpha_hat),
        "r2": float(r2),
        "n": int(valid.sum()),
        "intercept": float(intercept),
        "method": "theory_consistent",
        "mean_neg_log_p1": float(np.mean(neg_log_p1_valid)),
        "rate_type": rate_label,  # "lambda" or "domega/(2pi)"
        "r_type": r_label,  # "r_inter_at_hit", "r_inter", or "r_global"
        "tau_source": tau_source  # "KM" or "naive"
    }


def plot_validation_figure(results: Dict[str, np.ndarray], epsilon: float,
                           fit: Dict[str, float], outpath: Optional[Path] = None) -> None:
    """Generate publication-quality validation figure."""
    M = results["M"]
    tau_med = results["tau_median"]
    tau_q25 = results["tau_q25"]
    tau_q75 = results["tau_q75"]
    hit_frac = results["hit_fraction"]
    rbar = results["r_mean"]
    domega = results["domega_mean"]

    alpha_hat = fit.get("alpha_hat", np.nan)
    r2 = fit.get("r2", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: τ vs M with IQR
    ax = axes[0]
    mask = np.isfinite(tau_med) & (tau_med > 0)

    # IQR as error bars
    yerr_low = tau_med[mask] - tau_q25[mask]
    yerr_high = tau_q75[mask] - tau_med[mask]

    ax.errorbar(M[mask], tau_med[mask], yerr=[yerr_low, yerr_high],
                fmt='o', capsize=4, markersize=8, color='#2E86AB',
                ecolor='#2E86AB', elinewidth=2, capthick=2, label='Simulation (median ± IQR)')
    ax.set_yscale("log")
    ax.set_xlabel("Coordination depth M", fontsize=12)
    ax.set_ylabel("Coherence time τ_coh (s)", fontsize=12)

    # Fitted line: use actual model predictions, not constant r₀
    if np.isfinite(alpha_hat) and mask.any() and 'intercept' in fit:
        intercept = fit['intercept']
        c = np.log(2 * np.pi / float(epsilon))

        # Compute attempt_rate the SAME WAY as fit functions
        # (Prefer lambda_attempt_mean, fallback to domega/(2π))
        if "lambda_attempt_mean" in results and not np.all(np.isnan(results["lambda_attempt_mean"])):
            attempt_rate = results["lambda_attempt_mean"]
        else:
            attempt_rate = domega / (2 * np.pi)

        # Interpolate r(M) and attempt_rate for prediction curve
        M_fit = M[mask]
        r_fit = rbar[mask]
        rate_fit = attempt_rate[mask]

        M_grid = np.linspace(M_fit.min(), M_fit.max(), 50)
        # Linear interpolation of r and attempt_rate across M
        r_interp = np.interp(M_grid, M_fit, r_fit)
        rate_interp = np.interp(M_grid, M_fit, rate_fit)

        # Predicted log(tau * attempt_rate) from fitted model
        x_pred = (1 - r_interp) * (M_grid - 1)
        log_tau_lambda_pred = alpha_hat * c * x_pred + intercept
        tau_pred = np.exp(log_tau_lambda_pred) / rate_interp

        ax.plot(M_grid, tau_pred, '--', linewidth=2, color='#E94F37',
                label=f'Fit: α̂ = {alpha_hat:.2f}, R² = {r2:.2f}')

    ax.legend(fontsize=10)
    ax.set_title("A. Coherence time scaling", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: Hit fraction
    ax = axes[1]
    colors = ['#28A745' if h >= 0.5 else '#DC3545' for h in hit_frac]
    bars = ax.bar(M, hit_frac, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    ax.set_xlabel("Coordination depth M", fontsize=12)
    ax.set_ylabel("Hit fraction", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("B. Fraction reaching coherence", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add hit count labels
    for bar, n_hit, n_tot in zip(bars, results["n_hit"], results["n_trials"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{int(n_hit)}/{int(n_tot)}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        # Also save PDF for LaTeX
        fig.savefig(outpath.with_suffix('.pdf'), bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate coherence-time scaling.")
    ap.add_argument("--outdir", type=str, default="validation_results")
    ap.add_argument("--topology", type=str, default="modular",
                    help="Network topology: modular, all-to-all, or sparse")
    ap.add_argument("--N", type=int, default=100, help="Total number of oscillators")
    ap.add_argument("--K", type=float, default=1.0, help="Global coupling gain")
    ap.add_argument("--omega-std", type=float, default=0.3, help="Natural frequency spread (rad/s)")
    ap.add_argument("--sigma", type=float, default=0.1, help="Phase noise strength (rad/sqrt(s))")
    ap.add_argument("--T", type=float, default=60.0, help="Simulation duration (s)")
    ap.add_argument("--dt", type=float, default=0.005, help="Time step (s)")
    ap.add_argument("--r-threshold", type=float, default=0.6, help="Module coherence threshold")
    ap.add_argument("--epsilon", type=float, default=2.0,
                    help="Phase alignment tolerance - FULL window width (rad). "
                         "Alignment requires all phases within ±ε/2 of mean.")
    ap.add_argument("--dwell", type=float, default=0.03, help="Required dwell time at coherence (s)")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials per M")
    ap.add_argument("--M-min", type=int, default=3, help="Minimum number of modules")
    ap.add_argument("--M-max", type=int, default=10, help="Maximum number of modules")
    ap.add_argument("--M-step", type=int, default=1, help="Step size for M sweep")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--save-json", action="store_true", help="Save results to JSON")
    # Modular topology parameters (for reproducibility)
    ap.add_argument("--K-intra", type=float, default=1.0,
                    help="Intra-module coupling strength (modular topology)")
    ap.add_argument("--K-inter", type=float, default=0.15,
                    help="Inter-module coupling strength (modular topology)")
    ap.add_argument("--p-sparse", type=float, default=0.03,
                    help="Connection probability (sparse topology)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    base = SimParams(
        N=args.N, M=10, K=args.K, omega_std=args.omega_std, sigma=args.sigma,
        topology=args.topology, T=args.T, dt=args.dt, seed=args.seed,
        r_threshold=args.r_threshold, epsilon=args.epsilon, dwell_time=args.dwell,
        K_intra=args.K_intra, K_inter=args.K_inter, p_sparse=args.p_sparse,
    )

    print(f"Running sweep: topology={args.topology}, N={args.N}, trials={args.trials}")
    M_values = list(range(args.M_min, args.M_max + 1, args.M_step))
    results = sweep_M(base, M_values=M_values, n_trials=args.trials)

    # Two fitting approaches
    fit_surrogate = fit_alpha_from_sweep(results, epsilon=base.epsilon)
    fit_theory = fit_alpha_theory_consistent(results, epsilon=base.epsilon)

    # Use theory-consistent fit as primary result
    # Fall back to surrogate if theory fit fails (high-coherence regime)
    if np.isfinite(fit_theory.get("alpha_hat", float("nan"))):
        primary_fit = fit_theory
        fit_label = "theory-consistent"
    else:
        primary_fit = fit_surrogate
        fit_label = f"surrogate (theory failed: {fit_theory.get('note', 'unknown')})"

    # Generate figure with primary fit
    fig_path = outdir / f"fig_validation_{args.topology}_N{args.N}.png"
    plot_validation_figure(results, epsilon=base.epsilon, fit=primary_fit, outpath=fig_path)

    # Save JSON with both fits and primary selection
    if args.save_json:
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_params": asdict(base),
            "sweep": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()},
            "fit": primary_fit,  # primary result (theory-consistent if valid, else surrogate)
            "fit_label": fit_label,  # which fit was used as primary
            "fit_surrogate": fit_surrogate,
            "fit_theory_consistent": fit_theory,
        }
        (outdir / f"validation_{args.topology}_N{args.N}.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"\nPrimary fit: {fit_label}")
    print(f"  α̂ = {primary_fit.get('alpha_hat'):.3f}")
    print(f"  R² = {primary_fit.get('r2'):.3f}")
    print(f"  n  = {primary_fit.get('n')} valid M values")
    if 'rate_type' in primary_fit:
        print(f"  rate: {primary_fit['rate_type']}")
    if 'r_type' in primary_fit:
        print(f"  r: {primary_fit['r_type']}")
    if 'tau_source' in primary_fit:
        print(f"  τ estimation: {primary_fit['tau_source']} {'(survival analysis)' if primary_fit['tau_source'] == 'KM' else '(hits only)'}")

    print("\nCensoring summary:")
    for M_val, hf in zip(results["M"], results["hit_fraction"]):
        status = "✓" if hf >= 0.8 else "⚠" if hf >= 0.5 else "✗"
        print(f"  M={M_val}: {status} hit_fraction={hf:.2f}")

    print("\nSurrogate fit (linearized):")
    print(f"  α̂ = {fit_surrogate.get('alpha_hat'):.3f}")
    print(f"  R² = {fit_surrogate.get('r2'):.3f}")
    print(f"  n  = {fit_surrogate.get('n')} valid M values")

    print("\nTheory-consistent fit (exact p1):")
    if np.isfinite(fit_theory.get('alpha_hat', float('nan'))):
        print(f"  α̂ = {fit_theory.get('alpha_hat'):.3f}")
        print(f"  R² = {fit_theory.get('r2'):.3f}")
        print(f"  n  = {fit_theory.get('n')} valid M values")
    else:
        print(f"  {fit_theory.get('note', 'fit failed')}")
    print(f"\nFigure: {fig_path}")
    print(f"PDF:    {fig_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
