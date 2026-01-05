#!/usr/bin/env python3
"""
Kuramoto coherence-time validation suite.

Goal
----
Validate the scaling law used in the paper:

    τ_coh ≈ (1/Δω) (2π/ε)^[ α (1 - r̄) (M - 1) ]

where
- M: number of semi-independent modules that must align
- r̄: mean Kuramoto coherence (mean resultant length) over the run
- ε: phase alignment tolerance (radians)
- Δω: phase exploration rate (rad/s)
- α: topology factor (fit from simulation)

This script:
- Uses median/IQR for heavy-tailed first-passage times
- Reports hit fraction (fraction of trials reaching coherence)
- Generates publication-ready validation figure

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
        t = np.linspace(0.0, self.p.T, n_steps)
        theta = np.zeros((n_steps, self.p.N), dtype=float)
        theta[0] = self.theta0.copy()
        for k in range(1, n_steps):
            theta[k] = self.step(theta[k - 1])
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
        r_m, psi_m = self.module_stats(theta_t)
        if not np.all(r_m >= self.p.r_threshold):
            return False
        _, psi_bar = mean_resultant(psi_m)
        phase_err = np.abs(wrap_angle(psi_m - psi_bar))
        return bool(np.max(phase_err) <= self.p.epsilon)

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
        start_idx = int(self.p.r_avg_start * theta.shape[0])
        z = np.mean(np.exp(1j * theta[start_idx:]), axis=1)
        r = np.abs(z)
        return float(np.mean(r))

    def delta_omega_proxy(self) -> float:
        """Static proxy for phase exploration rate (used if no trajectory available)."""
        return float(np.std(self.omega) + 0.5 * (self.p.sigma ** 2))

    def measure_effective_domega(self, theta: np.ndarray) -> float:
        """
        Measure the effective phase exploration rate (Δω) dynamically.

        Rather than using the static natural frequency spread (omega_std),
        this measures the standard deviation of instantaneous frequencies
        averaged over the simulation. This accounts for coupling suppression
        of phase drift and noise diffusion.

        Returns the mean spread of instantaneous frequencies across oscillators.
        """
        # Calculate instantaneous phase velocities
        # theta is wrapped (0, 2pi), so we wrap the difference to (-pi, pi)
        d_theta = wrap_angle(np.diff(theta, axis=0)) / self.p.dt

        # Calculate the spread (std) of velocities across the population at each time step
        # Discard first 10% as transient
        start_idx = int(0.1 * d_theta.shape[0])
        instantaneous_spreads = np.std(d_theta[start_idx:], axis=1)

        return float(np.mean(instantaneous_spreads))


# ----------------------------
# Experiments with proper stats
# ----------------------------

def run_trials(base: SimParams, n_trials: int, seeds: Optional[List[int]] = None) -> Dict:
    """Run repeated trials and return summary stats with median/IQR."""
    taus: List[float] = []
    rs: List[float] = []
    dws_static: List[float] = []
    dws_effective: List[float] = []

    if seeds is None:
        seeds = list(range(base.seed, base.seed + n_trials))

    for s in seeds[:n_trials]:
        p = SimParams(**{**asdict(base), "seed": int(s)})
        net = KuramotoNetwork(p)
        t, theta = net.simulate()
        tau = net.measure_tau_coh(t, theta)
        if tau is not None:
            taus.append(float(tau))
        rs.append(net.mean_global_r(theta))
        dws_static.append(net.delta_omega_proxy())
        # NEW: Measure effective dynamic dispersion instead of static proxy
        dws_effective.append(net.measure_effective_domega(theta))

    taus_arr = np.array(taus) if taus else np.array([])

    return {
        "tau_median": float(np.median(taus_arr)) if len(taus_arr) > 0 else float("nan"),
        "tau_q25": float(np.percentile(taus_arr, 25)) if len(taus_arr) > 0 else float("nan"),
        "tau_q75": float(np.percentile(taus_arr, 75)) if len(taus_arr) > 0 else float("nan"),
        "tau_mean": float(np.mean(taus_arr)) if len(taus_arr) > 0 else float("nan"),
        "tau_std": float(np.std(taus_arr)) if len(taus_arr) > 0 else float("nan"),
        "n_hit": int(len(taus)),
        "n_trials": int(n_trials),
        "hit_fraction": float(len(taus) / n_trials),
        "r_mean": float(np.mean(rs)),
        "domega_static": float(np.mean(dws_static)),    # static proxy (omega_std + noise)
        "domega_effective": float(np.mean(dws_effective)),  # dynamic measurement
        "domega_mean": float(np.mean(dws_effective)),   # use effective for fitting (backward compat)
        "taus_raw": taus,  # keep raw values for detailed analysis
    }


def sweep_M(base: SimParams, M_values: List[int], n_trials: int) -> Dict[str, np.ndarray]:
    out = {
        "M": [], "tau_median": [], "tau_q25": [], "tau_q75": [],
        "tau_mean": [], "tau_std": [], "n_hit": [], "n_trials": [],
        "hit_fraction": [], "r_mean": [], "domega_static": [], "domega_effective": [], "domega_mean": []
    }
    for M in M_values:
        print(f"  M={M}...", end=" ", flush=True)
        p = SimParams(**{**asdict(base), "M": int(M)})
        stats = run_trials(p, n_trials=n_trials)
        out["M"].append(M)
        for k in out.keys():
            if k != "M":
                out[k].append(stats[k])
        print(f"hit={stats['n_hit']}/{stats['n_trials']}, τ_med={stats['tau_median']:.2f}")
    return {k: np.asarray(v) for k, v in out.items()}


def fit_alpha_from_sweep(results: Dict[str, np.ndarray], epsilon: float) -> Dict[str, float]:
    """Fit α using median τ values."""
    M = results["M"].astype(float)
    tau = results["tau_median"].astype(float)
    rbar = results["r_mean"].astype(float)
    domega = results["domega_mean"].astype(float)

    mask = np.isfinite(tau) & (tau > 0) & np.isfinite(domega) & (domega > 0)
    if mask.sum() < 3:
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(mask.sum())}

    x = (1.0 - rbar[mask]) * (M[mask] - 1.0)
    y = np.log(tau[mask] * domega[mask])
    c = np.log(2 * np.pi / float(epsilon))

    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = beta
    alpha_hat = float(slope / c) if c != 0 else float("nan")

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {"alpha_hat": alpha_hat, "r2": float(r2), "n": int(mask.sum()), "intercept": float(intercept)}


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

    # Fitted line
    if np.isfinite(alpha_hat) and mask.any():
        M0 = float(M[mask][0])
        tau0 = float(tau_med[mask][0])
        r0 = float(rbar[mask][0])
        c = np.log(2 * np.pi / float(epsilon))
        M_grid = np.linspace(M[mask].min(), M[mask].max(), 50)
        rel = np.exp(alpha_hat * c * (1 - r0) * (M_grid - M0))
        tau_pred = tau0 * rel
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
    ap.add_argument("--epsilon", type=float, default=2.0, help="Phase alignment tolerance (rad)")
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
    fit = fit_alpha_from_sweep(results, epsilon=base.epsilon)

    # Generate figure
    fig_path = outdir / f"fig_validation_{args.topology}_N{args.N}.png"
    plot_validation_figure(results, epsilon=base.epsilon, fit=fit, outpath=fig_path)

    # Save JSON
    if args.save_json:
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_params": asdict(base),
            "sweep": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()},
            "fit": fit,
        }
        (outdir / f"validation_{args.topology}_N{args.N}.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"  α̂ = {fit.get('alpha_hat'):.3f}")
    print(f"  R² = {fit.get('r2'):.3f}")
    print(f"  n  = {fit.get('n')} valid M values")
    print(f"  Figure: {fig_path}")
    print(f"  PDF:    {fig_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
