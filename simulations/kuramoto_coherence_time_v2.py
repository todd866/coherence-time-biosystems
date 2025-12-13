#!/usr/bin/env python3
"""
Kuramoto coherence-time validation suite.

Goal
----
Validate the scaling law used in the paper:

    τ_coh ≈ (1/Δω) (2π/ε)^[ α (1 - r̄) (M - 1) ]

where
- M: number of semi-independent modules that must align
- r̄: mean Kuramoto coherence (mean resultant length) over the run (or a window)
- ε: phase alignment tolerance (radians)
- Δω: phase exploration rate (rad/s) (frequency spread + diffusion)
- α: topology factor (fit from simulation)

This script improves on the initial stub by:
- implementing phase-alignment checks across modules (not only r_threshold)
- using an SDE-friendly Euler–Maruyama integrator (optional noise)
- measuring τ_coh as *first passage with dwell time* (robust to transient crossings)
- supporting topology sweeps and fitting α from simulated data
- providing a CLI and reproducible seed control
- avoiding O(N^2) Python loops when building modular adjacency

Author: Ian Todd (with code cleanup and validation utilities)
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


def participation_ratio_from_cov(X: np.ndarray, eps: float = 1e-12) -> float:
    """Compute participation ratio from covariance eigenvalues."""
    C = np.cov(X.T)
    eig = np.linalg.eigvalsh(C)
    eig = eig[eig > eps]
    if eig.size == 0:
        return 0.0
    return float((eig.sum() ** 2) / (np.square(eig).sum()))


# ----------------------------
# Model
# ----------------------------

@dataclass
class SimParams:
    N: int = 200                 # total oscillators
    M: int = 10                  # modules
    K: float = 1.2               # coupling gain (pre-normalization)
    omega_std: float = 0.5       # natural frequency spread
    sigma: float = 0.0           # phase diffusion strength (rad / sqrt(s))
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
    r_threshold: float = 0.75
    epsilon: float = np.pi / 2           # phase tolerance between module mean phases
    dwell_time: float = 0.050            # must hold coherence for this long (s)

    # how to compute r̄
    r_avg_start: float = 0.25            # fraction of trajectory to skip as transient (0-1)

    def validate(self) -> None:
        if self.M < 2:
            raise ValueError("M must be ≥ 2.")
        if self.N < self.M:
            raise ValueError("N must be ≥ M.")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and T must be positive.")
        if not (0 < self.r_threshold <= 1):
            raise ValueError("r_threshold must be in (0,1].")
        if not (0 < self.epsilon <= np.pi):
            raise ValueError("epsilon should be in (0, π].")
        if self.dwell_time < 0:
            raise ValueError("dwell_time must be ≥ 0.")
        if not (0 <= self.r_avg_start < 1):
            raise ValueError("r_avg_start must be in [0,1).")


class KuramotoNetwork:
    """
    Kuramoto network with an adjacency/coupling matrix A (row-normalized).

    Dynamics (Ito SDE if sigma>0):
        dθ_i = [ ω_i + K Σ_j A_ij sin(θ_j - θ_i) ] dt + sigma dW_i
    """

    def __init__(self, params: SimParams):
        params.validate()
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        # frequencies and initial phases
        self.omega = self.rng.normal(loc=0.0, scale=params.omega_std, size=params.N)
        self.theta0 = self.rng.uniform(0.0, 2 * np.pi, size=params.N)

        # module assignment: nearly equal sizes
        self.modules = np.repeat(np.arange(params.M), int(np.ceil(params.N / params.M)))[:params.N]

        # adjacency / coupling weights
        self.A = self._build_A()

    def _build_A(self) -> np.ndarray:
        N, M = self.p.N, self.p.M
        topo = self.p.topology.lower()

        if topo == "all-to-all":
            A = np.ones((N, N), dtype=float) - np.eye(N, dtype=float)
            A /= (N - 1)

        elif topo == "modular":
            # Vectorized modular weights via module equality
            same = (self.modules[:, None] == self.modules[None, :]).astype(float)
            np.fill_diagonal(same, 0.0)
            A = self.p.K_inter * (1.0 - same) + self.p.K_intra * same
            # remove self coupling
            np.fill_diagonal(A, 0.0)
            # row normalize
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
        """One Euler–Maruyama step."""
        # pairwise phase differences θ_j - θ_i
        diff = theta[None, :] - theta[:, None]
        drift = self.omega + self.p.K * np.sum(self.A * np.sin(diff), axis=1)

        if self.p.sigma > 0:
            noise = self.p.sigma * self.rng.standard_normal(self.p.N) * np.sqrt(self.p.dt)
        else:
            noise = 0.0

        theta_next = theta + drift * self.p.dt + noise
        return np.mod(theta_next, 2 * np.pi)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate and return (t, theta[t, i])."""
        n_steps = int(np.floor(self.p.T / self.p.dt)) + 1
        t = np.linspace(0.0, self.p.T, n_steps)
        theta = np.zeros((n_steps, self.p.N), dtype=float)
        theta[0] = self.theta0.copy()
        for k in range(1, n_steps):
            theta[k] = self.step(theta[k - 1])
        return t, theta

    def module_stats(self, theta_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a single timepoint phases (N,), return:
        - r_m: (M,) module mean resultant lengths
        - psi_m: (M,) module mean phases
        """
        r_m = np.zeros(self.p.M, dtype=float)
        psi_m = np.zeros(self.p.M, dtype=float)
        for m in range(self.p.M):
            idx = (self.modules == m)
            r, psi = mean_resultant(theta_t[idx])
            r_m[m] = r
            psi_m[m] = psi
        return r_m, psi_m

    def coherence_event(self, theta_t: np.ndarray) -> bool:
        """Check if coherence conditions are met at a timepoint."""
        r_m, psi_m = self.module_stats(theta_t)

        # 1) all modules internally coherent
        if not np.all(r_m >= self.p.r_threshold):
            return False

        # 2) module mean phases aligned within epsilon of their circular mean
        _, psi_bar = mean_resultant(psi_m)
        phase_err = np.abs(wrap_angle(psi_m - psi_bar))
        return bool(np.max(phase_err) <= self.p.epsilon)

    def measure_tau_coh(self, t: np.ndarray, theta: np.ndarray) -> Optional[float]:
        """
        Return first time where coherence holds for dwell_time continuously.
        If dwell_time == 0, returns first crossing time.
        """
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
        """Mean global order parameter r̄ over post-transient window."""
        start_idx = int(self.p.r_avg_start * theta.shape[0])
        z = np.mean(np.exp(1j * theta[start_idx:]), axis=1)
        r = np.abs(z)
        return float(np.mean(r))

    def delta_omega_proxy(self) -> float:
        """
        Proxy for phase exploration rate Δω.

        We treat Δω as the combination of:
        - natural frequency dispersion (std of ω)
        - diffusion from noise (sigma contributes to phase variance ~ sigma^2 t)

        A simple proxy with correct units (rad/s):
            Δω ≈ std(ω) + sigma^2 / 2

        The sigma term is a heuristic; for your paper you may want to
        estimate Δω from instantaneous frequency spread or phase diffusion fits.
        """
        return float(np.std(self.omega) + 0.5 * (self.p.sigma ** 2))


# ----------------------------
# Experiments
# ----------------------------

def run_trials(base: SimParams, n_trials: int, seeds: Optional[List[int]] = None) -> Dict[str, float]:
    """Run repeated trials and return summary stats."""
    taus: List[float] = []
    rs: List[float] = []
    dws: List[float] = []

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
        dws.append(net.delta_omega_proxy())

    return {
        "tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "tau_std": float(np.std(taus)) if taus else float("nan"),
        "tau_n": int(len(taus)),
        "r_mean": float(np.mean(rs)),
        "r_std": float(np.std(rs)),
        "domega_mean": float(np.mean(dws)),
    }


def sweep_M(base: SimParams, M_values: List[int], n_trials: int) -> Dict[str, np.ndarray]:
    out = {"M": [], "tau_mean": [], "tau_std": [], "tau_n": [], "r_mean": [], "domega_mean": []}
    for M in M_values:
        p = SimParams(**{**asdict(base), "M": int(M)})
        stats = run_trials(p, n_trials=n_trials)
        out["M"].append(M)
        for k in ["tau_mean", "tau_std", "tau_n", "r_mean", "domega_mean"]:
            out[k].append(stats[k])
    return {k: np.asarray(v) for k, v in out.items()}


def fit_alpha_from_sweep(results: Dict[str, np.ndarray], epsilon: float) -> Dict[str, float]:
    """
    Fit α from:
        log(τ Δω) = α (1 - r̄) (M - 1) log(2π/ε)

    Using tau_mean per M and domega_mean proxy.
    """
    M = results["M"].astype(float)
    tau = results["tau_mean"].astype(float)
    rbar = results["r_mean"].astype(float)
    domega = results["domega_mean"].astype(float)

    mask = np.isfinite(tau) & (tau > 0) & np.isfinite(domega) & (domega > 0)
    if mask.sum() < 3:
        return {"alpha_hat": float("nan"), "r2": float("nan"), "n": int(mask.sum())}

    x = (1.0 - rbar[mask]) * (M[mask] - 1.0)
    y = np.log(tau[mask] * domega[mask])
    c = np.log(2 * np.pi / float(epsilon))

    # model y = (alpha * c) x + b, allow intercept b to absorb pre-factors
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = beta
    alpha_hat = float(slope / c) if c != 0 else float("nan")

    # r^2
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {"alpha_hat": alpha_hat, "r2": float(r2), "n": int(mask.sum()), "intercept": float(intercept)}


def plot_tau_vs_M(results: Dict[str, np.ndarray], epsilon: float, alpha_hat: Optional[float],
                  outpath: Optional[Path] = None) -> None:
    M = results["M"]
    tau = results["tau_mean"]
    tau_std = results["tau_std"]
    rbar = results["r_mean"]
    domega = results["domega_mean"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(M, tau, yerr=tau_std, fmt="o", capsize=3, label="Simulation")
    ax.set_yscale("log")
    ax.set_xlabel("Coordination depth M")
    ax.set_ylabel("Coherence time τ_coh (s)")
    ax.grid(True, alpha=0.3)

    title = f"τ_coh vs M (ε={epsilon:.2f} rad)"
    if alpha_hat is not None and np.isfinite(alpha_hat):
        title += f", α̂={alpha_hat:.2f}"
    ax.set_title(title)

    # overlay fitted scaling curve using mean rbar and mean domega
    if alpha_hat is not None and np.isfinite(alpha_hat):
        m0 = np.mean(M)
        # use the model: τ ≈ (1/Δω) exp[ α (1-r̄) (M-1) ln(2π/ε) + b ]
        # we recover b from the regression intercept if desired; here we just show relative scaling
        # anchored at the first finite tau
        mask = np.isfinite(tau) & (tau > 0)
        if mask.any():
            M0 = float(M[mask][0])
            tau0 = float(tau[mask][0])
            r0 = float(rbar[mask][0])
            d0 = float(domega[mask][0])
            c = np.log(2 * np.pi / float(epsilon))
            M_grid = np.arange(int(M.min()), int(M.max()) + 1)
            rel = np.exp(alpha_hat * c * (1 - r0) * (M_grid - M0))
            tau_pred = tau0 * rel * (d0 / np.mean(domega))
            ax.plot(M_grid, tau_pred, "--", linewidth=2, label="Fitted scaling (anchored)")

    ax.legend()
    fig.tight_layout()
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate coherence-time scaling in modular Kuramoto networks.")
    ap.add_argument("--outdir", type=str, default="figures", help="Output directory for figures/results.")
    ap.add_argument("--topology", type=str, default="modular", choices=["all-to-all", "modular", "sparse"])
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--K", type=float, default=1.2)
    ap.add_argument("--omega-std", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--r-threshold", type=float, default=0.75)
    ap.add_argument("--epsilon", type=float, default=float(np.pi/2))
    ap.add_argument("--dwell", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--M-min", type=int, default=4)
    ap.add_argument("--M-max", type=int, default=20)
    ap.add_argument("--M-step", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-json", action="store_true", help="Save sweep results as JSON.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    base = SimParams(
        N=args.N, M=10, K=args.K, omega_std=args.omega_std, sigma=args.sigma,
        topology=args.topology, T=args.T, dt=args.dt, seed=args.seed,
        r_threshold=args.r_threshold, epsilon=args.epsilon, dwell_time=args.dwell,
    )

    M_values = list(range(args.M_min, args.M_max + 1, args.M_step))
    results = sweep_M(base, M_values=M_values, n_trials=args.trials)
    fit = fit_alpha_from_sweep(results, epsilon=base.epsilon)

    # Save figure + JSON
    fig_path = outdir / f"tau_vs_M_{args.topology}_N{args.N}.png"
    plot_tau_vs_M(results, epsilon=base.epsilon, alpha_hat=fit.get("alpha_hat"), outpath=fig_path)

    if args.save_json:
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_params": asdict(base),
            "sweep": {k: results[k].tolist() for k in results},
            "fit": fit,
        }
        (outdir / f"sweep_tau_vs_M_{args.topology}_N{args.N}.json").write_text(json.dumps(payload, indent=2))

    print("Sweep complete.")
    print(f"  topology={args.topology}, N={args.N}, trials={args.trials}")
    print(f"  α_hat={fit.get('alpha_hat'):.3f}, r2={fit.get('r2'):.3f}, n={fit.get('n')}")
    print(f"  Figure: {fig_path}")


if __name__ == "__main__":
    main()
