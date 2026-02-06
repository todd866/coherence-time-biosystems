#!/usr/bin/env python3
"""
Generate combined 3-panel figure showing M-scaling for all topologies.

Uses r_inter_mean (inter-module coherence) when available for better
alignment with Eq. (4); falls back to r_mean (global coherence) otherwise.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_results(topology: str) -> dict:
    path = Path(f"validation_results/validation_{topology}_N100.json")
    if not path.exists():
        print(f"  Warning: {path} not found, skipping {topology}")
        return None
    with open(path) as f:
        return json.load(f)

def main():
    topologies = ["modular", "all-to-all", "sparse"]
    titles = ["A. Modular", "B. All-to-all", "C. Sparse"]
    colors = ["#2E86AB", "#E94F37", "#28A745"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, topo, title, color in zip(axes, topologies, titles, colors):
        data = load_results(topo)
        if data is None:
            ax.text(0.5, 0.5, f"No data for {topo}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue

        sweep = data["sweep"]
        fit = data["fit"]

        M = np.array(sweep["M"])

        # Prefer KM estimates (handle censoring), fallback to naive
        if "tau_median_km" in sweep and any(np.isfinite(sweep["tau_median_km"])):
            tau_med = np.array(sweep["tau_median_km"])
            tau_q25 = np.array(sweep["tau_q25_km"])
            tau_q75 = np.array(sweep["tau_q75_km"])
            tau_label = "KM"
        else:
            tau_med = np.array(sweep["tau_median"])
            tau_q25 = np.array(sweep["tau_q25"])
            tau_q75 = np.array(sweep["tau_q75"])
            tau_label = "naive"

        hit_fraction = np.array(sweep.get("hit_fraction", [1.0]*len(M)))

        # Prefer unbiased r_inter_mean_all (all trials), then hit-only, then global
        if "r_inter_mean_all" in sweep and any(np.isfinite(sweep["r_inter_mean_all"])):
            rbar = np.array(sweep["r_inter_mean_all"])
            r_label = "r_inter_all"
        elif "r_inter_mean" in sweep and any(np.isfinite(sweep["r_inter_mean"])):
            rbar = np.array(sweep["r_inter_mean"])
            r_label = "r_inter"
        else:
            rbar = np.array(sweep["r_mean"])
            r_label = "r_global"

        alpha_hat = fit["alpha_hat"]
        r2 = fit["r2"]
        epsilon = data["base_params"]["epsilon"]

        # IQR error bars
        yerr_low = tau_med - tau_q25
        yerr_high = tau_q75 - tau_med

        # Plot points with error bars
        valid = np.isfinite(tau_med) & (tau_med > 0)
        ax.errorbar(M[valid], tau_med[valid], yerr=[yerr_low[valid], yerr_high[valid]],
                    fmt='o', capsize=4, markersize=8, color=color,
                    ecolor=color, elinewidth=2, capthick=2)

        # Add censoring indicators (upward arrows) when hit_fraction < 0.5
        censored = hit_fraction < 0.5
        if any(censored):
            # For censored points, plot at tau_med (lower bound) with upward arrow
            for m_val, tau_val, hf in zip(M[censored], tau_med[censored], hit_fraction[censored]):
                if np.isfinite(tau_val) and tau_val > 0:
                    ax.annotate('', xy=(m_val, tau_val * 1.5),
                               xytext=(m_val, tau_val),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2))
                    ax.text(m_val, tau_val * 1.7, f'{hf:.1f}',
                           ha='center', fontsize=8, color=color)

        ax.set_yscale("log")
        ax.set_xlabel("Coordination depth M", fontsize=11)
        ax.set_ylabel("Coherence time (s)", fontsize=11)

        # Fitted line: use actual model predictions with interpolated r(M)
        if np.isfinite(alpha_hat) and 'intercept' in fit:
            intercept = fit['intercept']
            fit_method = fit.get('method', 'surrogate')

            # Get attempt rate (prefer lambda_attempt_mean, fallback to domega)
            if "lambda_attempt_mean" in sweep:
                attempt_rate = np.array(sweep["lambda_attempt_mean"])
            else:
                domega = np.array(sweep.get("domega_mean", sweep.get("domega_effective", [1.0]*len(M))))
                attempt_rate = domega / (2 * np.pi)

            M_grid = np.linspace(M.min(), M.max(), 50)
            r_interp = np.interp(M_grid, M, rbar)
            rate_interp = np.interp(M_grid, M, attempt_rate)

            if fit_method == 'theory_consistent':
                # Theory-consistent: log(τ·λ) = α(M-1)·(-log p1) + intercept
                # Need to compute p1 for each r value
                from kuramoto_coherence_time import r_to_kappa, compute_p1
                neg_log_p1_interp = np.zeros_like(r_interp)
                for i, r_val in enumerate(r_interp):
                    kappa = r_to_kappa(float(r_val))
                    p1 = compute_p1(epsilon, kappa)
                    neg_log_p1_interp[i] = -np.log(p1) if p1 > 0 and p1 < 1 else 0.0

                x_pred = (M_grid - 1.0) * neg_log_p1_interp
                log_tau_lambda = alpha_hat * x_pred + intercept
                tau_pred = np.exp(log_tau_lambda) / rate_interp
            else:
                # Surrogate: log(τ·Δω) ≈ α·c·(1-r)(M-1) + intercept
                c = np.log(2 * np.pi / float(epsilon))
                x_pred = (1 - r_interp) * (M_grid - 1)
                log_tau_lambda = alpha_hat * c * x_pred + intercept
                tau_pred = np.exp(log_tau_lambda) / rate_interp

            ax.plot(M_grid, tau_pred, '--', linewidth=2, color='black', alpha=0.7)

        ax.set_title(f"{title}\n$\\hat{{\\alpha}}={alpha_hat:.2f}$, $R^2={r2:.2f}$", fontsize=11)
        ax.grid(True, alpha=0.3)
        # Dynamic y-limits based on data (with padding)
        valid_tau = tau_med[np.isfinite(tau_med) & (tau_med > 0)]
        if len(valid_tau) > 0:
            ymin = max(0.5, valid_tau.min() * 0.7)
            ymax = valid_tau.max() * 1.5
            ax.set_ylim(ymin, ymax)

    fig.tight_layout()

    outpath = Path("figures/fig_topology_comparison.pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.png'), dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)

if __name__ == "__main__":
    main()
