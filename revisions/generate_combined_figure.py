#!/usr/bin/env python3
"""
Generate combined 3-panel figure showing M-scaling for all topologies.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_results(topology: str) -> dict:
    path = Path(f"validation_results/validation_{topology}_N100.json")
    with open(path) as f:
        return json.load(f)

def main():
    topologies = ["modular", "all-to-all", "sparse"]
    titles = ["A. Modular", "B. All-to-all", "C. Sparse"]
    colors = ["#2E86AB", "#E94F37", "#28A745"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, topo, title, color in zip(axes, topologies, titles, colors):
        data = load_results(topo)
        sweep = data["sweep"]
        fit = data["fit"]

        M = np.array(sweep["M"])
        tau_med = np.array(sweep["tau_median"])
        tau_q25 = np.array(sweep["tau_q25"])
        tau_q75 = np.array(sweep["tau_q75"])
        rbar = np.array(sweep["r_mean"])

        alpha_hat = fit["alpha_hat"]
        r2 = fit["r2"]
        epsilon = data["base_params"]["epsilon"]

        # IQR error bars
        yerr_low = tau_med - tau_q25
        yerr_high = tau_q75 - tau_med

        ax.errorbar(M, tau_med, yerr=[yerr_low, yerr_high],
                    fmt='o', capsize=4, markersize=8, color=color,
                    ecolor=color, elinewidth=2, capthick=2)
        ax.set_yscale("log")
        ax.set_xlabel("Coordination depth M", fontsize=11)
        ax.set_ylabel("Coherence time (s)", fontsize=11)

        # Fitted line
        if np.isfinite(alpha_hat):
            M0 = float(M[0])
            tau0 = float(tau_med[0])
            r0 = float(rbar[0])
            c = np.log(2 * np.pi / float(epsilon))
            M_grid = np.linspace(M.min(), M.max(), 50)
            rel = np.exp(alpha_hat * c * (1 - r0) * (M_grid - M0))
            tau_pred = tau0 * rel
            ax.plot(M_grid, tau_pred, '--', linewidth=2, color='black', alpha=0.7)

        ax.set_title(f"{title}\n$\\hat{{\\alpha}}={alpha_hat:.2f}$, $R^2={r2:.2f}$", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(3, 25)  # consistent y-axis

    fig.tight_layout()

    outpath = Path("figures/fig_topology_comparison.pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.png'), dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)

if __name__ == "__main__":
    main()
