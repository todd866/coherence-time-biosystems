#!/usr/bin/env python3
"""
Supplementary analyses for coherence time paper.

Includes:
1. Conditional exponentiality test - show waiting times are exponential when conditioned on r
2. ε-scaling test - verify τ scales as ε^{-(M-1)}
3. Participation ratio vs fitted α comparison
4. von Mises distribution validation - empirical phase deviations vs theoretical PDF

Author: Ian Todd
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from dataclasses import asdict

from kuramoto_coherence_time import (
    SimParams, KuramotoNetwork, run_trials, sweep_M,
    r_to_kappa, compute_p1, fit_alpha_from_sweep, wrap_angle
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def test_conditional_exponentiality(n_long_trials: int = 100, T: float = 120.0):
    """
    Test that waiting times are approximately exponential when conditioned on r.

    The paper predicts that unconditional waiting times may be heavy-tailed
    (due to r(t) drift), but conditional on a narrow r bin, they should be
    approximately exponential.
    """
    print("\n" + "="*60)
    print("TEST 1: Conditional Exponentiality")
    print("="*60)

    # Run longer simulations to collect more hits
    params = SimParams(
        N=60, M=5, K=0.8, omega_std=0.4, sigma=0.15,
        T=T, epsilon=2.0, r_threshold=0.5, seed=0
    )

    print(f"Running {n_long_trials} trials (T={T}s each)...")
    results = run_trials(params, n_trials=n_long_trials)

    # Extract per-trial data
    trials = results.get("trials", [])
    if not trials:
        print("  No per-trial data available.")
        return

    # Get tau and r_inter_at_hit for hits
    hits = [t for t in trials if t["hit"]]
    if len(hits) < 20:
        print(f"  Only {len(hits)} hits, need more for statistical test.")
        return

    taus = np.array([t["tau"] for t in hits])
    # Use trajectory-mean r_inter (measured over entire trial, avoids outcome leakage)
    # NOT r_inter_at_hit, which could be biased by selection (r typically higher near hits)
    r_inters = np.array([t["r_inter"] for t in hits])

    print(f"  Collected {len(hits)} hit events")
    print(f"  τ range: [{taus.min():.2f}, {taus.max():.2f}] s")
    print(f"  r_inter (trajectory mean) range: [{r_inters.min():.3f}, {r_inters.max():.3f}]")

    # Bin by trajectory-mean r_inter and test exponentiality in each bin
    r_bins = np.percentile(r_inters, [0, 33, 67, 100])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (ax, r_lo, r_hi) in enumerate(zip(axes, r_bins[:-1], r_bins[1:])):
        mask = (r_inters >= r_lo) & (r_inters <= r_hi)
        tau_bin = taus[mask]

        if len(tau_bin) < 10:
            ax.text(0.5, 0.5, f"n={len(tau_bin)} (too few)", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        # Fit exponential and test
        rate = 1.0 / np.mean(tau_bin)
        _, p_value = stats.kstest(tau_bin, 'expon', args=(0, 1/rate))

        # Plot empirical CDF vs exponential CDF
        tau_sorted = np.sort(tau_bin)
        ecdf = np.arange(1, len(tau_bin) + 1) / len(tau_bin)
        tau_grid = np.linspace(0, tau_sorted.max(), 100)
        exp_cdf = 1 - np.exp(-rate * tau_grid)

        ax.step(tau_sorted, ecdf, where='post', label='Empirical', linewidth=2)
        ax.plot(tau_grid, exp_cdf, 'r--', label=f'Exp(λ={rate:.2f})', linewidth=2)
        ax.set_xlabel('τ (s)', fontsize=11)
        ax.set_ylabel('CDF', fontsize=11)
        ax.set_title(f'r ∈ [{r_lo:.3f}, {r_hi:.3f}]\nn={len(tau_bin)}, KS p={p_value:.3f}',
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Conditional Exponentiality Test: τ | r_inter (trajectory mean)', fontsize=12, fontweight='bold')
    fig.tight_layout()

    outpath = FIGURES_DIR / "fig_conditional_exponentiality.pdf"
    fig.savefig(outpath, bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close(fig)


def test_epsilon_scaling(n_trials: int = 20):
    """
    Test that τ scales as ε^{-(M-1)} by varying ε at fixed M.

    For low coherence (κ→0), p1 ≈ ε/(2π), so τ ∝ (2π/ε)^{α(M-1)}.
    We test this by running simulations at multiple ε values.
    """
    print("\n" + "="*60)
    print("TEST 2: ε-Scaling")
    print("="*60)

    M = 4  # Lower M for faster convergence
    epsilon_values = [0.8, 1.0, 1.3, 1.6, 2.0]  # radians (tighter range to see scaling)

    # Use moderate coupling - strong enough to hit, weak enough that ε matters
    base_params = SimParams(
        N=40, M=M, K=0.8, omega_std=0.3, sigma=0.08,
        T=45.0, r_threshold=0.55, seed=0
    )

    results_by_eps = {}
    for eps in epsilon_values:
        print(f"  ε={eps:.1f}...", end=" ", flush=True)
        params = SimParams(**{**asdict(base_params), "epsilon": eps})
        res = run_trials(params, n_trials=n_trials)
        results_by_eps[eps] = res
        tau_str = f"{res['tau_median']:.2f}" if np.isfinite(res['tau_median']) else ">T"
        print(f"τ_med={tau_str}, hit={res['n_hit']}/{res['n_trials']}")

    # Extract data for plotting (prefer KM estimates if available)
    eps_arr = np.array(list(results_by_eps.keys()))
    tau_arr = np.array([results_by_eps[e].get('tau_median_km', results_by_eps[e]['tau_median']) for e in eps_arr])
    hit_frac = np.array([results_by_eps[e]['hit_fraction'] for e in eps_arr])
    valid = np.isfinite(tau_arr) & (tau_arr > 0) & (hit_frac > 0.5)  # exclude heavily censored

    if valid.sum() < 3:
        print("  Insufficient valid data for ε-scaling test.")
        return

    # Theory predicts: log(τ) ∝ -(M-1) log(ε) + const (in low-κ limit)
    # So plot log(τ) vs log(1/ε)
    x = np.log(1.0 / eps_arr[valid])
    y = np.log(tau_arr[valid])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Expected slope = α(M-1); typically α ≈ 0.3-0.7 for modular networks
    expected_slope_alpha1 = M - 1
    expected_slope_alpha05 = 0.5 * (M - 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=80, c='#2E86AB', zorder=5)

    x_fit = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2,
            label=f'Fit: slope={slope:.2f}')

    ax.set_xlabel('log(1/ε)', fontsize=12)
    ax.set_ylabel('log(τ)', fontsize=12)
    ax.set_title(f'ε-Scaling Test (M={M})\nSlope={slope:.2f}, R²={r_value**2:.3f}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_epsilon_scaling.pdf"
    fig.savefig(outpath, bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    print(f"  Fitted slope: {slope:.3f}")
    print(f"  Expected: {expected_slope_alpha1} if α=1, or {expected_slope_alpha05:.1f} if α=0.5")
    print(f"  Implied α = slope/(M-1) = {slope/(M-1):.3f}")
    print(f"  R² = {r_value**2:.3f}")
    plt.close(fig)


def test_participation_ratio_vs_alpha(n_trials: int = 20):
    """
    Compare α from scaling fit to α estimated from participation ratio.

    The paper defines d_eff = α(M-1) and estimates it from:
    d_eff ≈ (tr C)² / tr(C²)

    where C is the covariance of module phase deviations.
    """
    print("\n" + "="*60)
    print("TEST 3: Participation Ratio vs Fitted α")
    print("="*60)

    # Run sweep to get fitted α
    base_params = SimParams(
        N=60, M=5, K=0.8, omega_std=0.3, sigma=0.1,
        T=60.0, epsilon=2.0, r_threshold=0.6, seed=0
    )

    M_values = [3, 4, 5, 6, 7, 8]
    print("Running M-sweep for fitted α...")
    sweep_results = sweep_M(base_params, M_values, n_trials=n_trials)
    fit = fit_alpha_from_sweep(sweep_results, epsilon=base_params.epsilon)
    alpha_fit = fit.get('alpha_hat', float('nan'))

    print(f"\n  Fitted α = {alpha_fit:.3f}")

    # Now estimate α from participation ratio at each M
    print("\nEstimating α from participation ratio at each M...")
    alpha_pr_estimates = []

    for M in M_values:
        params = SimParams(**{**asdict(base_params), "M": M})
        net = KuramotoNetwork(params)
        t, theta = net.simulate()

        # Compute module phases over time
        n_steps = theta.shape[0]
        start_idx = int(0.25 * n_steps)  # skip transient
        psi_m = np.zeros((n_steps - start_idx, M))

        for k, t_idx in enumerate(range(start_idx, n_steps)):
            _, psi = net.module_stats(theta[t_idx])
            psi_m[k] = psi

        # Compute phase deviations from mean
        mean_psi = np.angle(np.mean(np.exp(1j * psi_m), axis=1, keepdims=True))
        delta_psi = np.angle(np.exp(1j * (psi_m - mean_psi)))  # wrap to [-π, π]

        # Covariance matrix of phase deviations
        C = np.cov(delta_psi.T)

        # Participation ratio
        eigenvalues = np.linalg.eigvalsh(C)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # remove numerical zeros
        if len(eigenvalues) > 0:
            d_eff = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            alpha_pr = d_eff / (M - 1) if M > 1 else float('nan')
        else:
            alpha_pr = float('nan')

        alpha_pr_estimates.append(alpha_pr)
        print(f"  M={M}: d_eff={d_eff:.2f}, α_PR={alpha_pr:.3f}")

    alpha_pr_mean = np.nanmean(alpha_pr_estimates)
    alpha_pr_std = np.nanstd(alpha_pr_estimates)

    print(f"\n  Mean α from participation ratio: {alpha_pr_mean:.3f} ± {alpha_pr_std:.3f}")
    print(f"  α from scaling fit: {alpha_fit:.3f}")
    print(f"  Ratio (fit/PR): {alpha_fit/alpha_pr_mean:.2f}" if alpha_pr_mean > 0 else "  (PR estimate invalid)")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(M_values, alpha_pr_estimates, s=80, c='#2E86AB', label='α from participation ratio')
    ax.axhline(alpha_fit, color='#E94F37', linestyle='--', linewidth=2, label=f'α from scaling fit = {alpha_fit:.2f}')
    ax.axhline(alpha_pr_mean, color='#28A745', linestyle=':', linewidth=2, label=f'Mean α_PR = {alpha_pr_mean:.2f}')

    ax.set_xlabel('M (coordination depth)', fontsize=12)
    ax.set_ylabel('α (effective independence)', fontsize=12)
    ax.set_title('Participation Ratio vs Scaling Fit', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_participation_ratio_vs_alpha.pdf"
    fig.savefig(outpath, bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close(fig)


def test_von_mises_fit(n_trials: int = 30):
    """
    Validate that phase deviations follow von Mises distribution conditional on r.

    The paper assumes module phase deviations follow von Mises(0, κ) given
    mean resultant length r. This test checks the r ↔ κ mapping directly by
    binning samples by r_inter and testing von Mises fit within each bin.

    This is a stronger test than pooling: it validates "model closure."
    """
    print("\n" + "="*60)
    print("TEST 4: Von Mises Distribution Fit (Conditional on r)")
    print("="*60)

    # Run simulations to collect phase deviations
    params = SimParams(
        N=60, M=5, K=0.8, omega_std=0.3, sigma=0.1,
        T=60.0, epsilon=2.0, r_threshold=0.6, seed=0
    )

    print(f"Running {n_trials} trials to collect phase deviations...")
    results = run_trials(params, n_trials=n_trials)

    trials = results.get("trials", [])
    if not trials:
        print("  No per-trial data available.")
        return

    # Collect phase deviations PAIRED with their r_inter values
    # (not pooled, so we can bin by r later)
    phase_dev_samples = []  # list of (r_inter, delta_psi_array) tuples

    for trial_dict in trials:
        # Re-run simulation to get phase trajectory using ORIGINAL seed
        original_seed = trial_dict.get("seed", 0)
        p = SimParams(**{**asdict(params), "seed": int(original_seed)})
        net = KuramotoNetwork(p)
        t, theta = net.simulate()

        start_idx = int(0.25 * len(theta))
        end_idx = int(0.75 * len(theta))

        for i in range(start_idx, end_idx, 10):  # sample every 10 steps
            _, psi_m = net.module_stats(theta[i])
            r_inter_val = net.r_inter_at_time(theta[i])

            # Phase deviations from mean
            psi_mean = np.angle(np.mean(np.exp(1j * psi_m)))
            delta_psi = wrap_angle(psi_m - psi_mean)

            phase_dev_samples.append((r_inter_val, delta_psi))

    # Convert to arrays
    r_vals = np.array([s[0] for s in phase_dev_samples])

    print(f"  Collected {len(phase_dev_samples)} time-point samples")
    print(f"  r_inter range: [{r_vals.min():.3f}, {r_vals.max():.3f}]")

    # Bin by r_inter (4 bins)
    r_bins = np.percentile(r_vals, [0, 25, 50, 75, 100])
    n_bins = len(r_bins) - 1

    # Create figure with one panel per bin
    fig, axes = plt.subplots(1, n_bins, figsize=(12, 3.5))
    from scipy.stats import vonmises

    kappa_expected_list = []
    kappa_fitted_list = []
    r_bin_centers = []

    for bin_idx, (r_lo, r_hi) in enumerate(zip(r_bins[:-1], r_bins[1:])):
        ax = axes[bin_idx]

        # Select samples in this r bin
        mask = (r_vals >= r_lo) & (r_vals <= r_hi)
        samples_in_bin = [phase_dev_samples[i] for i in range(len(phase_dev_samples)) if mask[i]]

        if len(samples_in_bin) < 20:
            ax.text(0.5, 0.5, f'n={len(samples_in_bin)}\n(too few)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'r ∈ [{r_lo:.2f}, {r_hi:.2f}]', fontsize=10)
            continue

        # Pool phase deviations from all time points in this bin
        r_in_bin = [s[0] for s in samples_in_bin]
        phase_devs_in_bin = np.concatenate([s[1] for s in samples_in_bin])
        r_mean_bin = np.mean(r_in_bin)

        # Expected κ from r → κ mapping
        kappa_expected = r_to_kappa(r_mean_bin)

        # Fitted κ via MLE
        kappa_mle, _, _ = vonmises.fit(phase_devs_in_bin, fscale=1)

        kappa_expected_list.append(kappa_expected)
        kappa_fitted_list.append(kappa_mle)
        r_bin_centers.append(r_mean_bin)

        # Plot histogram
        ax.hist(phase_devs_in_bin, bins=25, density=True, alpha=0.6,
               color='#2E86AB', label='Empirical')

        # Overlay von Mises PDFs
        theta_grid = np.linspace(-np.pi, np.pi, 200)
        pdf_expected = vonmises.pdf(theta_grid, kappa_expected)
        pdf_fitted = vonmises.pdf(theta_grid, kappa_mle)

        ax.plot(theta_grid, pdf_expected, 'r--', linewidth=2,
               label=f'κ(r)={kappa_expected:.2f}')
        ax.plot(theta_grid, pdf_fitted, 'g-', linewidth=1.5, alpha=0.7,
               label=f'κ_MLE={kappa_mle:.2f}')

        ax.set_xlabel('δψ (rad)', fontsize=9)
        if bin_idx == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'r ∈ [{r_lo:.2f}, {r_hi:.2f}]\n(r̄={r_mean_bin:.2f}, n={len(samples_in_bin)})',
                    fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Von Mises Fit Conditional on r_inter (Model Closure Test)',
                fontsize=11, fontweight='bold')
    fig.tight_layout()

    outpath = FIGURES_DIR / "fig_von_mises_validation.pdf"
    fig.savefig(outpath, bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")

    # Summary of κ comparison
    print(f"\n  Summary of r ↔ κ mapping:")
    print(f"  {'r_mean':<10} {'κ_expected':<12} {'κ_fitted':<12} {'Ratio':<8}")
    for r_m, k_exp, k_fit in zip(r_bin_centers, kappa_expected_list, kappa_fitted_list):
        ratio = k_fit / k_exp if k_exp > 0 else float('nan')
        print(f"  {r_m:<10.3f} {k_exp:<12.3f} {k_fit:<12.3f} {ratio:<8.3f}")

    # Overall assessment
    ratios = [k_fit / k_exp for k_exp, k_fit in zip(kappa_expected_list, kappa_fitted_list) if k_exp > 0]
    if ratios:
        mean_ratio = np.mean(ratios)
        print(f"\n  Mean κ_fitted/κ_expected = {mean_ratio:.3f}")
        if 0.8 < mean_ratio < 1.2:
            print(f"  ✓ von Mises r ↔ κ mapping validated (within 20%)")
        else:
            print(f"  ⚠ κ mapping shows systematic bias")

    plt.close(fig)


def main():
    print("="*60)
    print("SUPPLEMENTARY ANALYSES FOR COHERENCE TIME PAPER")
    print("="*60)

    # Run all tests
    test_conditional_exponentiality(n_long_trials=50, T=60.0)
    test_epsilon_scaling(n_trials=15)
    test_participation_ratio_vs_alpha(n_trials=15)
    test_von_mises_fit(n_trials=30)

    print("\n" + "="*60)
    print("ALL SUPPLEMENTARY ANALYSES COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
