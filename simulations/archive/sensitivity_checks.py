#!/usr/bin/env python3
"""
Sensitivity checks for reviewer concerns:
1. Does r_intra (within-module coherence) decline with M at fixed N=100?
2. Is alpha_hat robust to doubling T_dwell?

Runs modular topology only (the validated regime).
"""

import json
import numpy as np
from pathlib import Path
from kuramoto_coherence_time import SimParams, KuramotoNetwork, r_to_kappa, compute_p1

# Load base params from the saved validation
with open('validation_results/validation_modular_N100.json') as f:
    saved = json.load(f)
bp = saved['base_params']


def check_r_intra():
    """Check 1: r_intra vs M at fixed N=100."""
    print("=" * 60)
    print("CHECK 1: Within-module coherence r_intra vs M")
    print("  N=100 fixed, M varies, module size = N/M")
    print("=" * 60)

    M_vals = list(range(3, 11))
    n_trials = 10  # enough to get stable means

    print(f"\n{'M':>3} {'N/M':>5} {'r_intra mean':>12} {'r_intra std':>12} {'r_inter mean':>12}")
    print("-" * 50)

    results = []
    for M in M_vals:
        r_intra_all = []
        r_inter_all = []
        for trial in range(n_trials):
            p = SimParams(
                N=bp['N'], M=M, K=bp['K'],
                omega_std=bp['omega_std'], sigma=bp['sigma'],
                topology='modular', K_intra=bp['K_intra'], K_inter=bp['K_inter'],
                T=20.0, dt=bp['dt'], seed=trial * 100 + M,
                r_threshold=bp['r_threshold'], epsilon=bp['epsilon'],
                dwell_time=bp['dwell_time']
            )
            net = KuramotoNetwork(p)
            t, theta = net.simulate()

            # r_intra: mean within-module coherence
            start = int(p.r_avg_start * theta.shape[0])
            r_intras = []
            for k in range(start, theta.shape[0], 50):  # sample every 50 steps
                r_m, _ = net.module_stats(theta[k])
                r_intras.append(np.mean(r_m))
            r_intra_all.append(np.mean(r_intras))

            # r_inter for comparison
            r_inter_all.append(net.mean_inter_module_r(theta))

        mean_ri = np.mean(r_intra_all)
        std_ri = np.std(r_intra_all)
        mean_re = np.mean(r_inter_all)
        results.append((M, bp['N'] // M, mean_ri, std_ri, mean_re))
        print(f"{M:3d} {bp['N']//M:5d} {mean_ri:12.4f} {std_ri:12.4f} {mean_re:12.4f}")

    # Check for systematic decline
    r_intras = [r[2] for r in results]
    correlation = np.corrcoef(M_vals, r_intras)[0, 1]
    print(f"\nCorrelation(M, r_intra) = {correlation:.3f}")
    if abs(correlation) < 0.5:
        print("=> No systematic decline in r_intra with M.")
    else:
        print(f"=> {'Decline' if correlation < 0 else 'Increase'} detected — worth reporting.")


def check_dwell_sensitivity():
    """Check 2: alpha_hat robustness to T_dwell variation."""
    print("\n" + "=" * 60)
    print("CHECK 2: Dwell time sensitivity")
    print("  Base T_dwell = 30 ms, testing 15 ms and 60 ms")
    print("=" * 60)

    dwell_values = [0.015, 0.03, 0.06]
    n_trials = 20
    M_vals = [3, 4, 5, 6, 7]  # use the 5 M values from the R²=0.97 fit

    for dwell in dwell_values:
        print(f"\n--- T_dwell = {dwell*1000:.0f} ms ---")
        tau_by_M = {}

        for M in M_vals:
            taus = []
            for trial in range(n_trials):
                p = SimParams(
                    N=bp['N'], M=M, K=bp['K'],
                    omega_std=bp['omega_std'], sigma=bp['sigma'],
                    topology='modular', K_intra=bp['K_intra'], K_inter=bp['K_inter'],
                    T=bp['T'], dt=bp['dt'], seed=trial * 1000 + M,
                    r_threshold=bp['r_threshold'], epsilon=bp['epsilon'],
                    dwell_time=dwell
                )
                net = KuramotoNetwork(p)
                t, theta = net.simulate()
                tau = net.measure_tau_coh(t, theta)
                if tau is not None:
                    taus.append(tau)
            tau_by_M[M] = taus

        # Compute alpha_hat via simple linear regression on log(median_tau) vs M
        Ms_fit, log_taus = [], []
        for M in M_vals:
            if len(tau_by_M[M]) >= 5:
                median_tau = np.median(tau_by_M[M])
                Ms_fit.append(M)
                log_taus.append(np.log(median_tau))
                hit_frac = len(tau_by_M[M]) / n_trials
                print(f"  M={M}: median_tau={median_tau:.2f}s, hits={len(tau_by_M[M])}/{n_trials}")

        if len(Ms_fit) >= 3:
            Ms_fit = np.array(Ms_fit)
            log_taus = np.array(log_taus)
            # Simple slope estimate: log(tau) ~ slope * M + intercept
            slope, intercept = np.polyfit(Ms_fit, log_taus, 1)
            ss_res = np.sum((log_taus - (slope * Ms_fit + intercept))**2)
            ss_tot = np.sum((log_taus - np.mean(log_taus))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  => slope = {slope:.3f}, R² = {r2:.3f}")
        else:
            print(f"  => Too few M-values with hits for fit")


if __name__ == '__main__':
    check_r_intra()
    check_dwell_sensitivity()
