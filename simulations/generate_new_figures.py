#!/usr/bin/env python3
"""
Generate new figures for the coherence time paper (R1 revision).

  fig_spatial_deff.pdf        — D_eff vs assembly size (analytical)
  fig_pharmacological_space.pdf — Drug classes on D_eff × Gamma_commit space

Usage:
    python generate_new_figures.py          # both figures
    python generate_new_figures.py spatial  # spatial only
    python generate_new_figures.py pharm    # pharmacological only

Author: Ian Todd
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BLUE = '#2E86AB'
RED = '#E94F37'
GREEN = '#28A745'
ORANGE = '#E8850C'
PURPLE = '#7B68EE'

OUTDIR = Path('figures')


def deff_analytical(N, r):
    """Participation ratio for compound-symmetry covariance at coherence r.

    For N oscillators with von Mises phases at concentration giving coherence r,
    C_ij = (1/2)[(1-r^2)delta_ij + r^2], yielding:
        D_eff = N^2 / [(1 + r^2(N-1))^2 + (N-1)(1-r^2)^2]
    """
    num = N**2
    denom = (1 + r**2 * (N - 1))**2 + (N - 1) * (1 - r**2)**2
    return num / denom


def save(fig, name):
    OUTDIR.mkdir(exist_ok=True)
    fig.savefig(OUTDIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUTDIR / f'{name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved figures/{name}.pdf")
    plt.close(fig)


def spatial_deff():
    """Panel A: D_eff vs N curves. Panel B: gamma vs alpha bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    N_range = np.logspace(np.log10(5), np.log10(2000), 300)

    for r, color, label in [
        (0.3, BLUE,  r'$r = 0.3$ (loose coupling)'),
        (0.5, GREEN, r'$r = 0.5$ (moderate)'),
        (0.7, RED,   r'$r = 0.7$ (tight coupling)'),
    ]:
        D = np.array([deff_analytical(N, r) for N in N_range])
        ax1.plot(N_range, D, '-', color=color, lw=2.5, label=label)
        sat = 1 / r**4
        ax1.axhline(sat, ls=':', color=color, alpha=0.25, lw=1)
        ax1.text(2200, sat, f'{sat:.0f}', fontsize=8, color=color, va='center')

    ax1.plot(N_range, N_range, ':', color='gray', alpha=0.3,
             label=r'$D_{\mathrm{eff}} = N$ (independent)')
    ax1.axhline(1, ls=':', color='gray', alpha=0.3)

    ax1.axvspan(20, 100, alpha=0.06, color=RED, zorder=0)
    ax1.text(45, 1.4, 'Local\n(gamma)', fontsize=8, color=RED,
             ha='center', alpha=0.8, style='italic')
    ax1.axvspan(300, 2000, alpha=0.06, color=BLUE, zorder=0)
    ax1.text(800, 1.4, 'Distributed\n(alpha/theta)', fontsize=8, color=BLUE,
             ha='center', alpha=0.8, style='italic')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(5, 2500)
    ax1.set_ylim(1, 500)
    ax1.set_xlabel('Assembly size $N$', fontsize=12)
    ax1.set_ylabel(r'$D_{\mathrm{eff}}$ (participation ratio)', fontsize=12)
    ax1.set_title(r'A.  $D_{\mathrm{eff}}$ scales with assembly size'
                  '\n     at fixed coherence', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # Panel B: bar chart
    gamma_N, gamma_r = 50, 0.7
    alpha_N, alpha_r = 500, 0.3
    gamma_D = deff_analytical(gamma_N, gamma_r)
    alpha_D = deff_analytical(alpha_N, alpha_r)

    ax2.bar([0, 1], [gamma_D, alpha_D], color=[RED, BLUE],
            alpha=0.85, edgecolor='black', linewidth=0.5, width=0.5)

    fold = alpha_D / gamma_D
    y_top = alpha_D * 1.08
    ax2.plot([0, 0, 1, 1],
             [gamma_D * 1.05, y_top, y_top, alpha_D * 1.02],
             color='black', lw=1)
    ax2.text(0.5, y_top * 1.05, f'{fold:.0f}\u00d7', ha='center',
             fontsize=13, fontweight='bold')

    ax2.text(0, gamma_D / 2, f'{gamma_D:.1f}', ha='center', va='center',
             fontsize=11, color='white', fontweight='bold')
    ax2.text(1, alpha_D / 2, f'{alpha_D:.0f}', ha='center', va='center',
             fontsize=11, color='white', fontweight='bold')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f'Gamma-like\n($N$={gamma_N}, $r$={gamma_r})',
         f'Alpha-like\n($N$={alpha_N}, $r$={alpha_r})'],
        fontsize=10)
    ax2.set_ylabel(r'$D_{\mathrm{eff}}$', fontsize=12)
    ax2.set_title('B.  Biological comparison:\n'
                  '     assembly size dominates', fontsize=11)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(0, y_top * 1.2)

    fig.tight_layout()
    save(fig, 'fig_spatial_deff')

    print(f"\nKey values:")
    print(f"  Gamma-like (N={gamma_N}, r={gamma_r}): D_eff = {gamma_D:.1f}")
    print(f"  Alpha-like (N={alpha_N}, r={alpha_r}): D_eff = {alpha_D:.1f}")
    print(f"  Fold change: {fold:.0f}\u00d7")
    print(f"\nSaturation values (1/r^4):")
    for r_val in [0.3, 0.5, 0.7]:
        print(f"  r={r_val}: D_eff \u2192 {1/r_val**4:.1f}")


def pharmacological_space():
    """D_eff vs Gamma_commit with four drug class arrows."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 11)
    ax.set_xlabel(r'$D_{\mathrm{eff}}$ (effective dimensionality)', fontsize=13)
    ax.set_ylabel(r'$\Gamma_{\mathrm{commit}}$ (commit rate)', fontsize=13)

    # Iso-tau contours
    for tau in [0.4, 0.8, 1.5, 3.0, 6.0, 12.0]:
        d = np.linspace(0.2, 11, 300)
        g = d / tau
        mask = (g <= 11) & (g >= 0)
        ax.plot(d[mask], g[mask], '-', color='#eeeeee', lw=0.8, zorder=1)

    ax.annotate('', xy=(10.2, 0.8), xytext=(7.0, 4.0),
                arrowprops=dict(arrowstyle='->', color='#cccccc', lw=1.5,
                                linestyle='--'))
    ax.text(9.5, 2.2, r'increasing $\tau_{\mathrm{subj}}$',
            fontsize=8, color='#aaaaaa', ha='center', style='italic',
            rotation=-48)

    # Normal waking
    nx, ny = 5.0, 5.5
    ax.plot(nx, ny, 'o', color='black', ms=12, zorder=10)
    ax.text(nx + 0.5, ny + 0.5, 'Normal\nwaking', fontsize=11,
            fontweight='bold', ha='left', va='bottom', zorder=10)

    # Drug classes: (name, dx, dy, color, phenom, lx, ly, ha, va)
    drugs = [
        ('Psychedelics',  8.5, 2.0, ORANGE,
         'Time dilation\n+ richness',          9.5, 0.5, 'center', 'top'),
        ('Deliriants',    3.8, 2.0, PURPLE,
         'Time confusion',                     2.5, 0.5, 'center', 'top'),
        ('Anaesthetics',  1.5, 3.5, BLUE,
         'Time compression\n\u2192 unconsciousness', 1.5, 6.0, 'center', 'bottom'),
        ('Stimulants',    6.0, 9.0, GREEN,
         '\u2191 Temporal\nresolution',         8.0, 9.5, 'left', 'bottom'),
    ]

    for name, dx, dy, color, phenom, lx, ly, ha, va in drugs:
        ax.annotate('', xy=(dx, dy), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2.5, mutation_scale=15),
                    zorder=5)
        ax.plot(dx, dy, 'o', color=color, ms=9, zorder=6,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(f'{name}\n{phenom}',
                    xy=(dx, dy), xytext=(lx, ly),
                    fontsize=9, color=color, fontweight='bold',
                    ha=ha, va=va,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=color, alpha=0.92),
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.8, linestyle=':'),
                    zorder=7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, -0.3, 'Low', fontsize=10, color='gray', ha='center', va='top')
    ax.text(10.5, -0.3, 'High', fontsize=10, color='gray', ha='center', va='top')
    ax.text(-0.3, 0.0, 'Low', fontsize=10, color='gray', ha='right', va='center')
    ax.text(-0.3, 10.5, 'High', fontsize=10, color='gray', ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, 'fig_pharmacological_space')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('spatial', 'both'):
        spatial_deff()
    if which in ('pharm', 'both'):
        pharmacological_space()
