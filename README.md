# Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration

**Distributed biological computation is bottlenecked by the waiting time for oscillator modules to simultaneously align.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

When multiple semi-independent oscillator modules must synchronise within a tolerance window before a system can register a state transition, the waiting time grows exponentially with coordination depth. We derive the scaling law:

$$\tau_{\text{coh}} = \frac{1}{\lambda_{\text{attempt}}} \cdot p_1(\varepsilon, \kappa(r))^{-\alpha(M-1)}$$

where *M* is coordination depth, *r* is inter-module coherence, *p*&#8321; is single-variable alignment probability under von Mises closure, *alpha* is effective independence (topology-dependent), and *lambda*_attempt is the phase exploration rate. The exponent is (*M*-1), not *M*, because alignment is rotationally invariant.

## Key Results

- **Exponential scaling**: Coherence time grows exponentially with number of coordinating modules, validated in Kuramoto simulations (R^2 = 0.97 for modular networks)
- **Speed-flexibility trade-off**: Increasing *M* expands combinatorial flexibility but slows commits exponentially; increasing *r* speeds commits but restricts dynamics to low-dimensional attractors
- **Binding windows**: Visual binding (30-50 ms) reproduced from independently constrained parameters
- **Thermodynamic dominance**: Coherence time exceeds quantum speed limits by 11 orders of magnitude in neural systems
- **Pharmacological predictions**: The ratio D_eff / Gamma_commit predicts subjective temporal scaling across four drug classes (psychedelics, deliriants, anaesthetics, stimulants) from a two-parameter structure
- **Regime boundaries**: Modular architecture is a necessary condition; sparse random networks fail the scaling law (R^2 = 0.28), confirming the framework predicts its own regime of validity

## Running Simulations

```bash
cd simulations
python3 kuramoto_coherence_time.py       # Main Kuramoto validation (all topologies)
python3 generate_combined_figure.py      # Generate Figure 1 (topology comparison)
python3 generate_new_figures.py          # Generate Figures 2-3 (D_eff scaling, pharmacological space)
python3 supplementary_analyses.py        # Supplementary validation figures
```

## Paper

**Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration**

Todd, I. (2026). *BioSystems* (R1 revision, BIOSYS-D-25-00981).

Companion paper: [Alignment Probabilities on Product Statistical Manifolds](https://github.com/todd866/alignment-geometry)

## Citation

```bibtex
@article{todd2026coherence,
  author  = {Todd, Ian},
  title   = {Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration},
  journal = {BioSystems},
  year    = {2026},
  note    = {Under review, R1 revision}
}
```

## License

MIT License
