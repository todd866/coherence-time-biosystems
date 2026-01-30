# Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration

**Distributed biological computation is bottlenecked by the waiting time for oscillator modules to simultaneously align.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

When multiple semi-independent oscillator modules must synchronise within a tolerance window before a system can register a state transition, the waiting time grows exponentially with coordination depth. We derive the scaling law (Eq. 4):

$$\tau_{\text{coh}} \approx \frac{1}{\Delta\omega} \cdot p_1(\varepsilon, \kappa)^{-\alpha(M-1)}$$

where M is coordination depth, p_1 is single-variable alignment probability, alpha is effective independence (topology-dependent), and Delta-omega is the phase exploration rate. The exponent is (M-1), not M, because alignment is rotationally invariant.

## Key Results

- **Exponential scaling**: Coherence time grows exponentially with number of coordinating modules, explaining why neural binding is slow relative to spike transmission
- **Topology dependence**: All-to-all, modular, and sparse coupling topologies produce distinct effective independence parameters alpha, validated by Kuramoto simulation
- **Quantitative predictions**: Visual binding (30-70 ms), cross-modal integration (100-150 ms), and 1000x flicker fusion range across taxa all follow from the scaling law
- **Thermodynamic dominance**: Coherence time exceeds quantum speed limits and Landauer bounds by 11 orders of magnitude in neural systems

## Running Simulations

```bash
cd revisions/simulations
python kuramoto_coherence_time.py       # Main Kuramoto validation (all topologies)
python generate_combined_figure.py      # Generate Figure 1 (combined panel)
python cerebellar_takens_sim.py         # Generate Figure 2 (cerebellar delay embedding)
python supplementary_analyses.py        # Supplementary validation figures
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
