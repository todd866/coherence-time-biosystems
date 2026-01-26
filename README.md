# Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status:** R1 revision submitted to *BioSystems* (BIOSYS-D-25-00981)

## Overview

Distributed biological computation is bottlenecked by **coherence time**: the waiting time for multiple semi-independent oscillator modules to simultaneously align within a tolerance window.

We derive the scaling law:

$$\tau_{\text{coh}} \approx \frac{1}{\Delta\omega} \cdot p_1(\varepsilon, \kappa)^{-\alpha(M-1)}$$

where:
- **M** = coordination depth (modules requiring alignment)
- **p₁** = single-variable alignment probability
- **α** = effective independence (topology-dependent)
- **Δω** = phase exploration rate

The exponent is **(M−1)**, not M, because alignment is rotationally invariant—one phase serves as reference.

## R1 Revision (Jan 2026)

Reviewer feedback addressed:
- Replaced "speed of thought" with "rate of state registration" (Reviewer #2)
- Added explicit discussion of Landauer bound, Margolus-Levitin quantum speed limit, and energy-time uncertainty (Reviewers #1 & #2)
- Cited Bormashenko (2024) Entropy paper on Landauer bound
- Cited 4 BioSystems papers on quantum limits of computation
- Generalized scope to "biological oscillator assemblies" (not just neural)
- Abstract trimmed to <250 words, no references

Key new section: **§2.3 Quantum and thermodynamic limits of computation** — shows that QSL and Landauer bounds are non-binding in neural systems; coherence time dominates by 11 orders of magnitude.

## Key Predictions

| Phenomenon | Predicted | Observed |
|------------|-----------|----------|
| Visual binding | 30–70 ms | 30–50 ms |
| Cross-modal integration | 100–150 ms | 100–200 ms |
| Flicker fusion range | 1000× across taxa | ~1000× |

## Repository Structure

```
├── coherence_time.tex       # Original submission
├── coherence_time.pdf       # Original PDF
├── figures/                 # Manuscript figures
├── revisions/               # Post-submission work
│   ├── coherence_time_r1.*  # R1 revision (addressing reviewer feedback)
│   ├── figures/             # R1 figures
│   └── simulations/         # Validation code
└── README.md
```

## Mathematical Foundation

The **(M−1)** exponent and the geometric derivation are detailed in a companion paper:

> Todd, I. (2025). *Alignment Probabilities on Product Statistical Manifolds: Fisher Information and Coordination Depth.* Information Geometry (in preparation).
> GitHub: [todd866/alignment-geometry](https://github.com/todd866/alignment-geometry)

## Related Work

This paper extends:
> Todd, I. (2026). *Intelligence as High-Dimensional Coherence.* BioSystems. DOI: 10.1016/j.biosystems.2026.105704

The Intelligence paper establishes *what* intelligence is (high-D coherent dynamics); this paper establishes *how fast* such systems can act.

## Building

```bash
cd revisions
pdflatex coherence_time_r1.tex
```

## Citation

```bibtex
@article{todd2026coherence,
  title={Coherence Time in Biological Oscillator Assemblies Bounds the Rate of State Registration},
  author={Todd, Ian},
  journal={BioSystems},
  year={2026},
  note={Under review, R1 revision}
}
```

## Author

Ian Todd
Sydney Medical School, University of Sydney
ORCID: [0009-0002-6994-0917](https://orcid.org/0009-0002-6994-0917)

## License

MIT License
