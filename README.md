# Coherence Time in Neural Oscillator Assemblies Sets the Speed of Thought

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status:** Under review at *BioSystems*

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

## Key Predictions

| Phenomenon | Predicted | Observed |
|------------|-----------|----------|
| Visual binding | 30–70 ms | 30–50 ms |
| Cross-modal integration | 100–150 ms | 100–200 ms |
| Flicker fusion range | 1000× across taxa | ~1000× |

## Repository Structure

```
├── coherence_time.tex       # Submitted manuscript
├── coherence_time.pdf       # Compiled PDF
├── figures/                 # Manuscript figures
├── revisions/               # Post-submission work
│   ├── coherence_time_r1.*  # R1 revision (improved formula)
│   ├── REVISION_PLAN*.md    # Revision notes
│   └── simulations/         # Validation code
└── README.md
```

## Mathematical Foundation

The **(M−1)** exponent and the geometric derivation are detailed in a companion paper:

> Todd, I. (2025). *Alignment Probabilities on Product Statistical Manifolds: Fisher Information and Coordination Depth.* Information Geometry (in preparation).
> GitHub: [todd866/alignment-geometry](https://github.com/todd866/alignment-geometry)

The companion provides the rigorous quotient-geometry proof; this paper focuses on neural applications and empirical validation.

## Building

```bash
pdflatex coherence_time.tex
```

## Citation

```bibtex
@article{todd2025coherence,
  title={Coherence Time in Neural Oscillator Assemblies Sets the Speed of Thought},
  author={Todd, Ian},
  journal={BioSystems},
  year={2025},
  note={Under review}
}
```

## Author

Ian Todd
Sydney Medical School, University of Sydney
ORCID: [0009-0002-6994-0917](https://orcid.org/0009-0002-6994-0917)

## License

MIT License
