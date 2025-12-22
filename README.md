# Coherence Time in Neural Oscillator Assemblies

**Repository:** todd866/coherence-time-biosystems
**Paper status:** Under review at *BioSystems* (submitted 2025-11-15)
**This repo:** Post-submission validation + revisions

## One-line thesis

Distributed biological computation is bottlenecked by **coherence time**: the waiting time for multiple semi-independent oscillator modules to align within a tolerance window.

## Core result

We derive the scaling law:

```
τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r)(M-1)]
```

where:
- **M** = coordination depth (number of semi-independent modules requiring alignment)
- **r** = Kuramoto coherence (mean resultant length; circular variance is 1-r)
- **ε** = phase alignment tolerance (radians)
- **Δω** = phase exploration rate (frequency spread + diffusion)
- **α** = topology-dependent coordination parameter

## What the paper explains (order-of-magnitude)

- Perceptual binding windows (~30–50 ms)
- Arousal-driven time dilation with stable reaction time (tachypsychia)
- Alpha frequency correlates of temporal acuity
- Metabolic scaling of temporal resolution across species

## Repository structure

```
4_coherence_time/
├── coherence_time_SUBMITTED.tex   # Frozen submitted version (DO NOT EDIT)
├── coherence_time.tex             # Development version
├── coherence_time.pdf             # Compiled output
├── revisions/                     # ← R1 REVISION WORK LIVES HERE
│   ├── coherence_time.tex         # Revised manuscript (consolidated, no ESM)
│   ├── coherence_time.pdf         # Compiled revision
│   ├── kuramoto_coherence_time_v2.py  # Updated validation code
│   ├── validation_results/        # JSON + figures for all topologies
│   └── figures/                   # Publication figures
├── simulations/                   # Original Kuramoto validation
│   └── kuramoto_coherence_time.py
├── figures/                       # Submitted figures
└── build_clean.sh                 # LaTeX build script
```

## Simulation validation

Kuramoto modular-network simulations (N=100, 20 trials per M) test the predicted scaling of τ_coh with coordination depth M.

| Topology   | r̄    | α̂    | R²   | Interpretation |
|------------|------|------|------|----------------|
| Modular    | 0.94 | 0.35 | 0.71 | Exponential scaling (formula's target regime) |
| All-to-all | 0.95 | 0.62 | 0.88 | Strong scaling but lacks independent modules |
| Sparse     | 0.79 | 0.26 | 0.96 | Scaling present; lower coherence |

**Key takeaway:** The formula applies to hierarchically modular networks where modules are internally coherent but not globally phase-locked—precisely the architecture of biological neural networks.

## Build

```bash
# Main folder (submitted version)
./build_clean.sh
# or: pdflatex coherence_time.tex && pdflatex coherence_time.tex

# Revisions folder (R1 work)
cd revisions && pdflatex coherence_time.tex && pdflatex coherence_time.tex
```

## Related papers

- **Intelligence paper** (`3_intelligence/`): High-D substrates, observable dimensionality bound
- **Coupling identification** (`36_coupling_identification/`): Sync vs measurement timescales
- **Falsifiability** (`1_falsifiability/`): Sub-Landauer domain, measurement thresholds

## Citation

```bibtex
@article{todd2025coherencetime,
  title={Coherence Time in Neural Oscillator Assemblies Sets the Speed of Thought},
  author={Todd, Ian},
  journal={BioSystems},
  year={2025},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
