# Coherence Time in Neural Oscillator Assemblies

**Repository:** todd866/coherence-time-biosystems
**Paper status:** Submitted to *BioSystems* (2025-11-15)
**This repo:** Post-submission validation + revisions (transparent, versioned)

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
- **α** = effective coordination regime parameter (depends on topology × coupling)

## What the paper explains (order-of-magnitude)

- Perceptual binding windows (~30–50 ms)
- Arousal-driven time dilation with stable reaction time (tachypsychia; dual-loop dissociation)
- Alpha frequency correlates of temporal acuity
- Metabolic scaling of temporal resolution across species

## Repository structure

```
4_coherence_time/
├── coherence_time_SUBMITTED.tex    # Frozen submitted version (DO NOT EDIT)
├── coherence_time.tex              # Development version
├── coherence_time.pdf              # Compiled output
├── revisions/                      # R1 materials when reviews arrive
│   ├── coherence_time_r1.tex       # Revised manuscript
│   └── README.md                   # Revision notes
├── simulations/                    # Kuramoto validation code
│   ├── kuramoto_coherence_time_v1.py
│   └── kuramoto_coherence_time_v2.py
├── figures/                        # Generated plots
└── build_clean.sh                  # LaTeX build script
```

## Build

```bash
./build_clean.sh
# or manually:
pdflatex coherence_time.tex && pdflatex coherence_time.tex
```

## Simulation validation (ESM 3)

Kuramoto modular-network simulations (N=100, 10 trials per M) test the predicted log-linear scaling of τ_coh with M.

| Topology   | α̂    | r²   | Interpretation                                            |
|------------|------|------|-----------------------------------------------------------|
| Modular    | 0.65 | 0.46 | Clear exponential scaling                                 |
| All-to-all | 0.15 | 0.71 | Weak M-dependence (global sync collapses effective depth) |
| Sparse     | 0.01 | 0.05 | No scaling (outside model assumptions)                    |

**Key takeaway:** The exponential coordination penalty emerges in the modular regime, consistent with biological networks being hierarchically modular rather than fully-connected or randomly sparse.

## Related papers

- **Intelligence paper** (`3_intelligence/`): High-D substrates, measurement-theoretic tracking bound
- **Abiogenesis paper** (`5_abiogenesis_chemistry/`): Code formation, dimensional mismatch
- **Coherence conditions** (`1_coherence_conditions/`): Coherence requirements for biological computation

## Citation

If you use this work, please cite:

```bibtex
@article{todd2025coherencetime,
  title={Coherence Time in Neural Oscillator Assemblies},
  author={Todd, Ian},
  journal={BioSystems},
  year={2025},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
