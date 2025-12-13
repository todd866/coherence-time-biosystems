# Coherence Time in Neural Oscillator Assemblies

**Status:** Submitted for peer review (2025-11-15)
**Current version:** Post-submission improvements in progress

## Overview

This paper derives a quantitative framework showing that coherence time in coupled oscillator networks sets the fundamental speed limit for biological information processing.

**Core formula:**
```
τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r)(M-1)]
```

Where:
- `M` = coordination depth (number of semi-independent modules requiring phase alignment)
- `r` = Kuramoto coherence (mean resultant length)
- `ε` = phase alignment tolerance (radians)
- `Δω` = phase-exploration rate (rad/s)
- `α` = network topology parameter (0.6-1.0)

## Key Claims

1. **Exponential scaling with coordination depth:** Processing speed slows exponentially as more modules must synchronize
2. **Speed-flexibility trade-off:** High coherence (high r) → fast commits but low flexibility; low coherence → slow commits but high flexibility
3. **Universal across biological oscillators:** Neural, molecular, genetic, circadian systems all governed by same physics
4. **Quantitative predictions:** Visual binding (30-50ms), tachypsychia dissociation, metabolic scaling of temporal acuity

## Directory Structure

```
4_coherence_time/
├── coherence_time.tex              # Main paper (current development version)
├── coherence_time_SUBMITTED.tex    # Submitted version (backup, DO NOT EDIT)
├── build_clean.sh                  # Compilation script
├── commentary/
│   └── codex.txt                   # Development log, changelog
├── figures/                        # Plots and diagrams
├── simulations/                    # Kuramoto network simulations
└── data/                          # Empirical validation data
```

## Recent Additions (Post-Submission)

### Igamberdiev Connection (2025-11-16)
Added historical context connecting to:
- **Vernadsky (1945):** Biological space-time geometry
- **Igamberdiev (1985):** Biological time as logarithmic function of physical time
- **Igamberdiev (1993):** Quantum non-demolition measurements in biosystems

**Key insight:** Our coherence time formula is the quantitative realization of Igamberdiev's phenomenological "biological time" framework. The exponential scaling with M corresponds to logarithmic biological time: t_bio ~ ln(τ_coh) ∝ M.

This completes a 40-year theoretical arc from Vernadsky's intuition → Igamberdiev's formalization → quantitative mechanistic formula.

## Compilation

```bash
cd /Users/iantodd/Desktop/highdimensional/4_coherence_time
pdflatex coherence_time.tex
pdflatex coherence_time.tex  # Second pass for cross-references
```

Output: `coherence_time.pdf` (26 pages)

## TODO

### High Priority
- [ ] Kuramoto network simulations validating coherence time formula
- [ ] Parameter sweeps: M (5-20), r (0.2-0.9), ε (π/4 to 2π)
- [ ] Topology effects: validate α parameter (all-to-all vs sparse)
- [ ] Phase diffusion rate Δω estimation protocols

### Medium Priority
- [ ] Reanalysis of existing neural datasets with coherence time predictions
- [ ] Alpha entrainment experiments → temporal window shift
- [ ] Arousal manipulation → perceptual vs motor timing dissociation
- [ ] Generate figures for key results

### Low Priority
- [ ] Extended comparison to other temporal frameworks
- [ ] Applications to non-neural biological oscillators
- [ ] Connection to criticality/avalanche dynamics

## Related Papers

- **Intelligence paper** (`3_intelligence/`): High-D substrates, measurement-theoretic tracking bound
- **Abiogenesis paper** (`2_abiogenesis_chemistry/`): Code formation, dimensional mismatch
- **Coherence conditions** (`1_coherence_conditions/`): Coherence requirements for biological computation
