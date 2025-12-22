# Revision Materials

This folder contains post-submission validation and extensions of the coherence-time framework. The submitted version (`coherence_time_SUBMITTED.tex`) should be considered the authoritative statement of the theory.

## Status

Submitted to BioSystems (2025-11-15). Awaiting reviews (expected ~Feb 2026).

## Post-Submission Development

These materials are being prepared in anticipation of reviewer requests and for future extensions. **Note:** These post-submission materials are not part of the peer-reviewed record unless explicitly incorporated into a formal revision.

### Simulation Validation
- `simulations/kuramoto_coherence_time_v2.py` - Paper-grade validation suite
  - Phase alignment check across modules (not just r_threshold)
  - Dwell time for robust τ_coh measurement
  - Euler-Maruyama integrator (supports noise)
  - Sweep M and fit α from simulated data
  - CLI, reproducible seeds, JSON outputs

### Key Validation Target
```
τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r̄)(M-1)]
```

Primary goal: demonstrate log τ_coh ∝ (M-1) scaling in simulation.

## Revision Checklist

- [ ] Receive reviews
- [ ] Run τ_coh vs M validation sweep
- [ ] Generate single clean validation figure
- [ ] Draft point-by-point response letter
- [ ] Address each reviewer comment
- [ ] Update manuscript with tracked changes
- [ ] Verify all citations/references
- [ ] Recompile and proofread
- [ ] Submit revision

## Files

```
revisions/
├── README.md                    # This file
├── coherence_time_r1.tex        # Revised manuscript (when reviews arrive)
└── response_letter.tex          # Point-by-point response (when reviews arrive)
```

## Running Validation

```bash
cd simulations
python kuramoto_coherence_time_v2.py \
  --topology modular --N 200 --K 1.2 --omega-std 0.5 --sigma 0.2 \
  --epsilon 1.57 --r-threshold 0.75 --dwell 0.05 \
  --M-min 4 --M-max 20 --M-step 2 --trials 20 \
  --outdir ../figures --save-json
```
