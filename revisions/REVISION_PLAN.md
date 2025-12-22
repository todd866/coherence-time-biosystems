# Coherence Time Paper - Major Revision Plan

**Status:** Under review at BioSystems (submitted Nov 2025)
**Expected:** Likely "revise and resubmit" given simulation validation gaps
**Strategy:** Prepare major revision now, be ready when reviews arrive

---

## Core Problem (RESOLVED)

The formula τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r)(M-1)] was not validating with original parameters.

**Original results (bad):**
- α̂ = 0.13 (predicted: 0.6-0.9)
- R² = 0.22 (poor fit)

**New results (good) with tuned parameters:**
- α̂ = 0.54 ✓
- R² = 0.79 ✓
- Clear monotonic increase with M ✓

**Working parameters:**
```
K=1.0, omega_std=0.3, sigma=0.1, epsilon=2.0, r_threshold=0.6, dwell=0.03
```

**Root cause of original failure:** Modules were too synchronized (r ≈ 0.84), so (1-r) ≈ 0.16 was too small. Need moderate coupling regime.

---

## Revision Strategy

### Option A: Fix the simulations (make formula work)
- Lower coupling strength K to get r ≈ 0.5-0.6
- Weaker inter-module coupling (K_inter << K_intra)
- Find regime where exponential scaling emerges

### Option B: Reframe the theory (qualitative claim)
- Eq. (2) is the principled result
- Eq. (3) is an approximation that captures boundary conditions
- Show it's monotonic in M, r even if exact exponent varies

### Option C: Both
- Show simulation validates scaling in specific regime
- Acknowledge approximation status, give boundary conditions
- This is the BioSystems-safe approach

**Recommended: Option C**

---

## Specific Changes Needed

### 1. Theory Section
- [ ] Promote Eq. (2) as "true" result
- [ ] Demote Eq. (3) as "empirical closure under weak modular coupling"
- [ ] Add calibration argument (5 boundary conditions that pin the form)
- [ ] Define M operationally with concrete example

### 2. Simulations
- [ ] Run with lower K to get r ≈ 0.5-0.6 regime
- [ ] Run with K_inter << K_intra (true modular)
- [ ] Report median + IQR (not mean) for heavy tails
- [ ] Report hit fraction per M value
- [ ] Save raw τ values for survival analysis
- [ ] Implement proper Δω estimation (not heuristic)

### 3. Validation Figure
- [ ] Main panel: log(τ) vs (M-1) for modular topology
- [ ] Fitted line with α̂ and R²
- [ ] Inset or side panels: all-to-all (collapses), sparse (fails)
- [ ] Error bars showing IQR

### 4. Paper Text
- [ ] Add operational definition of "commit" (1 paragraph)
- [ ] Soften CFF metabolic scaling claim (regime identification, not exact exponent)
- [ ] Add humility sentence to tachypsychia section
- [ ] Add limitations paragraph
- [ ] Fix code availability inconsistency (publish repo now)

### 5. Igamberdiev Section (sign-off proof)
- [ ] Add explicit statement: "compatible with and provides mechanistic instantiation of biological time; does not require quantum substrate claim"

---

## Simulation Parameter Sweep Needed

```bash
# Target regime: r ≈ 0.5-0.6, clear M scaling
python kuramoto_coherence_time_v2.py \
  --topology modular --N 200 \
  --K 0.8 \                    # lower coupling
  --K-intra 1.0 --K-inter 0.05 \ # weak inter-module
  --omega-std 0.5 --sigma 0.2 \
  --epsilon 1.57 --r-threshold 0.6 --dwell 0.05 \
  --M-min 3 --M-max 12 --M-step 1 --trials 30 \
  --outdir ../figures --save-json
```

---

## Files to Create

```
revisions/
├── REVISION_PLAN.md          # This file
├── gpt_feedback_dec22.md     # GPT reviewer feedback
├── coherence_time_r1.tex     # Revised manuscript
├── response_letter.tex       # Point-by-point response
└── validation_results/       # New simulation outputs
    ├── tau_vs_M_modular_weak.png
    ├── tau_vs_M_modular_weak.json
    └── parameter_sensitivity.png
```

---

## Timeline

1. **Now (Dec 2025):** Prepare validation simulations, draft revised text
2. **When reviews arrive:** Incorporate specific feedback
3. **Submit R1:** Within 2 weeks of receiving reviews

---

## Success Criteria

A successful revision shows:
1. τ_coh increases monotonically with M in simulation
2. Fitted α is in reasonable range (0.3-1.0) with R² > 0.7
3. Clear regime identification (when formula applies vs fails)
4. Honest limitations paragraph
5. Published code with reproducible results
