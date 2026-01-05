# Coherence Time R1 Revision Plan

**Date:** Jan 2026
**Feedback source:** Detailed reviewer-style analysis (Jan 5)
**Key asset:** Companion IG paper now complete (alignment_geometry.tex, 15 pages)

---

## Strategic Overview

The IG paper we just wrote solves the core mathematical weakness. The revision strategy:

1. **Reference the IG paper** for rigorous derivation of (M-1) exponent
2. **Fix Eq (2) closure** to match boundary conditions
3. **Align ε convention** throughout
4. **Add scaling collapse validation**
5. **Strengthen cognitive applications** with sensitivity analysis

---

## Major Issues and Fixes

### Major Issue A: The p_align closure is the weakest link

**Problem:** Eq (2) gives p_align → (ε/2π)^α when r→0, but should give p_align → ε/2π.
The boundary conditions don't match unless α=1.

**Solution (from IG paper):** Keep p_align explicit. The rigorous result is:

```
P(alignment) = p₁(ε, κ)^(M-1)
```

where p₁ is the single-variable window probability (computed explicitly via von Mises integral).
Move α to the exponent as "effective independence":

```
τ_coh ≈ (1/Δω) · p₁(ε, κ(r))^{-α(M-1)}
```

where:
- α = 1 for independent modules
- α < 1 for coupled modules (captures correlation)

**Specific changes:**
1. Replace Eq (2) with explicit p_align from von Mises
2. Reframe Eq (3) as: α multiplies (M-1), not the whole exponent
3. Add remark: "The rigorous derivation appears in [companion IG paper]"
4. The (1-r) dependence comes through κ(r), not as a separate factor

**New formulation:**
```latex
p_{\rm align}(\varepsilon, \kappa) = \int_{-\varepsilon/2}^{\varepsilon/2} \frac{e^{\kappa\cos\theta}}{2\pi I_0(\kappa)} d\theta
```

For small ε: p₁ ≈ (ε/2π) · (e^κ / I_0(κ))

Approximate closure (preserving limits):
```latex
p_{\rm align}(\varepsilon, r) \approx \frac{\varepsilon}{2\pi} \cdot g(r)
```
where g(0)=1, g(1)→∞ (concentrated limit).

The simplest form with correct limits:
```latex
p_{\rm align} \approx \frac{\varepsilon}{2\pi} \cdot (1 + \beta r / (1-r))
```
for some β > 0. Or use the explicit von Mises integral.

---

### Major Issue B: ε convention mismatch

**Problem:**
- Theory: ε is window width (probability ≈ ε/2π)
- Simulation: ε is half-width (max deviation ≤ ε)

This factor of 2 propagates into fitted α.

**Solution:** Align to window width everywhere.

**Changes:**
1. In paper: clarify "ε denotes the full allowable phase window width"
2. In simulation code: change `phase_err <= epsilon` to `phase_err <= epsilon/2`
3. Or: keep simulation as-is, but use (2π/2ε) in theory formulas

**Recommended:** Keep ε as full width in theory, fix simulation to use half-width check.

---

### Major Issue C: Validation too narrow

**Problem:** One modular regime with α̂ ≈ 0.35, R² = 0.71 is not enough to justify using α = 0.35 in cognitive applications.

**Solution:** Parameter sweep + scaling collapse figure.

**New validation protocol:**
1. Vary K_inter/K_intra ratio to produce multiple r values
2. For each (r, M) combination, measure τ_coh
3. Plot: ln(τ_coh · Δω) / ln(2π/ε) vs (M-1)
4. If theory is correct, all points collapse onto lines with slope ≈ α·(1-r)
5. Or: plot vs α(1-r)(M-1) for common α, check collapse

**Figure to add:** "Scaling collapse" showing data from multiple r conditions collapsing onto predicted line.

**Table to add:** α̂ values across different r regimes, showing stability.

---

### Major Issue D: "M scaling is universal" sounds tautological

**Problem:** "More constraints → longer wait" is obvious. The headline shouldn't be this.

**Solution:** Reframe what's nontrivial.

**What IS nontrivial:**
1. The specific geometric base ln(2π/ε) - not arbitrary
2. The coupling dependence via r (or κ) - tunable
3. The speed-flexibility frontier with interpretable parameters
4. The (M-1) exponent from quotient geometry (not M)

**Text change in abstract/intro:**
```
"We show that coherence time follows a precise geometric scaling law
with coordination depth M, where the exponent arises from quotient
geometry on the product torus T^M. The (M-1) form—not M—reflects
rotational invariance: one phase serves as reference."
```

---

### Major Issue E: Cognitive applications need grounding

#### §3.1 Visual binding (sensitivity analysis)

**Problem:** One parameter combination (M=10, r=0.6, ε=π, Δω=2π×10) gives 30-50 ms. Looks cherry-picked.

**Solution:** Add sensitivity table.

```
| M    | r    | ε     | Δω (rad/s)  | τ_coh (ms) |
|------|------|-------|-------------|------------|
| 6    | 0.5  | π/2   | 2π×5        | 180        |
| 8    | 0.6  | π     | 2π×10       | 45         |
| 10   | 0.7  | π     | 2π×15       | 25         |
| 12   | 0.8  | 3π/2  | 2π×20       | 15         |
```

Show: 30-50 ms binding windows occupy robust region of parameter space.

#### §3.2 Tachypsychia (label as hypothesis)

**Add sentence:** "This is a mechanistic hypothesis consistent with the dissociation; alternative accounts exist (attentional sampling, memory density). The key empirical discriminator is predicted independence between temporal-order thresholds and simple RT under arousal."

#### §3.4 Metabolic scaling (add derivation)

**Problem:** Claim f_CFF ∝ P_meta^0.6 without showing derivation.

**Add paragraph:**
```
Derivation: If neural power allocation scales as P_neural ∝ P_meta^β
(with β ≈ 1 for brain-to-body scaling), and ΔD varies weakly across taxa,
then τ_power = λ_D·ΔD/P_neural ∝ P_meta^{-β}, giving f_CFF ∝ P_meta^β.
The observed exponent ≈ 0.6 suggests β < 1, consistent with brain power
not scaling isometrically with whole-organism metabolic rate.
```

---

## Writing/Structure Improvements

### A) Move §1.5 (Igamberdiev context) later

Currently delays main physics. Options:
- Keep 1 paragraph in intro, move extended discussion to Discussion
- Or: keep if BioSystems expects this (Igamberdiev-friendly)

### B) Add schematic figure

**New Figure 0:** Show:
- M modules, each with internal coherence
- Mean phases wandering/aligning
- "Commit event" when all within ε
- Dwell time requirement

This makes M, ε, r, Δω concrete.

### C) Distinguish module-level vs oscillator-level coherence

Add notation:
- r_intra or r_m: coherence within module
- r_inter: coherence among module mean phases
- r_global: coherence across all oscillators

### D) Improve Figure 2 readability

Make full-width with larger fonts, or split into separate panels.

---

## Code/Validation Changes

### A) Fix ε convention in code

```python
# Current (half-width):
return bool(np.max(phase_err) <= self.p.epsilon)

# Fixed (full-width):
return bool(np.max(phase_err) <= self.p.epsilon / 2)
```

### B) Add r-sweep and ε-sweep validation

Beyond M-sweep, add:
- τ vs ε at fixed (M, r)
- τ vs r at fixed (M, ε)

These test the specific functional form.

### C) Clarify Δω measurement

Add text: "Δω is measured as time-averaged standard deviation of instantaneous phase velocity after discarding transients. This is an operational definition of exploration/mixing rate, not identical to carrier frequency."

### D) Consider moving all-to-all results to supplement

All-to-all doesn't have intrinsic "modules" - comparing to modular is apples-to-oranges unless carefully explained.

---

## Minimal Revision Plan (Highest Impact, Lowest Effort)

1. **Fix ε convention** - clarify in text, align code
2. **Replace Eq (2)** with explicit p_align that preserves r→0 limit
3. **Add scaling collapse figure** showing (1-r)(M-1) dependence
4. **Add sensitivity table** for §3.1 binding window calculation
5. **Reference IG paper** for rigorous (M-1) derivation

These four changes address the most critical reviewer vulnerabilities.

---

## Files to Modify

```
coherence_time.tex:
- §2.1: Replace Eq (2), add explicit p_align
- §2.1: Clarify ε as full width
- §2.2: Add scaling collapse figure reference
- §3.1: Add sensitivity table
- §3.2: Add "hypothesis" labeling
- §3.4: Add derivation paragraph
- Discussion: Add limitations paragraph

revisions/kuramoto_coherence_time_v2.py:
- Fix epsilon half-width check
- Add r-sweep capability
- Add ε-sweep capability

New files:
- figures/scaling_collapse.pdf
- figures/sensitivity_table.pdf (or inline table)
```

---

## Timeline

1. Fix Eq (2) and ε convention (1 hour)
2. Run parameter sweeps for scaling collapse (2 hours compute + analysis)
3. Add sensitivity table (30 min)
4. Polish text, add figure legends (1 hour)
5. Final compile and review (30 min)

**Total:** ~5 hours focused work
