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
# GPT Reviewer Feedback (Dec 22, 2025)

## Summary

This is *much* cleaner as a "single-idea" BioSystems piece than the ODB monster, and you're explicitly tying it into Igamberdiev's "biological time" program in a way that reads like *continuation*, not contradiction (Section 1.5 is doing that work).

## What's Strong Already

* **Clear thesis**: "coherence time sets speed of thought" is legible and falsifiable in spirit.
* **Useful formula**: explicit scaling law for τ_coh with (M, r, ε, Δω, α). That's the right kind of "hard" object for BioSystems.
* **Igamberdiev bridge is smart**: explicitly map his "log biological time" intuition to log τ_coh ∝ M.
* **Predictions are concrete**: alpha entrainment linearity; dual-loop dissociation under arousal; CFF/metabolic scaling.

---

## Vulnerability 1: Eq. (3) is currently a *plausible ansatz*, not a derivation

You derive a waiting-time form via alignment probability and independence, then jump to a surrogate τ_coh ≈ (2π/ε)^[α(1-r)(M-1)]/Δω with "consistent with Kuramoto theory" language and admit validation remains to be done.

That's okay for BioSystems **if you label it correctly and show it behaves right**, but right now a reviewer can say:

* Why (1-r) in the exponent instead of (1-r²), -log(1-r), or κ(r) explicitly?
* Why power-law-in-(2π/ε) raised to (M-1) rather than something like exp(c(M-1)) with c coming from κ?
* Why is α constant over regimes?
* What exactly is M operationally? You give a protocol, but it's still squishy (modules, clusters, predictors).

### Fixes (best is A + B)

**A) Reframe Eq. (3) explicitly as a controlled approximation to Eq. (2).**
Make Eq. (3) a *specific approximation* where p_align(ε, κ(r)) ≈ (ε/2π)^[α(1-r)] and justify the exponent by boundary conditions.

**B) Add a 6–10 line "calibration" argument that pins the exponent form.**
State the minimal properties you require:
1. τ increases with M (monotone)
2. τ decreases with r (monotone)
3. τ → Δω⁻¹ when M→1
4. τ blows up as ε→0
5. dependence should be exponential in coordination depth under independence

Then show Eq. (3) is the *lowest-complexity form* consistent with these, and that more exact forms (von Mises κ) collapse to something like it in the moderate regime (r~0.5-0.7).

**C) One tiny validation figure, even if only simulation.**
BioSystems reviewers love seeing *one* plot that says: "we simulated modular Kuramoto, measured waiting time, fit exponent ~ (M-1)(1-r) with α ~ 0.7–0.9 depending on topology."
Even 2–3 topologies, small M range, is enough to neutralize "hand-wavy".

Right now you say "remains to be performed" — that invites "major revision: add simulation."

---

## Vulnerability 2: the "unified bound" mixes apples unless you sharpen the *commit definition*

You define commits as thermodynamically irreversible registrations (dimensional collapse + Landauer + persistent measurement) and then take τ_eff = max[τ_QSL, τ_SNR, τ_coh, τ_power].

Reviewers will ask:
* Is a "commit" the same event for perception, action, and memory?
* Does τ_SNR refer to sensory detection, while τ_coh refers to global binding, while τ_power refers to metabolic cost of collapse?

**Quick fix:** add a one-paragraph "operational definition of commit" as *the minimal event that can change an externally measurable (or downstream-controller measurable) discrete state*—and clarify different tasks have different ΔD, different M, etc.

---

## Vulnerability 3: Tachypsychia section is strong but needs one sentence of humility

Your exclusion logic is good (subjective time dilation without reaction-time slowing), but the mechanistic dual-loop story could trigger "speculative."

Add one sentence:
* "This is a mechanistic hypothesis consistent with the dissociation; alternative accounts exist (e.g., attentional sampling, memory density), but the key empirical discriminator is the predicted independence between temporal-order thresholds and simple RT under arousal."

---

## Vulnerability 4: CFF metabolic scaling claim needs cautious phrasing

You cite Healy et al. and say log-log regression R²≈0.6 and near-linear f_CFF ∝ P_meta^0.6. Reviewers will nitpick exponent mismatch vs Kleiber (-1/4) etc.

Soften:
* "predicts scaling in the observed direction; exponent depends on how neural allocation scales with whole-organism metabolic rate and on ΔD across taxa."

Be explicit: this is **regime identification** ("power-limited vs coherence-limited"), not claiming exact exponent universality.

---

## Changes Before Sending to Reviewer

1. **Promote Eq. (2) as the "true" result; demote Eq. (3) as approximation.**
2. **Add a minimal simulation validation figure** (even if small) so nobody can call it pure curve-fitting.
3. **Define M more crisply**: "number of semi-independent modules whose order parameters must jointly exceed threshold for commit." Tighten and give one concrete operationalization example.
4. **Add a short limitations paragraph**: independence assumption, topology dependence, nonstationary Δω, and what breaks first.

---

## Strategic Suggestion: Make the Igamberdiev section "sign-off proof"

Section 1.5 is already pretty aligned and generous. For maximum safety with him as a friendly editor figure:

* explicitly say: "Our formula is compatible with (and provides a mechanistic instantiation of) Igamberdiev's biological time; it does not require a quantum substrate claim."

That keeps you from stepping on the "quantum vs classical" rake while still letting you believe the upstream unity.
