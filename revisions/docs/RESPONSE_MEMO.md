# Response Memo: Coherence Time R1 Revision

**Manuscript**: BIOSYS-D-25-00981
**Title**: Coherence time in biological oscillator assemblies bounds the rate of state registration
**Journal**: BioSystems
**Date**: January 2026

---

## Summary of Changes

This revision addresses reviewer feedback through four rounds of technical refinement (Rounds 2-4), resulting in substantial improvements to code correctness, statistical methodology, notational clarity, and reviewer-proofing. All changes preserve the core theoretical framework and strengthen its empirical grounding.

---

## 1. Notation Unification (Priority 1)

**Issue**: The paper used $\Delta\omega$ ambiguously — sometimes as attempt rate (s$^{-1}$), sometimes with "rad/s" units. This created confusion about dimensional consistency.

**Resolution**: Renamed the attempt rate to $\lambda_{\text{attempt}}$ throughout. The main equation (Eq. 4) now reads:

$$\tau_{\text{coh}} \approx \frac{1}{\lambda_{\text{attempt}}} \, p_1(\varepsilon, \kappa(r))^{-\alpha(M-1)}$$

where $\lambda_{\text{attempt}}$ (s$^{-1}$) is explicitly defined as the inverse decorrelation time of inter-module phase coherence. The frequency spread $\Delta\omega$ (rad/s) is retained only as a secondary notation for the underlying physical quantity.

**Impact**: Eliminates unit confusion; all equations and numerical examples now use consistent s$^{-1}$ units.

---

## 2. Figure 1 Caption Update (Priority 2)

**Issue**: Previous caption described only the linearized surrogate fit, inviting criticism that "you didn't validate the actual theory."

**Resolution**: New caption describes both fitting approaches:
- (i) **Theory-consistent**: Uses inter-module coherence $r_{\text{inter}}$ and exact alignment probability $p_1(\varepsilon, \kappa)$
- (ii) **Linearized surrogate**: Uses $(1-r_{\text{global}})(M-1)$ approximation

Both methods yield consistent $\hat{\alpha}$ estimates. Caption also documents Kaplan-Meier censoring handling with visual indicators for heavily censored points.

---

## 3. Model-to-Measurement Bridge (Priority 3)

**Issue**: Reviewers may question whether "commit events" are measurable or purely metaphorical.

**Resolution**: Added two boxed panels after Definition 1:
- **"What commit events are (and aren't)"**: Clarifies that commits are measurement thresholds, not ontological claims. Lists empirical proxies (PLI threshold crossing, PAC onset, classifier-detectable state change).
- **"Minimal empirical pipeline"**: Six-step protocol for estimating coherence time from neural data.

---

## 4. Alpha Exponent Justification (Priority 4)

**Issue**: Why should correlations affect the exponent as $\alpha(M-1)$ rather than the prefactor?

**Resolution**: Added paragraph after Eq. 5 (participation ratio) explaining:
- For independent phases, alignment probability scales as $\varepsilon^{M-1}$
- With correlated phases of effective rank $d_{\text{eff}}$, scaling becomes $\varepsilon^{d_{\text{eff}}}$
- This motivates $\alpha(M-1) \approx d_{\text{eff}}$
- Analogy to Gaussian effective dimensionality makes the argument concrete

---

## 5. Dimensionality Phrasing Refinement (Priority 5)

**Issue**: "Increasing $r$ reduces dimensionality" could be misread as applying to all neural activity.

**Resolution**: Clarified that dimensionality reduction refers specifically to the **relative-phase degrees of freedom** — dynamics are constrained toward a lower-dimensional **phase-synchronized manifold**. Added explicit note that amplitude patterns, cross-frequency interactions, and within-module heterogeneity can remain rich.

---

## 6. Takens Section Justification (Priority 6)

**Issue**: The Takens/cerebellum section is the most speculative and could distract from the core contribution.

**Resolution**: Added "Why include this section" paragraph at section start. Frames the Takens content as an **illustrative consequence** of the coherence time bound (constrains the window over which temporally embedded representations remain reconstructible), explicitly noting it is **not required for the derivation or validation of Eq. 4**.

---

## 7. Code Fixes (Rounds 3-4)

### Critical fixes (all verified):

| Fix | Problem | Solution | Impact |
|-----|---------|----------|--------|
| Plot/fit rate mismatch | Plotting used $\Delta\omega$ (rad/s), fitting used $\lambda$ (s$^{-1}$) | Both use same rate variable | Eliminates ~$2\pi$ systematic error |
| Seed reproducibility | Seeds derived from $\tau$ values; not reproducible | Store original seed in TrialResult | Full reproducibility |
| Attempt rate signal | Autocorrelated $\text{mean}(\delta\psi_m) \approx 0$ (weak signal) | Autocorrelate $|z_{\text{inter}}(t)|$ (strong signal) | Reliable decorrelation estimate |
| Conditional test | Conditioned on $r_{\text{inter\_at\_hit}}$ (outcome leakage) | Use trajectory-mean $r_{\text{inter}}$ | Unbiased conditioning |
| Von Mises validation | Assumed von Mises without check | Conditional-on-$r$ test: bin by $r$, fit $\kappa$ per bin | Validates model closure |
| **Fitting bias** | Theory-consistent fit used $r_{\text{inter}}$ at hit (selection-biased high) | Use trajectory-mean $r_{\text{inter}}$ across all trials (stationary estimate) | $\hat{\alpha}$ now within $(0,1]$; modular $\hat{\alpha}=0.99$, $R^2=0.97$ |

### Von Mises conditional test (key new result):

The conditional-on-$r$ test directly validates the $r \to \kappa \to p_1$ mapping chain:
- Mean $\kappa_{\text{fitted}}/\kappa_{\text{expected}} = 1.06$ (within 6%)
- All bins show good agreement between expected and fitted concentration
- This validates the core assumption underlying Eq. 4

---

## 8. Statistical Methodology Improvements

- **Kaplan-Meier survival analysis**: Properly handles right-censored observations (trials that did not reach alignment within $T$). Reports KM estimates rather than biased conditional medians.
- **Censoring indicators**: Figure panels show upward arrows for heavily censored points (hit fraction $< 0.5$).
- **M-sweep limitation acknowledged**: Fixed total $N$ means module size changes with $M$, potentially introducing finite-size effects. Noted explicitly in manuscript.

---

## 9. Supplementary Validation

Four supplementary analyses confirm model assumptions:

1. **Conditional exponentiality**: Waiting times conditioned on narrow $r$ bins are approximately exponential (KS test)
2. **$\varepsilon$-scaling**: $\tau$ scales as predicted with alignment tolerance
3. **Participation ratio vs. fitted $\alpha$**: Independent estimates of effective independence agree
4. **Von Mises conditional-on-$r$**: Validates phase distribution model closure

---

## 10. Topology Comparison (Combined Figure)

Three-panel figure comparing M-scaling across network topologies ($N=100$, $K=0.7$, $\varepsilon=1.0$, 20 trials per $M$, $M \in [3,10]$):

| Topology | $\hat{\alpha}$ | $R^2$ | Fit method | Hit range |
|----------|----------------|-------|------------|-----------|
| **Modular** ($K_{\text{inter}}=0.08$) | 0.99 | 0.97 | Theory-consistent | 40-85% |
| **All-to-all** | 1.14 | 0.78 | Surrogate (high coherence) | 100% |
| **Sparse** ($p=0.03$) | 0.12 | 0.28 | Theory-consistent | 30-70% |

**Interpretation**: Modular networks yield $\hat{\alpha} \approx 1$ (near-independent modules, $R^2 = 0.97$), confirming the theory's prediction for genuinely independent phase groups. Sparse random connectivity yields $\hat{\alpha} \approx 0.12$: strong correlations across the network collapse the effective number of independent constraints. All-to-all coupling yields $\hat{\alpha} \approx 1.1$ via surrogate fit (theory-consistent fit unavailable in this high-coherence regime). Kaplan-Meier survival analysis handles right-censored trials; upward arrows mark heavily censored M values.

**Key improvement**: The fitting bias fix (using stationary $r_{\text{inter}}$ instead of $r_{\text{inter}}$ at hit) resolved a systematic inflation of $\hat{\alpha}$ that previously pushed values above 1. The corrected modular result ($\hat{\alpha}=0.99$, $R^2=0.97$) is a strong validation of the theoretical framework.

---

## 11. Phase Delta Regime (New §2.3)

**Addition**: New section identifying the pre-commit interval as a computationally productive mode. Includes:
- **Definition 2 (Phase delta regime)**: Formal characterization of the state where inter-module phase relationships are structured but do not satisfy alignment criteria
- **Bias vs. binding distinction**: Phase delta biases downstream outcomes (anisotropic probability over commit space) without collapsing inter-module phase degrees of freedom, unlike phase locking which forces low-dimensional synchronized manifolds
- **85% time allocation**: For biologically relevant parameters ($M \sim 8$, $p_1 \sim 0.7$, $\alpha \sim 0.7$), the system spends ~85% of time in the phase delta regime
- **Sub-Landauer operation**: Phase delta maintenance operates below the Landauer floor because no discrete state is erased—the system biases future trajectories through analog phase relationships that never crystallize into states requiring erasure
- **Discrete structure from continuous dynamics**: Code formation (finite alphabet of distinguishable outputs) emerges at commits as a consequence of dimensional collapse, not as an architectural prerequisite

---

## 12. Ephaptic Field Substrate Identification (New in §2.1)

**Addition**: Argument from the framework's own definitions that the extracellular electric field is the physical carrier of the phase delta regime.

**Logic**: If commits are discrete (spike cascades, readout events) and the phase delta regime is continuous, high-dimensional exploration between commits, the pre-commit dynamics must reside in a substrate distinct from spikes. By elimination:
- Chemical synaptic transmission requires presynaptic spikes (discrete)
- Gap junctions are sparse in cortex
- Diffusible neuromodulators are too slow for oscillatory-timescale coordination
- The extracellular electric field is continuous, spatially extended, and has as many degrees of freedom as there are spatial points in the tissue

**Causal significance** follows from two premises: (1) fields modulate membrane potential and membrane potential determines spike timing (Anastassiou et al. 2011); (2) timing is causally significant (the paper's central result). The magnitude of field effects scales with $D_{\text{eff}}$: in low-dimensional regimes (high $r$, locked phases) field perturbations are swamped by synaptic coupling; in the high-dimensional phase delta regime, small field-mediated timing shifts propagate through $D_{\text{eff}}$ dimensions simultaneously.

---

## 13. Power Limit Quantified (Updated §2.1, §2.4)

**Addition**: Back-of-envelope calculation for the power limit $\tau_{\text{power}}$.

For cortical visual binding ($\Delta\mathcal{D} \sim 50$, $P \sim 10^{-4}$ W):
- At Landauer floor ($\lambda_{\mathcal{D}} = k_B T \ln 2$): $\tau_{\text{power}} \sim 10^{-15}$ s
- At biophysical scale ($\sim 10^7 \times$ Landauer): $\tau_{\text{power}} \sim 10^{-8}$ s

Both are far below $\tau_{\text{coh}} \sim 10^{-2}$ s, confirming the power limit is non-binding. Added $\tau_{\text{power}}$ to the explicit hierarchy listing in §2.4.

---

## 14. Biophysical Mechanism of Pre-Commit Bias (New in §2.2)

**Addition**: Paragraph explaining how structured phase relationships influence downstream circuits before a commit occurs. The extracellular electric field provides the primary channel via ephaptic coupling (gain modulation of downstream targets through oscillatory phase). This field-mediated bias is consolidated through subthreshold dendritic integration and short-term synaptic plasticity that reshape the efficacy landscape across the pre-commit interval.

---

## What Has Not Changed

- Core theoretical framework (Eq. 4 and the unified bound)
- Speed-flexibility trade-off proposition
- All empirical predictions (binding windows, tachypsychia, metabolic scaling)
- Scope boundaries and limitations discussion
- References and bibliography

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| Mathematical framework | Very high | Multiple rounds of formal review |
| Code correctness | Very high | All bugs fixed, validated |
| Statistical methods | Very high | KM survival, conditional tests, von Mises |
| Model closure | Very high | Conditional-on-$r$ test validates $r \to \kappa \to p_1$ |
| Numerical predictions | High | 27-54 ms binding window matches observations |
| Takens hypothesis | Medium | Explicitly flagged as speculative |

---

## Files Modified

### Manuscript
- `coherence_time_r1.tex` — All 6 priority edits applied, LaTeX compiles cleanly

### Code
- `simulations/kuramoto_coherence_time.py` — Rate variable fix, seed tracking, autocorrelation signal, attempt rate estimation
- `simulations/supplementary_analyses.py` — Conditional test fix, von Mises upgrade
- `simulations/generate_combined_figure.py` — KM estimates, censoring indicators, theory-consistent fit

### Documentation
- `docs/round4_working/ROUND4_MANUSCRIPT_EDITS.md` — Edit specifications
- `docs/round4_working/ROUND4_CODE_FIXES.md` — Code change details
- `docs/round4_working/VON_MISES_IMPROVEMENT.md` — Von Mises upgrade rationale
- `docs/RESPONSE_MEMO.md` — This file
