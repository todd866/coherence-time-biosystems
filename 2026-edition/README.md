# Coherence time: 2026 author edition

This edition rewrites *Coherence time in biological oscillator assemblies bounds the rate of state registration*, published in **BioSystems 263, 105755 (2026)**, [doi:10.1016/j.biosystems.2026.105755](https://doi.org/10.1016/j.biosystems.2026.105755). The published article remains the version of record and the one to cite. Its files are preserved alongside this folder.

Coherence time sets the pace at which an oscillator assembly registers a new state. This edition leads with the resulting speed–flexibility trade-off: independently variable phase relationships expand the biological repertoire while increasing the coordination needed for a joint readout.

The argument is biology first, with a sharper structure and clearer mathematics. Neural coordination motivates the problem; target geometry, registration opportunities and persistence explain its timing. Scope conditions sit beside the results they govern, leaving the discussion free to develop the biological consequences.

The numerical account brings the network architectures together and makes the neural timescale tangible: a 39 ms characteristic wait rises to 163 ms (162.883 ms) with a narrower phase window, while joint parameter variation spans 9.768–282.686 ms. The mathematics distinguishes mean waits from hard minima and covariance participation from target-probability scaling, and explicitly includes the dependence of dwell survival on module count.

The final sections develop the paper's wider ambition: structured phase dynamics between registrations help determine the next outcome, and the richness of those dynamics offers a route to understanding temporal experience. Independent survival predictions and phase perturbations connect these ideas to experiments.

This edition was rewritten with AI assistance (Claude, Codex) for drafting, mathematical checking and critical review; the author takes responsibility for the content.

Build from this folder with:

```sh
latexmk -pdf -interaction=nonstopmode coherence_time_2026.tex
latexmk -c
```
