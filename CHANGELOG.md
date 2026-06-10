# Changelog

The version of record is the published paper (BioSystems 263, 105755; DOI in the
README). This repository tracks ongoing development and corrections.

## 2026-06-11 — corrections

- **Table 1 (sensitivity analysis), "Narrower ε = π/3" row.** The entry reads 105 ms.
  Recomputing with the exact `p1` integral the paper states it uses for all numerical
  estimates (κ(0.75) = 2.37, `p1(π/3, κ)` = 0.540, α = 0.7, M = 8, λ_attempt = 126 s⁻¹)
  gives ~163 ms; the small-ε approximation gives ~99 ms. Every other cell in the table
  reproduces. The "20–80 ms binding window" summary already excludes this row, so the
  reported headline is unaffected.
