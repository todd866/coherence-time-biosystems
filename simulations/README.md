# Simulations

Kuramoto oscillator simulations validating the coherence-time scaling law.

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

```bash
python kuramoto_coherence_time.py --topology modular --M_values 2 3 4 5 6
```

This generates figures showing τ_coh vs M for different network topologies.

## Output

Figures are saved to `../figures/` by default.
