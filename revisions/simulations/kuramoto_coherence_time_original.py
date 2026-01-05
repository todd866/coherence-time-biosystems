#!/usr/bin/env python3
"""
Kuramoto network simulations to validate coherence time formula.

This script simulates coupled Kuramoto oscillators and measures the time
required for M semi-independent modules to achieve phase alignment within
tolerance ε.

Expected relationship:
    τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r)(M-1)]

Where:
    M = number of modules requiring coordination
    r = Kuramoto order parameter (mean resultant length)
    ε = phase alignment tolerance (radians)
    Δω = phase-exploration rate (rad/s)
    α = topology parameter (all-to-all ~ 0.9, sparse ~ 0.6)

Author: Ian Todd
Date: 2025-11-16
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

# TODO: Implement Kuramoto model
# TODO: Module detection (clustering by local coherence)
# TODO: Coherence time measurement (threshold crossing)
# TODO: Parameter sweeps
# TODO: Topology comparison (all-to-all vs sparse)

class KuramotoNetwork:
    """
    Kuramoto oscillator network with modular structure.

    Parameters
    ----------
    N : int
        Total number of oscillators
    M : int
        Number of semi-independent modules
    K : float
        Global coupling strength
    omega_std : float
        Standard deviation of natural frequencies
    topology : str
        Network topology: 'all-to-all', 'modular', 'sparse'
    """

    def __init__(self, N=100, M=10, K=1.0, omega_std=0.1, topology='modular'):
        self.N = N
        self.M = M
        self.K = K
        self.omega_std = omega_std
        self.topology = topology

        # Natural frequencies: Gaussian distribution
        self.omega = np.random.randn(N) * omega_std

        # Initial phases: uniform random
        self.theta0 = np.random.uniform(0, 2*np.pi, N)

        # Module assignments (N/M oscillators per module)
        self.modules = np.repeat(np.arange(M), N // M)[:N]

        # Coupling matrix
        self.A = self._build_coupling_matrix()

    def _build_coupling_matrix(self):
        """Build coupling matrix based on topology."""
        A = np.zeros((self.N, self.N))

        if self.topology == 'all-to-all':
            A = np.ones((self.N, self.N)) - np.eye(self.N)
            A /= (self.N - 1)  # Normalize

        elif self.topology == 'modular':
            # Strong intra-module coupling, weak inter-module
            K_intra = 1.0
            K_inter = 0.1
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        if self.modules[i] == self.modules[j]:
                            A[i, j] = K_intra
                        else:
                            A[i, j] = K_inter
            # Normalize by degree
            A = A / A.sum(axis=1, keepdims=True)

        elif self.topology == 'sparse':
            # Random sparse coupling
            prob = 0.1
            A = (np.random.rand(self.N, self.N) < prob).astype(float)
            A *= (1 - np.eye(self.N))
            A = A / (A.sum(axis=1, keepdims=True) + 1e-10)

        return A

    def dynamics(self, theta, t):
        """Kuramoto dynamics: dθ_i/dt = ω_i + K Σ_j A_ij sin(θ_j - θ_i)."""
        sin_diff = np.sin(theta[:, None] - theta[None, :])
        coupling = self.K * (self.A * sin_diff).sum(axis=1)
        return self.omega + coupling

    def simulate(self, T=100, dt=0.01):
        """
        Simulate Kuramoto network.

        Parameters
        ----------
        T : float
            Total simulation time
        dt : float
            Time step

        Returns
        -------
        t : array
            Time points
        theta : array
            Phase trajectories (shape: [len(t), N])
        """
        t = np.arange(0, T, dt)
        theta = odeint(self.dynamics, self.theta0, t)
        return t, theta

    def compute_order_parameter(self, theta):
        """
        Compute Kuramoto order parameter r(t).

        r = |⟨e^{iθ}⟩| = |(1/N) Σ_j e^{iθ_j}|
        """
        z = np.mean(np.exp(1j * theta), axis=1)
        r = np.abs(z)
        return r

    def compute_module_order_parameters(self, theta):
        """
        Compute order parameter for each module separately.

        Returns
        -------
        r_modules : array
            Order parameters for each module (shape: [len(t), M])
        """
        t_len = theta.shape[0]
        r_modules = np.zeros((t_len, self.M))

        for m in range(self.M):
            mask = (self.modules == m)
            z = np.mean(np.exp(1j * theta[:, mask]), axis=1)
            r_modules[:, m] = np.abs(z)

        return r_modules

    def measure_coherence_time(self, theta, epsilon=np.pi, r_threshold=0.8):
        """
        Measure time for all M modules to achieve coherence within tolerance ε.

        A "commit" occurs when all module order parameters exceed r_threshold
        AND remain aligned within phase tolerance ε.

        Parameters
        ----------
        theta : array
            Phase trajectories (shape: [T, N])
        epsilon : float
            Phase alignment tolerance (radians)
        r_threshold : float
            Minimum order parameter for each module

        Returns
        -------
        coherence_times : list
            Times at which all modules achieve coherence
        """
        r_modules = self.compute_module_order_parameters(theta)

        # Find times when all modules exceed threshold
        all_coherent = np.all(r_modules > r_threshold, axis=1)

        # TODO: Add phase alignment check (modules within ε of each other)
        # For now, just track when all r_m > threshold

        coherence_indices = np.where(all_coherent)[0]

        if len(coherence_indices) == 0:
            return []

        # Find first crossing
        first_coherence = coherence_indices[0]

        return [first_coherence]


def parameter_sweep_M(N=100, M_values=[5, 10, 15, 20], K=1.0, n_trials=10):
    """
    Sweep coordination depth M and measure coherence time.

    Expected: τ_coh grows exponentially with M.
    """
    results = {
        'M': M_values,
        'tau_coh_mean': [],
        'tau_coh_std': [],
        'r_mean': []
    }

    for M in tqdm(M_values, desc="Sweeping M"):
        coherence_times = []
        r_values = []

        for trial in range(n_trials):
            net = KuramotoNetwork(N=N, M=M, K=K, topology='modular')
            t, theta = net.simulate(T=100, dt=0.01)

            # Measure average order parameter (after transient)
            r = net.compute_order_parameter(theta[5000:])  # Skip first 50s
            r_values.append(np.mean(r))

            # Measure coherence time (time to first all-module coherence)
            coh_times = net.measure_coherence_time(theta)
            if len(coh_times) > 0:
                coherence_times.append(coh_times[0] * 0.01)  # Convert to time

        results['tau_coh_mean'].append(np.mean(coherence_times))
        results['tau_coh_std'].append(np.std(coherence_times))
        results['r_mean'].append(np.mean(r_values))

    return results


def plot_coherence_time_vs_M(results, epsilon=np.pi, alpha=0.9, Delta_omega=1.0):
    """
    Plot coherence time vs M and compare to theoretical prediction.

    Theory: τ_coh ≈ (1/Δω)(2π/ε)^[α(1-r)(M-1)]
    """
    M = np.array(results['M'])
    tau_mean = np.array(results['tau_coh_mean'])
    tau_std = np.array(results['tau_coh_std'])
    r_mean = np.mean(results['r_mean'])

    # Theoretical prediction
    tau_theory = (1/Delta_omega) * (2*np.pi/epsilon)**(alpha * (1-r_mean) * (M-1))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulation results
    ax.errorbar(M, tau_mean, yerr=tau_std, fmt='o', label='Simulation', capsize=5)

    # Theoretical curve
    ax.plot(M, tau_theory, 'r--', label='Theory', linewidth=2)

    ax.set_xlabel('Coordination depth M', fontsize=12)
    ax.set_ylabel('Coherence time τ_coh (s)', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Coherence Time vs Coordination Depth (r={r_mean:.2f})')

    plt.tight_layout()
    plt.savefig('../figures/coherence_time_vs_M.png', dpi=300)
    plt.show()

    return fig, ax


if __name__ == '__main__':
    print("Kuramoto coherence time validation")
    print("=" * 60)

    # TODO: Run parameter sweeps
    # TODO: Validate α for different topologies
    # TODO: Test phase diffusion rate Δω

    print("\nTODO: Implement full simulation pipeline")
    print("1. Parameter sweep over M")
    print("2. Coupling strength K → order parameter r relationship")
    print("3. Topology comparison (α parameter validation)")
    print("4. Phase alignment tolerance ε effects")
    print("5. Generate validation figures")
