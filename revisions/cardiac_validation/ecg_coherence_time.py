#!/usr/bin/env python3
"""
Coherence time analysis of multi-lead ECG data.

Tests the prediction: τ_coh scales exponentially with M (number of leads).

Uses PTB-XL dataset (12-lead ECG, 500 Hz, 10 seconds each).
"""

import numpy as np
from scipy import signal
from pathlib import Path
import wfdb
from typing import List, Tuple
import json


def bandpass_filter(data: np.ndarray, fs: float, low: float = 1.0, high: float = 40.0) -> np.ndarray:
    """Bandpass filter ECG signal."""
    nyq = fs / 2
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=0)


def compute_instantaneous_phase(data: np.ndarray) -> np.ndarray:
    """Compute instantaneous phase via Hilbert transform."""
    analytic = signal.hilbert(data, axis=0)
    return np.angle(analytic)


def kuramoto_order_parameter(phases: np.ndarray) -> np.ndarray:
    """
    Compute Kuramoto order parameter r(t) across leads.

    phases: (n_samples, n_leads)
    returns: r(t) array of shape (n_samples,)
    """
    z = np.exp(1j * phases)
    r = np.abs(np.mean(z, axis=1))
    return r


def measure_first_passage_times(r: np.ndarray, threshold: float, fs: float) -> List[float]:
    """
    Measure first-passage times to coherence events.

    A coherence event is when r crosses above threshold.
    Returns list of waiting times (in seconds) between events.
    """
    above = r > threshold

    # Find rising edges (transitions from below to above threshold)
    edges = np.diff(above.astype(int))
    event_indices = np.where(edges == 1)[0]

    if len(event_indices) < 2:
        return []

    # Compute inter-event intervals
    intervals = np.diff(event_indices) / fs
    return intervals.tolist()


def analyze_record(record_path: str, M_values: List[int] = [3, 6, 9, 12],
                   threshold: float = 0.7) -> dict:
    """
    Analyze a single ECG record for coherence time scaling.

    Args:
        record_path: Path to WFDB record (without extension)
        M_values: List of M (number of leads) to test
        threshold: Coherence threshold for event detection

    Returns:
        Dictionary with results for each M
    """
    # Load record
    record = wfdb.rdrecord(record_path)
    data = record.p_signal  # (n_samples, n_leads)
    fs = record.fs
    lead_names = record.sig_name

    # Preprocess: bandpass filter
    data_filt = bandpass_filter(data, fs)

    # Compute instantaneous phase for all leads
    phases = compute_instantaneous_phase(data_filt)

    results = {
        'record': str(record_path),
        'fs': fs,
        'n_samples': data.shape[0],
        'n_leads': data.shape[1],
        'lead_names': lead_names,
        'threshold': threshold,
        'M_results': {}
    }

    for M in M_values:
        if M > data.shape[1]:
            continue

        # Select first M leads (could also try random subsets)
        phases_M = phases[:, :M]

        # Compute Kuramoto order parameter
        r = kuramoto_order_parameter(phases_M)

        # Measure first-passage times
        fpts = measure_first_passage_times(r, threshold, fs)

        results['M_results'][M] = {
            'r_mean': float(np.mean(r)),
            'r_std': float(np.std(r)),
            'n_events': len(fpts) + 1 if fpts else 0,
            'fpt_list': fpts,
            'fpt_mean': float(np.mean(fpts)) if fpts else None,
            'fpt_median': float(np.median(fpts)) if fpts else None,
            'fpt_std': float(np.std(fpts)) if fpts else None
        }

    return results


def run_validation(data_dir: str, n_records: int = 10, threshold: float = 0.7):
    """
    Run coherence time validation across multiple records.
    """
    data_path = Path(data_dir)

    # Find all .hea files (record headers)
    hea_files = sorted(data_path.rglob("*.hea"))[:n_records]

    print(f"Found {len(hea_files)} records")

    all_results = []
    M_values = [3, 6, 9, 12]

    for hea_file in hea_files:
        record_path = str(hea_file.with_suffix(''))
        print(f"Processing: {hea_file.name}")

        try:
            result = analyze_record(record_path, M_values, threshold)
            all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Aggregate results by M
    aggregate = {M: {'fpts': [], 'r_means': []} for M in M_values}

    for result in all_results:
        for M, m_result in result['M_results'].items():
            M = int(M)
            if m_result['fpt_list']:
                aggregate[M]['fpts'].extend(m_result['fpt_list'])
            aggregate[M]['r_means'].append(m_result['r_mean'])

    print("\n" + "="*60)
    print("COHERENCE TIME SCALING RESULTS")
    print("="*60)
    print(f"{'M':<6} {'n_FPT':<8} {'τ_median':<12} {'τ_mean':<12} {'r_mean':<10}")
    print("-"*60)

    summary = {}
    for M in M_values:
        fpts = aggregate[M]['fpts']
        r_means = aggregate[M]['r_means']

        if fpts:
            tau_med = np.median(fpts)
            tau_mean = np.mean(fpts)
        else:
            tau_med = tau_mean = None

        r_mean = np.mean(r_means) if r_means else None

        summary[M] = {
            'n_fpts': len(fpts),
            'tau_median': tau_med,
            'tau_mean': tau_mean,
            'r_mean': r_mean
        }

        tau_med_str = f"{tau_med:.4f}" if tau_med else "N/A"
        tau_mean_str = f"{tau_mean:.4f}" if tau_mean else "N/A"
        print(f"{M:<6} {len(fpts):<8} {tau_med_str:<12} {tau_mean_str:<12} {r_mean:.3f}")

    # Test for exponential scaling: log(τ) should be linear in M
    M_arr = np.array([M for M in M_values if summary[M]['tau_median'] is not None])
    tau_arr = np.array([summary[M]['tau_median'] for M in M_arr])

    if len(M_arr) >= 2 and np.all(tau_arr > 0):
        log_tau = np.log(tau_arr)
        slope, intercept = np.polyfit(M_arr, log_tau, 1)
        r_squared = 1 - np.sum((log_tau - (slope*M_arr + intercept))**2) / np.sum((log_tau - np.mean(log_tau))**2)

        print("\n" + "-"*60)
        print(f"Exponential fit: τ ~ exp({slope:.3f} * M)")
        print(f"R² = {r_squared:.3f}")
        print("-"*60)

        summary['fit'] = {'slope': slope, 'intercept': intercept, 'r2': r_squared}

    return summary, all_results


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "ptbxl_subset"
    n_records = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    summary, results = run_validation(data_dir, n_records)

    # Save results
    output = {
        'summary': summary,
        'individual_records': results
    }

    with open('cardiac_coherence_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to cardiac_coherence_results.json")
