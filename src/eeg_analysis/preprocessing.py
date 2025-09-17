from loaddata import load_edf, load_hypno
from plot_eeg import (plot_artifact_summary, plot_epochs, 
                      plot_stage_distribution, plot_psd_by_stage)

import numpy as np
import mne
from scipy import stats
from scipy.signal import welch
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter
import pandas as pd


# TODO: convert into proper config file later
channels = ['EEG F7-O1', 'EEG F8-O2', 'EEG F8-F7', 'EEG F8-O1', 'EEG F7-O2']
sfreq = 100
epoch_len = 30
# frequency bands - delta, theta, alpha (might add beta later)
bands = [(0.5, 4), (4, 8), (8, 12)]  

# filter settings
highpass = 0.5
lowpass = 32
notch = 50  # EU power line freq


def apply_filters(raw) -> mne.io.Raw:
    """Basic EEG filtering - nothing fancy"""
    print("Filtering...")
    raw_filt = raw.copy()
    
    # high pass first
    raw_filt.filter(l_freq=highpass, h_freq=None, verbose=False)
    # then low pass
    raw_filt.filter(l_freq=None, h_freq=lowpass, verbose=False)  
    # notch filter for line noise
    raw_filt.notch_filter(freqs=notch, verbose=False)
    
    return raw_filt


def make_epochs(raw, hypno=None) -> mne.Epochs:
    """Create 30s epochs from raw data"""
    raw_filtered = apply_filters(raw)
    
    events = mne.make_fixed_length_events(raw_filtered, id=1, duration=epoch_len)
    picks = mne.pick_channels(raw_filtered.info["ch_names"], include=channels)
    
    epochs = mne.Epochs(raw_filtered, events, tmin=0, tmax=epoch_len, 
                       baseline=None, picks=picks, preload=True, verbose=False)
    
    return epochs


def detect_bad_epochs(epochs) -> tuple[mne.Epochs, mne.Epochs, list[int]]:
    """Find artifacts - bit of a mess but works"""
    print("Looking for artifacts...")
    
    data = epochs.get_data()
    n_epochs, n_channels, n_samples = data.shape
    bad_epochs = np.zeros(n_epochs, dtype=bool)
    
    # convert to microvolts for thresholding
    data_uv = data * 1e6
    
    # 1. Amplitude artifacts (>150 uV)
    max_amps = np.max(np.abs(data_uv), axis=(1,2))
    amp_bad = max_amps > 150
    bad_epochs |= amp_bad
    print(f"Amplitude artifacts: {np.sum(amp_bad)}")
    
    # 2. Gradient artifacts (steep changes)
    gradients = np.diff(data, axis=2)
    max_grads = np.max(np.abs(gradients), axis=(1,2))
    grad_bad = max_grads > 50e-6
    bad_epochs |= grad_bad
    print(f"Gradient artifacts: {np.sum(grad_bad)}")
    
    # 3. Flat line detection (variance too low)
    variances = np.var(data, axis=2)
    min_vars = np.min(variances, axis=1)
    flat_bad = min_vars < 1e-15
    bad_epochs |= flat_bad
    print(f"Flat line artifacts: {np.sum(flat_bad)}")
    
    # 4. High frequency noise (rough implementation)
    hf_bad = np.zeros(n_epochs, dtype=bool)
    for i in range(n_epochs):
        hf_ratios = []
        for ch in range(n_channels):
            freqs, psd = welch(data[i, ch, :], fs=sfreq, nperseg=min(256, n_samples))
            
            # ratio of high freq (20-35 Hz) to total power
            hf_mask = (freqs >= 20) & (freqs <= 35)
            total_mask = (freqs >= 0.5) & (freqs <= 35)
            
            if np.sum(total_mask) > 0 and np.sum(hf_mask) > 0:
                hf_power = np.sum(psd[hf_mask])
                total_power = np.sum(psd[total_mask])
                if total_power > 0:
                    hf_ratios.append(hf_power / total_power)
        
        if hf_ratios and np.mean(hf_ratios) > 0.3:  # threshold found by trial/error
            hf_bad[i] = True
    
    bad_epochs |= hf_bad
    print(f"High freq artifacts: {np.sum(hf_bad)}")
    
    # 5. Statistical outliers (z-score approach)
    # compute some stats for each epoch
    epoch_means = np.mean(data, axis=(1,2))
    epoch_stds = np.std(data, axis=(1,2))
    
    # flag extreme outliers
    mean_z = np.abs(stats.zscore(epoch_means))
    std_z = np.abs(stats.zscore(epoch_stds))
    
    outlier_bad = (mean_z > 4) | (std_z > 4)
    bad_epochs |= outlier_bad
    print(f"Statistical outliers: {np.sum(outlier_bad)}")
    
    # return clean epochs and rejected ones
    clean_epochs = epochs[~bad_epochs]
    rejected_epochs = epochs[bad_epochs] if np.sum(bad_epochs) > 0 else None
    rejected_indices = np.where(bad_epochs)[0].tolist()
    
    print(f"\nTotal rejected: {np.sum(bad_epochs)}/{n_epochs} ({np.sum(bad_epochs)/n_epochs*100:.1f}%)")
    print(f"Clean epochs: {len(clean_epochs)}")
    
    return clean_epochs, rejected_epochs, rejected_indices


def align_with_hypno(epochs, hypno_df) -> list[str]:
    """Match epochs to sleep stages - assumes 30s resolution for both"""
    if hypno_df is None:
        return ['UNKNOWN'] * len(epochs)
    
    epoch_stages = []
    for i in range(len(epochs)):
        # each epoch is 30s, hypno is also 30s
        hypno_idx = i
        
        if hypno_idx < len(hypno_df):
            stage = hypno_df.iloc[hypno_idx]['Event']
            epoch_stages.append(stage)
        else:
            epoch_stages.append('UNKNOWN')
    
    return epoch_stages


def compute_bandpower(epochs) -> dict[str, np.ndarray]:
    """Calculate power in different frequency bands"""
    print("Computing bandpower...")
    
    bp = {}
    for fmin, fmax in bands:
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            epochs.get_data(), sfreq=sfreq,
            fmin=fmin, fmax=fmax, verbose=False
        )
        
        band_name = f"{fmin}-{fmax}Hz"
        bp[band_name] = psds.mean(axis=2) if psds.ndim == 3 else psds
    
    return bp


def compute_psd_by_stage(epochs, stages) -> tuple[dict[str, dict], np.ndarray]:
    """PSD analysis per sleep stage"""
    unique_stages = list(set([s for s in stages if s != 'UNKNOWN']))
    print(f"Computing PSD for stages: {unique_stages}")
    
    psd_results = {}
    
    for stage in unique_stages:
        # get epochs for this stage
        stage_idx = [i for i, s in enumerate(stages) if s == stage]
        
        if len(stage_idx) == 0:
            continue
        
        print(f"  {stage}: {len(stage_idx)} epochs")
        
        stage_epochs = epochs[stage_idx]
        stage_data = stage_epochs.get_data()
        
        # compute PSD
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            stage_data, sfreq=sfreq, fmin=0.5, fmax=35, verbose=False
        )
        
        # average across epochs
        mean_psd = np.mean(psds, axis=0)
        
        psd_results[stage] = {
            'psd': mean_psd,
            'freqs': freqs,
            'n_epochs': len(stage_idx),
            'channels': epochs.ch_names
        }
    
    return psd_results, freqs


def compute_bandpower_by_stage(epochs, stages) -> dict[str, dict]:
    """Bandpower per sleep stage"""
    unique_stages = list(set([s for s in stages if s != 'UNKNOWN']))
    bp_results = {}
    
    for stage in unique_stages:
        stage_idx = [i for i, s in enumerate(stages) if s == stage]
        
        if len(stage_idx) == 0:
            continue
        
        stage_epochs = epochs[stage_idx]
        stage_data = stage_epochs.get_data()
        stage_bp = {}
        
        for fmin, fmax in bands:
            psds, freqs = mne.time_frequency.psd_array_multitaper(
                stage_data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False
            )
            
            band_name = f"{fmin}-{fmax}Hz"
            # average across frequencies first, then get stats
            epoch_bp = psds.mean(axis=2)  
            
            stage_bp[band_name] = {
                'mean': np.mean(epoch_bp, axis=0),
                'std': np.std(epoch_bp, axis=0),
                'all': epoch_bp,
                'n_epochs': len(stage_idx)
            }
        
        bp_results[stage] = stage_bp
    
    return bp_results





def process_patient(patient_id, raw_data, hypno=None) -> dict:
    """Main processing function per patient"""
    print(f"\n{'='*50}")
    print(f"Processing: {patient_id}")
    print(f"{'='*50}")
    
    # make epochs
    epochs = make_epochs(raw_data)
    print(f"Created {len(epochs)} epochs")
    
    # artifact detection
    clean_epochs, bad_epochs, bad_idx = detect_bad_epochs(epochs)
    
    # basic spectral analysis
    bandpower = compute_bandpower(clean_epochs)
    
    # sleep stage stuff (if hypno available)
    stages = None
    psd_by_stage = None
    bp_by_stage = None
    
    if hypno is not None:
        print("Aligning with hypnogram...")
        stages = align_with_hypno(clean_epochs, hypno)
        psd_by_stage, _ = compute_psd_by_stage(clean_epochs, stages)
        bp_by_stage = compute_bandpower_by_stage(clean_epochs, stages)
    
    # store results in a dict (lazy but works)
    results = {
        'clean_epochs': clean_epochs,
        'bad_epochs': bad_epochs,
        'bad_indices': bad_idx,
        'bandpower': bandpower,
        'stages': stages,
        'psd_by_stage': psd_by_stage,
        'bp_by_stage': bp_by_stage
    }
    
    return results