from loaddata import load_edf, load_hypno

import numpy as np
import mne
from scipy import stats
from scipy.signal import welch
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd

# =============================================================================
# CORE DATA CLASSES
# =============================================================================

class EEGConfig:
    """Configuration for EEG processing parameters"""
    def __init__(self, channels: List[str], sfreq: float, epoch_len: float, 
                 bands: List[Tuple[float, float]], apply_filters: bool = True):
        self.channels = channels
        self.sfreq = sfreq
        self.epoch_len = epoch_len
        self.bands = bands
        self.apply_filters = apply_filters
        
        # Filter parameters
        self.highpass_freq = 0.5
        self.lowpass_freq = 32
        self.notch_freq = 50

class ProcessingResults:
    """Container for all processing results"""
    def __init__(self):
        self.epochs = None
        self.rejected_epochs_obj = None
        self.rejected_indices = []
        self.epoch_sleep_stages = []
        self.artifact_log = {}
        self.bandpower = {}
        self.psd_per_stage = {}
        self.bandpower_per_stage = {}

# =============================================================================
# PREPROCESSING MODULE
# =============================================================================

class EEGPreprocessor:
    """Handles filtering and basic preprocessing"""
    
    def __init__(self, config: EEGConfig):
        self.config = config
    
    def filter_raw(self, raw):
        """Apply standard EEG filters"""
        if not self.config.apply_filters:
            return raw.copy()
            
        print("Applying filters...")
        raw_filtered = raw.copy()
        
        # High-pass filter
        raw_filtered.filter(l_freq=self.config.highpass_freq, h_freq=None, 
                           fir_design='firwin', verbose=False)
        
        # Low-pass filter  
        raw_filtered.filter(l_freq=None, h_freq=self.config.lowpass_freq, 
                           fir_design='firwin', verbose=False)
        
        # Notch filter
        raw_filtered.notch_filter(freqs=self.config.notch_freq, 
                                 fir_design='firwin', verbose=False)
        
        print("Filtering complete.")
        return raw_filtered
    
    def create_epochs(self, raw, hypnogram=None):
        """Create epochs from raw data"""
        raw_filtered = self.filter_raw(raw)
        
        events = mne.make_fixed_length_events(raw_filtered, id=1, 
                                            duration=self.config.epoch_len)
        picks = mne.pick_channels(raw_filtered.info["ch_names"], 
                                include=self.config.channels)
        
        epochs = mne.Epochs(raw_filtered, events, tmin=0, tmax=self.config.epoch_len, 
                           baseline=None, picks=picks, preload=True, verbose=False)
        
        return epochs

# =============================================================================
# ARTIFACT DETECTION MODULE
# =============================================================================

class ArtifactDetector(ABC):
    """Abstract base class for artifact detectors"""
    
    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        """Detect artifacts and return boolean array of rejected epochs"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class AmplitudeArtifactDetector(ArtifactDetector):
    def __init__(self, threshold_uv: float = 150):
        self.threshold_uv = threshold_uv
    
    @property
    def name(self) -> str:
        return "amplitude"
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        data_uv = data * 1e6
        max_amplitudes = np.max(np.abs(data_uv), axis=(1, 2))
        return max_amplitudes > self.threshold_uv

class GradientArtifactDetector(ArtifactDetector):
    def __init__(self, threshold: float = 50e-6):
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "gradient"
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        gradients = np.diff(data, axis=2)
        max_gradients = np.max(np.abs(gradients), axis=(1, 2))
        return max_gradients > self.threshold

class VarianceArtifactDetector(ArtifactDetector):
    def __init__(self, min_var: float = 1e-15, max_var: float = 1e-8):
        self.min_var = min_var
        self.max_var = max_var
    
    @property
    def name(self) -> str:
        return "variance"
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        variances = np.var(data, axis=2)
        min_var_per_epoch = np.min(variances, axis=1)
        max_var_per_epoch = np.max(variances, axis=1)
        
        flat_artifacts = min_var_per_epoch < self.min_var
        noisy_artifacts = max_var_per_epoch > self.max_var
        
        return flat_artifacts | noisy_artifacts

class HighFrequencyArtifactDetector(ArtifactDetector):
    def __init__(self, sfreq: float, hf_threshold: float = 0.3):
        self.sfreq = sfreq
        self.hf_threshold = hf_threshold
    
    @property
    def name(self) -> str:
        return "high_frequency"
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        hf_artifacts = np.zeros(data.shape[0], dtype=bool)
        
        for epoch_idx in range(data.shape[0]):
            hf_ratios = []
            for ch_idx in range(data.shape[1]):
                freqs, psd = welch(data[epoch_idx, ch_idx, :], 
                                 fs=self.sfreq, nperseg=min(256, data.shape[2]))
                
                hf_mask = (freqs >= 20) & (freqs <= 35)
                total_mask = (freqs >= 0.5) & (freqs <= 35)
                
                if np.sum(total_mask) > 0 and np.sum(hf_mask) > 0:
                    hf_power = np.sum(psd[hf_mask])
                    total_power = np.sum(psd[total_mask])
                    
                    if total_power > 0:
                        hf_ratio = hf_power / total_power
                        hf_ratios.append(hf_ratio)
            
            if hf_ratios and np.mean(hf_ratios) > self.hf_threshold:
                hf_artifacts[epoch_idx] = True
        
        return hf_artifacts

class StatisticalOutlierDetector(ArtifactDetector):
    def __init__(self, z_threshold: float = 4):
        self.z_threshold = z_threshold
    
    @property
    def name(self) -> str:
        return "outliers"
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        epoch_stats = []
        
        for epoch_idx in range(data.shape[0]):
            epoch_data = data[epoch_idx, :, :].flatten()
            
            mean_val = np.mean(epoch_data)
            std_val = np.std(epoch_data)
            skewness = stats.skew(epoch_data)
            kurtosis = stats.kurtosis(epoch_data)
            
            epoch_stats.append([mean_val, std_val, skewness, kurtosis])
        
        epoch_stats = np.array(epoch_stats)
        outlier_flags = np.zeros(data.shape[0], dtype=bool)
        
        for stat_idx in range(epoch_stats.shape[1]):
            z_scores = np.abs(stats.zscore(epoch_stats[:, stat_idx]))
            outlier_flags |= (z_scores > self.z_threshold)
        
        return outlier_flags

class ArtifactDetectionPipeline:
    """Manages multiple artifact detectors"""
    
    def __init__(self, config: EEGConfig):
        self.config = config
        self.detectors = [
            AmplitudeArtifactDetector(),
            GradientArtifactDetector(),
            VarianceArtifactDetector(),
            HighFrequencyArtifactDetector(config.sfreq),
            StatisticalOutlierDetector()
        ]
    
    def detect_artifacts(self, epochs):
        """Run all artifact detectors and return clean epochs"""
        print("Starting artifact detection...")
        
        data = epochs.get_data()
        n_epochs = data.shape[0]
        reject_flags = np.zeros(n_epochs, dtype=bool)
        artifact_log = {}
        
        # Run each detector
        for detector in self.detectors:
            print(f"Running {detector.name} artifact detection...")
            detector_flags = detector.detect(data)
            reject_flags |= detector_flags
            artifact_log[detector.name] = np.sum(detector_flags)
        
        # Create results
        rejected_indices = np.where(reject_flags)[0].tolist()
        rejected_epochs_obj = epochs[reject_flags] if len(rejected_indices) > 0 else None
        clean_epochs = epochs[~reject_flags]
        
        self._print_summary(n_epochs, artifact_log, len(rejected_indices))
        
        return clean_epochs, rejected_epochs_obj, rejected_indices, artifact_log
    
    def _print_summary(self, total_epochs, artifact_log, n_rejected):
        print("\n" + "="*50)
        print("ARTIFACT DETECTION SUMMARY")
        print("="*50)
        print(f"Total epochs: {total_epochs}")
        print(f"Rejected epochs: {n_rejected} ({n_rejected/total_epochs*100:.1f}%)")
        print(f"Clean epochs: {total_epochs - n_rejected} ({(total_epochs - n_rejected)/total_epochs*100:.1f}%)")
        print("\nRejection breakdown:")
        for artifact_type, count in artifact_log.items():
            print(f"  {artifact_type.capitalize()}: {count} epochs")
        print("="*50 + "\n")

# =============================================================================
# SLEEP STAGE ANALYSIS MODULE
# =============================================================================

class SleepStageAnalyzer:
    """Handles sleep stage alignment and analysis"""
    
    def __init__(self, config: EEGConfig):
        self.config = config
    
    def align_epochs_with_hypnogram(self, epochs, hypno_df: pd.DataFrame) -> List[str]:
        """Align epochs with sleep stages"""
        epoch_sleep_stages = []
        
        for epoch_idx in range(len(epochs)):
            start_time = epoch_idx * self.config.epoch_len
            hypno_idx = int(start_time // 30)  # 30s hypnogram resolution
            
            if hypno_idx < len(hypno_df):
                sleep_stage = hypno_df.iloc[hypno_idx]['Event']
                epoch_sleep_stages.append(sleep_stage)
            else:
                epoch_sleep_stages.append('UNKNOWN')
        
        return epoch_sleep_stages
    
    def compute_psd_per_stage(self, epochs, epoch_sleep_stages: List[str], 
                             freq_range: Tuple[float, float] = (0.5, 35)):
        """Compute PSD for each sleep stage"""
        unique_stages = [s for s in set(epoch_sleep_stages) if s != 'UNKNOWN']
        print(f"Computing PSD for stages: {unique_stages}")
        
        psd_per_stage = {}
        
        for stage in unique_stages:
            stage_indices = [i for i, s in enumerate(epoch_sleep_stages) if s == stage]
            
            if len(stage_indices) == 0:
                continue
            
            print(f"Processing {len(stage_indices)} epochs for {stage}")
            
            stage_epochs = epochs[stage_indices]
            stage_data = stage_epochs.get_data()
            
            psds, freqs = mne.time_frequency.psd_array_multitaper(
                stage_data, sfreq=self.config.sfreq,
                fmin=freq_range[0], fmax=freq_range[1],
                normalization='full', verbose=False
            )
            
            mean_psd = np.mean(psds, axis=0)
            
            psd_per_stage[stage] = {
                'psd': mean_psd,
                'freqs': freqs,
                'n_epochs': len(stage_indices),
                'channel_names': epochs.ch_names
            }
        
        return psd_per_stage, freqs
    
    def compute_bandpower_per_stage(self, epochs, epoch_sleep_stages: List[str]):
        """Compute bandpower for each sleep stage"""
        unique_stages = [s for s in set(epoch_sleep_stages) if s != 'UNKNOWN']
        bandpower_per_stage = {}
        
        for stage in unique_stages:
            stage_indices = [i for i, s in enumerate(epoch_sleep_stages) if s == stage]
            
            if len(stage_indices) == 0:
                continue
            
            stage_epochs = epochs[stage_indices]
            stage_data = stage_epochs.get_data()
            bandpower_stage = {}
            
            for fmin, fmax in self.config.bands:
                psds, freqs = mne.time_frequency.psd_array_multitaper(
                    stage_data, sfreq=self.config.sfreq,
                    fmin=fmin, fmax=fmax, normalization='full', verbose=False
                )
                
                band_name = f"{fmin}-{fmax}Hz"
                epoch_bandpower = psds.mean(axis=2)  # Average across frequencies
                
                bandpower_stage[band_name] = {
                    'mean': np.mean(epoch_bandpower, axis=0),
                    'std': np.std(epoch_bandpower, axis=0),
                    'all_epochs': epoch_bandpower,
                    'n_epochs': len(stage_indices)
                }
            
            bandpower_per_stage[stage] = bandpower_stage
        
        return bandpower_per_stage

# =============================================================================
# SPECTRAL ANALYSIS MODULE
# =============================================================================

class SpectralAnalyzer:
    """Handles spectral analysis computations"""
    
    def __init__(self, config: EEGConfig):
        self.config = config
    
    def compute_bandpower(self, epochs):
        """Compute bandpower for epochs"""
        print("Computing bandpower...")
        bandpower = {}
        
        for fmin, fmax in self.config.bands:
            psds, freqs = mne.time_frequency.psd_array_multitaper(
                epochs.get_data(), sfreq=self.config.sfreq,
                fmin=fmin, fmax=fmax, normalization='full', verbose=False
            )
            
            band_name = f"{fmin}-{fmax}Hz"
            bandpower[band_name] = psds.mean(axis=2) if psds.ndim == 3 else psds
        
        return bandpower

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class EEGVisualizer:
    """Handles all EEG plotting functions"""
    
    @staticmethod
    def plot_artifact_summary(artifact_log: Dict[str, int]):
        """Plot artifact detection summary"""
        if not artifact_log:
            print("No artifact detection performed yet.")
            return
        
        fig = go.Figure()
        
        artifact_types = list(artifact_log.keys())
        counts = list(artifact_log.values())
        
        fig.add_trace(go.Bar(
            x=artifact_types, y=counts, marker_color='red',
            text=counts, textposition='auto'
        ))
        
        fig.update_layout(
            title='Artifact Detection Summary',
            xaxis_title='Artifact Type',
            yaxis_title='Number of Rejected Epochs',
            showlegend=False
        )
        
        pyo.plot(fig)
    
    @staticmethod
    def plot_epochs_comparison(epochs_clean, epochs_rejected=None, 
                             n_clean_epochs=3, n_rejected_epochs=3):
        """Plot clean vs rejected epochs across all channels"""
        if epochs_clean is None:
            print("No clean epochs to plot")
            return
        
        clean_data = epochs_clean.get_data()
        n_channels = clean_data.shape[1]
        channel_names = epochs_clean.ch_names
        
        fig = make_subplots(
            rows=n_channels, cols=1,
            subplot_titles=[f"Channel: {name}" for name in channel_names],
            shared_xaxes=True, vertical_spacing=0.05
        )
        
        # Plot clean epochs
        n_clean_to_plot = min(n_clean_epochs, len(clean_data))
        for ch_idx in range(n_channels):
            for epoch_idx in range(n_clean_to_plot):
                fig.add_trace(
                    go.Scatter(
                        x=epochs_clean.times,
                        y=clean_data[epoch_idx, ch_idx, :] * 1e6,
                        mode='lines',
                        name=f'Clean {epoch_idx+1}' if ch_idx == 0 else None,
                        line=dict(color='blue', width=1.5),
                        opacity=0.8,
                        showlegend=(ch_idx == 0),
                        legendgroup='clean'
                    ),
                    row=ch_idx + 1, col=1
                )
        
        # Plot rejected epochs
        if epochs_rejected is not None and len(epochs_rejected) > 0:
            rejected_data = epochs_rejected.get_data()
            n_rejected_to_plot = min(n_rejected_epochs, len(rejected_data))
            
            for ch_idx in range(n_channels):
                for epoch_idx in range(n_rejected_to_plot):
                    fig.add_trace(
                        go.Scatter(
                            x=epochs_rejected.times,
                            y=rejected_data[epoch_idx, ch_idx, :] * 1e6,
                            mode='lines',
                            name=f'Rejected {epoch_idx+1}' if ch_idx == 0 else None,
                            line=dict(color='red', width=1.5, dash='dash'),
                            opacity=0.8,
                            showlegend=(ch_idx == 0),
                            legendgroup='rejected'
                        ),
                        row=ch_idx + 1, col=1
                    )
        
        fig.update_layout(
            height=200 * n_channels,
            title_text="EEG Epochs: Clean vs Rejected Comparison",
            title_x=0.5, showlegend=True
        )
        
        for ch_idx in range(n_channels):
            fig.update_yaxes(title_text="Amplitude (µV)", row=ch_idx + 1, col=1)
            if ch_idx == n_channels - 1:
                fig.update_xaxes(title_text="Time (s)", row=ch_idx + 1, col=1)
        
        pyo.plot(fig)
    
    @staticmethod
    def plot_psd_per_sleep_stage(psd_per_stage: Dict, channel_idx: int = 0, 
                               log_scale: bool = True):
        """Plot PSD comparison across sleep stages"""
        stage_colors = {
            'SLEEP-S0': 'blue', 'SLEEP-S1': 'lightblue', 'SLEEP-S2': 'green',
            'SLEEP-S3': 'darkgreen', 'SLEEP-REM': 'red', 'SLEEP-MT': 'orange',
            'SLEEP-UNSCORED': 'gray'
        }
        
        fig = go.Figure()
        
        if not psd_per_stage:
            print("No PSD data to plot")
            return
        
        channel_name = list(psd_per_stage.values())[0]['channel_names'][channel_idx]
        
        for stage, data in psd_per_stage.items():
            freqs = data['freqs']
            psd = data['psd'][channel_idx, :]
            n_epochs = data['n_epochs']
            color = stage_colors.get(stage, 'black')
            
            fig.add_trace(go.Scatter(
                x=freqs,
                y=10 * np.log10(psd) if log_scale else psd,
                mode='lines',
                name=f'{stage} (n={n_epochs})',
                line=dict(color=color, width=2)
            ))
        
        y_title = 'Power Spectral Density (dB)' if log_scale else 'Power Spectral Density (µV²/Hz)'
        
        fig.update_layout(
            title=f'PSD Comparison Across Sleep Stages - {channel_name}',
            xaxis_title='Frequency (Hz)', yaxis_title=y_title,
            showlegend=True, width=900, height=600
        )
        
        pyo.plot(fig)
    
    @staticmethod
    def plot_sleep_stage_distribution(epoch_sleep_stages: List[str]):
        """Plot distribution of epochs across sleep stages"""
        stage_counts = Counter([s for s in epoch_sleep_stages if s != 'UNKNOWN'])
        
        stage_colors = {
            'SLEEP-S0': 'blue', 'SLEEP-S1': 'lightblue', 'SLEEP-S2': 'green',
            'SLEEP-S3': 'darkgreen', 'SLEEP-REM': 'red', 'SLEEP-MT': 'orange',
            'SLEEP-UNSCORED': 'gray'
        }
        
        stages = list(stage_counts.keys())
        counts = list(stage_counts.values())
        colors = [stage_colors.get(stage, 'black') for stage in stages]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stages, y=counts, marker_color=colors,
            text=counts, textposition='auto'
        ))
        
        fig.update_layout(
            title='Distribution of Clean Epochs Across Sleep Stages',
            xaxis_title='Sleep Stage', yaxis_title='Number of Epochs',
            showlegend=False
        )
        
        pyo.plot(fig)

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

class EEGProcessor:
    """Main processing pipeline that orchestrates all components"""
    
    def __init__(self, config: EEGConfig):
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        self.artifact_detector = ArtifactDetectionPipeline(config)
        self.sleep_analyzer = SleepStageAnalyzer(config)
        self.spectral_analyzer = SpectralAnalyzer(config)
        self.visualizer = EEGVisualizer()
        self.results = ProcessingResults()
    
    def process(self, raw_data, hypnogram: Optional[pd.DataFrame] = None):
        """Main processing pipeline"""
        print(f"Starting EEG processing pipeline...")
        
        # 1. Preprocessing
        print("Step 1: Preprocessing...")
        epochs = self.preprocessor.create_epochs(raw_data, hypnogram)
        
        # 2. Artifact detection
        print("Step 2: Artifact detection...")
        (self.results.epochs, 
         self.results.rejected_epochs_obj, 
         self.results.rejected_indices, 
         self.results.artifact_log) = self.artifact_detector.detect_artifacts(epochs)
        
        # 3. Basic spectral analysis
        print("Step 3: Spectral analysis...")
        self.results.bandpower = self.spectral_analyzer.compute_bandpower(self.results.epochs)
        
        # 4. Sleep stage analysis (if hypnogram provided)
        if hypnogram is not None:
            print("Step 4: Sleep stage analysis...")
            self.results.epoch_sleep_stages = self.sleep_analyzer.align_epochs_with_hypnogram(
                self.results.epochs, hypnogram)
            
            self.results.psd_per_stage, _ = self.sleep_analyzer.compute_psd_per_stage(
                self.results.epochs, self.results.epoch_sleep_stages)
            
            self.results.bandpower_per_stage = self.sleep_analyzer.compute_bandpower_per_stage(
                self.results.epochs, self.results.epoch_sleep_stages)
        
        print("Processing complete!")
        return self.results
    
    def create_all_plots(self):
        """Generate all visualization plots"""
        print("Creating visualizations...")
        
        # Basic plots
        self.visualizer.plot_artifact_summary(self.results.artifact_log)
        self.visualizer.plot_epochs_comparison(
            self.results.epochs, self.results.rejected_epochs_obj)
        
        # Sleep stage plots (if available)
        if self.results.epoch_sleep_stages:
            self.visualizer.plot_sleep_stage_distribution(self.results.epoch_sleep_stages)
            
        if self.results.psd_per_stage:
            self.visualizer.plot_psd_per_sleep_stage(self.results.psd_per_stage, channel_idx=0)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    # Configuration
    config = EEGConfig(
        channels=['EEG F7-O1', 'EEG F8-O2', 'EEG F8-F7', 'EEG F8-O1', 'EEG F7-O2'],
        sfreq=100,
        epoch_len=30,
        bands=[(0.5, 4), (4, 8), (8, 12)],
        apply_filters=True
    )
    
    # Load data
    p = Path.cwd() / "data"
    d_edf = load_edf(p)  # You need to implement this
    
    # Process each patient
    for patient_id, raw_data in d_edf.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING PATIENT: {patient_id}")
        print(f"{'='*60}")
        
        # Load hypnogram
        hypno = load_hypno(p, [patient_id])['N1'] if load_hypno(p, [patient_id]) else None
        
        # Create processor and run pipeline
        processor = EEGProcessor(config)
        results = processor.process(raw_data, hypno)
        
        # Print summary
        print(f"\nSUMMARY FOR {patient_id}:")
        print("-" * 40)
        print(f"Clean epochs: {len(results.epochs)}")
        print(f"Rejected epochs: {len(results.rejected_indices)}")
        
        for band, power in results.bandpower.items():
            print(f"{band}: Mean = {np.mean(power):.2e}, Std = {np.std(power):.2e}")
        
        if results.bandpower_per_stage:
            print("\nBANDPOWER PER SLEEP STAGE:")
            for stage, stage_data in results.bandpower_per_stage.items():
                print(f"\n{stage}:")
                for band, band_data in stage_data.items():
                    mean_val = np.mean(band_data['mean'])
                    print(f"  {band}: {mean_val:.2e} (n={band_data['n_epochs']})")
        
        # Create all plots
        processor.create_all_plots()

if __name__ == "__main__":
    main()