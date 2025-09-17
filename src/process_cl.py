from loaddata import load_edf, load_hypno
from plot_eeg import plot_eeg_with_artifacts, plot_channel_quality
from pathlib import Path

import numpy as np
import mne
from scipy import stats
import plotly.graph_objects as go
import plotly.offline as pyo
from pathlib import Path

class EEGProcessing:
    def __init__(self, channel, sfreq, epoch_len, bands, apply_filters=True):
        self.channel = channel
        self.sfreq = sfreq
        self.epoch_len = epoch_len
        self.bands = bands
        self.epochs = None
        self.apply_filters = apply_filters
        self.rejected_epochs = []
        self.artifact_log = {}

    def create_epochs(self, raw, hypnogram=None):
        """Create epochs with filtering and artifact detection"""
        # Filter de data eerst
        if self.apply_filters:
            raw_filtered = raw.copy()
            print("Applying filters...")
            # High-pass filter (0.5 Hz)
            raw_filtered.filter(l_freq=0.5, h_freq=None, fir_design='firwin', verbose=False)
            # Low-pass filter (35 Hz)
            raw_filtered.filter(l_freq=None, h_freq=32, fir_design='firwin', verbose=False)
            # Notch filter voor 50Hz (netspanning)
            raw_filtered.notch_filter(freqs=50, fir_design='firwin', verbose=False)
            print("Filtering complete.")
        else:
            raw_filtered = raw
            
        # Create events and epochs
        events = mne.make_fixed_length_events(raw_filtered, id=1, duration=self.epoch_len)
        picks = mne.pick_channels(raw_filtered.info["ch_names"], include=self.channel)
        
        # Create epochs without preload first for artifact detection
        epochs_temp = mne.Epochs(raw_filtered, events, tmin=0, tmax=self.epoch_len, 
                                baseline=None, picks=picks, preload=True, verbose=False)
        
        # Apply artifact detection
        print(f"Initial epochs: {len(epochs_temp)}")
        clean_epochs, clean_ids = self.detect_artifacts(epochs_temp)
        print(f"Clean epochs after artifact rejection: {len(clean_epochs)}")
        
        self.epochs = clean_epochs
        self.clean_ids = clean_ids

    def detect_artifacts(self, epochs):
        """Comprehensive artifact detection for Dreem 3 EEG data"""
        print("Starting artifact detection...")
        
        # Get epoch data (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Initialize rejection flags
        reject_flags = np.zeros(n_epochs, dtype=bool)
        
        # 1. Amplitude-based rejection
        print("1. Amplitude-based artifact detection...")
        amplitude_artifacts = self._detect_amplitude_artifacts(data)
        reject_flags |= amplitude_artifacts
        self.artifact_log['amplitude'] = np.sum(amplitude_artifacts)
        
        # 2. Gradient-based rejection (sudden jumps)
        print("2. Gradient-based artifact detection...")
        gradient_artifacts = self._detect_gradient_artifacts(data)
        reject_flags |= gradient_artifacts
        self.artifact_log['gradient'] = np.sum(gradient_artifacts)
        
        # 3. Variance-based rejection (flat line detection)
        print("3. Variance-based artifact detection...")
        variance_artifacts = self._detect_variance_artifacts(data)
        reject_flags |= variance_artifacts
        self.artifact_log['variance'] = np.sum(variance_artifacts)
        
        # 4. High-frequency noise detection
        print("4. High-frequency noise detection...")
        hf_artifacts = self._detect_high_frequency_artifacts(data)
        reject_flags |= hf_artifacts
        self.artifact_log['high_frequency'] = np.sum(hf_artifacts)
        
        # 5. Statistical outlier detection (z-score based)
        print("5. Statistical outlier detection...")
        outlier_artifacts = self._detect_statistical_outliers(data)
        reject_flags |= outlier_artifacts
        self.artifact_log['outliers'] = np.sum(outlier_artifacts)
        
        # Store rejected epoch indices
        self.rejected_epochs = np.where(reject_flags)[0].tolist()
        
        # Create epochs object for rejected data (for comparison plotting)
        if len(self.rejected_epochs) > 0:
            self.rejected_epochs_obj = epochs[reject_flags]
        else:
            self.rejected_epochs_obj = None

        # Print rejection summary
        self._print_rejection_summary(n_epochs)     
        
        # Return clean epochs
        clean_indices = ~reject_flags
        return epochs[clean_indices], clean_indices
    
    def _detect_amplitude_artifacts(self, data, threshold_uv=150):
        """Detect epochs with extreme amplitudes"""
        # Convert to microvolts (assume data is in volts)
        data_uv = data * 1e6
        
        # Check for extreme amplitudes
        max_amplitudes = np.max(np.abs(data_uv), axis=(1, 2))
        amplitude_artifacts = max_amplitudes > threshold_uv
        return amplitude_artifacts

    def _detect_gradient_artifacts(self, data, threshold=50e-6):
        """Detect sudden jumps/gradients in the signal"""
        # Calculate gradient across time
        gradients = np.diff(data, axis=2)
        max_gradients = np.max(np.abs(gradients), axis=(1, 2))
        
        gradient_artifacts = max_gradients > threshold
        return gradient_artifacts

    def _detect_variance_artifacts(self, data, min_var=1e-15, max_var=1e-8):
        """Detect epochs with too low (flat) or too high variance"""
        # Calculate variance across time for each epoch and channel
        variances = np.var(data, axis=2)
        
        # Check for minimum variance (flat line)
        min_var_per_epoch = np.min(variances, axis=1)
        flat_artifacts = min_var_per_epoch < min_var
        
        # Check for maximum variance (excessive noise)
        max_var_per_epoch = np.max(variances, axis=1)
        noisy_artifacts = max_var_per_epoch > max_var
        
        variance_artifacts = flat_artifacts | noisy_artifacts
        return variance_artifacts

    def _detect_high_frequency_artifacts(self, data, hf_threshold=0.3):
        """Detect epochs with excessive high-frequency content"""
        # Calculate power in high frequency band (20-35 Hz) vs total power
        from scipy.signal import welch
        
        hf_artifacts = np.zeros(data.shape[0], dtype=bool)
        
        for epoch_idx in range(data.shape[0]):
            hf_ratios = []
            for ch_idx in range(data.shape[1]):
                # Calculate PSD
                freqs, psd = welch(data[epoch_idx, ch_idx, :], 
                                    fs=self.sfreq, nperseg=min(256, data.shape[2]))
                
                # Calculate ratio of high-frequency power
                hf_mask = (freqs >= 20) & (freqs <= 35)
                total_mask = (freqs >= 0.5) & (freqs <= 35)
                
                if np.sum(total_mask) > 0 and np.sum(hf_mask) > 0:
                    hf_power = np.sum(psd[hf_mask])
                    total_power = np.sum(psd[total_mask])
                    
                    if total_power > 0:
                        hf_ratio = hf_power / total_power
                        hf_ratios.append(hf_ratio)
            
            if hf_ratios and np.mean(hf_ratios) > hf_threshold:
                hf_artifacts[epoch_idx] = True
        
        return hf_artifacts

    def _detect_statistical_outliers(self, data, z_threshold=4):
        """Detect epochs that are statistical outliers"""
        # Calculate statistical measures for each epoch
        epoch_stats = []
        
        for epoch_idx in range(data.shape[0]):
            epoch_data = data[epoch_idx, :, :].flatten()
            
            # Calculate various statistics
            mean_val = np.mean(epoch_data)
            std_val = np.std(epoch_data)
            skewness = stats.skew(epoch_data)
            kurtosis = stats.kurtosis(epoch_data)
            
            epoch_stats.append([mean_val, std_val, skewness, kurtosis])
        
        epoch_stats = np.array(epoch_stats)
        
        # Calculate z-scores for each statistic
        outlier_flags = np.zeros(data.shape[0], dtype=bool)
        
        for stat_idx in range(epoch_stats.shape[1]):
            z_scores = np.abs(stats.zscore(epoch_stats[:, stat_idx]))
            outlier_flags |= (z_scores > z_threshold)
        
        return outlier_flags

    def _print_rejection_summary(self, total_epochs):
        """Print summary of artifact detection results"""
        print("\n" + "="*50)
        print("ARTIFACT DETECTION SUMMARY")
        print("="*50)
        print(f"Total epochs: {total_epochs}")
        print(f"Rejected epochs: {len(self.rejected_epochs)} ({len(self.rejected_epochs)/total_epochs*100:.1f}%)")
        print(f"Clean epochs: {total_epochs - len(self.rejected_epochs)} ({(total_epochs - len(self.rejected_epochs))/total_epochs*100:.1f}%)")
        print("\nRejection breakdown:")
        for artifact_type, count in self.artifact_log.items():
            print(f"  {artifact_type.capitalize()}: {count} epochs")
        print("="*50 + "\n")

    def compute_bandpower(self):
        """Compute bandpower for clean epochs"""
        if self.epochs is None:
            raise ValueError("Eerst create_epochs() aanroepen")

        print("Computing bandpower for clean epochs...")
        bandpower = {}
        
        for fmin, fmax in self.bands:
            psds, freqs = mne.time_frequency.psd_array_multitaper(
                self.epochs.get_data().squeeze() if self.epochs.get_data().ndim == 3 else self.epochs.get_data(),
                sfreq=self.sfreq,
                fmin=fmin,
                fmax=fmax,
                normalization='full',
                verbose=False
            )
            
            # Gemiddelde power per epoch
            band_name = f"{fmin}-{fmax}Hz"
            if psds.ndim == 2:
                bandpower[band_name] = psds.mean(axis=1)
            else:
                bandpower[band_name] = psds
                
        return bandpower
    
    def plot_artifact_summary(self):
        """Plot artifact detection summary"""
        if not self.artifact_log:
            print("No artifact detection performed yet.")
            return
            
        fig = go.Figure()
        
        artifact_types = list(self.artifact_log.keys())
        counts = list(self.artifact_log.values())
        
        fig.add_trace(go.Bar(
            x=artifact_types,
            y=counts,
            marker_color='red',
            text=counts,
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Artifact Detection Summary',
            xaxis_title='Artifact Type',
            yaxis_title='Number of Rejected Epochs',
            showlegend=False
        )
        
        pyo.plot(fig)

def main():
    # Example usage
    p = Path.cwd() / "data"
    d_edf = load_edf(p)

    # For loop if more patients; d_edf is a dictionary with patient keys and raw data
    for patient_id, raw_data in d_edf.items():
        print(f"\nProcessing patient: {patient_id}")
        
        hypno = load_hypno(p, [patient_id])['N1'] if load_hypno(p, [patient_id]) else None

        eeg_processor = EEGProcessing(
            channel=['EEG F7-O1', 'EEG F8-O2', 'EEG F8-F7', 'EEG F8-O1', 'EEG F7-O2'], 
            sfreq=100, 
            epoch_len=30, 
            bands=[(0.5, 4), (4, 8), (8, 12)],
            apply_filters=True  # Enable filtering
        )
        
        # Create epochs with artifact detection
        eeg_processor.create_epochs(raw_data, hypnogram=hypno)

        hypno_lean = hypno[eeg_processor.clean_ids] 

        
        # Compute bandpower for clean epochs
        bandpower = eeg_processor.compute_bandpower()
        
        print(f"\nBandpower results for {patient_id}:")
        for band, power in bandpower.items():
            print(f"{band}: Mean = {np.mean(power):.2e}, Std = {np.std(power):.2e}")
        
        # Plot artifact summary
        eeg_processor.plot_artifact_summary()
        
        # Plot clean epochs with artifact comparison
        if eeg_processor.epochs is not None:
            plot_eeg_with_artifacts(eeg_processor.epochs, eeg_processor.rejected_epochs_obj)
            
        # Plot channel quality analysis
        # plot_channel_quality(eeg_processor)

if __name__ == "__main__":
    main()