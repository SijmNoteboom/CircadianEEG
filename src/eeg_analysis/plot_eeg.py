import plotly.graph_objects as go
import plotly.offline as pyo    
from plotly.subplots import make_subplots
import mne
import numpy as np
from collections import Counter



def plot_eeg_with_artifacts(epochs_clean, epochs_rejected=None, n_clean_epochs=3, n_rejected_epochs=3):
    """Plot clean vs rejected epochs comparison across all channels"""

    
    if epochs_clean is None:
        print("No clean epochs to plot")
        return
    
    clean_data = epochs_clean.get_data()  # [n_epochs, n_channels, n_times]
    n_channels = clean_data.shape[1]
    channel_names = epochs_clean.ch_names
    
    # Create subplots - one row per channel
    fig = make_subplots(
        rows=n_channels, 
        cols=1,
        subplot_titles=[f"Channel: {name}" for name in channel_names],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Plot clean epochs
    if len(clean_data) > 0:
        n_clean_to_plot = min(n_clean_epochs, len(clean_data))
        for ch_idx in range(n_channels):
            for epoch_idx in range(n_clean_to_plot):
                fig.add_trace(
                    go.Scatter(
                        x=epochs_clean.times,
                        y=clean_data[epoch_idx, ch_idx, :] * 1e6,  # Convert to µV
                        mode='lines',
                        name=f'Clean {epoch_idx+1}' if ch_idx == 0 else None,  # Only show legend for first channel
                        line=dict(color='blue', width=1.5),
                        opacity=0.8,
                        showlegend=(ch_idx == 0),  # Only show legend for first channel
                        legendgroup='clean'
                    ),
                    row=ch_idx + 1, col=1
                )
    
    # Plot rejected epochs if available
    if epochs_rejected is not None and len(epochs_rejected) > 0:
        rejected_data = epochs_rejected.get_data()
        n_rejected_to_plot = min(n_rejected_epochs, len(rejected_data))
        
        for ch_idx in range(n_channels):
            for epoch_idx in range(n_rejected_to_plot):
                fig.add_trace(
                    go.Scatter(
                        x=epochs_rejected.times,
                        y=rejected_data[epoch_idx, ch_idx, :] * 1e6,  # Convert to µV
                        mode='lines',
                        name=f'Rejected {epoch_idx+1}' if ch_idx == 0 else None,
                        line=dict(color='red', width=1.5, dash='dash'),
                        opacity=0.8,
                        showlegend=(ch_idx == 0),
                        legendgroup='rejected'
                    ),
                    row=ch_idx + 1, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=200 * n_channels,  # Adjust height based on number of channels
        title_text="EEG Epochs: Clean vs Rejected Comparison",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x and y axis labels
    for ch_idx in range(n_channels):
        fig.update_yaxes(title_text="Amplitude (µV)", row=ch_idx + 1, col=1)
        if ch_idx == n_channels - 1:  # Only add x-axis label to bottom subplot
            fig.update_xaxes(title_text="Time (s)", row=ch_idx + 1, col=1)
    
    pyo.plot(fig)


def plot_channel_quality(eeg_processor):
    """Plot signal quality metrics per channel"""
    if eeg_processor.epochs is None:
        print("No epochs available for quality analysis")
        return
    

    data = eeg_processor.epochs.get_data()  # [n_epochs, n_channels, n_times]
    n_epochs, n_channels, n_times = data.shape
    channel_names = eeg_processor.epochs.ch_names
    
    # Calculate quality metrics per channel
    channel_metrics = {}
    for ch_idx, ch_name in enumerate(channel_names):
        ch_data = data[:, ch_idx, :] * 1e6  # Convert to µV
        
        # Calculate metrics
        mean_amplitude = np.mean(np.abs(ch_data), axis=1)  # Per epoch
        variance = np.var(ch_data, axis=1)  # Per epoch
        snr_estimate = np.mean(mean_amplitude) / np.std(mean_amplitude) if np.std(mean_amplitude) > 0 else 0
        
        channel_metrics[ch_name] = {
            'mean_amplitude': np.mean(mean_amplitude),
            'std_amplitude': np.std(mean_amplitude),
            'mean_variance': np.mean(variance),
            'snr_estimate': snr_estimate,
            'amplitude_per_epoch': mean_amplitude,
            'variance_per_epoch': variance
        }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Amplitude per Channel', 'Signal Variance per Channel', 
                       'Amplitude over Time', 'Variance over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Mean amplitude per channel
    channels = list(channel_metrics.keys())
    mean_amps = [channel_metrics[ch]['mean_amplitude'] for ch in channels]
    std_amps = [channel_metrics[ch]['std_amplitude'] for ch in channels]
    
    fig.add_trace(
        go.Bar(x=channels, y=mean_amps, error_y=dict(type='data', array=std_amps),
               name='Mean Amplitude', marker_color='blue'),
        row=1, col=1
    )
    
    # Plot 2: Mean variance per channel
    mean_vars = [channel_metrics[ch]['mean_variance'] for ch in channels]
    fig.add_trace(
        go.Bar(x=channels, y=mean_vars, name='Mean Variance', marker_color='green'),
        row=1, col=2
    )
    
    # Plot 3: Amplitude over epochs
    epoch_indices = range(n_epochs)
    for ch_idx, ch_name in enumerate(channels):
        fig.add_trace(
            go.Scatter(
                x=epoch_indices,
                y=channel_metrics[ch_name]['amplitude_per_epoch'],
                mode='lines+markers',
                name=f'{ch_name}',
                showlegend=(ch_idx < 3)  # Limit legend entries
            ),
            row=2, col=1
        )
    
    # Plot 4: Variance over epochs
    for ch_idx, ch_name in enumerate(channels):
        fig.add_trace(
            go.Scatter(
                x=epoch_indices,
                y=channel_metrics[ch_name]['variance_per_epoch'],
                mode='lines+markers',
                name=f'{ch_name}',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="EEG Signal Quality Analysis",
        title_x=0.5
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Channel", row=1, col=1)
    fig.update_xaxes(title_text="Channel", row=1, col=2)
    fig.update_xaxes(title_text="Epoch Number", row=2, col=1)
    fig.update_xaxes(title_text="Epoch Number", row=2, col=2)
    
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Variance (µV2)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (µV)", row=2, col=1)
    fig.update_yaxes(title_text="Variance (µV2)", row=2, col=2)
    
    pyo.plot(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("CHANNEL QUALITY SUMMARY")
    print("="*60)
    for ch_name, metrics in channel_metrics.items():
        print(f"{ch_name}:")
        print(f"  Mean Amplitude: {metrics['mean_amplitude']:.2f} ± {metrics['std_amplitude']:.2f} µV")
        print(f"  Mean Variance: {metrics['mean_variance']:.2e} µV2")
        print(f"  SNR Estimate: {metrics['snr_estimate']:.2f}")
        print()
    print("="*60)

# plotting functions - could be cleaner but whatever
def plot_artifact_summary(rejected_counts):
    """Quick bar plot of artifacts"""
    if not rejected_counts:
        print("No artifact data to plot")
        return
    
    fig = go.Figure()
    
    types = list(rejected_counts.keys())
    counts = list(rejected_counts.values())
    
    fig.add_trace(go.Bar(x=types, y=counts, marker_color='red'))
    
    fig.update_layout(
        title='Artifacts Found',
        xaxis_title='Type',
        yaxis_title='Count'
    )
    
    pyo.plot(fig)


def plot_epochs(clean_epochs, bad_epochs=None, n_show=3):
    """Plot some example epochs"""
    if clean_epochs is None:
        print("No epochs to plot")
        return
    
    clean_data = clean_epochs.get_data()
    n_ch = clean_data.shape[1]
    ch_names = clean_epochs.ch_names
    
    fig = make_subplots(rows=n_ch, cols=1, 
                       subplot_titles=ch_names,
                       shared_xaxes=True)
    
    # plot clean epochs
    n_clean = min(n_show, len(clean_data))
    for ch in range(n_ch):
        for ep in range(n_clean):
            fig.add_trace(
                go.Scatter(
                    x=clean_epochs.times,
                    y=clean_data[ep, ch, :] * 1e6,  # to uV
                    mode='lines',
                    name=f'Clean {ep+1}' if ch == 0 else None,
                    line=dict(color='blue', width=1),
                    showlegend=(ch == 0)
                ),
                row=ch+1, col=1
            )
    
    # plot bad epochs if available
    if bad_epochs is not None and len(bad_epochs) > 0:
        bad_data = bad_epochs.get_data()
        n_bad = min(n_show, len(bad_data))
        
        for ch in range(n_ch):
            for ep in range(n_bad):
                fig.add_trace(
                    go.Scatter(
                        x=bad_epochs.times,
                        y=bad_data[ep, ch, :] * 1e6,
                        mode='lines',
                        name=f'Bad {ep+1}' if ch == 0 else None,
                        line=dict(color='red', width=1, dash='dash'),
                        showlegend=(ch == 0)
                    ),
                    row=ch+1, col=1
                )
    
    fig.update_layout(height=200 * n_ch, title="Clean vs Bad Epochs")
    
    # add y-axis labels
    for ch in range(n_ch):
        fig.update_yaxes(title_text="µV", row=ch+1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
    
    pyo.plot(fig)


def plot_psd_by_stage(psd_data, ch_idx=0):
    """PSD comparison across sleep stages"""
    # colors for different stages (hardcoded, sue me)
    colors = {'SLEEP-S0': 'blue', 'SLEEP-S1': 'lightblue', 'SLEEP-S2': 'green',
              'SLEEP-S3': 'darkgreen', 'SLEEP-REM': 'red', 'SLEEP-MT': 'orange'}
    
    fig = go.Figure()
    
    if not psd_data:
        print("No PSD data")
        return
    
    ch_name = list(psd_data.values())[0]['channels'][ch_idx]
    
    for stage, data in psd_data.items():
        freqs = data['freqs']
        psd = data['psd'][ch_idx, :]
        n_ep = data['n_epochs']
        color = colors.get(stage, 'black')
        
        fig.add_trace(go.Scatter(
            x=freqs,
            y=10 * np.log10(psd),  # convert to dB
            mode='lines',
            name=f'{stage} (n={n_ep})',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title=f'PSD by Sleep Stage - {ch_name}',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (dB)',
        width=800, height=500
    )
    
    pyo.plot(fig)


def plot_stage_distribution(stages):
    """Bar plot of sleep stage counts"""
    stage_counts = Counter([s for s in stages if s != 'UNKNOWN'])
    
    colors = {'SLEEP-S0': 'blue', 'SLEEP-S1': 'lightblue', 'SLEEP-S2': 'green',
              'SLEEP-S3': 'darkgreen', 'SLEEP-REM': 'red', 'SLEEP-MT': 'orange'}
    
    stage_list = list(stage_counts.keys())
    count_list = list(stage_counts.values())
    color_list = [colors.get(s, 'gray') for s in stage_list]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stage_list, y=count_list, marker_color=color_list))
    
    fig.update_layout(
        title='Sleep Stage Distribution',
        xaxis_title='Stage',
        yaxis_title='# Epochs'
    )
    
    pyo.plot(fig)