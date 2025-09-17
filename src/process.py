from pathlib import Path
import mne

from eeg_analysis.loaddata import load_edf, load_hypno
from eeg_analysis.preprocessing import process_patient

import plotly.graph_objects as go   
import plotly.offline as pyo    


class EEGProcessing:
    def __init__(self, channel, sfreq, epoch_len, bands):
        self.channel = channel
        self.sfreq = sfreq
        self.epoch_len = epoch_len
        self.bands = bands
        self.epochs = None
        self.apply_filters = True  # Default to True, can be set to False if no filtering is neededr

    def create_epochs(self, raw, hypnogram=None):

        if self.apply_filters:
            raw_filtered = raw.copy()
            # High-pass filter
            raw_filtered.filter(l_freq=0.5, h_freq=None, fir_design='firwin')
            # Low-pass filter  
            raw_filtered.filter(l_freq=None, h_freq=35, fir_design='firwin')
            # Notch filter voor 50Hz
            raw_filtered.notch_filter(freqs=50, fir_design='firwin')
        else:
            raw_filtered = raw

        # Stel: hypnogram is optioneel, maar je kan wel per 30s knippen
        events = mne.make_fixed_length_events(raw, id=1, duration=self.epoch_len)
        picks = mne.pick_channels(raw.info["ch_names"], include=self.channel)
        self.epochs = mne.Epochs(raw, events, tmin=0, tmax=self.epoch_len, baseline=None, picks=picks, preload=True)
        return self, raw_filtered

    def compute_bandpower(self):
        if self.epochs is None:
            raise ValueError("Eerst create_epochs() aanroepen")

        bandpower = {}
        for fmin, fmax in self.bands:
            psds, freqs = mne.time_frequency.psd_array_multitaper(
                self.epochs.get_data().squeeze(),  # (n_epochs, n_times)
                sfreq=self.sfreq,
                fmin=fmin,
                fmax=fmax,
                normalization='full'
            )
            # Gemiddelde power per epoch
            band_name = f"{fmin}-{fmax}Hz"
            bandpower[band_name] = psds.mean(axis=1)
        return bandpower


def plot_eeg(data, ch_no=None, raw_filtered=None):
    channel = data.ch_names[ch_no]

    fig = go.Figure()
    if ch_no is None:
        for channel in data.ch_names:
            if 'EEG' not in channel:
                continue
            fig.add_trace(go.Scatter(x=data.times, y=data[channel, 0:100000], mode='lines', name=channel))
    else:
        fig.add_trace(go.Scatter(x=data.times, y=data.get_data(channel)[0, 100000:500000], mode='lines', name=channel))

    if raw_filtered is not None:
        fig.add_trace(go.Scatter(x=raw_filtered.times, y=raw_filtered.get_data(channel)[0, 100000:500000], mode='lines', name=f"{channel} (filtered)")) 

    fig.update_layout(title='EEG Data', xaxis_title='Time (s)', yaxis_title='Amplitude (µV)')
    pyo.plot(fig)

def main():
    # Example usage
    p = Path.cwd() / "data"
    d_edf = load_edf(p)

    data_plot = d_edf['N1']

    plot_eeg(d_edf['N1'], ch_no=1)

    # for loop if more patients; d_edf is a dictionary with patient keys and raw data
    for patient_id, raw_data in d_edf.items():
        hypno = load_hypno(p, [patient_id])['N1']
        if hypno is None:
            hypno = None    

        eeg_processor = EEGProcessing(channel=['EEG F7-O1', 'EEG F8-O2', 'EEG F8-F7', 'EEG F8-O1', 'EEG F7-O2'], 
                                  sfreq=100, epoch_len=30, bands=[(0.5, 4), (4, 8), (8, 12)])
        eeg_processor, raw_filtered = eeg_processor.create_epochs(raw_data, hypnogram=hypno)

        plot_eeg(raw_data, ch_no=1, raw_filtered=raw_filtered)

        bandpower = eeg_processor.compute_bandpower()
    
        print(bandpower)


if __name__ == "__main__":
    main()