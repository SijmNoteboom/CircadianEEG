import mne
from loaddata import load_eeg_data, load_hypno, load_edf
from preprocessing import process_patient

from plot_eeg import (plot_artifact_summary, plot_epochs, 
                      plot_stage_distribution, plot_psd_by_stage,
                      plot_eeg_with_artifacts, plot_channel_quality)
import numpy as np

from pathlib import Path

# https://wearipedia.readthedocs.io/en/stable/notebooks/dreem_headband_3.html


def main() -> dict:
    # load data
    data_path = Path.cwd() / "data"
    edf_data = load_edf(data_path)  
    
    # process each patient
    all_results = {}
    
    for pid, raw in edf_data.items():
        # try to load hypnogram
        hypno_data = load_hypno(data_path, [pid])
        hypno = hypno_data.get('N1') if hypno_data else None
        
        # process
        results = process_patient(pid, raw, hypno)
        all_results[pid] = results
        
        # print some summary stats
        print(f"\nResults for {pid}:")
        print(f"Clean epochs: {len(results['clean_epochs'])}")
        print(f"Bad epochs: {len(results['bad_indices'])}")
        
        for band, power in results['bandpower'].items():
            mean_p = np.mean(power)
            print(f"{band}: mean = {mean_p:.2e}")
        
        if results['bp_by_stage']:
            print("\nBandpower by sleep stage:")
            for stage, data in results['bp_by_stage'].items():
                print(f"  {stage}:")
                for band, bp in data.items():
                    print(f"    {band}: {np.mean(bp['mean']):.2e} (n={bp['n_epochs']})")
        
        
        # artifact summary (need to reconstruct counts - bit hacky)
        artifact_counts = {'total': len(results['bad_indices'])}  # simplified
        plot_artifact_summary(artifact_counts)
        
        # epoch comparison
        plot_epochs(results['clean_epochs'], results['bad_epochs'])
        
        # sleep stage plots
        if results['stages']:
            plot_stage_distribution(results['stages'])
            
        if results['psd_by_stage']:
            plot_psd_by_stage(results['psd_by_stage'], ch_idx=0)
    return all_results



if __name__ == "__main__":
    results = main()