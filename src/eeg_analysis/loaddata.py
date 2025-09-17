from pathlib import Path
import os
import glob

import h5py
import yasa
import mne

import pandas as pd


def load_eeg_data(p):

    pids = os.listdir(p)

    for pid in pids:
        p_data = p / pid

        filedir = glob.glob(p_data.as_posix() + '/*.h5')[0]

        with h5py.File(filedir, 'r') as f:
            d = load_hdf5_group(f)
    return d

def load_edf(p) -> dict:
    eeg = dict()
    edf_files = sorted(p.glob('*/*.edf'))

    for file in edf_files:
        patient = file.parent.name
        raw = mne.io.read_raw_edf(file, preload=True)
        eeg[patient] = raw

    return eeg


def load_hypno(p, keys):
    # Load hypnogram data

    hypno = dict()

    for i in keys:
        d1 = p / i
        if not d1.exists():
            continue

        hypno_file = sorted(d1.glob("*hypnogram.txt"))
        if not hypno_file:
            continue
            
        with open(hypno_file[0]) as fp:
            hd = fp.readlines()

        start_index = 0
        for j, line in enumerate(hd):
            if line.startswith('Sleep Stage'):
                start_index = j + 1
                break

        # Extract the column headers and rows
        columns = hd[start_index - 1].strip().split('\t')
        rows = [line.strip().split('\t') for line in hd[start_index:] if line.strip()]

        hypno[i] = pd.DataFrame(rows, columns=columns)
    return hypno
