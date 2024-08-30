from pathlib import Path
import os
import glob

import h5py
import yasa
import mne

import pandas as pd


def load_hdf5_group(f, path='.', data_dict=None):
    """
    Recursively loads data from an HDF5 file starting from the given path.
    
    :param hdf: An open h5py.File object.
    :param path: The current path in the HDF5 file (default is root '/').
    """
    if data_dict is None:
        data_dict = {}

    # Get the item at the current path
    item = f[path]
    
    if isinstance(item, h5py.Group):
        for key in item.keys():
            load_hdf5_group(f, f'{path}/{key}', data_dict)

    elif isinstance(item, h5py.Dataset):
        data = item[:]
        data_dict[path] = data

    
    return data_dict


def loaddata():
    p = Path.cwd() / "data"

    pids = os.listdir(p)

    for pid in pids:
        p_data = p / pid

        filedir = glob.glob(p_data.as_posix() + '/*.h5')[0]

        with h5py.File(filedir, 'r') as f:
            load_hdf5_group(f)


def loaddata_mne(): 
    p = Path.cwd() / "data"

    edf_files = sorted(p.glob('*/*.edf'))

    data_dict = {}

    for file in edf_files:
        patient = file.parent.name
        raw = mne.io.read_raw_edf(file, preload=True)
        data_dict[patient] = raw


    return data_dict

def load_hypno():
    # Load hypnogram data
    
    hypno_file = sorted(d1.parent.glob("*hypnogram.txt"))[0]

    with open(hypno_file) as fp:
        hd = fp.readlines()

    start_index = 0
    for i, line in enumerate(hd):
        if line.startswith('Sleep Stage'):
            start_index = i + 1
            break

    # Extract the column headers and rows
    columns = hd[start_index - 1].strip().split('\t')
    rows = [line.strip().split('\t') for line in hd[start_index:] if line.strip()]

    hypno = pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    # loaddata()
    d = loaddata_mne()
    print(9)

    yasa.sleep_statistics()