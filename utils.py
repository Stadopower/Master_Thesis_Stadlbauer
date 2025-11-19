# some function to make the end result more readable:
#filename has to be the entire path to the file so "data//P101//filename"
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pyxdf
import pandas as pd
import re
import mne
from sklearn.decomposition import fastica
from pyprep import PrepPipeline, NoisyChannels
from matplotlib.colors import TwoSlopeNorm
import matplotlib
from mne_icalabel import label_components
from autoreject import AutoReject

def get_data(filename, pilot=False):    # Pilot = True for all pilots except Pilot117
    print('Reading data')

    # Because of the software switch there are now 3 ACC channels and one Marker channel
    nr_non_eeg = 4
    if pilot:
        nr_non_eeg = 3

    # Reading in the xdf files and extracting the marker and eeg streams
    streams, fileheader = pyxdf.load_xdf(filename, select_streams=[{'type': 'EEG'}, {'name':'LSL4Unity.OmnideckWaiterVR'}] , synchronize_clocks=False)
    marker_stream = next(s for s in streams if 'LSL4Unity.OmnideckWaiterVR' in s['info']['name'][0])
    eeg_stream = next(s for s in streams if "EEG" in s['info']['type'][0])
    eeg_data = np.array(eeg_stream['time_series']).T
    #eeg_timestamps = np.array(eeg_stream['time_stamps'])
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    # Collection all ch names and renaming TP9, TP10
    ch_names = []
    for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']:
        if ch['label'][0] == 'TP10':
            ch_names.append('FCz')
        elif ch['label'][0] == 'TP9':
            ch_names.append('Fpz')
        elif ch['label'][0] == 'FPz':
            ch_names.append('Fpz')
        else:
            ch_names.append(ch['label'][0])
    info = mne.create_info(
        ch_names=ch_names[:-nr_non_eeg],
        sfreq=sfreq,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(eeg_data[:-nr_non_eeg]/10e5, info)

    # Creating the Montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Calculating the event samples for the annotations
    event_samples = (marker_stream['time_stamps'] - eeg_stream['time_stamps'][0])*sfreq
    event_samples = event_samples.astype(int)
    event_labels = [int(marker[0]) for marker in marker_stream['time_series']]


    annotations = mne.Annotations(onset=event_samples / sfreq,
                                  duration=[0] * len(event_samples),  # Instantaneous events
                                  description=list(event_labels))
    raw.set_annotations(annotations)

    return raw, marker_stream, event_samples

#apply a notch filter at 50hz and filter between 0.01 and 100Hz. Filter above 1Hz is applied when creating the epochs; lowpass filter at 40Hz is applied before saving the epochs
def filter_data(raw):
    # Notch Filter at 50 Hz
    raw = raw.notch_filter(50, method='fir', phase='zero',verbose=False)
    # Filter data between 0.01 and 100 Hz for the ICA algorithm
    iir_params = dict(order=2, ftype='butter',verbose=False)
    raw = raw.filter(l_freq=0.1, h_freq=100, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    return raw


# making epochs around a specified marker, Applying 1Hz Highpass filter if epoch is used for ICA
def get_epochs(raw: mne.io.Raw, marker_stream, event_samples , marker_id:int, tmin:int, tmax:int, preload=False, ica=True):
    if ica:
        iir_params = dict(order=2, ftype='butter',verbose=False)
        raw = raw.copy()
        raw = raw.filter(l_freq=1, h_freq=None, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    event_labels = [int(marker[0]) for marker in marker_stream['time_series']]
    events = np.array([[sample, 0, label] for sample, label in zip(event_samples, event_labels)])
    selected_events = events[events[:, 2] == marker_id]
    epochs = mne.Epochs(raw, np.array(selected_events), event_id=int(marker_id),baseline=None, tmin=tmin, tmax=tmax, reject_by_annotation=False, verbose=False, preload=preload) # , decim=decim
    return epochs

# Counts the markers and saves them in a file.
def count_markers(marker_stream, match = False):
    flat_list = [int(marker[0]) for marker in marker_stream['time_series']]
    marker_count = {}
    for marker in flat_list:
        if marker in marker_count:
            marker_count[marker] += 1
        else:
            marker_count[marker] = 1
    marker_count = dict(sorted(marker_count.items()))
    if match:
        with open(f'markers\\{match}_markers.txt', 'w') as file:
            for key, value in marker_count.items():
                file.write(f"{key}: {value}\n")
    #return marker_count

# ICA applied to individual epochs. Outdated function
def apply_ica_indiv(epochs: mne.Epochs):
    cleaned_epochs = []
    all_rej = []
    i = 0
    while i < np.shape(epochs.events)[0]:
        epoch = epochs[i]
        ica = mne.preprocessing.ICA(n_components=32, method='fastica', random_state=42)
        ica.fit(epoch)
        source = ica.get_sources(epoch).get_data()
        std = np.std(source)
        rej_ch = []
        for j, channel in enumerate(source[0]):
            if np.max(channel) > 5*std:     # This has been chosen somewhat arbitrarily now as it results in
                rej_ch.append(j)
        all_rej.append(rej_ch)

        cleaned_epochs.append(ica.apply(epoch, exclude=rej_ch))
        i+=1
    cleaned_epochs = mne.EpochsArray(
        data = np.squeeze([e.get_data() for e in cleaned_epochs]),
        info = epochs.info,
        events= epochs.events
    )
    return cleaned_epochs

# Naive way of ICA component rejection, based on maximum amplitude of component.
def get_ica(epochs: mne.Epochs, n_components=32, plot=False, save_path=None, match=None):
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter= 1500, method='fastica', verbose=False)
    ica.fit(epochs, verbose=False)
    #std = np.std(ica.get_sources(epochs).get_data())
    rej_ch = []
    for i, channel in enumerate(ica.get_sources(epochs).get_data()[0]):
        std = np.std(channel)
        if np.max(np.absolute(channel)) > 6*std:
            rej_ch.append(i)
    if plot and len(rej_ch) != 0:
        #print(len(rej_ch))
        with open(f'figures\\meeting\\ica\\{match}.txt', 'w') as file:
            file.write(", ".join(map(str, rej_ch)))
            file.write(f"\n{len(rej_ch)} rejected channels")
            file.close()
        fig = ica.plot_components(title=f'{match} all ICA components', show=False);
        for j, f in enumerate(fig):
            f.savefig(save_path+f'{match}_{j}_all_ica.png')
            plt.close(f)
        fig = ica.plot_components(picks=rej_ch,title=f'{match} rejected channels', show=False);
        fig.savefig(save_path+f'{match}_rejected_ica.png')
        plt.close(fig)
    return ica, rej_ch
    #ica.apply(epochs, exclude=rej_ch, verbose=False)

# ICA method using the mne implementation of icalabel
def get_icalabel(epochs: mne.Epochs, n_components=32, plot=False, save_path=None, match=None):
    """"
    A function extracting the ICA components and rejecting all components that are not labeled Brain by the icalabel function. Using a certainty threshhold of 70%

    epochs | mne epoch for which to calculate the ICA
    n_components | number of ICA components
    plot | whether to plot the ICA components
    save_path | path to save the ICA components
    match | the match regex that indicates the participant, trial and condition
    """
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter= 500, method='infomax', fit_params=dict(extended=True), verbose=False)
    ica.fit(epochs, verbose=False)
    rej_ch = []
    a = label_components(epochs, ica, method='iclabel')
    for i, ic in enumerate(a['labels']):
        #print(f'Component {i} is {ic} with proba {a['y_pred_proba'][i]}')
        if ic != 'brain' : #and a['y_pred_proba'][i] > 0.6:
            rej_ch.append(i)

    # Plotting the Components as well as Sources
    print('Rejected components:', len(rej_ch))
    if plot and len(rej_ch) != 0:
        # Saving Rejected channels in txt
        with open(f'figures\\meeting\\ica\\{match}.txt', 'w') as file:
            file.write(", ".join(map(str, rej_ch)))
            file.write(f"\n{len(rej_ch)} rejected channels")
            file.close()

        # Plotting Components
        fig = ica.plot_components(title=f'{match} all ICA components', show=False);
        for j, f in enumerate(fig):
            f.savefig(save_path+f'{match}_{j}_all_ica.png')
            plt.close(f)
        fig = ica.plot_components(picks=rej_ch,title=f'{match} rejected channels', show=False);
        fig.savefig(save_path+f'{match}_rejected_ica.png')
        plt.close(fig)

        # Plotting Sources
        fig = ica.plot_sources(epochs, picks=slice(0,32,1))
        fig.savefig(save_path+f'{match}_ica_sources.png')
        plt.close(fig)
    return ica, rej_ch

# Filters the epochs and creates the MRCP Event from the epochs
def mrcp(epochs):
    frontal_channels = ['F3', 'Fz', 'F4', 'FC1', 'FCz', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4']
    iir_params = dict(order=2, ftype='butter')
    epochs = epochs.filter(l_freq =0.1 ,h_freq=1, method='iir', iir_params=iir_params, phase='zero')
    evoked = epochs.average()
    evoked = evoked.pick(frontal_channels)
    return evoked

# Outdated function
def drop_bad_epochs(epochs):
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    std_per_channel = np.std(epoch_data, axis=(0, 2))  # Standard deviation per channel
    threshold = 10 * std_per_channel[:, np.newaxis]  # Expand dims to match shape
    # Find epochs where any channel exceeds the threshold
    bad_epochs = np.any(np.abs(epoch_data) > threshold[np.newaxis, :, :], axis=(1, 2))
    print(f'Bad epochs: {len(bad_epochs)}')
    # Drop bad epochs
    epochs_clean = epochs[~bad_epochs]
    return epochs_clean