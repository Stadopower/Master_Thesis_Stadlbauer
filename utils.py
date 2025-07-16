# some function to make the end result more readable:
#filename has to be the entire path to the file so "data//P101//filename"
import numpy as np
import matplotlib.pyplot as plt
import mne
import pyxdf

def get_data(filename):
    print('Reading data')
    streams, fileheader = pyxdf.load_xdf(filename, synchronize_clocks=False)
    marker_stream = next(s for s in streams if 'LSL4Unity.OmnideckWaiterVR' in s['info']['name'][0])
    eeg_stream = next(s for s in streams if "EEG" in s['info']['type'][0])
    eeg_data = np.array(eeg_stream['time_series']).T
    #eeg_timestamps = np.array(eeg_stream['time_stamps'])
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    ch_names = []
    for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']:
        ch_names.append(ch['label'][0])
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(eeg_data/10e5, info)

    # TODO: Is it a 10-20 montage?
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    event_samples = (marker_stream['time_stamps'] - eeg_stream['time_stamps'][0])*sfreq
    event_samples = event_samples.astype(int)
    event_labels = [int(marker[0]) for marker in marker_stream['time_series']]

    annotations = mne.Annotations(onset=event_samples / sfreq,
                                  duration=[0] * len(event_samples),  # Instantaneous events
                                  description=list(event_labels))
    raw.set_annotations(annotations)

    return raw, marker_stream, event_samples

#apply a notch filter at 50hz and filter between 0.01 and 30hz
def filter_data(raw):
    raw = raw.notch_filter(50, method='fir', phase='zero',verbose=False)
    iir_params = dict(order=2, ftype='butter',verbose=False)
    raw = raw.filter(l_freq=0.01, h_freq=40, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    return raw


# making epochs around a specified marker
def get_epochs(raw: mne.io.Raw, marker_stream, event_samples , marker_id:int, tmin:int, tmax:int, preload=False, ica=True):
    if ica:
        iir_params = dict(order=2, ftype='butter',verbose=False)
        raw = raw.copy()
        raw = raw.filter(l_freq=1, h_freq=None, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    event_labels = [int(marker[0]) for marker in marker_stream['time_series']]
    events = np.array([[sample, 0, label] for sample, label in zip(event_samples, event_labels)])
    selected_events = events[events[:, 2] == marker_id]
    epochs = mne.Epochs(raw, np.array(selected_events), event_id=int(marker_id),baseline=None, tmin=tmin, tmax=tmax, reject_by_annotation=False, verbose=False, preload=preload)
    return epochs

# a function to plot and save the beginning 30 sec of a epoch. To quickly check the data if it looks alright
def count_markers(marker_stream, match = 0):
    flat_list = [int(marker[0]) for marker in marker_stream['time_series']]
    marker_count = {}
    for marker in flat_list:
        if marker in marker_count:
            marker_count[marker] += 1
        else:
            marker_count[marker] = 1
    marker_count = dict(sorted(marker_count.items()))
    if match != 0:
        with open(f'markers\\{match}_markers.txt', 'w') as file:
            for key, value in marker_count.items():
                file.write(f"{key}: {value}\n")
    #return marker_count

# for ica data should be highpassed filtered at 1hz
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

# it returns the ica and rej channels so that it can be used to apply it to the epochs that are not highpass filtered at 1Hz
def get_ica(epochs: mne.Epochs, n_components=32, plot=False, save_path=None, match=None):
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter= 1500, method='fastica', verbose=False)
    ica.fit(epochs, verbose=False)
    std = np.std(ica.get_sources(epochs).get_data())
    rej_ch = []
    for i, channel in enumerate(ica.get_sources(epochs).get_data()[0]):
        if np.max(channel) > 4*std:
            rej_ch.append(i)
    if plot and len(rej_ch) != 0:
        #print(len(rej_ch))
        with open(f'figures\\meeting\\ica\\{match}.txt', 'w') as file:
            file.write(", ".join(map(str, rej_ch)))
            file.write(f"\n{len(rej_ch)} rejected channels")
            file.close()
        fig = ica.plot_components(title=f'{match} all ICA components', show=False);
        for j, f in enumerate(fig):
            f.savefig(save_path+f'_{j}_all_ica.png')
            plt.close(f)
        fig = ica.plot_components(picks=rej_ch,title=f'{match} rejected channels', show=False);
        fig.savefig(save_path+'rejected_ica.png')
        plt.close(fig)
    return ica, rej_ch
    #ica.apply(epochs, exclude=rej_ch, verbose=False)


# load data, filter, epoch, ICA, average, plot
def mrcp(epochs):
    frontal_channels = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2']
    iir_params = dict(order=2, ftype='butter')
    epochs = epochs.filter(l_freq =0.1 ,h_freq=1, method='iir', iir_params=iir_params, phase='zero')
    evoked = epochs.average()
    evoked = evoked.pick(frontal_channels)
    return evoked


def drop_bad_epochs(epochs):
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    std_per_channel = np.std(epoch_data, axis=(0, 2))  # Standard deviation per channel
    threshold = 4 * std_per_channel[:, np.newaxis]  # Expand dims to match shape
    # Find epochs where any channel exceeds the threshold
    bad_epochs = np.any(np.abs(epoch_data) > threshold[np.newaxis, :, :], axis=(1, 2))

    # Drop bad epochs
    epochs_clean = epochs[~bad_epochs]
    return epochs_clean