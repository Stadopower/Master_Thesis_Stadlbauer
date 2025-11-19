import os
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
from utils import *

# Data-Preprocessing, extracting epochs and saving
# Filter Line-Noise, Find-noisy channels, interpolate channels, re-reference, filter target frequencies, create ica for this apply highpass at 1hz, save this, epoch data, save again
# Participant 22 and 33 had an error
def main():
    print("Currently pre-processing data")
    baseline_marker = 1000
    target_markers = [1009, 1029]     #1009 = Handgrabbedbottle, 1029 Departing starts
    iir_params = dict(order=2, ftype='butter',verbose=False)
    drop_list = ['P003','P004','P007','P013','P029', 'P033']
    participants = [f'P{str(i).zfill(3)}' for i in range(34, 37)]
    participants = [p for p in participants if p not in drop_list]
    plot = False
    for p in participants:
        path = f'data\\{p}\\'

        #### Creating folders to save ####
        base_fig_dir = f'complete_study\\figures\\{p}'
        os.makedirs('complete_study\\figures', exist_ok=True)
        os.makedirs(base_fig_dir, exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'psd'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'epochs'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'ica'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'ica', 'baseline'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'ERDS'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'topomap'), exist_ok=True)
        os.makedirs(os.path.join(base_fig_dir, 'mrcp'), exist_ok=True)
        os.makedirs('complete_study\\epochs', exist_ok=True)
        os.makedirs(os.path.join('complete_study\\epochs', p), exist_ok=True)

        for condition in os.listdir(path):
            condition_path = os.path.join(path, condition)
            eeg_path = os.path.join(condition_path, 'eeg')
            if os.path.isdir(eeg_path):
                for filename in os.listdir(eeg_path):
                    # Excluding not working files P003 Omnideck2, P007 joystick (leaning does not exist)
                    files_to_exclude = ['sub-P003_ses-Omnideck2_task-Default_run-001_eeg.xdf', 'sub-P007_ses-Joystick1_task-Default_run-001_eeg.xdf']
                    if filename in files_to_exclude:
                        continue

                    item_path = os.path.join(eeg_path, filename)
                    if filename.endswith('.xdf') and os.path.isfile(item_path):
                        print(f"Working on File: {filename}")
                        match = re.search(r'(sub-[^_]+_ses-[^_\\]+)', filename)[0]
                        raw, marker_stream, event_samples = get_data(item_path)
                        raw = filter_data(raw)

                        # Find Noisy channels using Perp pipeline
                        noisy = NoisyChannels(raw, do_detrend=True)
                        noisy.find_all_bads(ransac=True)
                        # Save how many channels will be interpolated to adjust for ICA
                        nr_bads = len(noisy.get_bads())
                        raw.info['bads'].extend(noisy.get_bads())
                        raw.interpolate_bads(reset_bads=True)
                        with open(f'{base_fig_dir}\\{match}_rejecton_info.txt', 'w') as f:
                            f.write(f'Nr of bad channels: {nr_bads}\n')
                            f.write(f'Bad channels: {noisy.get_bads()}\n')

                        ### Plotting psd ###
                        fig = raw.plot_psd(average=True, fmax=50, show=False)
                        fig.savefig(f'{base_fig_dir}\\psd\\{match}_raw_psd.png')
                        plt.close(fig)

                        # re-reference to average
                        raw.set_eeg_reference('average', projection=False, ch_type='eeg')

                        # Epochs for ICA
                        baseline_epochs = get_epochs(raw, marker_stream, event_samples, marker_id=baseline_marker, tmin=0, tmax=60, ica=True, preload=True)
                        ica, rej_ch = get_icalabel(baseline_epochs.load_data(),n_components=31-nr_bads, plot=plot, save_path=f'{base_fig_dir}\\ica\\baseline\\', match = match)
                        ica.apply(baseline_epochs, exclude=rej_ch, verbose=False)
                        suffix = 'before'

                        # Ploting baseline PSD
                        if plot:
                            for i, base in enumerate(baseline_epochs):
                                if i != 0:
                                    suffix = 'after'
                                fig = baseline_epochs[i].plot_psd(average=True, fmax=50)
                                fig.savefig(f'{base_fig_dir}\\psd\\{match}_{suffix}_baseline_psd.png')
                                plt.close(fig)

                        # Filtering Baseline Epoch and Saving
                        baseline_epochs = baseline_epochs.filter(l_freq=0.01, h_freq=40, method='iir', iir_params=iir_params, phase='zero', verbose=False)
                        baseline_epochs.save(f'complete_study\\epochs\\{p}\\{match}_epochs_baseline_epo.fif', overwrite=True)

                        ### Loop for Target Markers ###
                        for target in target_markers:

                            # First load epochs with ICA=True to remove slow drifts and then apply to non high pass filtered data so it can be used for MRCP
                            target_epochs = get_epochs(raw, marker_stream, event_samples, marker_id=target, tmin=-6, tmax=4, ica=True, preload=True)
                            nr_epochs = len(target_epochs)

                            ### Epoch Rejection ###
                            ar = AutoReject(n_interpolate=np.array([1,2,3,4]), n_jobs=-1, verbose=True)
                            ar.fit(target_epochs)
                            target_epochs, log = ar.transform(target_epochs, return_log=True)
                            print(f'{target} | {nr_epochs-len(target_epochs)} epochs out of {nr_epochs} removed (filtered)')

                            ### ICA ###
                            ica, rej_ch = get_icalabel(target_epochs.load_data(), n_components=31-nr_bads, plot=plot, save_path=f'{base_fig_dir}\\ica\\', match = f'{match}_{target}')
                            target_epochs = get_epochs(raw, marker_stream, event_samples, marker_id=target, tmin=-6, tmax=4, ica=False, preload=True)

                            ### Removing the Epochs found earlier and Applying ICA ###
                            print('Removing bad Epochs from filtered Data')
                            nr_epochs = len(target_epochs)
                            target_epochs.drop(log.bad_epochs)
                            print(f'{target} | {nr_epochs-len(target_epochs)} epochs out of {nr_epochs} removed')
                            ica.apply(target_epochs, exclude=rej_ch, verbose=False)

                            # Plotting PSD
                            if plot:
                                fig = target_epochs.plot_psd(average=True, fmax=50)
                                fig.savefig(f'{base_fig_dir}\\psd\\{match}_{target}_psd.png')
                                plt.close(fig)
                                # Plotting Epochs
                                fig = target_epochs.plot(scalings=5e-5)
                                fig.savefig(f'{base_fig_dir}\\epochs\\{match}_{target}_epochs.png')
                                plt.close(fig)
                            target_epochs = target_epochs.filter(l_freq=0.01,h_freq=40, method='iir', iir_params=iir_params, phase='zero', verbose=False)
                            target_epochs.save(f'complete_study\\epochs\\{p}\\{match}_epochs_{target}_epo.fif', overwrite=True)

                            # Extracting first occurence of 1009
                            # This is done by finding the first 1009 marker after a 1025 marker (approaching starts)
                            if target == 1009:
                                events, event_id = mne.events_from_annotations(raw)
                                event_1025 = events[events[:, 2] == event_id['1025']]
                                event_1009 = events[events[:, 2] == event_id['1009']]
                                # These [np.array(marker_onset), ...]
                                first_1009_after_1025 = []

                                # Extracting first 1009
                                for i in range(len(event_1025)):
                                    start = event_1025[i, 0]

                                    # Define search end: either next 1025 or end of data
                                    if i + 1 < len(event_1025):
                                        stop = event_1025[i + 1, 0]
                                    else:
                                        stop = np.inf
                                    # Filter 1009s between this 1025 and the next
                                    candidates = event_1009[(event_1009[:, 0] > start) & (event_1009[:, 0] < stop)]

                                    if len(candidates) > 0:
                                        first_1009_after_1025.append(candidates[0][0])  # Only the first one

                                mask = np.isin(target_epochs.events[:, 0], first_1009_after_1025)
                                first_1009_epochs = target_epochs[mask]
                                first_1009_epochs = first_1009_epochs.filter(l_freq=0.01, h_freq=40, method='iir', iir_params=iir_params, phase='zero', verbose=False)
                                first_1009_epochs.save(f'complete_study\\epochs\\{p}\\{match}_epochs_first_1009_epo.fif', overwrite=True)
    print("Data pre-processing complete")


if __name__ == '__main__':
    os.chdir('D:\\RUG\\Master_Thesis\\Master_Thesis_Stadlbauer')
    mne.set_log_level('warning')
    main()