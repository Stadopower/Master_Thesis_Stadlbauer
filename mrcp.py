from utils import *
def main():
    print("Currently Running the MRCP")
    drop_list = ['P003','P004','P007','P013', 'P022', 'P033','P029']
    participants = [f'P{str(i).zfill(3)}' for i in range(1, 37)]
    participants = [p for p in participants if p not in drop_list]
    marker_ids = [1009, 1029]
    # Walking, Omnideck, Leaning, Joystick
    cond_names = ['Walking', 'Omnideck', 'Leaning', 'Joystick']
    grand_average_1009 = [[],[],[],[]]
    grand_average_first_1009 = [[],[],[],[]]
    grand_average_1029 = [[],[],[],[]]
    plot = True
    plot_individual_trials = True
    for p in participants:
        path = f'complete_study\\epochs\\{p}\\'
        base_fig_dir = f'complete_study\\figures\\{p}\\mrcp'
        for marker in marker_ids:
            # Creating Folders to store single trials
            os.makedirs(base_fig_dir, exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'walking'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'omnideck'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'leaning'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'joystick'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'walking_first'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'omnideck_first'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'leaning_first'), exist_ok=True)
            os.makedirs(os.path.join(base_fig_dir+f'\\{marker}', 'joystick_first'), exist_ok=True)
            # Walking, Omnideck, Leaning, Joystick
            evoked_lists = [[],[],[],[]]
            evoked_list_first = [[],[],[],[]]
            for filename in sorted(os.listdir(path)):
                os.makedirs(os.path.join(base_fig_dir, str(marker)), exist_ok=True)
                os.makedirs(os.path.join(base_fig_dir, f'{marker}\\ga'), exist_ok=True)
                if str(marker) in filename:
                    match = re.search(r'sub-[^-]+-([^-_]+)_epochs_([^-_]+)', filename, re.IGNORECASE)
                    sub_cond = re.search(r'(sub-[^_]+_ses-[^_\\]+)', filename)[0]
                    str_addon = ''
                    if 'first' in filename:
                        str_addon = 'first'
                    print('Working on:', filename)
                    epochs = mne.read_epochs(path+filename)
                    epochs = epochs.pick(['F3', 'Fz', 'F4', 'FC1', 'FCz', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4'])
                    iir_params = dict(order=2, ftype='butter')
                    epochs = epochs.filter(l_freq =0.1 ,h_freq=1, method='iir', iir_params=iir_params, phase='zero')


                    if plot_individual_trials and 'first' in filename:
                        # =============== Single Trial Plotting =============== #
                        # Trial index here does not take into account the actual number of the trail but only the pos after rejecting bad trials
                        for trial_index, epoch in enumerate(epochs):
                            single_evoked = mrcp(epochs[trial_index])
                            fig = mne.viz.plot_compare_evokeds(single_evoked, ci=True, axes='topo', ylim=dict(eeg=[-12, 12]), show=False);
                            epochs_original_index = epochs.selection
                            if 'first' in filename:
                                if 'Walking' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\walking_first\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                elif 'Omnideck' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\omnideck_first\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                elif 'Leaning' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\leaning_first\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                else:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\joystick_first\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                            else:
                                if 'Walking' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\walking\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                elif 'Omnideck' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\omnideck\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                elif 'Leaning' in filename:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\leaning\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])
                                else:
                                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\joystick\\{sub_cond}_trial_{epochs_original_index[trial_index]}.png')
                                    plt.close(fig[0])



                    # =============== Saving Evokeds to average later =============== #
                    epochs = epochs.filter(l_freq =0.1 ,h_freq=1, method='iir', iir_params=iir_params, phase='zero')
                    if not 'first' in filename:
                        if 'Walking' in filename:
                            evoked_lists[0].append(epochs)
                        elif 'Omnideck' in filename:
                            evoked_lists[1].append(epochs)
                        elif 'Leaning' in filename:
                            evoked_lists[2].append(epochs)
                        else:
                            evoked_lists[3].append(epochs)
                    else:
                        if 'Walking' in filename:
                            evoked_list_first[0].append(epochs)
                        elif 'Omnideck' in filename:
                            evoked_list_first[1].append(epochs)
                        elif 'Leaning' in filename:
                            evoked_list_first[2].append(epochs)
                        else:
                            evoked_list_first[3].append(epochs)


                    if plot:
                        # =============== Plotting =============== #
                        fig = mne.viz.plot_compare_evokeds({'Test' : list(epochs.iter_evoked())}, ci=True, axes='topo', ylim=dict(eeg=[-12, 12]),show=False);
                        fig[0].savefig(f'{base_fig_dir}\\{marker}\\{sub_cond}_{str_addon}_topo_mrcp.png') #
                        plt.close(fig[0])
                        #fig, ax = plt.subplots(figsize=(10, 6))
                        #evoked.plot(titles=f'{marker}_evoked', ylim=dict(eeg=[-12, 12]), show=False, axes=ax)
                        #ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        #fig.savefig(f'{base_fig_dir}\\{marker}\\{sub_cond}_{marker}_{str_addon}_interpolated_ch_mrcp.png')
                        #plt.close(fig)

                        # =============== Loop through individual channels =============== #
                        #for ch in evoked.info['ch_names']:
                        #    fig, ax = plt.subplots(figsize=(10, 6))
                        #    evoked.plot(titles=f'{sub_cond}_{ch}_evoked', picks=[ch], ylim=dict(eeg=[-12, 12]), axes=ax, show=False);
                        #    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        #    # Create the directory if it doesn't exist
                        #    if not os.path.exists(f'{base_fig_dir}\\{marker}\\indiv_ch\\'):
                        #        os.makedirs(f'{base_fig_dir}\\{marker}\\indiv_ch\\')
                        #    fig.savefig(f'{base_fig_dir}\\{marker}\\indiv_ch\\{sub_cond}_{marker}_{str_addon}_{ch}_interpolated_ch_mrcp.png')
                        #    plt.close(fig)

            # =============== Cond Averages ================ #
            for cond_nr, ev in enumerate(evoked_lists):
                if not ev or len(ev) < 2:
                    continue
                ev = mne.concatenate_epochs([ev[0],ev[1]])
                if marker == 1009:
                    grand_average_1009[cond_nr].append(ev)
                else:
                    grand_average_1029[cond_nr].append(ev)
                fig = mne.viz.plot_compare_evokeds({f'{marker}' : list(ev.iter_evoked())}, ci=True, axes='topo', ylim=dict(eeg=[-12, 12]),show=False);
                fig[0].savefig(f'{base_fig_dir}\\{marker}\\ga\\{cond_names[cond_nr]}_ga_topo_mrcp.png')
                plt.close(fig[0])
                #fig = ga.plot(titles=f'{cond_names[cond_nr]}_{marker}_evoked_average', ylim=dict(eeg=[-6, 6]), show=False)
                #fig.savefig(f'{base_fig_dir}\\{marker}\\ga\\{cond_names[cond_nr]}_{marker}_mrcp_average_interpolated.png')
                #plt.close(fig)

            if 'first' in filename:
                # Condition wise GA for only the first 1009 marker
                for cond_nr, ev in enumerate(evoked_list_first):
                    if not ev or len(ev) < 2:
                        continue
                    ev = mne.concatenate_epochs([ev[0],ev[1]])
                    grand_average_first_1009[cond_nr].append(ev)
                    fig = mne.viz.plot_compare_evokeds({f'{marker}' : list(ev.iter_evoked())}, ci=True, axes='topo', ylim=dict(eeg=[-12, 12]),show=False);
                    fig[0].savefig(f'{base_fig_dir}\\{marker}\\ga\\{cond_names[cond_nr]}_first_ga_topo_mrcp.png')
                    plt.close(fig[0])
                    #fig = ga.plot(titles=f'{cond_names[cond_nr]}_{marker}_evoked_average', ylim=dict(eeg=[-6, 6]), show=False)
                    #fig.savefig(f'{base_fig_dir}\\{marker}\\ga\\{cond_names[cond_nr]}_{marker}_first_mrcp_average_interpolated.png')
                    #plt.close(fig)

    # ================ Grand averages over all participants ================ #
    for cond_nr, ev in enumerate(grand_average_1009):
        ev = mne.concatenate_epochs(ev)
        fig = mne.viz.plot_compare_evokeds({f'{marker}' : list(ev.iter_evoked())}, ci=.95, axes='topo', ylim=dict(eeg=[-12, 12]), vlines=[-2,0],show=False)
        for ax in fig[0].axes:
            for text in ax.texts:
                text.set_fontsize(20)  # Set desired font size
        fig[0].savefig(f'complete_study\\figures\\{cond_names[cond_nr]}_1009_GA_all_participants_topo_mrcp.png')
        plt.close(fig[0])
        #fig = ga.plot(titles=f'{cond_names[cond_nr]}_evoked_average', ylim=dict(eeg=[-6, 6]), show=False)
        #fig.savefig(f'figures\\meeting\\{cond_names[cond_nr]}_1009_GA_all_participants_mrcp.png')
        #plt.close(fig)

    for cond_nr, ev in enumerate(grand_average_1029):
        ev = mne.concatenate_epochs(ev)
        fig = mne.viz.plot_compare_evokeds({f'{marker}' : list(ev.iter_evoked())}, ci=.95, axes='topo', ylim=dict(eeg=[-12, 12]), vlines=[-1, 1.0] ,show=False)
        for ax in fig[0].axes:
            for text in ax.texts:
                text.set_fontsize(20)  # Set desired font size
        fig[0].savefig(f'complete_study\\figures\\{cond_names[cond_nr]}_1029_GA_all_participants_topo_mrcp.png')
        plt.close(fig[0])
        #fig = ga.plot(titles=f'{cond_names[cond_nr]}_evoked_average', ylim=dict(eeg=[-6, 6]), show=False)
        #fig.savefig(f'figures\\meeting\\{cond_names[cond_nr]}_1029_GA_all_participants_mrcp.png')
        #plt.close(fig)

    for cond_nr, ev in enumerate(grand_average_first_1009):
        ev = mne.concatenate_epochs(ev)
        fig = mne.viz.plot_compare_evokeds({f'{marker}' : list(ev.iter_evoked())}, ci=.95, axes='topo', ylim=dict(eeg=[-12, 12]), vlines=[-2,0],show=False)
        for ax in fig[0].axes:
            for text in ax.texts:
                text.set_fontsize(20)  # Set desired font size
        fig[0].savefig(f'complete_study\\figures\\{cond_names[cond_nr]}_first_1009_GA_all_participants_topo_mrcp.png')
        plt.close(fig[0])
        #fig = ga.plot(titles=f'{cond_names[cond_nr]}_evoked_average', ylim=dict(eeg=[-6, 6]), show=False)
        #fig.savefig(f'figures\\meeting\\{cond_names[cond_nr]}_first_1009_GA_all_participants_mrcp.png')
        #plt.close(fig)

if __name__ == '__main__':
    mne.set_log_level('warning')
    os.chdir('D:\\RUG\\Master_Thesis\\Master_Thesis_Stadlbauer')
    main()
    print('MRCP Done')