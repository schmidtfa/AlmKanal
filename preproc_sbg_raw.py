

from obob_mne.raw import Raw as RawTemplate
import numpy as np
import mne
from raw_cleaner import raw_cleaner


class Raw(RawTemplate):
    sinuhe_root = '/home/schmidtfa/experiments/ncc/data/'
    study_acronym = 'aw_ncc'
    file_glob_patterns = ['%s_block%02d.fif',
                          '%s_block%d.fif']


    def run_cleaner(subject_id, 
                    maxfilter = True,
                    ica = True,
                    ica_threshold = 0.5,
                    notch = False,
                    downsample_f=None,
                    l_pass = 40,
                    h_pass = 0.1):

        
        n_blocks = Raw.get_number_of_runs(subject_id)
        
        #get average head pos
        block_pos_l = []
        for block in np.arange(1, n_blocks + 1):
            raw = Raw(subject_id, block_nr=block, preload=False)
            block_pos_l.append(raw.info["dev_head_t"]['trans'][:3, 3])

        blocks_pos = np.array(block_pos_l)
        all_distances = np.sqrt(blocks_pos[:,0]**2 + blocks_pos[:,1]**2 + blocks_pos[:,2]**2)
        mean_distance = np.median(all_distances)
        mean_d_idx = (np.abs(all_distances - mean_distance)).argmin()
        #TODO: maybe do np.median(np.array(block_pos_l), axis=0)
                        
        raw_all, first_samples = [], []
        for block in np.arange(1, n_blocks + 1):
            raw_tmp = Raw(subject_id, block_nr=block, preload=True)
            

        #% Salzburg specific: make sure that if channels are set as bio that they get added correctly
        if 'BIO003' in raw.ch_names:
            raw.set_channel_types({'BIO001': 'eog',
                                'BIO002': 'eog',
                                'BIO003': 'ecg',})

            mne.rename_channels(raw.info, {'BIO001': 'EOG001',
                                            'BIO002': 'EOG002',
                                            'BIO003': 'ECG003',})
                
        #TODO: Only if trans average or trans default
        eog_list = ['MEG0121','MEG0311' ,'MEG1211' ,'MEG1411']
        eog_list.extend([name for name in ['EOG001', 'EOG002'] if name in raw.ch_names])

        # Determine the destination parameter for maxwell_filter
        if trans_option == 'standard':
            destination = (0, 0, 0.05)
        elif trans_option == 'average' and input_files:
            destination = compute_average_position(input_files)
        else:
            destination = None

        raw = raw_cleaner()