import mne
import numpy as np
from datetime import datetime
from os import listdir
from os.path import join

#from rm_train import rm_train_ica

def raw_cleaner(cur_data, 
                 l_pass=None,
                 h_pass=None,
                 #ica part  
                 ica=False,
                 ic_eog=True,
                 ic_ecg=True,
                 ic_muscle=False,
                 ic_train=False,
                 ic_train_freq=16.6,
                 ic_threshold=0.5,
                 #maxwell part
                 mw=True,
                 mw_coord_frame='head',
                 mw_destination=None,
                 mw_calibration_file = './template_files/maxwell/sss_cal_sbg.dat',
                 mw_cross_talk_file = './template_files/maxwell/ct_sparse_sbg.fif',
                 mw_st_duration=None,):
    

    #maxfilter
    if mw:
        from preproc_utils.maxwell_utils import run_maxwell
        maxwell_kwargs = {'coord_frame': mw_coord_frame,
                          'destination': mw_destination,
                          'calibration_file': mw_calibration_file,
                          'cross_talk_file': mw_cross_talk_file,
                          'st_duration': mw_st_duration}

        # find bad channels first
        raw = run_maxwell(raw, **maxwell_kwargs)

    if np.logical_or(l_pass != None, h_pass != None):
        raw.filter(l_freq=h_pass, h_freq=l_pass)

    #ica
    if ica:
        from preproc_utils.ica_utils import run_ica
        ica_kwargs = {'eog': ic_eog,
                      'ecg': ic_ecg,
                      'muscle':ic_muscle,
                      'train':ic_train,
                      'train_freq': ic_train_freq,
                      'ica_corr_thresh': ic_threshold}
        
        raw = run_ica(raw, ica_kwargs)


    return raw