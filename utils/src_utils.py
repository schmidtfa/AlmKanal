#%%imports
from os import listdir
from os.path import join
from datetime import datetime
import numpy as np
import mne
from raw_cleaner import raw_cleaner

import warnings
from mne._fiff.pick import _contains_ch_type

#%%
def get_nearest_empty_room(info, 
                            empty_room_path = '/home/schmidtfa/empty_room_data/subject_subject'):
    """
    This function finds the empty room file with the closest date to the current measurement.
    The file is used for the noise covariance estimation.
    """
    
    all_empty_room_dates = np.array([datetime.strptime(date, '%y%m%d') for date in listdir(empty_room_path)])

    cur_date = info['meas_date']
    cur_date_truncated = datetime(cur_date.year, cur_date.month, cur_date.day)  # necessary to truncate

    def _nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    while True:
        nearest_date_datetime = _nearest(all_empty_room_dates, cur_date_truncated)
        nearest_date = nearest_date_datetime.strftime("%y%m%d")

        cur_empty_path = join(empty_room_path, nearest_date)

        if 'supine' in listdir(cur_empty_path)[0]:
            all_empty_room_dates = np.delete(all_empty_room_dates,
                                                all_empty_room_dates == nearest_date_datetime)
        elif np.logical_and('68' in listdir(cur_empty_path)[0],
                            'sss' not in listdir(cur_empty_path)[0].lower()):
            break

    fname_empty_room = join(cur_empty_path, listdir(cur_empty_path)[0])

    return fname_empty_room


def raw2source(raw, 
               fwd,
               pick_dict,
               data_cov=None,
               noise_cov=None, 
               preproc_settings=None,
               empty_room_path=None,
               ):

    '''This function does source reconstruction using lcmv beamformers based on raw data.'''           

    #%Compute covariance matrices
    #data covariance based on the actual recording
    #noise covariance based on empyt rooms
    #select only meg channels from raw

    if pick_dict['meg'] not in [True, 'mag', 'grad']:
        raise ValueError('Source Projection with the AlmKanal pipeline currently only works with MEG data.')

    if pick_dict['eeg']:
        pick_dict['eeg'] = False
        warnings.warn('WARNING: Source Projection with the AlmKanal pipeline currently only works with MEG data. \
                       Removing EEG here.')
    
    picks = mne.pick_types(raw.info, **pick_dict)
    raw.pick(picks=picks)
    info = raw.info

    #check if multiple channel types are present after picking
    n_ch_types = np.sum([_contains_ch_type(raw.info, ch_type) for ch_type in ['mag', 'grad', 'eeg']])

    # compute a data covariance matrix
    if data_cov is None:
        data_cov = mne.compute_raw_covariance(raw, rank=None, method='auto')

    #if you have mixed sensor types we need a noise covariance matrix
    #per default we take this from an empty room recording
    #importantly this should be preprocessed similarly to the actual data (except for ICA)
    if np.logical_and(n_ch_types > 1, noise_cov==None):
        print('Multiple channel types and no noise covariance matrix detected. \
               Trying to build noise covariance matrix using empty room recordings.')
        if preproc_settings is None:
            raise ValueError('You need to supply your preprocessing settings, if you are doing beamforming \
                       using a combination of different sensor types. Either you pick "mag" or "grad" or you supply your preprocessing settings.')
            
        if empty_room_path is None:
            fname_empty_room = get_nearest_empty_room(info, empty_room_path)
        else:
            fname_empty_room = empty_room_path
        empty_room = raw_cleaner(fname_empty_room, **preproc_settings)
        empty_room.pick(picks=picks)
        noise_cov = mne.compute_raw_covariance(empty_room, rank=None, method='auto')
        # when using noise cov rank should be based on noise cov
        true_rank = mne.compute_rank(noise_cov, info=empty_room.info)  # inferring true rank
    elif n_ch_types == 1:
        # when we dont have a noise cov we just use the data cov for rank comp
        true_rank = mne.compute_rank(data_cov, info=info)

    #build and apply filters
    filters = mne.beamformer.make_lcmv(info, 
                                       fwd, 
                                       data_cov, 
                                       reg=0.05,
                                       noise_cov=noise_cov, 
                                       pick_ori='max-power',
                                       weight_norm='nai', 
                                       rank=true_rank)
        
    stc = mne.beamformer.apply_lcmv_raw(raw, filters)

    return stc