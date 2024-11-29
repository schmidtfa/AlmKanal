#%%imports
import os
from datetime import datetime
import numpy as np
import mne
from pathlib import Path

import warnings
from mne._fiff.pick import _contains_ch_type

from preproc_utils.maxwell_utils import run_maxwell



#%%
def get_nearest_empty_room(info, 
                            empty_room_path = '/home/schmidtfa/empty_room_data/subject_subject'):
    """
    This function finds the empty room file with the closest date to the current measurement.
    The file is used for the noise covariance estimation.
    """
    
    all_empty_room_dates = np.array([datetime.strptime(date, '%y%m%d') for date in os.listdir(empty_room_path)])

    cur_date = info['meas_date']
    cur_date_truncated = datetime(cur_date.year, cur_date.month, cur_date.day)  # necessary to truncate

    def _nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    while True:
        nearest_date_datetime = _nearest(all_empty_room_dates, cur_date_truncated)
        nearest_date = nearest_date_datetime.strftime("%y%m%d")

        cur_empty_path = os.path.join(empty_room_path, nearest_date)

        if 'supine' in os.listdir(cur_empty_path)[0]:
            all_empty_room_dates = np.delete(all_empty_room_dates,
                                                all_empty_room_dates == nearest_date_datetime)
        elif np.logical_and('68' in os.listdir(cur_empty_path)[0],
                            'sss' not in os.listdir(cur_empty_path)[0].lower()):
            break

    fname_empty_room = os.path.join(cur_empty_path, os.listdir(cur_empty_path)[0])

    return fname_empty_room


def process_empty_room(data,
                       info,
                       pick_dict,
                       icas,
                       ica_ids,
                       empty_room_path,
                       preproc_info):
        
        # print('Multiple channel types and no noise covariance matrix detected. \
        #        Trying to build noise covariance matrix using empty room recordings.')
        # # if preproc_info is None:
        # #     raise ValueError('You need to supply your preprocessing settings, if you are doing beamforming \
        # #                using a combination of different sensor types. Either you pick "mag" or "grad" or you supply your preprocessing settings.')
            
        if empty_room_path is None:
            fname_empty_room = get_nearest_empty_room(info, empty_room_path)
        else:
            fname_empty_room = empty_room_path

        raw_er = mne.io.read_raw(fname_empty_room, preload=True)
        
        if preproc_info['maxwell'] is not None:
            if isinstance(data, mne.epochs.Epochs):
                raw = mne.io.RawArray(np.empty([len(data.info.ch_names), 100]), info=data.info)
            elif isinstance(data, mne.io.fiff.raw.Raw):
                raw = data

            raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(raw_er=raw_er, raw=raw)
            raw_er = run_maxwell(raw_er, **preproc_info['maxwell'])

        #Add filtering here -> i.e. check if deviation between empty and real data and then filter
        if np.logical_and(np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) == False,  
                          (np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01),) == False):
            raw_er.filter(l_freq=data.info['highpass'], h_freq=data.info['lowpass'])
        elif np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) == False:
            raw_er.filter(l_freq=data.info['highpass'], h_freq=None,)
        elif np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01) == False:
            raw_er.filter(l_freq=None, h_freq=data.info['lowpass'])
        else:
            print('No filtering applied')

        #TODO: Also make sure that the sampling rate is the same
        if np.isclose(data.info['sfreq'], raw_er.info['sfreq'], atol=.9) == False:
            #adjust for small floating point differences
            raw_er.resample(data.info['sfreq'])

        if preproc_info['ica'] is not None:
            #we loop here, because you could have done more than one ica
            for ica, ica_id in zip(icas, ica_ids):
                ica.apply(raw_er, exclude=ica_id)

        picks = mne.pick_types(raw_er.info, **pick_dict)
        raw_er.pick(picks=picks)
        if isinstance(data, mne.io.fiff.raw.Raw):
            noise_cov = mne.compute_raw_covariance(raw_er, rank=None, method='auto')

        elif isinstance(data, mne.epochs.Epochs):

            t_length = np.abs(data.epoched.tmax - data.epoched.tmin)
            raw_er = mne.make_fixed_length_epochs(raw_er, duration=t_length)
            noise_cov = mne.compute_covariance(raw_er, rank=None, method='auto')
        # when using noise cov rank should be based on noise cov
        true_rank = mne.compute_rank(noise_cov, info=raw_er.info)  # inferring true rank

        return true_rank, noise_cov

def data2source(data, 
               fwd,
               pick_dict,
               icas=None,
               ica_ids=None,
               data_cov=None,
               noise_cov=None, 
               preproc_info=None,
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

    picks = mne.pick_types(data.info, **pick_dict)
    data.pick(picks=picks)
    info = data.info

    #check if multiple channel types are present after picking
    n_ch_types = np.sum([_contains_ch_type(data.info, ch_type) for ch_type in ['mag', 'grad', 'eeg']])

    # compute a data covariance matrix
    if np.logical_and(data_cov is None, isinstance(data, mne.io.fiff.raw.Raw)):
        data_cov = mne.compute_raw_covariance(data, rank=None, method='auto')
    elif np.logical_and(data_cov is None, isinstance(data, mne.epochs.Epochs)):
        data_cov = mne.compute_covariance(data, rank=None, method='auto')
    else:
        print('Data covariance matrix not computed as it was supplied by the analyst.')

    #if you have mixed sensor types we need a noise covariance matrix
    #per default we take this from an empty room recording
    #importantly this should be preprocessed similarly to the actual data (except for ICA)
    if np.logical_and(n_ch_types > 1, noise_cov==None):

        true_rank, noise_cov = process_empty_room(data,
                                                    info,
                                                    pick_dict,
                                                    icas,
                                                    ica_ids,
                                                    empty_room_path,
                                                    preproc_info)

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
    

    if isinstance(data, mne.io.fiff.raw.Raw):

        stc = mne.beamformer.apply_lcmv_raw(data, filters)

    elif isinstance(data, mne.epochs.Epochs):
        
        stc = mne.beamformer.apply_lcmv_epochs(data, filters)

    return stc, filters



def src2parc(stc,
             subject_id,
             subjects_dir,
             atlas='glasser',
             source='surface'):
        
        if atlas == 'dk':
             vol_atlas = 'aparc+aseg'
             surf_atlas = 'aparc'
        elif atlas == 'destrieux':
             vol_atlas = 'aparc.a2009s+aseg'
             surf_atlas = 'aparc.a2009s'
        elif atlas == 'glasser':
             if source == 'volume':
                ValueError('No volumetric model for the glasser atlas available')
             surf_atlas = 'HCPMMP1'

        fs_dir = Path(os.path.join(subjects_dir, 'freesurfer'))
        # mean flip time series costs significantly less memory than averaging the irasa'd spectra
        if source == 'surface':

            src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-ico-4-src.fif'
            src = mne.read_source_spaces(src_file)
            labels_mne = mne.read_labels_from_annot(f'{subject_id}_from_template', 
                                                    parc=surf_atlas, 
                                                    subjects_dir=fs_dir)
            names_order_mne = np.array([label.name[:-3] for label in labels_mne])

            rh = [True if label.hemi == 'rh' else False for label in labels_mne]
            lh = [True if label.hemi == 'lh' else False for label in labels_mne]

            parc = {'lh': lh,
                            'rh': rh,
                            'parc': surf_atlas,
                            'names_order_mne': names_order_mne}
            parc.update({'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode='mean_flip')})
        elif source == 'volume':

            src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-vol-10-src.fif'
            src = mne.read_source_spaces(src_file)
            labels_mne = os.path.join(fs_dir, f'{subject_id}_from_template', 'mri/' + vol_atlas + '.mgz')
            label_names = mne.get_volume_labels_from_aseg(labels_mne)

            ctx_logical = [True if 'ctx' in label else False for label in label_names]
            sctx_logical = [True if f == False else False for f in ctx_logical]
            
            ctx_labels = [label[4:] for label in label_names if 'ctx' in label]
            sctx_labels = list(np.array(label_names)[sctx_logical])
            rh = [True if label[:2] == 'rh' else False for label in ctx_labels]
            lh = [True if label[:2] == 'lh' else False for label in ctx_labels]

            parc = {'lh': lh,
                    'rh': rh,
                    'parc': vol_atlas + '.mgz',
                    'ctx_labels': ctx_labels,
                    'ctx_logical': ctx_logical,
                    'sctx_logical': sctx_logical,
                    'sctx_labels': sctx_labels}
            parc.update({'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode='auto')})

        else:
            raise ValueError('the only valid options for source are `surface` and `volume`.')

        return parc