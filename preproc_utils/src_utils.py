#%%imports
from os import listdir
from os.path import join
from datetime import datetime
import numpy as np
import mne
import joblib
from pathlib import Path
from raw_cleaner import raw_cleaner
import matplotlib as mpl

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


def make_fwd(info, source, trans_path, subjects_dir, subject_id, template_mri):
    
    ###### MAKE FORWARD SOLUTION AND INVERSE OPERATOR
    if template_mri:
        fpath_add_on = '_from_template'
    else:
        fpath_add_on = ''

    fs_path = join(subjects_dir, f'{subject_id}{fpath_add_on}')
    bem_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-5120-5120-5120-bem.fif'

    if source == 'volume':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-vol-10-src.fif'
        
    elif source == 'surface':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-ico-4-src.fif'
    
    fname_trans = join(trans_path, subject_id, subject_id + '-trans.fif')

    bem_sol = mne.make_bem_solution(bem_file, 
                                    solver='mne', 
                                    verbose=True) 
    fwd = mne.make_forward_solution(info=info, 
                                    trans=fname_trans, 
                                    src=src_file, 
                                    bem=bem_sol)

    return fwd


def raw2source(raw, 
               subject_id, 
               subjects_dir,
               preproc_settings,
               picks='meg', 
               source='surface',
               template_mri=True,
               trans_path = '/home/schmidtfa/experiments/brain_age/data/data_cam_can/headmodels/',
               nearest_empty = True,
               empty_room_path='/home/schmidtfa/empty_room_data/subject_subject',
               
               ):

    '''This function does source reconstruction using lcmv beamformers based on raw data.'''           

    #%Compute covariance matrices
    #data covariance based on the actual recording
    #noise covariance based on empyt rooms
    #select only meg channels from raw
    raw.pick(picks=[picks])
    info = raw.info

    if nearest_empty:
        fname_empty_room = get_nearest_empty_room(info, empty_room_path)
    else:
        fname_empty_room = empty_room_path

    empty_room = raw_cleaner(fname_empty_room, **preproc_settings)

    if picks == 'meg':
        noise_cov = mne.compute_raw_covariance(empty_room, rank=None, picks=picks, method='auto')
        # when using noise cov rank should be based on noise cov
        true_rank = mne.compute_rank(noise_cov, info=empty_room.info)  # inferring true rank

    data_cov = mne.compute_raw_covariance(raw, rank=None, picks=picks, method='auto')

    fwd = make_fwd(info, source, trans_path, subjects_dir, subject_id, template_mri)

    filters = mne.beamformer.make_lcmv(info, fwd, data_cov, 
                                           reg=0.05,
                                           noise_cov=noise_cov, 
                                           pick_ori='max-power',
                                           weight_norm='nai', 
                                           rank=true_rank)
        
    stc = mne.beamformer.apply_lcmv_raw(raw, filters)

    return stc



def plot_parc(stc_parc, stc_mask, labels_mne, 
                subjects_dir, cmap, clevels, plot_kwargs, 
                parc='HCPMMP1'):

    #mpl.use('Qt5Agg')

    labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

    names_order_mne = np.array([label.name[:-3] for label in labels_mne])

    rh = [True if label.hemi == 'rh' else False for label in labels_mne]
    lh = [True if label.hemi == 'lh' else False for label in labels_mne]

    import nibabel as nib
    Brain = mne.viz.get_brain_class() #doesnt work directly from pysurfer

    brain = Brain("fsaverage", **plot_kwargs)

    #mask locations based on percentile
    for hemi in ["lh", "rh"]:

        annot_file = subjects_dir + f'/fsaverage/label/{hemi}.{parc}.annot'
        labels, _, nib_names = nib.freesurfer.read_annot(annot_file)

        names_order_nib = np.array([str(name)[2:-1] for name in nib_names])

        if hemi == "lh":
            names_mne = names_order_mne[lh]
            cur_stc = stc_parc[lh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[lh]
        else:
            names_mne = names_order_mne[rh]
            cur_stc = stc_parc[rh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[rh]

        # Create a dictionary to map strings to their indices in array1
        index_dict = {value: index for index, value in enumerate(names_mne)}

        # Find the indices of strings in array1 corresponding to array2
        right_order = [index_dict[value] for value in names_order_nib]

        cur_stc_ordered = cur_stc[right_order]
        cur_mask_ordered = cur_mask[right_order]
        
        cur_stc_ordered[cur_mask_ordered] = np.nan

        vtx_data = cur_stc_ordered[labels]
        vtx_data[labels == -1] = -1

        brain.add_data(vtx_data, hemi=hemi, fmin=clevels[0], fmid=clevels[1],
                       fmax=clevels[2], colormap=cmap, #np.nanmax(stc_parc)
                       colorbar=False, alpha=.8)

    
    screenshot = brain.screenshot()
    #brain.close()
    
    return screenshot