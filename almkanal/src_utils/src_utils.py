# %%imports
import os
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from mne._fiff.pick import _contains_ch_type
from numpy.typing import NDArray

from almkanal.data_utils.data_classes import InfoClass, PickDictClass
from almkanal.preproc_utils.maxwell_utils import run_maxwell


# %%
def get_nearest_empty_room(info: mne.Info, empty_room_dir: str) -> Path:
    """
    Find the empty room recording closest in date to the current measurement.

    This function locates the empty room file, which is used for noise covariance estimation,
    with a date nearest to the current MEG measurement.

    Parameters
    ----------
    info : mne.Info
        The MEG data information structure, including measurement date.
    empty_room_dir : str
        Directory containing the empty room recordings.

    Returns
    -------
    Path
        Path to the nearest empty room file.
    """

    all_empty_room_dates = np.array([datetime.strptime(date, '%y%m%d') for date in os.listdir(empty_room_dir)])

    cur_date = info['meas_date']
    cur_date_truncated = datetime(cur_date.year, cur_date.month, cur_date.day)  # necessary to truncate

    def _nearest(items: NDArray, pivot: datetime) -> datetime:
        return min(items, key=lambda x: abs(x - pivot))

    while True:
        nearest_date_datetime = _nearest(all_empty_room_dates, cur_date_truncated)
        nearest_date = nearest_date_datetime.strftime('%y%m%d')

        cur_empty_path = Path(empty_room_dir) / nearest_date

        if 'supine' in os.listdir(cur_empty_path)[0]:
            all_empty_room_dates = np.delete(all_empty_room_dates, all_empty_room_dates == nearest_date_datetime)
        elif np.logical_and('68' in os.listdir(cur_empty_path)[0], 'sss' not in os.listdir(cur_empty_path)[0].lower()):
            break

    fname_empty_room = Path(cur_empty_path) / os.listdir(cur_empty_path)[0]

    return fname_empty_room


def preproc_empty_room(
    raw_er: mne.io.Raw,
    data: mne.io.Raw | mne.Epochs,
    preproc_info: InfoClass,
    icas: None | list,
    ica_ids: None | list,
    pick_dict: PickDictClass,
) -> mne.io.Raw:
    """
    Preprocess an empty room recording to match the preprocessing of the original MEG data.

    Parameters
    ----------
    raw_er : mne.io.Raw
        The raw empty room MEG data.
    data : mne.io.Raw | mne.Epochs
        The original MEG data or epochs for comparison and preprocessing alignment.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    icas : None | list, optional
        List of ICA objects to apply to the empty room data. Defaults to None.
    ica_ids : None | list, optional
        List of ICA component indices to exclude for each ICA object. Defaults to None.
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.

    Returns
    -------
    mne.io.Raw
        The preprocessed empty room MEG data.
    """

    # do channel picking here -> we need to disallow dropping bad
    # channels as this can result in problems
    picks = mne.pick_types(raw_er.info, exclude=[], **pick_dict)
    raw_er.pick(picks=picks)
    if preproc_info.maxwell is not None:
        if isinstance(data, mne.epochs.Epochs):
            raw = mne.io.RawArray(np.empty([len(data.info.ch_names), 100]), info=data.info)
        elif isinstance(data, mne.io.fiff.raw.Raw):
            raw = data

        raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(raw_er=raw_er, raw=raw)
        raw_er = run_maxwell(raw_er, **preproc_info.maxwell)

    picks = mne.pick_types(raw_er.info, **pick_dict)
    raw_er.pick(picks=picks)
    # Add filtering here -> i.e. check if deviation between empty and real data and then filter
    if bool(
        np.logical_and(
            np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) is False,
            (np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01),) is False,
        )
    ):
        raw_er.filter(l_freq=data.info['highpass'], h_freq=data.info['lowpass'])
    elif np.isclose(data.info['highpass'], raw_er.info['highpass'], atol=0.01) is False:
        raw_er.filter(
            l_freq=data.info['highpass'],
            h_freq=None,
        )
    elif np.isclose(data.info['lowpass'], raw_er.info['lowpass'], atol=0.01) is False:
        raw_er.filter(l_freq=None, h_freq=data.info['lowpass'])
    else:
        print('No filtering applied')

    # TODO: Also make sure that the sampling rate is the same
    if np.isclose(data.info['sfreq'], raw_er.info['sfreq'], atol=0.9) is False:
        # adjust for small floating point differences
        raw_er.resample(data.info['sfreq'])

    if preproc_info.ica is not None:
        # we loop here, because you could have done more than one ica
        assert isinstance(icas, list)
        assert isinstance(ica_ids, list)
        for ica, ica_id in zip(icas, ica_ids):
            ica.apply(raw_er, exclude=ica_id)

    return raw_er


def process_empty_room(
    data: mne.io.Raw | mne.Epochs,
    info: mne.Info,
    pick_dict: PickDictClass,
    icas: None | list,
    ica_ids: None | list,
    preproc_info: InfoClass,
    empty_room: str | mne.io.Raw,
    get_nearest: bool = False,
) -> tuple[NDArray, NDArray]:
    """
    Process the empty room MEG data for noise covariance estimation.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        The original MEG data or epochs for alignment with the empty room data.
    info : mne.Info
        The MEG data information structure, including measurement metadata.
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.
    icas : None | list, optional
        List of ICA objects to apply to the empty room data. Defaults to None.
    ica_ids : None | list, optional
        List of ICA component indices to exclude for each ICA object. Defaults to None.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    empty_room : str | mne.io.Raw
        Path to the empty room recording or preloaded empty room raw data.
    get_nearest : bool, optional
        If True, finds the nearest empty room recording based on the measurement date. Defaults to False.

    Returns
    -------
    tuple[NDArray, NDArray]
        - `true_rank`: The true rank of the noise covariance matrix.
        - `noise_cov`: The computed noise covariance matrix.
    """

    if np.logical_and(get_nearest, isinstance(empty_room, str)):
        fname_empty_room = get_nearest_empty_room(info, empty_room_dir=empty_room)
        raw_er = mne.io.read_raw(fname_empty_room, preload=True)
    elif np.logical_and(not get_nearest, isinstance(empty_room, str)):
        raw_er = mne.io.read_raw(empty_room, preload=True)
    elif isinstance(empty_room, mne.io.Raw):
        raw_er = empty_room

    raw_er = preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        icas=icas,
        ica_ids=ica_ids,
        pick_dict=pick_dict,
    )

    # if isinstance(data, mne.io.fiff.raw.Raw):
    noise_cov = mne.compute_raw_covariance(raw_er, rank=None, method='auto')

    # TODO: This seems unecessary think about whether i can just drop this elif
    # elif isinstance(data, mne.epochs.Epochs):
    #     t_length = np.abs(data.epoched.tmax - data.epoched.tmin)
    #     raw_er = mne.make_fixed_length_epochs(raw_er, duration=t_length)
    #     noise_cov = mne.compute_covariance(raw_er, rank=None, method='auto')
    # when using noise cov rank should be based on noise cov
    true_rank = mne.compute_rank(noise_cov, info=raw_er.info)  # inferring true rank

    return true_rank, noise_cov


def comp_spatial_filters(
    data: mne.io.Raw | mne.Epochs,
    fwd: mne.Forward,
    pick_dict: PickDictClass,
    preproc_info: InfoClass,
    icas: None | list = None,
    ica_ids: None | list = None,
    data_cov: None | NDArray = None,
    noise_cov: None | NDArray = None,
    empty_room: None | str | mne.io.Raw = None,
    get_nearest_empty_room: bool = False,
) -> tuple[mne.SourceEstimate, mne.beamformer.Beamformer]:
    """
    Compute spatial filters for source reconstruction using LCMV beamformers.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        MEG data for source reconstruction.
    fwd : mne.Forward
        The forward model.
    pick_dict : PickDictClass
        Dictionary specifying channel selection criteria.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    icas : None | list, optional
        List of ICA objects to apply to the data, if needed. Defaults to None.
    ica_ids : None | list, optional
        List of ICA component indices to exclude for each ICA object. Defaults to None.
    data_cov : None | NDArray, optional
        Data covariance matrix. If None, it is computed automatically. Defaults to None.
    noise_cov : None | NDArray, optional
        Noise covariance matrix. If None, it is computed from the empty room recording or ad-hoc. Defaults to None.
    empty_room : None | str | mne.io.Raw, optional
        Path to the empty room recording or preloaded empty room raw data. Defaults to None.
    get_nearest_empty_room : bool, optional
        Whether to find the nearest empty room recording for noise covariance. Defaults to False.

    Returns
    -------
    mne.beamformer.Beamformer
        LCMV spatial filters for source projection.
    """

    # %Compute covariance matrices
    # data covariance based on the actual recording
    # noise covariance based on empyt rooms
    # select only meg channels from raw

    if pick_dict['meg'] not in [True, 'mag', 'grad']:
        raise ValueError('Source Projection with the AlmKanal pipeline currently only works with MEG data.')

    if pick_dict['eeg']:
        pick_dict['eeg'] = False
        warnings.warn(
            'WARNING: Source Projection with the AlmKanal pipeline currently only works with MEG data. \
                       Removing EEG here.'
        )

    picks = mne.pick_types(data.info, **pick_dict)
    data.pick(picks=picks)
    info = data.info

    # check if multiple channel types are present after picking
    n_ch_types = np.sum([_contains_ch_type(data.info, ch_type) for ch_type in ['mag', 'grad', 'eeg']])

    # compute a data covariance matrix
    if np.logical_and(data_cov is None, isinstance(data, mne.io.fiff.raw.Raw)):
        data_cov = mne.compute_raw_covariance(data, rank=None, method='auto')
    elif np.logical_and(data_cov is None, isinstance(data, mne.epochs.Epochs)):
        data_cov = mne.compute_covariance(data, rank=None, method='auto')
    else:
        print('Data covariance matrix not computed as it was supplied by the analyst.')

    # if you have mixed sensor types we need a noise covariance matrix
    # per default we take this from an empty room recording
    # importantly this should be preprocessed similarly to the actual data
    if np.logical_and(
        np.logical_and(n_ch_types > 1, noise_cov is None),
        np.logical_or(isinstance(empty_room, str), isinstance(empty_room, mne.io.Raw)),
    ):
        # assert np.logical_or(isinstance(empty_room, str), isinstance(empty_room, mne.io.Raw)), """Please
        # supply either a mne.io.raw object, a path that leads directly
        #  to an empty_room recording or a folder with a bunch of empty room recordings"""
        true_rank, noise_cov = process_empty_room(
            data=data,
            info=info,
            pick_dict=pick_dict,
            preproc_info=preproc_info,
            icas=icas,
            ica_ids=ica_ids,
            empty_room=empty_room,
            get_nearest=get_nearest_empty_room,
        )

    elif np.logical_and(
        np.logical_and(n_ch_types > 1, noise_cov is None),
        np.logical_and(empty_room is None, not get_nearest_empty_room),
    ):
        warnings.warn("""You have multiple sensor types, but did neither specify a noise covariance
                      matrix or supply a path to an empty room file. Computing an ad-hoc covariance matrix..
                      ATTENTION: THIS IS NOT THE OPTIMAL WAY OF DOING THINGS!!!""")
        noise_cov = mne.make_ad_hoc_cov(info)
        # TODO: check in with thomas if rank should be computed on data_cov if ad-hoc cov is created
        true_rank = mne.compute_rank(data_cov, info=info)

    elif n_ch_types == 1:
        # when we dont have a noise cov we just use the data cov for rank comp
        true_rank = mne.compute_rank(data_cov, info=info)
        noise_cov = None

    lcmv_settings = {
        'reg': 0.05,
        'noise_cov': noise_cov,
        'pick_ori': 'max-power',
        'weight_norm': 'nai',
        'rank': true_rank,
    }

    # build and apply filters
    filters = mne.beamformer.make_lcmv(info, fwd, data_cov, **lcmv_settings)

    return filters


def src2parc(
    stc: mne.SourceEstimate,
    subject_id: str,
    subjects_dir: str,
    atlas: str = 'glasser',
    source: str = 'surface',
    label_mode: str = 'mean_flip',
) -> dict:
    """
    Parcellate source data into predefined brain regions using an atlas.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate object containing the source data to parcellate.
    subject_id : str
        Subject identifier for the source data.
    subjects_dir : str
        Path to the directory containing FreeSurfer subject data.
    atlas : str, optional
        Atlas to use for parcellation ('glasser', 'dk', or 'destrieux'). Defaults to 'glasser'.
    source : str, optional
        Source space type ('surface' or 'volume'). Defaults to 'surface'.
    label_mode : str, optional
        Mode for extracting label time courses ('mean', 'mean_flip', etc.). Defaults to 'mean_flip'.

    Returns
    -------
    dict
        Dictionary containing parcellation information, including labels, hemisphere assignments,
        and extracted time courses for each region.
    """

    if atlas == 'dk':
        vol_atlas = 'aparc+aseg'
        surf_atlas = 'aparc'
    elif atlas == 'destrieux':
        vol_atlas = 'aparc.a2009s+aseg'
        surf_atlas = 'aparc.a2009s'
    elif atlas == 'glasser':
        if source == 'volume':
            raise ValueError('No volumetric model for the glasser atlas available')
        surf_atlas = 'HCPMMP1'

    fs_dir = Path(subjects_dir) / 'freesurfer'
    # mean flip time series costs significantly less memory than averaging the irasa'd spectra
    if source == 'surface':
        src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-ico-4-src.fif'
        src = mne.read_source_spaces(src_file)
        labels_mne = mne.read_labels_from_annot(f'{subject_id}_from_template', parc=surf_atlas, subjects_dir=fs_dir)
        names_order_mne = np.array([label.name[:-3] for label in labels_mne])

        rh = [label.hemi == 'rh' for label in labels_mne]
        lh = [label.hemi == 'lh' for label in labels_mne]

        parc = {'lh': lh, 'rh': rh, 'parc': surf_atlas, 'names_order_mne': names_order_mne}
        parc.update({'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode=label_mode)})
    elif source == 'volume':
        src_file = f'{fs_dir}/{subject_id}_from_template/bem/{subject_id}_from_template-vol-5-src.fif'
        src = mne.read_source_spaces(src_file)
        labels_mne = (
            fs_dir / f'{subject_id}_from_template' / 'mri' / (vol_atlas + '.mgz')
        )  # os.path.join(fs_dir, f'{subject_id}_from_template', 'mri/' + vol_atlas + '.mgz')
        label_names = mne.get_volume_labels_from_aseg(labels_mne)

        ctx_logical = 'ctx' in label_names  # [True if 'ctx' in label else False for label in label_names]
        sctx_logical = not ctx_logical  # [True if not f else False for f in ctx_logical]

        ctx_labels = np.array([label[4:] for label in label_names if 'ctx' in label])
        sctx_labels = list(np.array(label_names)[sctx_logical])
        rh = ctx_labels[:2] == 'rh'  # [True if label[:2] == 'rh' else False for label in ctx_labels]
        lh = ctx_labels[:2] == 'lh'  # [True if label[:2] == 'lh' else False for label in ctx_labels]

        parc = {
            'lh': lh,
            'rh': rh,
            'parc': vol_atlas + '.mgz',
            'ctx_labels': ctx_labels,
            'ctx_logical': ctx_logical,
            'sctx_logical': sctx_logical,
            'sctx_labels': sctx_labels,
        }
        parc.update(
            {'label_tc': mne.extract_label_time_course(stc, labels_mne, src, mode='auto')}
        )  # NOTE: This needs to be auto

    else:
        raise ValueError('the only valid options for source are `surface` and `volume`.')

    return parc
