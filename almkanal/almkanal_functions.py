from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray

from almkanal.data_utils.check_data import check_raw_epoch
from almkanal.data_utils.data_classes import ICAInfoDict, InfoClass, PickDictClass
from almkanal.preproc_utils.bio_utils import run_bio_preproc
from almkanal.preproc_utils.ica_utils import run_ica

# all them utility functions
from almkanal.preproc_utils.maxwell_utils import run_maxwell
from almkanal.src_utils.headmodel_utils import compute_headmodel, make_fwd
from almkanal.src_utils.spatial_filter_utils import comp_spatial_filters
from almkanal.src_utils.src_recon_utils import src2parc
from almkanal.data_utils.almkanal_objs import AlmkanalRaw, AlmkanalEpochs
from almkanal.src_utils.headmodel_utils import plot_head_model

def do_maxwell(
    ak_raw: AlmkanalRaw,
    mw_coord_frame: str = 'head',
    mw_destination: str | None = None,
    mw_calibration_file: str | None = None,
    mw_cross_talk_file: str | None = None,
    mw_st_duration: int | None = None,
) -> None:
    """
    Apply Maxwell filtering to the raw MEG data.

    Parameters
    ----------
    mw_coord_frame : str, optional
        Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
    mw_destination : str | None, optional
        Destination coordinate frame for alignment. Defaults to None.
    mw_calibration_file : str | None, optional
        Path to the calibration file. Defaults to None.
    mw_cross_talk_file : str | None, optional
        Path to the cross-talk file. Defaults to None.
    mw_st_duration : int | None, optional
        Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

    Returns
    -------
    None
    """

    # this should do maxwell filtering
    # should only be possible on raw data and only if no other preprocessing apart from filtering was done

    maxwell_settings = {
        'coord_frame': mw_coord_frame,
        'destination': mw_destination,
        'calibration_file': mw_calibration_file,
        'cross_talk_file': mw_cross_talk_file,
        'st_duration': mw_st_duration,
    }

    raw_max = run_maxwell(raw=ak_raw.raw, **maxwell_settings)
    ak_raw.report.add_raw(raw_max, 
                           butterfly=False, 
                           psd=True, 
                           title='raw_maxfiltered')
    
    ak_raw.from_mne_raw(raw_max)

    return ak_raw

def do_ica(
    ak_raw: AlmkanalRaw,
    n_components: None | int | float = None,
    method: str = 'picard',
    random_state: None | int = 42,
    fit_params: dict | None = None,
    ica_hp_freq: None | float = 1.0,
    ica_lp_freq: None | float = None,
    resample_freq: int = 200,  # downsample to 200hz per default
    eog: bool = True,
    surrogate_eog_chs: None | dict = None,
    eog_corr_thresh: float = 0.5,
    ecg: bool = True,
    ecg_corr_thresh: float = 0.5,
    emg: bool = False,
    emg_thresh: float = 0.5,
    train: bool = True,
    train_freq: int = 16,
    train_thresh: float = 2.0,
    img_path: None | str = None,
    fname: None | str = None,
) -> tuple[AlmkanalRaw, mne.preprocessing.ICA]:
    """
    Perform ICA to identify and remove peripheral physiological signals like
    EOG and ECG as well as an artifact caused by our local train in Salzburg.

    Parameters
    ----------
    n_components : int | float | None, optional
        Number of ICA components to compute. Defaults to None.
    method : str, optional
        ICA method to use ('picard', etc.). Defaults to 'picard'.
    random_state : int | None, optional
        Random seed for reproducibility. Defaults to 42.
    fit_params : dict | None, optional
        Additional fitting parameters for ICA. Defaults to None.
    ica_hp_freq : float | None, optional
        High-pass filter frequency for ICA preprocessing. Defaults to 1.0 Hz.
    ica_lp_freq : float | None, optional
        Low-pass filter frequency for ICA preprocessing. Defaults to None.
    resample_freq : int, optional
        Downsampling frequency before ICA. Defaults to 200 Hz.
    eog : bool, optional
        Whether to detect and remove EOG artifacts. Defaults to True.
    eog_corr_thresh : float, optional
        Correlation threshold for EOG artifact detection. Defaults to 0.5.
    ecg : bool, optional
        Whether to detect and remove ECG artifacts. Defaults to True.
    ecg_corr_thresh : float, optional
        Correlation threshold for ECG artifact detection. Defaults to 0.5.
    emg : bool,
        Whether to detect and remove EMG artifacts. Defaults to False.
    emg_thresh:
        Value above which a component should be marked as muscle-related, relative to a typical muscle component.
    train : bool, optional
        Whether to detect and remove train-related artifacts. Defaults to True.
    train_freq : int, optional
        Frequency for train artifact detection. Defaults to 16 Hz.
    img_path : str | None, optional
        Path to save ICA plots. Defaults to None.
    fname : str | None, optional
        Filename for ICA plots. Defaults to None.

    Returns
    -------
    None
    """

    # this should do an ica
    ica_info = ICAInfoDict(
        n_components=n_components,
        method=method,
        random_state=random_state,
        fit_params=fit_params,
        ica_hp_freq=ica_hp_freq,
        ica_lp_freq=ica_lp_freq,
        resample_freq=resample_freq,
        eog=eog,
        eog_corr_thresh=eog_corr_thresh,
        surrogate_eog_chs=surrogate_eog_chs,
        ecg=ecg,
        ecg_corr_thresh=ecg_corr_thresh,
        emg=emg,
        emg_thresh=emg_thresh,
        train=train,
        train_freq=train_freq,
        train_thresh=train_thresh,
        img_path=img_path,
        fname=fname,
    )

    raw, ica, components_dict, eog_scores, ecg_scores = run_ica(ak_raw.raw, **ica_info)

    titles = {}
    for key, vals in components_dict.items():
        for val in vals:
            titles.update({int(val): f'{key}'})

    ak_raw.report.add_ica(ica, inst=raw, title='ICA', 
                          ecg_scores=ecg_scores,
                          eog_scores=eog_scores,
                          picks=list(titles.keys()), 
                          tags=list(titles.values()))
    ak_raw.from_mne_raw(raw)

    return ak_raw, ica

def do_bio_process(
    ak_raw: AlmkanalRaw,
    ecg: None | str | list = None,
    resp: None | str | list = None,
    eog: None | str | list = None,
    emg: None | str | list = None,
) -> mne.io.Raw:
    """
    Preprocess physiological signals (ECG, EOG, RESP, EMG) in an MNE raw object.

    This method extracts specified physiological channels, preprocesses them using `neurokit2`,
    and returns a new raw object containing the cleaned physiological data along with stimulus channels.

    Parameters
    ----------
    ecg : str | list | None, optional
        Name(s) of the ECG channel(s) to preprocess. Defaults to None.
    resp : str | list | None, optional
        Name(s) of the respiratory channel(s) to preprocess. Defaults to None.
    eog : str | list | None, optional
        Name(s) of the EOG channel(s) to preprocess. Defaults to None.
    emg : str | list | None, optional
        Name(s) of the EMG channel(s) to preprocess. Defaults to None.

    Returns
    -------
    mne.io.Raw
        A new raw object containing the preprocessed physiological signals and stimulus channels.
    """

    return run_bio_preproc(
        raw=ak_raw.raw,
        ecg=ecg,
        resp=resp,
        eog=eog,
        emg=emg,
    )


def do_fwd_model(
    data: AlmkanalRaw | AlmkanalEpochs,
    subject_id: str,
    subjects_dir: str,
    pick_dict: dict,
    source: str = 'surface',
    template_mri: bool = True,
    redo_hdm: bool = True,
) -> mne.Forward:
    """
    Generate a forward model for source reconstruction.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    source : str, optional
        Type of source space ('surface' or 'volume'). Defaults to 'surface'.
    template_mri : bool, optional
        Whether to use a template MRI. Defaults to True.
    redo_hdm : bool, optional
        Whether to recompute the head model. Defaults to True.

    Returns
    -------
    mne.Forward
    """
    new_source_identifier = subject_id + '_from_template' if template_mri else subject_id

    # fetch fsaverage if subjects_dir and fsaverage is not yet there
    freesurfer_dir = Path(subjects_dir) / 'freesurfer'
    if (freesurfer_dir / 'fsaverage').is_dir() is False:
        print('Download missing freesurfer fsaverage data for source modelling.')
        mne.datasets.fetch_fsaverage(freesurfer_dir)
        # also build a downsampled version of the ico-5 to save some processing power
        src = mne.setup_source_space(
            subject='fsaverage',  # Subject name
            spacing='ico4',  # Use ico-4 source spacing
            add_dist=False,  # Avoid computing inter-source distances (optional)
            subjects_dir=freesurfer_dir,  # FreeSurfer's subjects directory
        )

        # Save the source space to a file
        mne.write_source_spaces(f'{freesurfer_dir}/fsaverage/bem/fsaverage-ico-4-src.fif', src)

    if redo_hdm:
        # recompute or take the saved one
        trans, fig = compute_headmodel(
            info=data.info,
            subject_id=new_source_identifier,
            subjects_dir=subjects_dir,
            pick_dict=pick_dict,
            template_mri=template_mri,
        )
           
    else:
        trans = Path(subjects_dir) / 'headmodels' / subject_id / (subject_id + '-trans.fif')
        fig = plot_head_model(trans, data.info, 
                        subject_id=new_source_identifier, 
                        subjects_dir=freesurfer_dir)
        
    data.report.add_figure(fig=fig, 
                            title='Coregistration',
                            image_format="PNG",
                            caption="",)

    fwd = make_fwd(
        data.info,
        source=source,
        fname_trans=trans,
        subjects_dir=subjects_dir,
        subject_id=new_source_identifier,
        template_mri=template_mri,
    )
    data.report.add_forward(fwd, title='ForwardModel')

    return fwd


def do_spatial_filters(
    data,
    fwd: mne.Forward,
    data_cov: None | NDArray = None,
    noise_cov: None | NDArray = None,
    empty_room: None | str | mne.io.Raw = None,
    get_nearest_empty_room: bool = False,
) -> mne.beamformer.Beamformer:
    """
    Compute spatial filters for source projection using LCMV beamformers.

    Parameters
    ----------
    fwd : mne.Forward | None, optional
        The forward model. Defaults to None.
    data_cov : NDArray | None, optional
        Data covariance matrix. Defaults to None.
    noise_cov : NDArray | None, optional
        Noise covariance matrix. Defaults to None.
    empty_room : str | mne.io.Raw | None, optional
        Path to or preloaded empty room recording. Defaults to None.
    get_nearest_empty_room : bool, optional
        Whether to find the nearest empty room recording. Defaults to False.

    Returns
    -------
    None
    """

    if self.fwd is not None:
        data = self.raw if self.raw is not None else self.epoched

        filters = comp_spatial_filters(
            data=data,
            fwd=self.fwd,
            pick_dict=self.pick_dict,
            icas=self.ica,
            ica_ids=self.ica_ids,
            data_cov=data_cov,
            noise_cov=noise_cov,
            preproc_info=data.info,
            empty_room=empty_room,
            get_nearest_empty_room=get_nearest_empty_room,
        )
    return filters


def do_src(
    data: AlmkanalRaw | AlmkanalEpochs,
    filters: mne.beamformer.Beamformer,
    return_parc: bool = False,
    label_mode: str = 'mean_flip',
    subject_id: None | str = None,
    subjects_dir: None | str = None,
    atlas: str = 'glasser',
    source: str = 'surface',
) -> None:
    """
    Perform source reconstruction and optional parcellation.

    Parameters
    ----------
    return_parc : bool, optional
        Whether to return parcellated source data. Defaults to False.
    label_mode : str, optional
        Mode for extracting label time courses ('mean_flip', etc.). Defaults to 'mean_flip'.
    subject_id : str | None, optional
        Subject identifier for parcellation. Required if `return_parc` is True.
    subjects_dir : str | None, optional
        Path to FreeSurfer subjects directory. Required if `return_parc` is True.
    atlas : str, optional
        Atlas for parcellation ('glasser', 'dk', etc.). Defaults to 'glasser'.
    source : str, optional
        Source space type ('surface' or 'volume'). Defaults to 'surface'.

    Returns
    -------
    mne.SourceEstimate | dict
        Source time courses or parcellated data.
    """

    data = self.raw if self.raw is not None else self.epoched

    if isinstance(data, mne.io.fiff.raw.Raw):
        stc = mne.beamformer.apply_lcmv_raw(data, filters)

    elif isinstance(data, mne.epochs.Epochs):
        stc = mne.beamformer.apply_lcmv_epochs(data, filters)

    if np.logical_and(return_parc, np.logical_and(subject_id is not None, subjects_dir is not None)):
        assert isinstance(
            subject_id, str
        ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
        assert isinstance(
            subjects_dir, str
        ), 'You need to set the correct name for the `subject_id` and `subjects_dir` if you want to parcels.'
        stc = src2parc(
            stc,
            subject_id=subject_id,
            subjects_dir=subjects_dir,
            atlas=atlas,
            source=source,
            label_mode=label_mode,
        )

    return stc
