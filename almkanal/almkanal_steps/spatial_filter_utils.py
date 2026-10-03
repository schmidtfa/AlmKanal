# %%imports
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from attrs import define, field

from almkanal import AlmKanalStep
from almkanal.almkanal_steps.channel_utils import run_maxwell
from almkanal.info import AlmKanalInfo, StepInfo


# %%
def get_nearest_empty_room(info: mne.Info, empty_room_dir: str) -> Path:
    """
    Find the empty room recording closest in date to the current measurement.

    This function looks for subdirectories named as dates in '%y%m%d' format
    in the given empty_room_dir. It selects the directory closest to the
    measurement date and returns the first file it contains.

    Parameters
    ----------
    info : mne.Info
        MEG data information containing the measurement date.
    empty_room_dir : str
        Directory containing dated empty room subdirectories.

    Returns
    -------
    Path
        Path to the nearest empty room recording.

    Raises
    ------
    ValueError
        If no valid empty room recording is found.
    """
    # Build list of valid dates from directory names
    empty_room_path = Path(empty_room_dir)

    dated_dirs = []
    for directory in empty_room_path.iterdir():
        if not directory.is_dir():
            continue

        try:
            date = datetime.strptime(directory.name, '%y%m%d')
        except ValueError:
            continue  # Skip entries that don't match the date format

        dated_dirs.append((date, directory))

    if not dated_dirs:
        raise ValueError(f'No valid empty room directories found in {empty_room_path}')
    # Truncate measurement date to day resolution
    meas_date = info['meas_date']
    meas_date = datetime(meas_date.year, meas_date.month, meas_date.day)

    dated_dirs.sort(key=lambda item: abs(item[0] - meas_date))

    for _, directory in dated_dirs:
        files = sorted(path for path in directory.iterdir() if path.is_file())

        if files:
            return files[0]

    raise ValueError(f'No empty room recordings found in {empty_room_path}')


def preproc_empty_room(  # noqa: C901
    raw_er: mne.io.Raw,
    data: mne.io.Raw | mne.Epochs,
    preproc_info: list[StepInfo],
    picks: list[str] | None,
) -> mne.io.Raw:
    """
    Preprocess an empty room recording to match the preprocessing of the
    experimental data.

    Parameters
    ----------
    raw_er : mne.io.BaseRaw
        The raw empty room recording.
    data : mne.io.BaseRaw | mne.BaseEpochs
        The experimental data whose preprocessing should be matched.
    preproc_info : dict
        Information about preprocessing steps already applied to the data.
    picks : list[str] | None
        Channel names to retain in the empty room recording. If None,
        no channel selection is applied.

    Returns
    -------
    mne.io.BaseRaw
        The preprocessed empty room recording.
    """

    if picks is not None:
        raw_er.pick(picks)

    replay_steps = [step for step in preproc_info if step.step in {'Maxwell', 'Filter', 'Resample', 'ICA'}]

    for step in replay_steps:
        if step.step == 'Maxwell':
            maxwell_info = step.info['maxwell_info']
            if isinstance(data, mne.BaseEpochs):
                raw = mne.io.RawArray(np.zeros((len(data.ch_names), 1)), info=data.info)
            else:
                raw = data

            raw_er = mne.preprocessing.maxwell_filter_prepare_emptyroom(raw_er=raw_er, raw=raw)
            raw_er = run_maxwell(raw_er, **maxwell_info)

        elif step.step == 'Filter':
            filter_info = step.info['filter_info'].copy()
            raw_er.filter(**filter_info)

        elif step.step == 'Resample':
            resample_info = step.info['resample_info'].copy()
            raw_er.resample(**resample_info)

        elif step.step == 'ICA':
            ica_info = step.info['ica_info']

            if not ica_info.get('fit_only', False):
                ica = ica_info['ica'].copy()
                ica.exclude = list(ica_info.get('applied_exclude', ica.exclude))
                ica.apply(raw_er)

    return raw_er


def process_empty_room(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    info: mne.Info,
    picks: list[str] | None,
    preproc_info: list[StepInfo],
    empty_room: str | mne.io.BaseRaw,
    get_nearest: bool = False,
) -> tuple[dict, mne.Covariance]:
    """
    Process empty room data for noise covariance estimation.

    Parameters
    ----------
    data : mne.io.BaseRaw | mne.BaseEpochs
        Experimental data whose preprocessing should be matched.
    info : mne.Info
        Measurement information used to identify the nearest empty room.
    picks : list[str] | None
        Channel names to retain in the empty room recording.
    preproc_info : dict
        Information about preprocessing applied to the experimental data.
    empty_room : str | mne.io.BaseRaw
        Path to an empty room recording, directory containing dated empty
        room recordings, or preloaded empty room data.
    get_nearest : bool, optional
        Find the recording closest to the measurement date when
        ``empty_room`` is a directory.

    Returns
    -------
    true_rank : dict
        Estimated rank of the noise covariance.
    noise_cov : mne.Covariance
        Noise covariance computed from the empty room recording.
    """

    if get_nearest and isinstance(empty_room, str):
        fname_empty_room = get_nearest_empty_room(info, empty_room_dir=empty_room)
        raw_er = mne.io.read_raw(fname_empty_room, preload=True)
    elif not get_nearest and isinstance(empty_room, str):
        raw_er = mne.io.read_raw(empty_room, preload=True)
    elif isinstance(empty_room, mne.io.BaseRaw):
        raw_er = empty_room

    raw_er = preproc_empty_room(
        raw_er=raw_er,
        data=data,
        preproc_info=preproc_info,
        picks=picks,
    )

    noise_cov = mne.compute_raw_covariance(raw_er, rank=None, method='auto')
    true_rank = mne.compute_rank(noise_cov, info=raw_er.info)

    return true_rank, noise_cov


def comp_spatial_filters(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    fwd: mne.Forward,
    pick_dict: dict | None,
    preproc_info: list[StepInfo],
    data_cov: None | mne.Covariance = None,
    noise_cov: None | mne.Covariance = None,
    empty_room: None | str | mne.io.BaseRaw = None,
    nearest_empty_room: bool = False,
    lcmv_reg: float = 0.05,
    lcmv_pick_ori: None | str = 'max-power',
    lcmv_weight_norm: str | None = 'nai',
    lcmv_reduce_rank: bool = False,
) -> tuple[
    mne.beamformer.Beamformer,
    dict,
    mne.Covariance | None,
    mne.Covariance,
]:
    """
    Compute spatial filters for source reconstruction using LCMV beamformers.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        MEG data for source reconstruction.
    fwd : mne.Forward
        The forward model.
    pick_dict : dict
        Dictionary specifying channel selection criteria.
    preproc_info : InfoClass
        Configuration object containing preprocessing details (e.g., Maxwell filter settings, ICA).
    data_cov : None | NDArray, optional
        Data covariance matrix. If None, it is computed automatically. Defaults to None.
    noise_cov : None | NDArray, optional
        Noise covariance matrix. If None, it is computed from the empty room recording or ad-hoc. Defaults to None.
    empty_room : None | str | mne.io.Raw, optional
        Path to the empty room recording or preloaded empty room raw data. Defaults to None.
    nearest_empty_room : bool, optional
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

    if isinstance(pick_dict, dict):
        picks = mne.pick_types(data.info, **pick_dict)
        picks = [data.ch_names[pick] for pick in picks]
        data.pick(picks=picks)
    elif pick_dict is None:
        picks = None

    info = data.info

    # check if multiple channel types are present after picking
    n_ch_types = len({'mag', 'grad', 'eeg'} & set(data.get_channel_types(unique=True)))

    # compute a data covariance matrix
    if data_cov is None:
        if isinstance(data, mne.io.BaseRaw):
            data_cov = mne.compute_raw_covariance(data, rank=None, method='auto')
        elif isinstance(data, mne.BaseEpochs):
            data_cov = mne.compute_covariance(data, rank=None, method='auto')

    # if you have mixed sensor types we need a noise covariance matrix
    # per default we take this from an empty room recording
    # importantly this should be preprocessed similarly to the actual data
    if noise_cov is not None:
        true_rank = mne.compute_rank(noise_cov, info=info)

    elif n_ch_types > 1 and isinstance(empty_room, str | mne.io.BaseRaw):
        # assert np.logical_or(isinstance(empty_room, str), isinstance(empty_room, mne.io.Raw)), """Please
        # supply either a mne.io.raw object, a path that leads directly
        #  to an empty_room recording or a folder with a bunch of empty room recordings"""
        true_rank, noise_cov = process_empty_room(
            data=data,
            info=info,
            picks=picks,
            preproc_info=preproc_info,
            empty_room=empty_room,
            get_nearest=nearest_empty_room,
        )

    elif n_ch_types > 1 and empty_room is None:
        warnings.warn("""You have multiple sensor types, but did neither specify a noise covariance
                      matrix or supply a path to an empty room file. Computing an ad-hoc covariance matrix!""")

        noise_cov = mne.make_ad_hoc_cov(info)
        true_rank = mne.compute_rank(data_cov, info=info)

    elif n_ch_types == 1:
        true_rank = mne.compute_rank(data_cov, info=info)
        noise_cov = None

    lcmv_settings = {
        'reg': lcmv_reg,
        'noise_cov': noise_cov,
        'pick_ori': lcmv_pick_ori,
        'weight_norm': lcmv_weight_norm,
        'rank': true_rank,
        'reduce_rank': lcmv_reduce_rank,
    }

    filters = mne.beamformer.make_lcmv(info, fwd, data_cov, **lcmv_settings)

    # build and apply filters
    return filters, lcmv_settings, noise_cov, data_cov


@define
class SpatialFilter(AlmKanalStep):
    """Compute LCMV spatial filters for source reconstruction.

    ``SpatialFilter`` computes data and noise covariance matrices as needed and
    constructs an LCMV beamformer for subsequent source reconstruction. The
    forward model and channel-selection parameters can either be supplied
    directly or obtained from preceding pipeline configuration.

    If an empty-room recording is used to estimate the noise covariance, the
    relevant preprocessing history is passed along so that compatible
    preprocessing can be applied to the empty-room data before covariance
    estimation.

    Parameters
    ----------
    fwd : mne.Forward | None, default=None
        Forward model used to construct the spatial filter. If ``None``, the
        forward solution produced by a preceding ``ForwardModel`` step is used.
    pick_dict : dict | None, default=None
        Channel-selection parameters used for spatial filtering. If ``None``,
        the pipeline-level ``pick_params`` stored in :class:`AlmKanalInfo` are
        used. A value must be available from one of these sources.
    data_cov : mne.Covariance | None, default=None
        Precomputed data covariance matrix. If ``None``, it is computed from
        the input data.
    noise_cov : mne.Covariance | None, default=None
        Precomputed noise covariance matrix. If provided, it is used directly.
        Otherwise, a covariance can be derived from ``empty_room`` or, when no
        empty-room data are supplied, according to the fallback behaviour of
        :func:`comp_spatial_filters`.
    empty_room : str | mne.io.BaseRaw | None, default=None
        Empty-room recording used to estimate the noise covariance. May be a
        path to a recording or an already loaded :class:`mne.io.BaseRaw`
        instance.
    nearest_empty_room : bool, default=False
        Whether an empty-room recording nearest in acquisition date should be
        selected when resolving empty-room data.
    chans2keep : list of str | None, default=None
        Channels to preserve separately before spatial-filter channel selection,
        for example stimulus features, ECG, EOG, or eye-tracking channels.
        Their data are stored in the returned spatial-filter metadata.
    lcmv_reg : float, default=0.05
        Regularization parameter passed to the LCMV beamformer computation.
    lcmv_pick_ori : str | None, default='max-power'
        Source-orientation selection passed to the LCMV beamformer.
    lcmv_weight_norm : str | None, default='nai'
        Weight-normalization method passed to the LCMV beamformer.
    lcmv_reduce_rank : bool, default=False
        Whether to reduce the rank during LCMV filter construction.

    Notes
    -----
    ``pick_dict`` takes precedence over pipeline-level ``pick_params``. If
    neither is available, :meth:`run` raises a :class:`ValueError`.

    Likewise, an explicitly supplied ``fwd`` takes precedence over the forward
    model stored by a preceding ``ForwardModel`` step.

    The input data are returned unchanged. Computed spatial-filter information
    is stored under ``'spatial_filter_info'`` and contains the beamformer
    filters, effective LCMV settings, data covariance, noise covariance, and any
    channels preserved through ``chans2keep``.

    When empty-room data are used, the pipeline's ordered processing history is
    passed to the empty-room preprocessing machinery so that relevant previous
    preprocessing steps can be replayed before estimating the noise covariance.

    ``SpatialFilter`` is intended to precede ``SourceReconstruction`` and can
    occur only once in a pipeline.

    See Also
    --------
    comp_spatial_filters
        Compute the covariance matrices and LCMV beamformer.
    SourceReconstruction
        Apply the resulting spatial filters to reconstruct source activity.
    ForwardModel
        Produce the forward solution used for spatial filtering.
    """

    fwd: mne.Forward | None = None
    pick_dict: dict | None = None
    data_cov: None | mne.Covariance = None
    noise_cov: None | mne.Covariance = None
    empty_room: None | str | mne.io.BaseRaw = None
    nearest_empty_room: bool = False
    chans2keep: list[str] | None = None
    lcmv_reg: float = 0.05
    lcmv_pick_ori: str | None = 'max-power'
    lcmv_weight_norm: str | None = 'nai'
    lcmv_reduce_rank: bool = False

    must_be_before: tuple = ('SourceReconstruction',)
    must_be_after: tuple = (
        'Maxwell',
        'ICA',
        'ForwardModel',
    )
    allow_repeated: bool = field(default=False, init=False)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: AlmKanalInfo) -> dict:
        pick_dict = self.pick_dict

        if pick_dict is None:
            pick_dict = info.pick_params

        if pick_dict is None:
            raise ValueError('pick_dict must be provided for spatial filtering.')

        # before picking data we want to keep our extra data (e.g. envelopes, ECG, EOG or eyetracker)
        extra_data = {ch: data.get_data(ch) for ch in self.chans2keep} if self.chans2keep is not None else None

        fwd = self.fwd

        if fwd is None:
            fwd = info.get_step_info('ForwardModel', required=True)['fwd_info']['fwd']

        data_cov_source = (
            'provided' if self.data_cov is not None else 'continuous' if isinstance(data, mne.io.BaseRaw) else 'epoched'
        )

        filters, lcmv_settings, noise_cov, data_cov = comp_spatial_filters(
            data=data,
            fwd=fwd,
            pick_dict=pick_dict,
            data_cov=self.data_cov,
            noise_cov=self.noise_cov,
            preproc_info=info.processing_history,
            empty_room=self.empty_room,
            nearest_empty_room=self.nearest_empty_room,
            lcmv_reg=self.lcmv_reg,
            lcmv_pick_ori=self.lcmv_pick_ori,
            lcmv_weight_norm=self.lcmv_weight_norm,
            lcmv_reduce_rank=self.lcmv_reduce_rank,
        )

        if self.noise_cov is not None:
            noise_cov_source = 'provided'
        elif self.empty_room is not None:
            noise_cov_source = 'nearest_empty_room' if self.nearest_empty_room else 'empty_room'
        elif noise_cov is not None:
            noise_cov_source = 'ad_hoc'
        else:
            noise_cov_source = 'none'

        return {
            'data': data,
            'spatial_filter_info': {
                'filters': filters,
                'lcmv_settings': lcmv_settings,
                'data_cov': data_cov,
                'noise_cov': noise_cov,
                'extra_data': extra_data,
                # reporting metadata
                'reg': self.lcmv_reg,
                'pick_ori': self.lcmv_pick_ori,
                'weight_norm': self.lcmv_weight_norm,
                'reduce_rank': self.lcmv_reduce_rank,
                'data_cov_source': data_cov_source,
                'noise_cov_source': noise_cov_source,
            },
        }

    def reports(self, data: mne.io.Raw, report: mne.Report, info: AlmKanalInfo) -> None:
        spatial_info = info.get_step_info('SpatialFilter', required=True)['spatial_filter_info']

        report.add_covariance(
            spatial_info['data_cov'],
            info=data.info,
            title='Data Covariance Matrix',
        )

        if spatial_info['noise_cov'] is not None:
            noise_cov = spatial_info['noise_cov']

            if noise_cov.data.ndim == 1:
                noise_cov = noise_cov.copy()._as_square()
            report.add_covariance(
                noise_cov,
                info=data.info,
                title='Noise Covariance Matrix',
            )
