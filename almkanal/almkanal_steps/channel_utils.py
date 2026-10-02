import mne
import numpy as np
from attrs import define, field
from autoreject import Ransac
from numpy.typing import ArrayLike

from almkanal.almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo


def run_maxwell(
    raw: mne.io.Raw,
    coord_frame: str = 'head',
    destination: None | ArrayLike = None,
    calibration_file: None | bool | str = None,
    cross_talk_file: None | bool | str = None,
    st_duration: float | None = None,
    st_correlation: float = 0.98
) -> mne.io.Raw:
    """
    Perform Maxwell filtering on raw MEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data to preprocess.
    coord_frame : str, optional
        Coordinate frame for Maxwell filtering ('head' or 'meg'). Defaults to 'head'.
    destination : None | ArrayLike, optional
        Destination coordinate frame for alignment. Defaults to None.
    calibration_file : None | str, optional
        Path to the calibration file. Defaults to None.
    cross_talk_file : None | str, optional
        Path to the cross-talk file. Defaults to None.
    st_duration : float | None, optional
        Duration (in seconds) for tSSS (temporal Signal Space Separation). Defaults to None.

    Returns
    -------
    mne.io.Raw
        The Maxwell-filtered raw MEG data.
    """

    # find bad channels first
    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(
        raw,
        coord_frame=coord_frame,
        calibration=calibration_file,
        cross_talk=cross_talk_file,  # noqa
    )
    raw.info['bads'] = list(dict.fromkeys(raw.info['bads'] + noisy_chs + flat_chs))

    raw = mne.preprocessing.maxwell_filter(
        raw,
        st_duration=st_duration,
        calibration=calibration_file,
        cross_talk=cross_talk_file,
        coord_frame=coord_frame,
        st_correlation=st_correlation,
        destination=destination,
    )

    return raw


@define
class Maxwell(AlmKanalStep):
    """Apply Maxwell filtering to a continuous raw MEG recording.

    ``Maxwell`` applies Signal Space Separation (SSS) and, optionally, temporal
    Signal Space Separation (tSSS) to a single :class:`mne.io.BaseRaw` recording.

    The step can also transform the data to a specified destination head position
    and use system-specific fine-calibration and cross-talk compensation files.

    Parameters
    ----------
    mw_coord_frame : str, default='head'
        Coordinate frame used for Maxwell filtering. Typically ``'head'`` or
        ``'meg'``.
    mw_destination : array-like | None, default=None
        Destination head position used during Maxwell filtering. ``None`` keeps the
        destination behavior defined by the underlying Maxwell-filtering routine.
    mw_st_duration : float | None, default=None
        Temporal Signal Space Separation (tSSS) buffer duration, in seconds.
        ``None`` disables temporal SSS.
    mw_calibration_file : str | bool | None, default=None
        Fine-calibration information passed to Maxwell filtering. A path
        specifies an external calibration file. With ``None``, calibration
        embedded in ``data.info`` is used when available. ``True`` requires
        embedded calibration information, whereas ``False`` disables fine
        calibration.

    mw_cross_talk_file : str | bool | None, default=None
        Cross-talk compensation passed to Maxwell filtering. A path specifies
        an external cross-talk file. With ``None``, cross-talk information
        embedded in ``data.info`` is used when available. ``True`` requires
        embedded cross-talk information, whereas ``False`` disables it.

    mw_st_correlation : float, default=0.98
        Correlation threshold used by temporal Signal Space Separation (tSSS).
        Only relevant when ``mw_st_duration`` is not ``None``.


    Notes
    -----
    The input must be a continuous :class:`mne.io.BaseRaw` object. Maxwell
    filtering is performed through :func:`run_maxwell` using the configured
    coordinate frame, destination, calibration file, cross-talk file, and tSSS
    settings.

    The processed raw object is returned under ``'data'``.

    The processing metadata returned by :meth:`run` is stored under
    ``'maxwell_info'`` and contains the effective coordinate frame, destination,
    calibration file, cross-talk file, and tSSS duration.

    ``Maxwell`` can occur only once in a pipeline and is intended to run before
    ICA, forward modelling, spatial filtering, and source reconstruction.

    See Also
    --------
    run_maxwell
        Apply Maxwell filtering to a raw MEG recording.
    MultiBlockMaxwell
        Apply Maxwell filtering to multiple recording blocks before concatenating
        them.
    """

    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    mw_coord_frame: str = 'head'
    mw_destination: None | ArrayLike = None
    mw_calibration_file: str | bool | None = None
    mw_cross_talk_file: str | bool | None = None
    mw_st_duration: float | None = None
    mw_st_correlation: float = 0.98

    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        calibration_applied = self.mw_calibration_file is not False and (
            self.mw_calibration_file is not None or data.info.get('fine_calibration') is not None
        )

        cross_talk_applied = self.mw_cross_talk_file is not False and (
            self.mw_cross_talk_file is not None or data.info.get('cross_talk') is not None
        )
        raw_max = run_maxwell(
            raw=data,
            coord_frame=self.mw_coord_frame,
            destination=self.mw_destination,
            calibration_file=self.mw_calibration_file,
            cross_talk_file=self.mw_cross_talk_file,
            st_duration=self.mw_st_duration,
            st_correlation=self.mw_st_correlation,
        )

        return {
            'data': raw_max,
            'maxwell_info': {
                'coord_frame': self.mw_coord_frame,
                'destination': self.mw_destination,
                'calibration_file': self.mw_calibration_file,
                'cross_talk_file': self.mw_cross_talk_file,
                'st_duration': self.mw_st_duration,
                'st_correlation': self.mw_st_correlation,
                'calibration_applied': calibration_applied,
                'cross_talk_applied': cross_talk_applied,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_maxfiltered')


@define
class MultiBlockMaxwell(AlmKanalStep):
    """Apply Maxwell filtering to multiple raw MEG recording blocks.

    ``MultiBlockMaxwell`` applies Maxwell filtering independently to each raw MEG
    block and then concatenates the filtered blocks into a single continuous
    recording.

    All blocks are transformed to a common destination position before
    concatenation. If ``mw_destination`` is not provided, the destination is
    computed from the median device-to-head translation across all input blocks.

    Parameters
    ----------
    mw_coord_frame : str, default='head'
        Coordinate frame used for Maxwell filtering. Typically ``'head'`` or
        ``'meg'``.
    mw_destination : array-like | None, default=None
        Destination head position used during Maxwell filtering. If ``None``, a
        common destination is derived by taking the median of the translation
        components of ``info['dev_head_t']`` across all input blocks.
    mw_calibration_file : str | None, default=None
        Path to the fine-calibration file used by Maxwell filtering.
    mw_cross_talk_file : str | None, default=None
        Path to the cross-talk compensation file used by Maxwell filtering.
    mw_st_duration : float | None, default=None
        Temporal Signal Space Separation (tSSS) buffer duration, in seconds.
        ``None`` disables temporal SSS.

    Notes
    -----
    The input to :meth:`run` must be a list of :class:`mne.io.BaseRaw` objects.
    Each block is Maxwell filtered separately using the same coordinate frame,
    destination, calibration file, cross-talk file, and tSSS configuration.

    When ``mw_destination`` is ``None``, the common destination is computed from
    the median three-dimensional translation of the device-to-head transforms
    across all blocks. This provides a shared head position to which all blocks
    are aligned before concatenation.

    After Maxwell filtering, the processed blocks are concatenated with
    :func:`mne.concatenate_raws` and returned as a single continuous raw object.

    The processing metadata returned by :meth:`run` is stored under
    ``'maxwell_info'`` and contains the effective coordinate frame, destination,
    calibration file, cross-talk file, and tSSS duration.

    ``MultiBlockMaxwell`` can occur only once in a pipeline and is intended to run
    before ICA, forward modelling, spatial filtering, and source reconstruction.

    See Also
    --------
    run_maxwell
        Apply Maxwell filtering to an individual raw recording.
    mne.concatenate_raws
        Concatenate the filtered raw blocks.
    """

    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    mw_coord_frame: str = 'head'
    mw_destination: None | ArrayLike = None
    mw_calibration_file: None | str = None
    mw_cross_talk_file: None | str = None
    mw_st_duration: float | None = None

    def run(
        self,
        data: list[mne.io.BaseRaw],
        info: AlmKanalInfo,
    ) -> dict:
        if self.mw_destination is None:
            block_pos_l = [raw.info['dev_head_t']['trans'][:3, 3] for raw in data]
            destination = np.median(block_pos_l, axis=0)
        else:
            destination = self.mw_destination

        raw_max_list = []
        for raw in data:
            raw_max_list.append(
                run_maxwell(
                    raw=raw,
                    coord_frame=self.mw_coord_frame,
                    destination=destination,
                    calibration_file=self.mw_calibration_file,
                    cross_talk_file=self.mw_cross_talk_file,
                    st_duration=self.mw_st_duration,
                )
            )

        raw_max = mne.concatenate_raws(raw_max_list)

        return {
            'data': raw_max,
            'maxwell_info': {
                'coord_frame': self.mw_coord_frame,
                'destination': destination,
                'calibration_file': self.mw_calibration_file,
                'cross_talk_file': self.mw_cross_talk_file,
                'st_duration': self.mw_st_duration,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_maxfiltered')


@define
class EEGRANSAC(AlmKanalStep):
    """Detect and interpolate bad EEG channels using RANSAC.

    ``EEGRANSAC`` uses :class:`autoreject.Ransac` to identify EEG channels whose
    signals are poorly predicted from other EEG sensors. Continuous data are first
    split into fixed-length epochs for RANSAC fitting. Channels identified as bad
    are then marked in the original recording and interpolated using MNE.

    Only EEG channels participate in RANSAC detection and interpolation. Existing
    bad channels of other channel types are preserved and excluded from the
    interpolation step.

    Parameters
    ----------
    ransac_epoch_duration : int | float, default=4
        Duration, in seconds, of the fixed-length epochs created from the
        continuous recording for RANSAC fitting.
    n_resample : int, default=50
        Number of random sensor subsets used by RANSAC to estimate channel
        predictability.
    min_channels : float, default=0.25
        Fraction of available EEG channels used for robust reconstruction during
        each RANSAC resampling iteration.
    min_corr : float, default=0.75
        Minimum correlation between the measured and RANSAC-predicted signal for a
        channel to be considered sufficiently predictable.
    unbroken_time : float, default=0.4
        Fraction of time for which a channel may fall below ``min_corr`` before it
        is classified as bad.
    n_jobs : int, default=1
        Number of parallel jobs used by :class:`autoreject.Ransac`.
    verbose : bool, default=False
        Whether RANSAC should emit progress and diagnostic output.

    Notes
    -----
    The continuous recording is segmented into fixed-length epochs solely for
    RANSAC fitting. These temporary epochs are not returned and do not replace the
    continuous pipeline data.

    Only EEG channels are retained in the temporary RANSAC epochs. ECG, EOG, MEG,
    and other channel types therefore do not contribute to the bad-channel
    detection procedure.

    Previously marked bad channels are preserved when the newly detected EEG bad
    channels are added. During interpolation, previously bad non-EEG channels are
    explicitly excluded so that this step only repairs EEG channels.

    Interpolation is performed on the supplied raw object using
    :meth:`mne.io.Raw.interpolate_bads`.

    The processing metadata returned by :meth:`run` is stored under
    ``'ransac_info'`` and contains the RANSAC epoch duration and effective RANSAC
    configuration.

    ``EEGRANSAC`` may occur multiple times in a pipeline and is intended to run
    before ICA, forward modelling, spatial filtering, and source reconstruction.

    See Also
    --------
    autoreject.Ransac
        RANSAC implementation used for bad-channel detection.
    mne.make_fixed_length_epochs
        Create the temporary epochs used for RANSAC fitting.
    mne.io.Raw.interpolate_bads
        Interpolate channels marked as bad.
    """

    must_be_before: tuple = ('ICA', 'ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = True

    ransac_epoch_duration: int | float = 4
    n_resample: int = 50
    min_channels: float = 0.25
    min_corr: float = 0.75
    unbroken_time: float = 0.4
    n_jobs: int = 1
    verbose: bool = False

    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        epo4ransac = mne.make_fixed_length_epochs(data, duration=self.ransac_epoch_duration)
        # NOTE: We only do this for eeg -> think if this is overall smart
        # So why not use picks from autoreject.Ransac -> I am not sure hwo this will handle eg. ECG or EOG signals
        # Idont want them to be part of the ransac eeg spiel TODO: Test this
        epo4ransac.load_data().pick_types(eeg=True, ecg=False, eog=False)
        ransac = Ransac(
            n_resample=self.n_resample,
            min_channels=self.min_channels,
            min_corr=self.min_corr,
            unbroken_time=self.unbroken_time,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        ransac.fit(epo4ransac)
        bad_chs_eeg = ransac.bad_chs_
        print(f'RANSAC detected the following bad channels: {bad_chs_eeg}')

        previous_bads = data.info['bads'].copy()

        eeg_picks = mne.pick_types(
            data.info,
            eeg=True,
            meg=False,
            exclude=[],
        )

        eeg_ch_names = {data.ch_names[pick] for pick in eeg_picks}
        non_eeg_bads = [ch for ch in previous_bads if ch not in eeg_ch_names]

        data.info['bads'] = list(dict.fromkeys(previous_bads + bad_chs_eeg))

        raw_ransac = data.interpolate_bads(
            reset_bads=True,
            exclude=non_eeg_bads,
        )

        return {
            'data': raw_ransac,
            'ransac_info': {
                'ransac_epoch_duration': self.ransac_epoch_duration,
                'bad_chs_eeg': bad_chs_eeg,
                'n_resample': self.n_resample,
                'min_channels': self.min_channels,
                'min_corr': self.min_corr,
                'unbroken_time': self.unbroken_time,
                'n_jobs': self.n_jobs,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_ransac')


@define
class ReReference(AlmKanalStep):
    """Re-reference EEG, ECoG, sEEG, or DBS channels.

    ``ReReference`` changes the reference of electrophysiological channels using
    :meth:`mne.io.Raw.set_eeg_reference`. It supports references based on one or
    more existing channels, average referencing, REST referencing, and explicit
    per-channel reference mappings.

    Parameters
    ----------
    ref_channels : str | list of str | dict, default='average'
        Reference specification passed to
        :meth:`mne.io.Raw.set_eeg_reference`.

        Supported forms include:

        - ``'average'`` to use the average of the selected channel type.
        - ``'REST'`` to apply the Reference Electrode Standardization Technique.
        - A channel name or list of channel names defining the reference.
        - A dictionary mapping data-channel names to one or more reference
        channels, allowing different references for individual channels.
        - An empty list to leave the data unchanged.

    projection : bool, default=False
        Whether an average reference should be added as a projection rather than
        applied directly to the data. When ``True`` with
        ``ref_channels='average'``, the projection is added but is not immediately
        applied. Other reference schemes require direct re-referencing.
    ch_type : str | list of str, default='auto'
        Channel type or types to re-reference. Supported types include ``'eeg'``,
        ``'ecog'``, ``'seeg'``, and ``'dbs'``. With ``'auto'``, MNE selects the
        first supported channel type present in the recording.
    forward : mne.Forward | None, default=None
        Forward solution used for REST referencing. Only relevant when
        ``ref_channels='REST'``.
    joint : bool, default=False
        Whether multiple channel types specified by ``ch_type`` should share a
        common reference. If ``False``, referencing is handled separately for each
        channel type.
    verbose : bool | str | int | None, default=False
        Verbosity setting passed to :meth:`mne.io.Raw.set_eeg_reference`.

    Notes
    -----
    Re-referencing is performed in place on the supplied raw object. The modified
    object is returned under ``'data'``.

    Except when adding an average-reference projection, the data must be preloaded
    before the reference can be applied.

    For average referencing, channels marked as bad in ``data.info['bads']`` are
    excluded from the reference calculation by MNE.

    REST referencing requires a suitable forward solution supplied through
    ``forward``.

    The processing metadata returned by :meth:`run` is stored under ``'ref_info'``
    and contains the reference channels, projection setting, channel types,
    forward solution, joint-reference setting, and verbosity configuration.

    ``ReReference`` may occur multiple times in a pipeline and is intended to run
    before forward modelling, spatial filtering, and source reconstruction.

    See Also
    --------
    mne.io.Raw.set_eeg_reference
        Change the reference of channels in a raw recording.
    mne.set_eeg_reference
        General MNE function for re-referencing Raw, Epochs, or Evoked data.
    """

    must_be_before: tuple = ('ForwardModel', 'SpatialFilter', 'SourceReconstruction')
    must_be_after: tuple = ()
    allow_repeated: bool = True

    ref_channels: str | list[str] | dict = 'average'
    projection: bool = False
    ch_type: str | list[str] = 'auto'
    forward: mne.Forward | None = None
    joint: bool = False
    verbose: bool | str | int | None = False

    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        supported_types = ('eeg', 'ecog', 'seeg', 'dbs')

        if self.ch_type == 'auto':
            present_types = set(data.get_channel_types())
            resolved_ch_type: str | list[str] = next(ch_type for ch_type in supported_types if ch_type in present_types)
        else:
            resolved_ch_type = self.ch_type

        reref = data.set_eeg_reference(
            ref_channels=self.ref_channels,
            projection=self.projection,
            ch_type=self.ch_type,
            forward=self.forward,
            joint=self.joint,
            verbose=self.verbose,
        )

        return {
            'data': reref,
            'ref_info': {
                'ref_channels': self.ref_channels,
                'projection': self.projection,
                'ch_type': self.ch_type,
                'forward': self.forward,
                'resolved_ch_type': resolved_ch_type,
                'joint': self.joint,
                'verbose': self.verbose,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        report.add_raw(data, butterfly=False, psd=True, title='raw_reref')
