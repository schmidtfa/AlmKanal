import warnings

import mne
import numpy as np
from attrs import define
from pyrasa.irasa import irasa
from pyrasa.utils.peak_utils import get_band_info

from almkanal.almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo


def eog_ica_from_meg(
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    left_eog_chs: list = ['MEG0121', 'MEG0311'],
    right_eog_chs: list = ['MEG1211', 'MEG1411'],
    threshold: float = 0.5,
    tol: float = 0.2,
) -> list:
    """
    Detect eye related activity in MEG data using ICA component correlation and dipolar pattern analysis.

    This function identifies ICA components corresponding to eye related activity by computing the correlation
    between ICA components and user-specified left/right MEG channels. A component is considered
    eye related if it surpasses the correlation threshold and exhibits a dipolar pattern across
    left and right surrogate EOG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data used for ICA decomposition.
    ica : mne.preprocessing.ICA
        The fitted ICA object containing independent components.
    left_eog_chs : list, optional
        List of MEG channels representing left-side eye movements. Defaults to ['MEG0121', 'MEG0311'].
    right_eog_chs : list, optional
        List of MEG channels representing right-side eye movements. Defaults to ['MEG1211', 'MEG1411'].
    threshold : float, optional
        Correlation threshold for detecting EOG-related ICA components. Defaults to 0.5.
    tol: float, optional
        Difference in correlation between left and right channels. Should be below threshold for a clear blink pattern.

    Returns
    -------
    list
        Indices of ICA components identified as eye related activity.
    """

    eog_list = left_eog_chs + right_eog_chs
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_list, measure='correlation', threshold=threshold)

    left_scores = np.mean([eog_scores[ix] for ix, _ in enumerate(left_eog_chs)], axis=0)
    right_scores = np.mean([eog_scores[ix + len(left_eog_chs)] for ix, _ in enumerate(right_eog_chs)], axis=0)

    for eog_ix in eog_indices.copy():
        if np.logical_or(
            np.logical_and(left_scores[eog_ix] > 0, right_scores[eog_ix] < 0),
            np.logical_and(left_scores[eog_ix] < 0, right_scores[eog_ix] > 0),
        ):
            if np.isclose(np.abs(left_scores[eog_ix]), np.abs(right_scores[eog_ix]), atol=tol):
                print('Dipolar pattern on candidate EOG component detected. The Component is considered valid')

            else:
                eog_indices.remove(eog_ix)

        else:
            eog_indices.remove(eog_ix)

    return eog_indices


def run_ica(  # noqa: C901, PLR0912, PLR0915
    raw: mne.io.Raw,
    fit_only: bool = False,
    n_components: None | int | float = None,
    method: str = 'picard',
    random_state: None | int = 42,
    fit_params: dict | None = None,
    resample_freq: None | int = None,
    ica_hp_freq: None | float = 1.0,
    ica_lp_freq: None | float = None,
    eog: bool = True,
    surrogate_eog_chs: None | dict = None,
    eog_corr_thresh: float = 0.5,
    ecg: bool = True,
    ecg_corr_thresh: float = 0.5,
    emg: bool = False,
    emg_thresh: float = 0.5,
    train: bool = True,
    train_freq: float = 16.666,
    train_thresh: float = 2,
) -> tuple[mne.io.Raw, mne.preprocessing.ICA, dict, list, list]:
    """
    Run ICA on raw MEG data to identify and remove artifacts (EOG, ECG, train).

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data to process.
    n_components : None | int | float, optional
        Number of ICA components to compute. Defaults to None.
    method : str, optional
        ICA method to use (default is 'picard').
    random_state : None | int, optional
        Random seed for reproducibility. Defaults to 42.
    fit_params : dict | None, optional
        Additional fitting parameters for ICA. Defaults to None.
    resample_freq : None | int, optional
        Resampling frequency before ICA. Defaults to None.
    ica_hp_freq : None | float, optional
        High-pass filter frequency for ICA preprocessing. Defaults to 1.0 Hz.
    ica_lp_freq : None | float, optional
        Low-pass filter frequency for ICA preprocessing. Defaults to None.
    eog : bool, optional
        Whether to identify and remove EOG artifacts. Defaults to True.
    eog_corr_thresh : float, optional
        Correlation threshold for EOG artifact detection. Defaults to 0.5.
    ecg : bool, optional
        Whether to identify and remove ECG artifacts. Defaults to True.
    ecg_corr_thresh : float, optional
        Correlation threshold for ECG artifact detection. Defaults to 0.5.
    train : bool, optional
        Whether to identify and remove train artifacts. Defaults to True.
    train_freq : int, optional
        Frequency to use for train artifact detection. Defaults to 16 Hz.

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed MEG data with ICA-applied artifact removal.
    ica : mne.preprocessing.ICA
        ICA object containing decomposition results.
    bad_ids : list
        List of identified and excluded component indices.
    """

    # we run ica on high-pass filtered data
    raw_copy = raw.copy().filter(l_freq=ica_hp_freq, h_freq=ica_lp_freq)
    if resample_freq is not None:
        raw_copy.resample(resample_freq)
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=random_state, method=method, fit_params=fit_params
    )
    ica.fit(raw_copy)

    bads = []
    components_dict = {}
    # find which ICs match the EOG/ECG pattern using correlation
    # check if ecg and eog channels are present in the data
    ch_dict = mne.channel_indices_by_type(raw_copy.info, picks='all')

    if eog:
        if np.logical_and(len(ch_dict['eog']) == 0, (len(ch_dict['mag']) + len(ch_dict['grad'])) > 0):
            if surrogate_eog_chs is not None:
                eog_idcs = eog_ica_from_meg(raw_copy, ica=ica, threshold=eog_corr_thresh, **surrogate_eog_chs)
                warnings.warn("""No EOG channels detected switching to an experimental method detecting EOG
                            related activity via correlation with user specified MEG channels.
                            Please plot the EOG components to verify that they are sensible!!!""")
            else:
                raise ValueError('You need to specify a dictionary of left and right surrogate EOG channels.')

        elif np.logical_and(len(ch_dict['eog']) == 0, (len(ch_dict['mag']) + len(ch_dict['grad'])) == 0):
            raise ValueError(
                """No EOG channels detected. You need to specify EOG channels,
                if you want to reject EOG components via correlation.
                The EOG detection method using surrogate channels only works for MEG data."""
            )
        else:
            eog_idcs, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eog_corr_thresh)

        components_dict.update({'eog': eog_idcs})
        bads.append(eog_idcs)

    if ecg:
        if len(ch_dict['ecg']) == 0:
            if (len(ch_dict['mag']) + len(ch_dict['grad'])) == 0:
                raise ValueError(
                    'No ECG or MEG channels detected. ECG component detection '
                    'requires either an ECG channel or MEG data.'
                )

            warnings.warn('No ECG channels detected. ECG channel is constructed from MEG data.')

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_copy)
        # find the ECG components,
        ecg_idcs, ecg_scores = ica.find_bads_ecg(
            ecg_epochs, method='correlation', measure='correlation', threshold=ecg_corr_thresh
        )
        components_dict.update({'ecg': ecg_idcs})
        bads.append(ecg_idcs)

    if emg:
        emg_idcs, _ = ica.find_bads_muscle(raw_copy, threshold=emg_thresh)
        components_dict.update({'emg': emg_idcs})
        bads.append(emg_idcs)

    # remove train based
    if train:
        train_idcs = find_train_ica(raw_copy, ica, train_freq, peak_threshold=train_thresh)
        components_dict.update({'train': train_idcs})
        bads.append(train_idcs)

    bad_ids = np.concatenate(bads).astype(int).tolist() if len(bads) > 0 else []
    bad_ids = sorted(set(bad_ids))

    ica.exclude = bad_ids

    if 'eog_scores' not in locals():
        eog_scores = None
    if 'ecg_scores' not in locals():
        ecg_scores = None

    # % drop physiological components
    if not fit_only:
        ica.apply(raw)

        note = f'# excluded components: {len(bad_ids)}; ' f'excluded ICA: {bad_ids}'

        previous_description = raw.info.get('description')

        raw.info['description'] = f'{previous_description}\n{note}' if previous_description else note

    return raw, ica, components_dict, eog_scores, ecg_scores


def find_train_ica(
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    train_freq: float,
    duration: int = 8,
    overlap: float = 0.5,
    hmax: float = 2,
    peak_threshold: float = 3,
) -> list:
    """
    Detect ICA components associated with train artifacts in MEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG data used during ICA fitting.
    ica : mne.preprocessing.ICA
        The ICA object containing the decomposition results.
    train_freq : int
        The target frequency of the train artifact (e.g., 16 Hz).
    duration : int, optional
        Window duration (in seconds) for PSD computation. Defaults to 4.
    overlap : float, optional
        Overlap ratio for PSD computation windows. Defaults to 0.5.
    hmax : float, optional
        Maximum up/downsampling factor for IRASA. Defaults to 2.
    peak_threshold : float, optional
        Standard deviation threshold for peak power detection. Defaults to 2.

    Returns
    -------
    list
        List of ICA component indices associated with train artifacts.
    """

    # get info for train timeseries
    ic_signal = ica.get_sources(raw).get_data()

    # run irasa
    fs = raw.info['sfreq']

    eps = 1e-4
    lower, upper = (raw.info['highpass'] * hmax) + eps, (raw.info['lowpass'] / hmax) - eps

    if not lower <= train_freq <= upper:
        raise ValueError(
            f'Train frequency ({train_freq:.2f} Hz) is outside the '
            f'IRASA frequency range ({lower:.2f}-{upper:.2f} Hz).'
        )

    ch_names = list(np.arange(ic_signal.shape[0]))

    irasa_out = irasa(
        ic_signal,
        fs=fs,
        band=(lower, upper),
        nperseg=int(duration * fs),
        noverlap=int(duration * fs * overlap),
        hset_info=(1, hmax, 0.05),
        ch_names=np.asarray(ch_names),
        filter_settings=(
            raw.info['highpass'],
            raw.info['lowpass'],
        ),
    )

    train_peaks = get_band_info(
        irasa_out.get_peaks(peak_threshold=peak_threshold,
                            peak_width_limits=(0.5, 2),
                            smooth=False),
        freq_range=(train_freq - 0.5, train_freq + 0.5),  # type: ignore[arg-type]
        ch_names=ch_names,

    ).dropna()

    bad_ics = train_peaks['ch_name'].astype(int).tolist()
    return bad_ics


@define
class ICA(AlmKanalStep):
    """Identify and optionally remove artifacts using independent component analysis.

    ``ICA`` fits an independent component analysis model to continuous raw data and
    identifies components associated with physiological or environmental artifacts.
    Supported artifact classes include eye movements (EOG), cardiac activity (ECG),
    muscle activity (EMG), and the approximately 16 Hz train-related artifact
    encountered in recordings acquired in Salzburg.

    The data used to fit ICA can be filtered and resampled independently of the
    original recording. Identified components can either be removed immediately or
    retained for inspection by setting ``fit_only=True``.

    Parameters
    ----------
    fit_only : bool, default=False
        If ``True``, fit ICA and identify artifact-related components without
        removing them from the data. If ``False``, identified components are
        excluded and ICA is applied to the original recording.
    n_components : int | float | None, default=None
        Number of ICA components to estimate. The interpretation follows
        :class:`mne.preprocessing.ICA`: an integer specifies an explicit number of
        components, while a float specifies the fraction of explained variance.
    method : str, default='picard'
        ICA fitting method passed to :class:`mne.preprocessing.ICA`.
    random_state : int | None, default=42
        Random seed used during ICA fitting.
    fit_params : dict | None, default=None
        Additional method-specific parameters passed to the ICA fitting procedure.
    ica_hp_freq : float | None, default=1.0
        High-pass cutoff frequency, in Hz, applied to the copy of the data used for
        ICA fitting. ``None`` disables high-pass filtering.
    ica_lp_freq : float | None, default=None
        Low-pass cutoff frequency, in Hz, applied to the copy of the data used for
        ICA fitting. ``None`` disables low-pass filtering.
    resample_freq : int, default=200
        Sampling frequency, in Hz, used for the data on which ICA is fitted.
    eog : bool, default=True
        Whether to identify components associated with ocular artifacts.
    surrogate_eog_chs : dict | None, default=None
        Definition of surrogate EOG channels used when dedicated EOG channels are
        unavailable.
    eog_corr_thresh : float, default=0.5
        Correlation threshold used to classify components as EOG-related.
    ecg : bool, default=True
        Whether to identify components associated with cardiac artifacts.
    ecg_corr_thresh : float, default=0.5
        Correlation threshold used to classify components as ECG-related.
    emg : bool, default=False
        Whether to identify components associated with muscle activity.
    emg_thresh : float, default=0.5
        Threshold used to classify components as EMG-related.
    train : bool, default=True
        Whether to identify components associated with the train-related artifact.
    train_freq : int, default=16
        Frequency, in Hz, around which the train-related artifact is detected.
    train_thresh : float, default=2.0
        Threshold used to classify components as train-related.
    img_path : str | None, default=None
        Optional image-output path retained as part of the ICA step configuration.
    fname : str | None, default=None
        Optional filename retained as part of the ICA step configuration.

    Notes
    -----
    ICA preprocessing is performed on a working copy of the recording. Filtering
    and resampling controlled by ``ica_hp_freq``, ``ica_lp_freq``, and
    ``resample_freq`` therefore do not replace the corresponding preprocessing of
    the pipeline data itself.

    When ``fit_only=False``, components identified as artifacts are excluded and
    the fitted ICA solution is applied to the original data. When
    ``fit_only=True``, the fitted ICA model and identified components are returned
    without modifying the input signal through component removal.

    The processing metadata returned by :meth:`run` is stored under ``'ica_info'``.
    It contains the fitted ICA object, identified components grouped by artifact
    type, the applied exclusion list, EOG and ECG scores, and the effective ICA
    configuration.

    Multiple ``ICA`` steps are permitted in a pipeline. Reporting uses the most
    recent occurrence.

    See Also
    --------
    run_ica
        Fit ICA, identify artifact-related components, and optionally apply the
        resulting exclusions.
    mne.preprocessing.ICA
        MNE-Python implementation of independent component analysis.
    """

    must_be_before: tuple = ()
    must_be_after: tuple = ('Maxwell',)
    allow_repeated: bool = True

    fit_only: bool = False
    n_components: None | int | float = None
    method: str = 'picard'
    random_state: None | int = 42
    fit_params: dict | None = None
    ica_hp_freq: None | float = 1.0
    ica_lp_freq: None | float = None
    resample_freq: int = 200  # downsample to 200hz per default
    eog: bool = True
    surrogate_eog_chs: None | dict = None
    eog_corr_thresh: float = 0.5
    ecg: bool = True
    ecg_corr_thresh: float = 0.5
    emg: bool = False
    emg_thresh: float = 0.5
    train: bool = True
    train_freq: float = 16.666
    train_thresh: float = 2.0
    img_path: None | str = None
    fname: None | str = None

    def run(
        self,
        data: mne.io.Raw,
        info: AlmKanalInfo,
    ) -> dict:
        raw, ica, components_dict, eog_scores, ecg_scores = run_ica(
            data,
            fit_only=self.fit_only,
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            fit_params=self.fit_params,
            ica_hp_freq=self.ica_hp_freq,
            ica_lp_freq=self.ica_lp_freq,
            resample_freq=self.resample_freq,
            eog=self.eog,
            eog_corr_thresh=self.eog_corr_thresh,
            surrogate_eog_chs=self.surrogate_eog_chs,
            ecg=self.ecg,
            ecg_corr_thresh=self.ecg_corr_thresh,
            emg=self.emg,
            emg_thresh=self.emg_thresh,
            train=self.train,
            train_freq=self.train_freq,
            train_thresh=self.train_thresh,
        )

        return {
            'data': raw,
            'ica_info': {
                'ica': ica,
                'component_ids': list(components_dict.values()),
                'components_dict': components_dict,
                'fit_only': self.fit_only,
                'applied_exclude': [] if self.fit_only else ica.exclude.copy(),
                'eog_scores': eog_scores,
                'ecg_scores': ecg_scores,
                'n_components': self.n_components,
                'method': self.method,
                'fit_params': self.fit_params,
                'ica_hp_freq': self.ica_hp_freq,
                'ica_lp_freq': self.ica_lp_freq,
                'resample_freq': self.resample_freq,
                'eog': self.eog,
                'eog_corr_thresh': self.eog_corr_thresh,
                'surrogate_eog_chs': self.surrogate_eog_chs,
                'ecg': self.ecg,
                'ecg_corr_thresh': self.ecg_corr_thresh,
                'emg': self.emg,
                'emg_thresh': self.emg_thresh,
                'train': self.train,
                'train_freq': self.train_freq,
                'train_thresh': self.train_thresh,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: AlmKanalInfo) -> None:
        titles = {}
        cur_ica_info = info.get_step_info('ICA', occurrence=-1, required=True)
        for key, vals in cur_ica_info['ica_info']['components_dict'].items():
            for val in vals:
                titles.update({int(val): f'{key}'})

        if len(titles) > 0:
            report.add_ica(
                cur_ica_info['ica_info']['ica'],
                inst=data,
                title='ICA',
                ecg_scores=cur_ica_info['ica_info']['ecg_scores'],
                eog_scores=cur_ica_info['ica_info']['eog_scores'],
                picks=list(titles.keys()),
                tags=list(titles.values()),
            )

            if isinstance(data, mne.io.BaseRaw):
                report.add_raw(data, butterfly=False, psd=True, title='Raw (ICA)')
        else:
            print('No bad ICA components detected.')
