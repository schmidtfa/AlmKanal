from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import mne
from attrs import define

from almkanal import AlmKanalStep

if TYPE_CHECKING:
    import numpy.typing as npt

    from almkanal.info import AlmKanalInfo


@define
class Filter(AlmKanalStep):
    """Apply temporal filtering to continuous or epoched M/EEG data.

    ``Filter`` applies high-pass, low-pass, band-pass, or band-stop filtering
    using MNE's filtering implementation. The configured filter is applied
    directly to the supplied :class:`mne.io.BaseRaw` or
    :class:`mne.BaseEpochs` object.

    In addition to performing the filter operation, the step records the
    effective filter settings in the processing history. When transition
    bandwidths are set to ``'auto'``, concrete values are resolved before
    filtering so that the reported preprocessing parameters remain explicit
    and reproducible.

    Parameters
    ----------
    highpass : float | None, default=0.1
        High-pass cutoff frequency, in Hz. Passed to MNE as ``l_freq``.
        ``None`` disables high-pass filtering.
    lowpass : float | None, default=40.0
        Low-pass cutoff frequency, in Hz. Passed to MNE as ``h_freq``.
        ``None`` disables low-pass filtering.
    picks : str | array-like | slice | None, default=None
        Channels to filter. Passed directly to MNE.
    filter_length : str | int, default='auto'
        Filter length passed to MNE.
    l_trans_bandwidth : float | {'auto'}, default='auto'
        Transition bandwidth, in Hz, below the high-pass cutoff. When set to
        ``'auto'``, a concrete value following MNE's default rule is computed
        before filtering and stored in the processing metadata.
    h_trans_bandwidth : float | {'auto'}, default='auto'
        Transition bandwidth, in Hz, above the low-pass cutoff. When set to
        ``'auto'``, a concrete value following MNE's default rule is computed
        before filtering and stored in the processing metadata.
    n_jobs : int | str | None, default=None
        Number of parallel jobs used for filtering.
    method : str, default='fir'
        Filtering method passed to MNE, for example ``'fir'`` or ``'iir'``.
    iir_params : dict | None, default=None
        Parameters used for IIR filtering. When ``method='iir'`` and
        ``iir_params`` is ``None``, MNE's effective default Butterworth
        configuration is recorded in the processing metadata.
    phase : str, default='zero'
        Phase response used for filtering.
    fir_window : str, default='hamming'
        Window used for FIR filter design.
    fir_design : str, default='firwin'
        FIR design method passed to MNE.
    skip_by_annotation : str | tuple of str | list of str, default=('edge', 'bad_acq_skip')
        Annotation descriptions over which filtering is skipped.
    pad : str, default='reflect_limited'
        Padding mode used at the signal boundaries.

    Notes
    -----
    The filter is applied in place to the supplied MNE object.

    ``highpass`` and ``lowpass`` correspond to MNE's ``l_freq`` and ``h_freq``
    parameters, respectively. Supplying only ``highpass`` produces a high-pass
    filter, supplying only ``lowpass`` produces a low-pass filter, and supplying
    both produces a band-pass filter.

    When transition bandwidths are set to ``'auto'``, the values used for
    filtering are resolved explicitly before the MNE call. This ensures that
    the processing history records concrete transition bandwidths rather than
    the symbolic ``'auto'`` setting.

    ``skip_by_annotation`` is normalized to a JSON-friendly representation
    before being stored in the processing metadata.

    The processing metadata returned by :meth:`run` is stored under
    ``'filter_info'`` and contains the effective cutoff frequencies,
    transition bandwidths, filter design parameters, channel selection, and
    annotation-skipping configuration.

    ``Filter`` may occur multiple times in a pipeline.

    See Also
    --------
    mne.io.Raw.filter
        Filter continuous M/EEG data.
    mne.Epochs.filter
        Filter epoched M/EEG data.
    """

    highpass: float | None = 0.1
    lowpass: float | None = 40.0
    picks: str | npt.ArrayLike | slice | None = None
    filter_length: str | int = 'auto'
    l_trans_bandwidth: float | Literal['auto'] = 'auto'
    h_trans_bandwidth: float | Literal['auto'] = 'auto'
    n_jobs: int | str | None = None
    method: str = 'fir'
    iir_params: dict[str, Any] | None = None
    phase: str = 'zero'
    fir_window: str = 'hamming'
    fir_design: str = 'firwin'
    skip_by_annotation: str | tuple[str, ...] | list[str] = ('edge', 'bad_acq_skip')
    pad: str = 'reflect_limited'

    must_be_before: tuple[str, ...] = ()
    must_be_after: tuple[str, ...] = ()
    allow_repeated: bool = True

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: AlmKanalInfo) -> dict[str, Any]:
        # --- compute transition bandwidths for REPORT (floats) and for API call (float | 'auto')
        l_tb_report: float | None = None
        h_tb_report: float | None = None

        if self.l_trans_bandwidth == 'auto':
            if self.highpass is not None:
                # exact value inspired by MNE default, but concrete for reporting
                l_tb_report = float(min(max(self.highpass * 0.25, 2.0), self.highpass))
            l_tb_param: float | str = l_tb_report if l_tb_report is not None else 'auto'
        else:
            l_tb_param = self.l_trans_bandwidth
            l_tb_report = float(self.l_trans_bandwidth)

        if self.h_trans_bandwidth == 'auto':
            if self.lowpass is not None:
                nyq_margin = float(data.info['sfreq']) / 2.0 - self.lowpass
                h_tb_report = float(min(max(self.lowpass * 0.25, 2.0), nyq_margin))
            h_tb_param: float | str = h_tb_report if h_tb_report is not None else 'auto'
        else:
            h_tb_param = self.h_trans_bandwidth
            h_tb_report = float(self.h_trans_bandwidth)

        # --- apply filter
        data.filter(
            l_freq=self.highpass,
            h_freq=self.lowpass,
            picks=self.picks,
            filter_length=self.filter_length,
            l_trans_bandwidth=l_tb_param,
            h_trans_bandwidth=h_tb_param,
            n_jobs=self.n_jobs,
            method=self.method,
            iir_params=self.iir_params,
            phase=self.phase,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
            skip_by_annotation=self.skip_by_annotation,
            pad=self.pad,
        )

        # normalize skip_by_annotation to a JSON-friendly list (keep str as-is)
        if isinstance(self.skip_by_annotation, str):
            skip_for_json: str | list[str] = self.skip_by_annotation
        else:
            skip_for_json = list(self.skip_by_annotation)

        # default iir_params only when needed
        iir_params_json: dict[str, Any] | None
        if self.method == 'iir' and self.iir_params is None:
            iir_params_json = {'order': 4, 'ftype': 'butter', 'output': 'sos'}
        else:
            iir_params_json = self.iir_params

        return {
            'data': data,
            'filter_info': {
                'l_freq': self.highpass,
                'h_freq': self.lowpass,
                'picks': self.picks,
                'filter_length': self.filter_length,
                'l_trans_bandwidth': l_tb_report,
                'h_trans_bandwidth': h_tb_report,
                'n_jobs': self.n_jobs,
                'method': self.method,
                'iir_params': iir_params_json,
                'phase': self.phase,
                'fir_window': self.fir_window,
                'fir_design': self.fir_design,
                'skip_by_annotation': skip_for_json,
                'pad': self.pad,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: AlmKanalInfo) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='Raw (filtered)')
        elif isinstance(data, mne.BaseEpochs):
            evokeds = data.average(by_event_type=True)
            report.add_evokeds(evokeds, n_time_points=5)


@define
class Resample(AlmKanalStep):
    """Resample continuous or epoched M/EEG data.

    ``Resample`` changes the sampling frequency of an
    :class:`mne.io.BaseRaw` or :class:`mne.BaseEpochs` object using MNE's
    resampling implementation.

    When downsampling, the step requires the data to have been explicitly
    low-pass filtered at or below the Nyquist frequency of the requested
    sampling rate. Although MNE applies anti-aliasing internally during
    resampling, AlmKanal requires an explicit preceding low-pass filter so that
    the effective preprocessing settings are recorded in the processing
    history and can be reported reproducibly.

    Parameters
    ----------
    sfreq : int
        Target sampling frequency, in Hz.
    npad : str, default='auto'
        Amount of padding used during resampling. Passed directly to MNE's
        ``resample`` method.
    window : str, default='auto'
        Window specification used for resampling. Passed directly to MNE's
        ``resample`` method.
    n_jobs : int | None, default=None
        Number of parallel jobs used during resampling.
    pad : str, default='auto'
        Padding mode used at the signal boundaries.
    method : str, default='fft'
        Resampling method passed to MNE.

    Notes
    -----
    Before resampling, the current low-pass frequency is compared with the
    Nyquist frequency of ``sfreq``. If the low-pass cutoff exceeds the new
    Nyquist frequency, :meth:`run` raises a :class:`ValueError` and instructs
    the user to apply an explicit anti-aliasing filter first.

    Resampling is performed in place on the supplied MNE object.

    The processing metadata returned by :meth:`run` is stored under
    ``'resample_info'`` and contains the target sampling frequency together
    with the effective resampling parameters.

    ``Resample`` may occur multiple times in a pipeline.

    See Also
    --------
    mne.io.Raw.resample
        Resample continuous data.
    mne.Epochs.resample
        Resample epoched data.
    """

    sfreq: int
    npad: str = 'auto'
    window: str = 'auto'
    n_jobs: int | None = None
    pad: str = 'auto'
    method: str = 'fft'
    must_be_before: tuple[str, ...] = ()
    must_be_after: tuple[str, ...] = ('Epochs',)
    allow_repeated: bool = True

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: AlmKanalInfo) -> dict[str, Any]:
        if not (self.sfreq / 2 >= float(data.info['lowpass'])):
            lowpass = float(data.info['lowpass'])
            raise ValueError(
                'You need to apply an anti-aliasing filter before downsampling the data.'
                f' Currently you lowpass the data at {lowpass} Hz. '
                'Note: MNE’s resampling applies an internal anti-aliasing filter, but this pipeline '
                'prefers explicit filter settings for reporting.'
            )

        data.resample(
            sfreq=self.sfreq,
            npad=self.npad,
            window=self.window,
            n_jobs=self.n_jobs,
            pad=self.pad,
            method=self.method,
        )

        return {
            'data': data,
            'resample_info': {
                'sfreq': self.sfreq,
                'npad': self.npad,
                'window': self.window,
                'pad': self.pad,
                'method': self.method,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: AlmKanalInfo) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='RawResample')
        elif isinstance(data, mne.BaseEpochs):
            evokeds = data.average(by_event_type=True)
            report.add_evokeds(evokeds, n_time_points=5)
