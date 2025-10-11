from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import mne
from attrs import define

from almkanal import AlmKanalStep

if TYPE_CHECKING:
    import numpy.typing as npt


@define
class Filter(AlmKanalStep):
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

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict[str, Any]) -> dict[str, Any]:
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
                'l_trans_bandwidth': None if l_tb_report is None else round(l_tb_report, 2),
                'h_trans_bandwidth': None if h_tb_report is None else round(h_tb_report, 2),
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

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict[str, Any]) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='Raw (filtered)')
        elif isinstance(data, mne.BaseEpochs):
            base_corr = data.copy()
            base_corr.apply_baseline(baseline=(None, 0))
            evokeds = base_corr.average(by_event_type=True)
            report.add_evokeds(evokeds, n_time_points=5)


@define
class Resample(AlmKanalStep):
    sfreq: int
    npad: str = 'auto'
    window: str = 'auto'
    n_jobs: int | None = None
    pad: str = 'auto'
    method: str = 'fft'
    must_be_before: tuple[str, ...] = ()
    must_be_after: tuple[str, ...] = ('Epochs',)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict[str, Any]) -> dict[str, Any]:
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
                'n_jobs': self.n_jobs,
                'pad': self.pad,
                'method': self.method,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict[str, Any]) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='RawResample')
        elif isinstance(data, mne.BaseEpochs):
            base_corr = data.copy()
            base_corr.apply_baseline(baseline=(None, 0))
            evokeds = base_corr.average(by_event_type=True)
            report.add_evokeds(evokeds, n_time_points=5)
