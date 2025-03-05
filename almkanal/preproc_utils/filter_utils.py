import mne
from attrs import define

from almkanal import AlmKanalStep


@define
class Filter(AlmKanalStep):
    highpass: float = 0.1
    lowpass: float = 40
    picks = None
    filter_length = 'auto'
    l_trans_bandwidth = 'auto'
    h_trans_bandwidth = 'auto'
    n_jobs = None
    method = 'fir'
    iir_params = None
    phase = 'zero'
    fir_window = 'hamming'
    fir_design = 'firwin'
    skip_by_annotation = ('edge', 'bad_acq_skip')
    pad = 'reflect_limited'

    must_be_before: tuple = ()
    must_be_after: tuple = ()

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        data.filter(
            l_freq=self.highpass,
            h_freq=self.lowpass,
            picks=self.picks,
            filter_length=self.filter_length,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth,
            n_jobs=self.n_jobs,
            method=self.method,
            iir_params=self.iir_params,
            phase=self.phase,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
            skip_by_annotation=self.skip_by_annotation,
            pad=self.pad,
        )

        return {
            'data': data,
            'filter_info': {
                'l_freq': self.highpass,
                'h_freq': self.lowpass,
                'picks': self.picks,
                'filter_length': self.filter_length,
                'l_trans_bandwidth': self.l_trans_bandwidth,
                'h_trans_bandwidth': self.h_trans_bandwidth,
                'n_jobs': self.n_jobs,
                'method': self.method,
                'iir_params': self.iir_params,
                'phase': self.phase,
                'fir_window': self.fir_window,
                'fir_design': self.fir_design,
                'skip_by_annotation': self.skip_by_annotation,
                'pad': self.pad,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='Raw (filtered)')

        elif isinstance(data, mne.BaseEpochs):
            report.add_epochs(data, title='Epoched (filtered)')


@define
class Resample(AlmKanalStep):
    sfreq: int
    npad = 'auto'
    window = 'auto'
    n_jobs = None
    pad = 'auto'
    method = 'fft'
    must_be_before: tuple = ()
    must_be_after: tuple = ('Epochs',)

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        if self.sfreq > data.info['lowpass'] // 2:
            raise ValueError(
                'You need to apply an anti-aliasing filter before downsampling the data.'
                f' Currently you lowpass the data at {data.info['lowpass']}Hz. '
                'Note: I am aware that the resampling method in MNE does that automatically, '
                'but I am not a big fan of that approach hence the error message.'
            )

        data.resample(
            sfreq=self.sfreq,
            npad=self.npad,
            window=self.window,
            stim_picks=None,
            n_jobs=self.n_jobs,
            events=None,
            pad=self.pad,
            method=self.method,
        )

        return {
            'data': data,
            'resample_info': {
                'sfreq': self.sfreq,
                'npad': self.npad,
                'window': self.window,
                'stim_picks': None,
                'n_jobs': self.n_jobs,
                'events': None,
                'pad': self.pad,
                'method': self.method,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='RawFilter')

        elif isinstance(data, mne.BaseEpochs):
            report.add_epochs(data, title='EpochedFilter')
