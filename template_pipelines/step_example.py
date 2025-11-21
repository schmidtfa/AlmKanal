import mne
from attrs import define

from almkanal import AlmKanalStep


@define
class Filter(AlmKanalStep):
    highpass: float = 0.1
    lowpass: float = 40

    must_be_before: tuple = ()
    must_be_after: tuple = ()

    def run(self, data: mne.io.BaseRaw | mne.BaseEpochs, info: dict) -> dict:
        data.filter(
            l_freq=self.highpass,
            h_freq=self.lowpass,
            picks=self.picks,
        )

        return {
            'data': data,
            'filter_info': {
                'l_freq': self.highpass,
                'h_freq': self.lowpass,
            },
        }

    def reports(self, data: mne.io.BaseRaw | mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title='Raw (filtered)')

        elif isinstance(data, mne.BaseEpochs):
            report.add_epochs(data, title='Epoched (filtered)')
