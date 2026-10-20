"""Teaching example: continuous-only notch filtering with minimal metadata."""
from typing import Any

import mne
from attrs import define, field
from almkanal import AlmKanalStep
from almkanal.info import AlmKanalInfo


@define
class NotchAtFrequency(AlmKanalStep):
    """Apply a single-frequency notch filter to continuous data.

    Extend the recorded filter parameters before using this teaching step
    as the basis of a complete scientific Methods description.
    """
    frequency: float = 50.0
    must_be_before: tuple[str, ...] = field(
        default=("SourceReconstruction",), init=False
    )

    def run(self, data: mne.io.BaseRaw, info: AlmKanalInfo) -> dict[str, Any]:
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("NotchAtFrequency requires continuous Raw data.")
        if not 0 < self.frequency < data.info["sfreq"] / 2:
            raise ValueError("Frequency must be between zero and Nyquist.")
        data.load_data().notch_filter(freqs=[self.frequency])
        return {
            "data": data,
            "notch_info": {"frequency_hz": self.frequency},
        }

    def reports(
        self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo
    ) -> None:
        report.add_raw(
            data, title="After notch filtering", butterfly=False, psd=True
        )
