import mne
from attrs import define, field


@define(kw_only=True, slots=False)
class ReportMixin:
    report: mne.Report = field(factory=mne.Report)


class AlmkanalRaw(mne.io.Raw, ReportMixin):
    """Container for the raw field of the AlmKanal workflow"""

    @classmethod
    def from_mne_raw(cls, raw: mne.io.Raw) -> 'AlmkanalRaw':
        new_raw = cls.__new__(cls)
        new_raw.__dict__.update(raw.__dict__)
        ReportMixin.__init__(new_raw)
        return new_raw
