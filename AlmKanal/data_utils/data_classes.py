from typing import TypedDict

from attrs import define


@define
class InfoClass:
    """Container for the info field of the AlmKanal workflow"""

    raw: bool = False
    epoched: bool = False
    maxwell: dict | None = None
    ica: dict | None = None
    epoched: dict | None = None
    trf_epochs: dict | None = None


class PickDictClass(TypedDict):
    """Container for data picking"""

    meg: bool = True
    eog: bool = True
    ecg: bool = True
    eeg: bool = False
    stim: bool = True
