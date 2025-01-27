from typing import TypedDict

from attrs import define


class PickDictClass(TypedDict):
    """Container for data picking"""

    meg: bool
    eog: bool
    ecg: bool
    eeg: bool
    stim: bool


class ICAInfoDict(TypedDict):
    """Container for Dict Info"""

    n_components: int | float | None
    method: str
    random_state: None | int
    fit_params: dict | None
    ica_hp_freq: None | float
    ica_lp_freq: None | float
    resample_freq: None | int
    eog: bool
    eog_corr_thresh: float
    ecg: bool
    ecg_corr_thresh: float
    train: bool
    train_freq: int
    img_path: None | str
    fname: None | str


@define
class InfoClass:
    """Container for the info field of the AlmKanal workflow"""

    raw: bool = False
    epoched: bool = False
    maxwell: dict | None = None
    ica: list[ICAInfoDict] | None = None
    trf_epochs: dict | None = None
