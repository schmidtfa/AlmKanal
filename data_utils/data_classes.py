from attrs import define
from typing import TypedDict

@define
class info_class:

    '''Container for the info field of the AlmKanal workflow'''

    raw: bool = False
    epoched: bool = False
    maxwell: dict | None = None
    ica: dict | None = None
    epoched: dict | None = None
    trf_epochs: dict | None = None


class pick_dict_class(TypedDict):

    '''Container for data picking'''

    meg: bool = True 
    eog: bool = True
    ecg: bool = True
    eeg: bool = False
    stim: bool = True 