
from almkanal.almkanal_functions import AlmkanalRaw, do_ica
import pytest


def test_ica_eog(gen_mne_data_raw):

    raw, data_path = gen_mne_data_raw
    raw = raw.pick(picks=['meg', 'stim']).copy()  

    ak_raw = AlmkanalRaw.from_mne_raw(raw=raw)
     
    do_ica(ak_raw,
           n_components=50,
            train=False,
            eog=True,
            surrogate_eog_chs = {'left_eog_chs': ['MEG 0121','MEG 0311'],
                                    'right_eog_chs': ['MEG 1211' ,'MEG 1411'],},
            ecg=True,
            emg=True,
            resample_freq=100,
            )
