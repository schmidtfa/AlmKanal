import mne
from almkanal.almkanal_tng import AlmkanalRaw
import numpy as np

def test_convert_raw_to_almkanal():
    # get fif file from mne testing
    raw = mne.io.read_raw_fif(mne.datasets.testing.data_path() / 'MEG/sample/sample_audvis_trunc_raw.fif')
    alm_raw = AlmkanalRaw.from_mne_raw(raw)
    assert isinstance(alm_raw, AlmkanalRaw)
    assert isinstance(alm_raw.report, mne.Report)
    assert alm_raw.info == raw.info
    assert np.all(alm_raw.get_data() == raw.get_data())