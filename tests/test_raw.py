from almkanal.almkanal import AlmKanal
import pytest
from .settings import CH_PICKS, ICA_TRAIN, ICA_EOG, ICA_ECG, ICA_THRESH, ICA_RESAMPLE, ICA_NCOMPS
import mne

def test_maxwell(gen_mne_data_raw):

    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_maxwell()


@pytest.mark.parametrize('resample_freq', ICA_RESAMPLE, scope='session')
@pytest.mark.parametrize('ecg', ICA_ECG, scope='session')
@pytest.mark.parametrize('eog', ICA_EOG, scope='session')
@pytest.mark.parametrize('threshold', ICA_THRESH, scope='session')
@pytest.mark.parametrize('train', ICA_TRAIN, scope='session')
#@pytest.mark.parametrize('n_components', ICA_NCOMPS, scope='session')
def test_ica(gen_mne_data_raw, train, eog, ecg, resample_freq, threshold):

    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_ica(n_components=10,
              train=train,
              eog=eog,
              ecg=ecg,
              resample_freq=resample_freq,
              threshold=threshold
              )

# @pytest.mark.parametrize('ch_picks', CH_PICKS, scope='session')
# def test_src(gen_mne_data_raw, ch_picks):
#     data_path = mne.datasets.sample.data_path()
#     meg_path = data_path / 'MEG' / 'sample'
#     ak = AlmKanal(raw=gen_mne_data_raw)

#     fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
#     fwd = mne.read_forward_solution(fwd_fname)
#     ak.pick_dict['meg'] = ch_picks
#     ak.fwd = fwd
#     if ch_picks:
#         ak.do_src(noise_cov=mne.make_ad_hoc_cov(ak.raw.info))
#     else:
#         ak.do_src()