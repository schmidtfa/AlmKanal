from almkanal.almkanal import AlmKanal
import pytest
from .settings import CH_PICKS, ICA_TRAIN, ICA_EOG, ICA_ECG, ICA_THRESH, ICA_RESAMPLE, ICA_NCOMPS, SOURCE
import mne



@pytest.mark.parametrize('ch_picks', CH_PICKS, scope='session')
def test_src(gen_mne_data_raw, ch_picks): #, ch_picks
    data_path = mne.datasets.sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'
    ak = AlmKanal(raw=gen_mne_data_raw)

    fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    ak.pick_dict['meg'] = ch_picks
    ak.fwd = fwd
    if ch_picks:
        ak.do_src(empty_room_path=raw_fname)
    else:
        ak.do_src()


def test_maxwell(gen_mne_data_raw):

    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_maxwell()


#@pytest.mark.parametrize('resample_freq', ICA_RESAMPLE, scope='session')
@pytest.mark.parametrize('ecg', ICA_ECG, scope='session')
@pytest.mark.parametrize('eog', ICA_EOG, scope='session')
#@pytest.mark.parametrize('threshold', ICA_THRESH, scope='session')
@pytest.mark.parametrize('train', ICA_TRAIN, scope='session')
#@pytest.mark.parametrize('n_components', ICA_NCOMPS, scope='session')
def test_ica(gen_mne_data_raw, train, eog, ecg):

    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_ica(n_components=10,
              train=train,
              eog=eog,
              ecg=ecg,
              resample_freq=200,
              threshold=0.4
              )
    

def test_double_ica(gen_mne_data_raw):

    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_ica(n_components=10,
              train=False,
              eog=True,
              ecg=False,
              resample_freq=100,
              threshold=0.4
              )
    
    ak.do_ica(n_components=10,
              train=False,
              eog=True,
              ecg=False,
              resample_freq=100,
              threshold=0.4
              )


def test_ica_plot(gen_mne_data_raw):
    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_ica(n_components=40,
              train=False,
              eog=True,
              ecg=True,
              resample_freq=100,
              threshold=0.4,
              fname='test',
              img_path='./'
              )
    


def test_epoching(gen_mne_data_raw):
    ak = AlmKanal(raw=gen_mne_data_raw)

    ak.do_events()

    event_dict = {
    'Auditory/Left': 1,
    'Auditory/Right': 2,
    'Visual/Left': 3,
    'Visual/Right': 4,
    }

    ak.do_epochs(tmin=-0.2, tmax=0.5, event_id=event_dict)

@pytest.mark.parametrize('source, atlas', SOURCE, scope='session')
def test_fwd(gen_mne_data_raw, source, atlas):
    ak = AlmKanal(raw=gen_mne_data_raw)
    ak.do_fwd_model(subject_id='sample',
                    subjects_dir='./data_old/')
    
    ak.pick_dict['meg'] = 'mag'
    ak.do_src(subject_id = 'sample',
              subjects_dir = './data_old/',
              source=source,
              atlas=atlas,
              return_parc=True,)
    
