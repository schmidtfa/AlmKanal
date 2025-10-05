from almkanal import (AlmKanal, Maxwell, ICA, ForwardModel, 
                      SpatialFilter, SourceReconstruction, PhysioCleaner,
                      RANSAC, ReReference, Filter, Resample)
import pytest
from .settings import CH_PICKS, ICA_TRAIN, ICA_EOG, ICA_ECG, ICA_THRESH, ICA_RESAMPLE, ICA_NCOMPS, SOURCE_SURF, SOURCE_VOL
import mne

#@pytest.mark.parametrize('ch_picks', CH_PICKS, scope='session')

def test_ransac(gen_mne_data_raw_eeg):

    raw, data_path = gen_mne_data_raw_eeg

    ak = AlmKanal(steps=[RANSAC(),
                         Filter(),
                         ReReference(),
                         Resample(100)])
    ak.run(raw)



def test_maxwell(gen_mne_data_raw):

    raw, data_path = gen_mne_data_raw

    ak = AlmKanal(steps=[Maxwell()])
    ak.run(raw)


#@pytest.mark.parametrize('resample_freq', ICA_RESAMPLE, scope='session')
@pytest.mark.parametrize('ecg', ICA_ECG, scope='session')
@pytest.mark.parametrize('eog', ICA_EOG, scope='session')
#@pytest.mark.parametrize('threshold', ICA_THRESH, scope='session')
@pytest.mark.parametrize('train', ICA_TRAIN, scope='session')
#@pytest.mark.parametrize('n_components', ICA_NCOMPS, scope='session')
def test_ica(gen_mne_data_raw, train, eog, ecg):

    raw, data_path = gen_mne_data_raw

    ak = AlmKanal(steps=[ICA(n_components=10,
                        train=train,
                        eog=eog,
                        surrogate_eog_chs=None,
                        ecg=ecg,
                        emg=True,
                        resample_freq=100,)])
        
    ak.run(raw)



@pytest.mark.parametrize('source, atlas', SOURCE_SURF, scope='session')
def test_fwd(gen_mne_data_raw, source, atlas):
    raw, data_path = gen_mne_data_raw

    pick_dict = {
        'meg': 'mag',
        'eog': False,
        'ecg': False,
        'eeg': False,
        'stim': False,
    }

    ak = AlmKanal(steps=[ForwardModel(pick_dict=pick_dict,
                                      subject_id='sample',
                                      subjects_dir='./data_old/',
                                      source=source),
                        SpatialFilter(pick_dict=pick_dict),
                        SourceReconstruction(subject_id = 'sample',
                                            subjects_dir = './data_old/',
                                            source=source,
                                            atlas=atlas,
                                            return_parc=True,)])

    ak.run(raw)
    

def test_src(): #, ch_picks
    
    data_path = mne.datasets.sample.data_path()

    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'
    #fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'

    raw = mne.io.read_raw_fif(raw_fname, preload=True)#.crop(tmin=0, tmax=60)

    #raw = raw.pick(picks=['meg', 'eog', 'stim'])

    
    #fwd = mne.read_forward_solution(fwd_fname)

    pick_dict = {
        'meg': 'mag',
        'eog': True,
        'ecg': False,
        'eeg': False,
        'stim': False,
    }
    
    ak = AlmKanal(pick_params=pick_dict,
                  steps=[Maxwell(),
                         ICA(),
                         ForwardModel(subject_id='sample', 
                                      subjects_dir='./data_old', 
                                      redo_hdm=True),
                         SpatialFilter( empty_room=raw.copy()),
                         SourceReconstruction(source='volume'),
                         ])
    
    ak.run(raw)


@pytest.mark.parametrize('source, atlas', SOURCE_VOL, scope='session')
def test_ad_hoc_cov(gen_mne_data_raw, source, atlas):
    raw, data_path = gen_mne_data_raw

    pick_dict = {
        'meg': True,
        'eog': False,
        'ecg': False,
        'eeg': False,
        'stim': False,
    }

    ak = AlmKanal(steps=[ForwardModel(
                                pick_dict=pick_dict,
                                subject_id='sample',
                                subjects_dir='./data_old/',
                                source=source),
                 SpatialFilter(pick_dict=pick_dict),
                 SourceReconstruction(subject_id = 'sample',
                                        subjects_dir = './data_old/',
                                        source=source,
                                        atlas=atlas,
                                        return_parc=True,)])
    ak.run(raw)


    ak.generate_json()


