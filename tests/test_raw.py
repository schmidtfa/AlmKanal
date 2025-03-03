from almkanal.almkanal_functions import AlmkanalRaw, do_maxwell, do_ica, do_fwd_model
import pytest
from .settings import CH_PICKS, ICA_TRAIN, ICA_EOG, ICA_ECG, ICA_THRESH, ICA_RESAMPLE, ICA_NCOMPS, SOURCE_SURF, SOURCE_VOL
import mne

#@pytest.mark.parametrize('ch_picks', CH_PICKS, scope='session')

def test_maxwell(gen_mne_data_raw):

    raw, data_path = gen_mne_data_raw

    ak_raw = AlmkanalRaw.from_mne_raw(raw=raw)
    ak_raw = do_maxwell(ak_raw)


#@pytest.mark.parametrize('resample_freq', ICA_RESAMPLE, scope='session')
@pytest.mark.parametrize('ecg', ICA_ECG, scope='session')
@pytest.mark.parametrize('eog', ICA_EOG, scope='session')
#@pytest.mark.parametrize('threshold', ICA_THRESH, scope='session')
@pytest.mark.parametrize('train', ICA_TRAIN, scope='session')
#@pytest.mark.parametrize('n_components', ICA_NCOMPS, scope='session')
def test_ica(gen_mne_data_raw, train, eog, ecg):

    raw, data_path = gen_mne_data_raw

    ak_raw = AlmkanalRaw.from_mne_raw(raw=raw)
        
    do_ica(ak_raw,
        n_components=10,
        train=train,
        eog=eog,
        surrogate_eog_chs=None,
        ecg=ecg,
        emg=True,
        resample_freq=100,
        )


@pytest.mark.parametrize('source, atlas', SOURCE_SURF, scope='session')
def test_fwd(gen_mne_data_raw, source, atlas):
    raw, data_path = gen_mne_data_raw

    ak_raw = AlmkanalRaw.from_mne_raw(raw=raw)

    fwd = do_fwd_model(ak_raw,
                        subject_id='sample',
                        subjects_dir='./data_old/',
                        source=source)

    
    ak.pick_dict['meg'] = 'mag'
    do_spatial_filters()
    ak.do_src(subject_id = 'sample',
              subjects_dir = './data_old/',
              source=source,
              atlas=atlas,
              return_parc=True,)
    

def test_src(): #, ch_picks
    
    data_path = mne.datasets.sample.data_path()

    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)#.crop(tmin=0, tmax=60)

    raw = raw.pick(picks=['meg', 'eog', 'stim'])

    ak = AlmKanal(raw=raw.copy())
    meg_path = data_path / 'MEG' / 'sample'

    ak.do_maxwell()
    # % you can always use common mne methods like filtering that modify
    # the raw and epoched objects in place
    ak.raw.filter(l_freq=0.1, h_freq=100)
    #  one shot call to ica
    ak.do_ica()

    fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    ak.pick_dict['meg'] = True
    ak.fwd = fwd
    ak.do_spatial_filters(empty_room=raw.copy())
    ak.do_src(source='volume')
    

@pytest.mark.parametrize('source, atlas', SOURCE_VOL, scope='session')
def test_ad_hoc_cov(gen_mne_data_raw, source, atlas):
    raw, data_path = gen_mne_data_raw

    ak = AlmKanal(raw=raw)


    ak.do_fwd_model(subject_id='sample',
                    subjects_dir='./data_old/',
                    source=source)

    
    ak.pick_dict['meg'] = True
    ak.do_spatial_filters()
    ak.do_src(subject_id = 'sample',
              subjects_dir = './data_old/',
              source=source,
              atlas=atlas,
              return_parc=True,)
    

def test_bio(gen_mne_data_raw):

    raw, _ = gen_mne_data_raw
    ak = AlmKanal(raw=raw)

    ak.do_bio_process(eog='EOG 061')


