import mne
from almkanal import AlmKanal, SpatialFilter, SourceReconstruction


def test_src(gen_mne_data_epochs): 

    data_path = mne.datasets.sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    
    pick_dict = {
    'meg': 'mag',
    'eog': False,
    'ecg': False,
    'eeg': False,
    'stim': False,
}

    ak = AlmKanal(steps=[
                         SpatialFilter(fwd=fwd, pick_dict=pick_dict),
                         SourceReconstruction()])

    ak.run(gen_mne_data_epochs)
